"""POST /guide — model-driven serial tool-calling loop."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from adapter.clock import SystemClock
from adapter.errors import AdapterError
from adapter.limits import LimitTracker, RunLimits
from adapter.recovery import chat_with_recovery
from adapter.runtime import current_model
from adapter.tokens import EVIDENCE_BUDGET, count_text
from app.errors import ValidationRejection
from app.errors import adapter_error_to_http
from app.prompt_loader import load_prompt
from app.schemas import GuideRequest, GuideResponse
from app.tools.context import ToolContext
from app.tools.dispatch import ToolDispatcher
from app.tools.schemas import TOOL_SCHEMAS
from ingest.chapters import load_chapter_map
from traces.schema import JSONLWriter, TraceEvent


router = APIRouter()
_TRACE_ROOT = Path("traces/task2")
_MAX_TOOL_TURNS = 10
#: Cap a single tool-result payload so guide turns honour the 8k evidence budget.
_MAX_TOOL_RESULT_CHARS = 6_000

_SYSTEM_PROMPT = load_prompt("guide_system.txt")


@router.post("/guide", response_model=GuideResponse)
def guide(
    request: GuideRequest,
    trace_check_name: str = Query("adhoc"),
) -> GuideResponse:
    """Run the actual model-driven tool loop until a pending draft or a
    terminal structured error is reached.

    ``completed`` is reserved for a genuinely finished goal. Ending the
    model turn without a valid pending draft after rejected saves is an
    ``error``, not a successful completion.
    """
    run_id = uuid.uuid4().hex
    is_task3 = trace_check_name.startswith("R")
    task = "t3" if is_task3 else "t2"
    trace_root = Path("traces/task3") if is_task3 else _TRACE_ROOT
    trace_path = trace_root / f"{trace_check_name}_{request.request_id}.jsonl"
    trace_path.unlink(missing_ok=True)
    writer = JSONLWriter()
    started = time.monotonic()

    chapter_map = load_chapter_map()
    corpus_max = len(chapter_map)
    if request.max_chapter > corpus_max:
        writer.append(
            trace_path,
            TraceEvent(
                request_id=request.request_id,
                run_id=run_id,
                task=task,  # type: ignore[arg-type]
                operation="validation",
                outcome="rejected",
                notes={
                    "reason": "max_chapter_out_of_range",
                    "max_chapter": request.max_chapter,
                    "corpus_max_chapter": corpus_max,
                },
            ),
        )
        raise ValidationRejection(
            code="max_chapter_out_of_range",
            message=(
                f"max_chapter must be between 1 and {corpus_max} "
                f"(got {request.max_chapter})"
            ),
            detail={"corpus_max_chapter": corpus_max},
        )

    context = ToolContext(
        request_id=request.request_id,
        max_chapter=request.max_chapter,
    )
    dispatcher = ToolDispatcher(context=context)
    model = current_model()
    tracker = getattr(model, "tracker", None) or LimitTracker(
        limits=RunLimits(),
        clock=getattr(model, "clock", None) or SystemClock(),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Goal: {request.goal}\n"
                f"Application chapter limit (inclusive): {request.max_chapter}"
            ),
        },
    ]

    rejected_tool_attempts = 0
    rejected_save_attempts = 0
    seen_chapter_ids: set[int] = set()

    for turn in range(_MAX_TOOL_TURNS):
        try:
            recovery = chat_with_recovery(
                model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                response_format=None,
                max_output_tokens=1_500,
                temperature=0.0,
                parallel_tool_calls=False,
                operation_id=f"guide:{request.request_id}:turn:{turn}",
                task=task,
                request_id=request.request_id,
                run_id=run_id,
                trace_path=trace_path,
                writer=writer,
                tracker=tracker,
                source_chapter_ids=sorted(seen_chapter_ids),
                reducer=_guide_context_reducer,
            )
        except AdapterError as exc:
            raise adapter_error_to_http(exc) from exc
        messages = recovery.messages
        response = recovery.response

        if not response.tool_calls:
            # Model stopped without a pending draft. That is never a
            # successful guide completion for this endpoint.
            writer.append(
                trace_path,
                TraceEvent(
                    request_id=request.request_id,
                    run_id=run_id,
                    task=task,  # type: ignore[arg-type]
                    operation="validation",
                    attempt=turn + 1,
                    source_chapter_ids=sorted(seen_chapter_ids),
                    outcome="error",
                    stubbed=response.stubbed,
                    notes={
                        "reason": "ended_without_pending_draft",
                        "rejected_tool_attempts": rejected_tool_attempts,
                        "rejected_save_attempts": rejected_save_attempts,
                    },
                ),
            )
            return GuideResponse(
                request_id=request.request_id,
                status="error",
                error={
                    "code": "guide_incomplete",
                    "message": (
                        "model ended without creating a valid pending "
                        "draft for approval"
                    ),
                    "rejected_tool_attempts": rejected_tool_attempts,
                    "rejected_save_attempts": rejected_save_attempts,
                },
                metrics={
                    "turns": turn + 1,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "stubbed": response.stubbed,
                    "rejected_tool_attempts": rejected_tool_attempts,
                    "rejected_save_attempts": rejected_save_attempts,
                },
            )

        assistant_message = {
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls,
        }
        messages.append(assistant_message)

        for call in response.tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name", ""))
            call_id = str(call.get("id", f"call_{turn}"))
            raw_arguments = function.get("arguments", {})
            arguments = _parse_arguments(raw_arguments)
            tracker.pre_call(op_id=f"tool:{call_id}")
            result = dispatcher.dispatch(
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
            chapter_ids = _chapters_from_tool_result(result.payload)
            seen_chapter_ids.update(chapter_ids)
            if not result.ok:
                rejected_tool_attempts += 1
                if name == "save_guide":
                    rejected_save_attempts += 1
            writer.append(
                trace_path,
                TraceEvent(
                    request_id=request.request_id,
                    run_id=run_id,
                    task=task,  # type: ignore[arg-type]
                    operation="tool_call",
                    attempt=turn + 1,
                    source_chapter_ids=chapter_ids or sorted(seen_chapter_ids),
                    outcome="ok" if result.ok else "rejected",
                    stubbed=response.stubbed,
                    notes={
                        "tool_name": name,
                        "call_id": call_id,
                        "arguments": arguments,
                        "result": result.payload,
                        "rejected_tool_attempts": rejected_tool_attempts,
                        "rejected_save_attempts": rejected_save_attempts,
                    },
                ),
            )
            tool_content = _bounded_tool_content(result.content())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": tool_content,
                }
            )
            if not result.ok and name == "save_guide":
                # Mirror /answer citation correction: give the model an
                # explicit recovery instruction instead of hoping it
                # infers next steps from a raw tool error alone.
                messages.append(
                    {
                        "role": "user",
                        "content": _save_guide_correction_prompt(result.payload),
                    }
                )

            if result.pending:
                pending = result.payload
                return GuideResponse(
                    request_id=request.request_id,
                    status="pending_approval",
                    operation_id=pending["operation_id"],
                    draft={
                        **pending["draft"],
                        "payload_hash": pending["payload_hash"],
                    },
                    metrics={
                        "turns": turn + 1,
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                        "stubbed": response.stubbed,
                        "rejected_tool_attempts": rejected_tool_attempts,
                        "rejected_save_attempts": rejected_save_attempts,
                    },
                )

    writer.append(
        trace_path,
        TraceEvent(
            request_id=request.request_id,
            run_id=run_id,
            task=task,  # type: ignore[arg-type]
            operation="limit_check",
            attempt=_MAX_TOOL_TURNS,
            outcome="error",
            recovery_decision="limit_exhausted",
            source_chapter_ids=sorted(seen_chapter_ids),
            notes={
                "max_tool_turns": _MAX_TOOL_TURNS,
                "rejected_tool_attempts": rejected_tool_attempts,
                "rejected_save_attempts": rejected_save_attempts,
            },
        ),
    )
    return GuideResponse(
        request_id=request.request_id,
        status="error",
        error={
            "code": "tool_loop_exhausted",
            "message": "model tool loop reached its finite turn limit",
            "rejected_tool_attempts": rejected_tool_attempts,
            "rejected_save_attempts": rejected_save_attempts,
        },
        metrics={
            "turns": _MAX_TOOL_TURNS,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "rejected_tool_attempts": rejected_tool_attempts,
            "rejected_save_attempts": rejected_save_attempts,
        },
    )


def _chapters_from_tool_result(payload: dict[str, Any]) -> list[int]:
    """Collect chapter IDs actually present in a tool result payload."""
    chapters: set[int] = set()
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("chapter"), int):
        chapters.add(payload["chapter"])
    for key in ("results", "citations"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("chapter"), int):
                chapters.add(item["chapter"])
    draft = payload.get("draft")
    if isinstance(draft, dict):
        for citation in draft.get("citations") or []:
            if isinstance(citation, dict) and isinstance(citation.get("chapter"), int):
                chapters.add(citation["chapter"])
    return sorted(chapters)


def _parse_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _bounded_tool_content(content: str) -> str:
    """Keep individual tool results within the evidence budget spirit."""
    if count_text(content) <= EVIDENCE_BUDGET and len(content) <= _MAX_TOOL_RESULT_CHARS:
        return content
    clipped = content[:_MAX_TOOL_RESULT_CHARS]
    return clipped + "\n...[truncated to enforce evidence budget]..."


def _save_guide_correction_prompt(payload: dict[str, Any]) -> str:
    detail = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return (
        "save_guide was rejected. Fix the draft and call save_guide again. "
        "Requirements: at least one citation; chunk_id/source_filename/"
        "chapter/pdf_pages must match a previously fetched passage exactly; "
        "excerpt must be a non-empty verbatim substring copied from that "
        "passage's content (do not paraphrase). Tool error payload: "
        f"{detail}"
    )


def _guide_context_reducer(
    messages: list[dict[str, Any]],
    reported_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shrink oversized guide histories by truncating older tool blobs.

    Preserves the system prompt, the original goal message, and the
    most recent tool/assistant turns so the model can still recover.
    """
    from adapter.tokens import (
        count_messages,
        count_output_reservation,
        count_tool_schemas,
    )

    if len(messages) <= 2:
        return messages, {"trimmed": False, "reason": "nothing_to_trim"}

    reduced = [dict(msg) for msg in messages]
    truncated = 0
    # Truncate oldest tool contents first (skip system + first user).
    for idx in range(2, len(reduced)):
        msg = reduced[idx]
        if msg.get("role") != "tool":
            continue
        content = msg.get("content") or ""
        if not isinstance(content, str) or len(content) <= 400:
            continue
        reduced[idx] = {
            **msg,
            "content": content[:400] + "\n...[truncated for context recovery]...",
        }
        truncated += 1
        total = (
            count_messages(reduced)
            + count_tool_schemas(TOOL_SCHEMAS)
            + count_output_reservation(1_500)
        )
        if total <= reported_limit:
            return reduced, {
                "trimmed": True,
                "truncated_tool_messages": truncated,
                "total_tokens_after": total,
            }

    # Last resort: drop middle turns, keep system, first user, last 4.
    if len(reduced) > 6:
        kept = reduced[:2] + reduced[-4:]
        return kept, {
            "trimmed": True,
            "dropped_middle_messages": len(reduced) - len(kept),
            "truncated_tool_messages": truncated,
        }

    # Even if still over the limit, return truncated content when we
    # changed anything — never discard useful reduction work.
    if truncated:
        total = (
            count_messages(reduced)
            + count_tool_schemas(TOOL_SCHEMAS)
            + count_output_reservation(1_500)
        )
        return reduced, {
            "trimmed": True,
            "truncated_tool_messages": truncated,
            "total_tokens_after": total,
            "still_over_limit": total > reported_limit,
        }
    return messages, {"trimmed": False, "truncated_tool_messages": 0}


__all__ = [
    "router",
    "_bounded_tool_content",
    "_guide_context_reducer",
    "_save_guide_correction_prompt",
]
