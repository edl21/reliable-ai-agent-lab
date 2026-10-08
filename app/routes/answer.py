"""POST /answer — grounded question answering over the corpus.

Flow (each step traced as one ``TraceEvent``):

1. **Pre-generation validation.** ``AnswerRequest`` is pydantic-parsed
   before we enter this function; here we additionally check
   ``max_chapter <= corpus_max_chapter``. Failure raises 422 via
   ``app/errors.py::ValidationRejection`` — critically, BEFORE any
   model or embedding call. This satisfies the source's required
   check on malformed ``max_chapter``.

2. **Retrieval.** ``ingest.index.retrieve`` returns top-k chunks
   filtered by ``$lte max_chapter`` in Chroma AND by the chapter
   guard in Python (defence in depth).

3. **Evidence assembly + token budgeting.** We build the model
   messages, count tokens with ``adapter/tokens.py``, and enforce
   both the 8,000 evidence-tokens budget and the total-context
   limit BEFORE calling the model. If evidence overflows, we drop
   the lowest-score chunk and re-check — bounded by the top-k.

4. **Model call.** Structured JSON via ``response_format`` — the model
   MUST return the schema in ``app/schemas.py``.

5. **Authoritative answer verification** (schema + grounding):
   a) Every ``[chunk_id]`` in ``answer`` appears in ``citations``.
   b) Every citation belongs to the retrieved evidence set and
      matches that chunk's ``source_filename``, ``chapter``, and
      allowed ``pdf_pages``; ``excerpt`` is a non-empty verbatim
      NFC substring of the chunk text with a minimum length.
   c) ``status=answered`` requires at least one citation, at least
      one inline ``[chunk_id]``, and an inline citation on every
      substantive sentence (no invented uncited claims).
   d) Contradiction positions obey the same membership/excerpt rules.
   e) ``insufficient_evidence`` responses carry zero citations.
   On failure the route re-prompts with the concrete errors within
   the shared bounded model-call budget. If correction still fails,
   it returns ``insufficient_evidence`` and never a successful
   answer with known-bad or missing citations.

6. **Trace** the full ordered message list (credentials omitted) to
   ``traces/task1/<any-caller-set-check-name>_<request_id>.jsonl``.
   Callers control the filename via the ``trace_check_name`` query
   param so each ``checks/task1_*.py`` script produces a
   convention-conforming path.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from pydantic import ValidationError

from adapter.clock import SystemClock
from adapter.errors import AdapterError
from adapter.limits import LimitTracker, RunLimits
from adapter.recovery import chat_with_recovery
from adapter.runtime import current_model
from adapter.tokens import (
    EVIDENCE_BUDGET,
    TOTAL_CONTEXT_LIMIT_DEFAULT,
    count_messages,
    count_output_reservation,
    count_text,
)
from app.errors import ValidationRejection, adapter_error_to_http
from app.prompt_loader import load_prompt
from app.schemas import (
    AnswerMetrics,
    AnswerRequest,
    AnswerResponse,
    Citation,
    Contradiction,
    ModelAnswer,
    model_response_json_schema,
)
from ingest.chapter_guard import enforce_chapter_limit
from ingest.chapters import load_chapter_map
from ingest.index import RetrievedChunk, retrieve
from traces.schema import (
    ContextCounts,
    JSONLWriter,
    TraceEvent,
)


router = APIRouter()

_TRACE_ROOT = Path("traces/task1")

# Reserved output tokens for the model's structured JSON response.
# Chosen so an "answered" response with citations and one or two
# contradictions fits comfortably; documented in README.
_MAX_OUTPUT_TOKENS: int = 1_500

_SYSTEM_PROMPT: str = load_prompt("answer_system.txt")


@router.post("/answer", response_model=AnswerResponse)
def answer(  # noqa: PLR0915 — end-to-end orchestration; splitting would obscure the flow
    request: AnswerRequest,
    trace_check_name: str = Query(
        "adhoc",
        description=(
            "Names the trace file: traces/task1/<trace_check_name>_"
            "<request_id>.jsonl. Callers use their check-script name."
        ),
    ),
) -> AnswerResponse:
    t0 = time.monotonic()
    is_task3 = trace_check_name.startswith("R")
    task = "t3" if is_task3 else "t1"
    trace_root = Path("traces/task3") if is_task3 else _TRACE_ROOT
    trace_path = trace_root / f"{trace_check_name}_{request.request_id}.jsonl"
    run_id = uuid.uuid4().hex
    # A request ID identifies the logical request; a run ID identifies
    # this execution. Clear an old same-name trace so repeating a
    # manual/check request does not append a second run to the first
    # run's evidence file.
    trace_path.unlink(missing_ok=True)
    writer = JSONLWriter()

    # --- Step 1: pre-generation validation ------------------------
    chapter_map = load_chapter_map()
    corpus_max = len(chapter_map)
    if request.max_chapter > corpus_max:
        writer.append(
            trace_path,
            TraceEvent(
                request_id=request.request_id,
                run_id=run_id,
                task=task,
                operation="validation",
                attempt=0,
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
                f"max_chapter must be between 1 and {corpus_max} (got "
                f"{request.max_chapter})"
            ),
            detail={"corpus_max_chapter": corpus_max},
        )

    # --- Step 2: retrieval ---------------------------------------
    t_retrieval_start = time.monotonic()
    retrieved = retrieve(request.question, request.max_chapter, top_k=8)
    retrieval_ms = int((time.monotonic() - t_retrieval_start) * 1000)
    writer.append(
        trace_path,
        TraceEvent(
            request_id=request.request_id,
            run_id=run_id,
            task=task,
            operation="retrieval",
            attempt=0,
            elapsed_time_ms=retrieval_ms,
            source_chapter_ids=sorted({c.chapter for c in retrieved}),
            outcome="ok",
            notes={
                "top_k": 8,
                "retrieved_count": len(retrieved),
                "chunk_ids": [c.chunk_id for c in retrieved],
            },
        ),
    )

    # --- Step 3: evidence assembly + token budgeting -------------
    evidence, evidence_tokens, total_tokens = _assemble_evidence(
        retrieved, request.question
    )
    writer.append(
        trace_path,
        TraceEvent(
            request_id=request.request_id,
            run_id=run_id,
            task=task,
            operation="limit_check",
            attempt=0,
            source_chapter_ids=sorted({c.chapter for c in evidence}),
            context_counts=ContextCounts(
                evidence_tokens=evidence_tokens,
                total_tokens=total_tokens,
                output_tokens_reserved=_MAX_OUTPUT_TOKENS,
            ),
            outcome="ok",
            notes={
                "evidence_budget": EVIDENCE_BUDGET,
                "total_context_limit": TOTAL_CONTEXT_LIMIT_DEFAULT,
                "surviving_chunk_ids": [c.chunk_id for c in evidence],
            },
        ),
    )

    # --- Step 4+5: model call with authoritative verification ------
    messages = _build_messages(request.question, evidence, request.max_chapter)
    model = current_model()
    tracker = getattr(model, "tracker", None) or LimitTracker(
        limits=RunLimits(),
        clock=getattr(model, "clock", None) or SystemClock(),
    )
    reducer = _answer_context_reducer(
        question=request.question,
        max_chapter=request.max_chapter,
        evidence=evidence,
    )
    operation_id = f"answer:{request.request_id}"
    chapter_ids = sorted({c.chapter for c in evidence})
    response_format = {
        "type": "json_schema",
        "json_schema": model_response_json_schema(),
    }

    t_model_start = time.monotonic()
    model_answer: ModelAnswer | None = None
    resp: Any = None
    validation_attempt = 0
    max_validation_rounds = tracker.limits.attempts_per_op

    while validation_attempt < max_validation_rounds:
        try:
            recovery = chat_with_recovery(
                model,
                messages=messages,
                tools=None,
                response_format=response_format,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,
                parallel_tool_calls=None,
                operation_id=operation_id,
                task=task,
                request_id=request.request_id,
                run_id=run_id,
                trace_path=trace_path,
                writer=writer,
                tracker=tracker,
                reducer=reducer,
                source_chapter_ids=chapter_ids,
            )
        except AdapterError as exc:
            raise adapter_error_to_http(exc) from exc
        messages = recovery.messages
        resp = recovery.response
        validation_attempt += 1

        try:
            parsed = json.loads(resp.content or "{}")
            candidate = ModelAnswer.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            writer.append(
                trace_path,
                TraceEvent(
                    request_id=request.request_id,
                    run_id=run_id,
                    task=task,
                    operation="validation",
                    attempt=validation_attempt,
                    outcome="schema_invalid",
                    recovery_decision="schema_reprompt",
                    stubbed=resp.stubbed,
                    notes={
                        "stage": "schema",
                        "raw_output": resp.content,
                        "error": str(exc),
                        "corrective_action": "schema_reprompt",
                        "will_retry": validation_attempt < max_validation_rounds,
                    },
                ),
            )
            if validation_attempt >= max_validation_rounds:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed schema validation: "
                        f"{exc}. Return only a valid answer JSON object "
                        "matching the required schema."
                    ),
                },
            ]
            continue

        verifier_notes = _run_post_hoc_verifiers(
            candidate,
            evidence,
            request.max_chapter,
        )
        errors = verifier_notes.get("verifier_errors") or []
        if not errors:
            writer.append(
                trace_path,
                TraceEvent(
                    request_id=request.request_id,
                    run_id=run_id,
                    task=task,
                    operation="validation",
                    attempt=validation_attempt,
                    outcome="ok",
                    stubbed=resp.stubbed,
                    notes=verifier_notes,
                ),
            )
            model_answer = candidate
            break

        writer.append(
            trace_path,
            TraceEvent(
                request_id=request.request_id,
                run_id=run_id,
                task=task,
                operation="validation",
                attempt=validation_attempt,
                outcome="error",
                recovery_decision=(
                    "schema_reprompt"
                    if validation_attempt < max_validation_rounds
                    else "retries_exhausted"
                ),
                stubbed=resp.stubbed,
                notes={
                    **verifier_notes,
                    "stage": "grounding",
                    "corrective_action": (
                        "citation_reprompt"
                        if validation_attempt < max_validation_rounds
                        else "insufficient_evidence_fallback"
                    ),
                    "will_retry": validation_attempt < max_validation_rounds,
                },
            ),
        )
        if validation_attempt >= max_validation_rounds:
            break
        messages = [
            *messages,
            {"role": "assistant", "content": resp.content},
            {
                "role": "user",
                "content": _citation_correction_prompt(errors),
            },
        ]

    model_ms = int((time.monotonic() - t_model_start) * 1000)
    if resp is None:
        raise ValidationRejection(
            code="model_call_failed",
            message="model call produced no response",
        )

    # Authoritative gate: never return status=answered with known-bad
    # citations. Exhausted correction budget → structured insufficient.
    if model_answer is None:
        writer.append(
            trace_path,
            TraceEvent(
                request_id=request.request_id,
                run_id=run_id,
                task=task,
                operation="validation",
                attempt=validation_attempt,
                outcome="insufficient_evidence",
                recovery_decision="retries_exhausted",
                stubbed=resp.stubbed,
                notes={
                    "reason": "authoritative_verification_failed",
                    "validation_rounds": validation_attempt,
                    "corrective_action": "insufficient_evidence_fallback",
                },
            ),
        )
        model_answer = ModelAnswer(
            status="insufficient_evidence",
            answer=(
                "Unable to produce a grounded answer with verifiable "
                "citations from the provided evidence."
            ),
            citations=[],
            contradictions=[],
        )

    # --- Step 6: build response ----------------------------------
    total_ms = int((time.monotonic() - t0) * 1000)
    return AnswerResponse(
        request_id=request.request_id,
        status=model_answer.status,
        answer=model_answer.answer,
        citations=model_answer.citations,
        contradictions=model_answer.contradictions,
        metrics=AnswerMetrics(
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            model_ms=model_ms,
            evidence_tokens=evidence_tokens,
            total_tokens=total_tokens,
            input_tokens=resp.usage.get("input_tokens"),
            output_tokens=resp.usage.get("output_tokens"),
            stubbed=resp.stubbed,
        ),
    )


# --- helpers --------------------------------------------------------------


def _answer_context_reducer(
    *,
    question: str,
    max_chapter: int,
    evidence: list[RetrievedChunk],
):
    """Return a reducer that learns R4's reported limit and removes only
    evidence, preserving the system instructions and chapter boundary."""

    def reduce(
        _current_messages: list[dict[str, Any]],
        reported_limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        reduced = enforce_chapter_limit(
            evidence,
            max_chapter,
            source="R4_reducer_output",
        )
        while reduced:
            candidate = _build_messages(question, reduced, max_chapter)
            total = count_messages(candidate) + count_output_reservation(
                _MAX_OUTPUT_TOKENS
            )
            if total <= reported_limit:
                return candidate, {
                    "retained_chunk_ids": [chunk.chunk_id for chunk in reduced],
                    "evidence_tokens_after": count_text(
                        _format_evidence_block(reduced)
                    ),
                    "total_tokens_after": total,
                }
            reduced.pop()

        candidate = _build_messages(question, [], max_chapter)
        total = count_messages(candidate) + count_output_reservation(
            _MAX_OUTPUT_TOKENS
        )
        if total <= reported_limit:
            return candidate, {
                "retained_chunk_ids": [],
                "evidence_tokens_after": 0,
                "total_tokens_after": total,
            }
        return _current_messages, {
            "retained_chunk_ids": [],
            "evidence_tokens_after": 0,
            "total_tokens_after": total,
        }

    return reduce


def _assemble_evidence(
    retrieved: list[RetrievedChunk],
    question: str,
) -> tuple[list[RetrievedChunk], int, int]:
    """Pick the highest-score chunks that fit under the 8,000-token
    evidence budget AND keep the total-context estimate under the
    configured limit. Returns (surviving, evidence_tokens, total_tokens).

    The budget counts the **formatted** evidence block the model sees
    (headers + passage text), not bare chunk bodies alone.
    """
    surviving: list[RetrievedChunk] = []

    for chunk in retrieved:
        candidate = surviving + [chunk]
        evidence_tokens = count_text(_format_evidence_block(candidate))
        if evidence_tokens > EVIDENCE_BUDGET:
            break
        surviving = candidate

    evidence_tokens = (
        count_text(_format_evidence_block(surviving)) if surviving else 0
    )
    # Estimate total tokens for the outbound message list. This is
    # what the R4 injector will compare against its threshold, so we
    # use the same counters. Chapter-limit text is small and fixed.
    approx_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Evidence passages:\n\n{_format_evidence_block(surviving)}"
                f"\n\nQuestion: {question}"
            ),
        },
    ]
    total = (
        count_messages(approx_messages)
        + count_output_reservation(_MAX_OUTPUT_TOKENS)
    )
    return surviving, evidence_tokens, total


def _build_messages(
    question: str,
    evidence: list[RetrievedChunk],
    max_chapter: int,
) -> list[dict[str, Any]]:
    """Compose the ordered chat messages sent to the model. Evidence
    lives in the USER role, not SYSTEM — the source's untrusted-content
    rule (book text can never be treated as instructions)."""
    evidence_block = _format_evidence_block(evidence)
    user_content = (
        f"Chapter limit (inclusive): {max_chapter}\n\n"
        f"Evidence passages (cite by chunk_id in square brackets):\n\n"
        f"{evidence_block}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _format_evidence_block(evidence: list[RetrievedChunk]) -> str:
    if not evidence:
        return "(no passages found under the chapter limit)"
    lines: list[str] = []
    for c in evidence:
        header = (
            f"[{c.chunk_id}] Chapter {c.chapter} "
            f"({c.chapter_original_label}: {c.chapter_title}) "
            f"pdf_pages={c.pdf_pages}"
        )
        lines.append(header)
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).rstrip()


#: Minimum NFC-normalised excerpt length for a citation to count as support.
_MIN_EXCERPT_CHARS = 20


def _run_post_hoc_verifiers(
    model_answer: ModelAnswer,
    evidence: list[RetrievedChunk],
    max_chapter: int,
) -> dict[str, Any]:
    """Authoritative grounding checks. Any non-empty ``verifier_errors``
    means the candidate answer must not be returned as ``answered``."""
    errors: list[dict[str, Any]] = []
    evidence_by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in evidence}

    inline_ids = set(re.findall(r"\[([a-f0-9]{16})\]", model_answer.answer))
    cited_ids = {c.chunk_id for c in model_answer.citations}
    missing_inline = inline_ids - cited_ids
    if missing_inline:
        errors.append(
            {
                "kind": "inline_citation_not_in_citations",
                "missing": sorted(missing_inline),
            }
        )

    if model_answer.status == "insufficient_evidence":
        if model_answer.citations:
            errors.append(
                {
                    "kind": "citations_present_on_insufficient_evidence",
                    "count": len(model_answer.citations),
                }
            )
        if inline_ids:
            errors.append(
                {
                    "kind": "inline_citations_on_insufficient_evidence",
                    "ids": sorted(inline_ids),
                }
            )
        # Contradiction positions still must be grounded if present.
        _validate_contradiction_positions(
            model_answer.contradictions, evidence_by_id, errors
        )
        return {
            "verifier_errors": errors,
            "errors": bool(errors),
        }

    # status == answered: invented / uncited answers must not pass.
    if not model_answer.citations:
        errors.append({"kind": "answered_without_citations"})
    if not inline_ids:
        errors.append({"kind": "answered_without_inline_citations"})
    uncited = _uncited_substantive_sentences(model_answer.answer)
    if uncited:
        errors.append(
            {
                "kind": "uncited_factual_sentences",
                "sentences": uncited[:5],
                "count": len(uncited),
            }
        )

    for cit in model_answer.citations:
        _validate_citation(cit, evidence_by_id, max_chapter, errors)

    _validate_contradiction_positions(
        model_answer.contradictions, evidence_by_id, errors
    )

    return {
        "verifier_errors": errors,
        "errors": bool(errors),
    }


def _uncited_substantive_sentences(answer: str) -> list[str]:
    """Return substantive sentences that lack an inline ``[chunk_id]``.

    A sentence is substantive when it has at least three alphabetic
    tokens after stripping citation markers. This is a deterministic
    grounding floor, not a semantic NLI judge.
    """
    if not (answer or "").strip():
        return []
    # Split on sentence terminators while keeping short clauses useful.
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    uncited: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if re.search(r"\[[a-f0-9]{16}\]", text):
            continue
        stripped = re.sub(r"\[[a-f0-9]{16}\]", " ", text)
        words = re.findall(r"[A-Za-z]{2,}", stripped)
        if len(words) >= 3:
            uncited.append(text)
    return uncited


def _validate_citation(
    cit: Citation,
    evidence_by_id: dict[str, RetrievedChunk],
    max_chapter: int,
    errors: list[dict[str, Any]],
) -> None:
    chunk = evidence_by_id.get(cit.chunk_id)
    if chunk is None:
        errors.append(
            {
                "kind": "citation_chunk_not_in_evidence",
                "chunk_id": cit.chunk_id,
            }
        )
        return

    if cit.chapter != chunk.chapter:
        errors.append(
            {
                "kind": "citation_chapter_mismatch",
                "chunk_id": cit.chunk_id,
                "cited_chapter": cit.chapter,
                "chunk_chapter": chunk.chapter,
            }
        )
    if cit.chapter > max_chapter:
        errors.append(
            {
                "kind": "citation_chapter_over_limit",
                "chunk_id": cit.chunk_id,
                "chapter": cit.chapter,
                "max_chapter": max_chapter,
            }
        )
    if cit.source_filename != chunk.source_filename:
        errors.append(
            {
                "kind": "citation_source_filename_mismatch",
                "chunk_id": cit.chunk_id,
                "cited": cit.source_filename,
                "expected": chunk.source_filename,
            }
        )
    if not cit.pdf_pages:
        errors.append(
            {
                "kind": "citation_pdf_pages_empty",
                "chunk_id": cit.chunk_id,
            }
        )
    else:
        disallowed = [p for p in cit.pdf_pages if p not in chunk.pdf_pages]
        if disallowed:
            errors.append(
                {
                    "kind": "citation_pdf_pages_not_allowed",
                    "chunk_id": cit.chunk_id,
                    "disallowed": disallowed,
                    "allowed": list(chunk.pdf_pages),
                }
            )

    needle = unicodedata.normalize("NFC", cit.excerpt or "")
    if not needle.strip():
        errors.append(
            {
                "kind": "citation_excerpt_empty",
                "chunk_id": cit.chunk_id,
            }
        )
        return
    if len(needle.strip()) < _MIN_EXCERPT_CHARS:
        errors.append(
            {
                "kind": "citation_excerpt_too_short",
                "chunk_id": cit.chunk_id,
                "length": len(needle.strip()),
                "min_length": _MIN_EXCERPT_CHARS,
            }
        )
    haystack = unicodedata.normalize("NFC", chunk.text)
    if needle not in haystack:
        errors.append(
            {
                "kind": "excerpt_not_substring",
                "chunk_id": cit.chunk_id,
                "needle_head": needle[:80],
            }
        )


def _validate_contradiction_positions(
    contradictions: list[Contradiction],
    evidence_by_id: dict[str, RetrievedChunk],
    errors: list[dict[str, Any]],
) -> None:
    for idx, contradiction in enumerate(contradictions):
        if not contradiction.positions:
            errors.append(
                {
                    "kind": "contradiction_positions_empty",
                    "index": idx,
                }
            )
            continue
        for pos in contradiction.positions:
            chunk = evidence_by_id.get(pos.chunk_id)
            if chunk is None:
                errors.append(
                    {
                        "kind": "contradiction_chunk_not_in_evidence",
                        "chunk_id": pos.chunk_id,
                        "index": idx,
                    }
                )
                continue
            needle = unicodedata.normalize("NFC", pos.excerpt or "")
            if not needle.strip():
                errors.append(
                    {
                        "kind": "contradiction_excerpt_empty",
                        "chunk_id": pos.chunk_id,
                        "index": idx,
                    }
                )
                continue
            haystack = unicodedata.normalize("NFC", chunk.text)
            if needle not in haystack:
                errors.append(
                    {
                        "kind": "contradiction_excerpt_not_substring",
                        "chunk_id": pos.chunk_id,
                        "index": idx,
                        "needle_head": needle[:80],
                    }
                )


def _citation_correction_prompt(errors: list[dict[str, Any]]) -> str:
    """Build a corrective user message listing concrete verifier errors."""
    summary = json.dumps(errors, ensure_ascii=False, separators=(",", ":"))
    return (
        "Your previous answer failed authoritative citation/contradiction "
        "validation. Fix every listed error. Rules: status=answered "
        "requires at least one citation and an inline [chunk_id] on every "
        "substantive sentence; every citation must reference an evidence "
        "chunk_id you were given; source_filename, chapter, and pdf_pages "
        "must match that chunk exactly; excerpt must be a verbatim NFC "
        "substring of the chunk text with at least "
        f"{_MIN_EXCERPT_CHARS} characters; every inline [chunk_id] in the "
        "answer must appear in citations; insufficient_evidence must have "
        "zero citations. Validation errors: "
        f"{summary}. Return only corrected JSON matching the schema."
    )


def _redacted_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the complete ordered messages for assessment evidence.

    The source requires the full ordered model messages for the
    two-chapter and before/after checks. These messages contain only
    the public system prompt, question, and book evidence; credentials
    are never inserted into them. Keep the structure lossless rather
    than storing only a preview, so a reviewer can reproduce exactly
    what the model saw."""
    return [dict(message) for message in messages]


__all__ = [
    "router",
    "_MIN_EXCERPT_CHARS",
    "_assemble_evidence",
    "_format_evidence_block",
    "_run_post_hoc_verifiers",
    "_citation_correction_prompt",
]
