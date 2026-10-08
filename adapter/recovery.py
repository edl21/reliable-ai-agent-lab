"""Shared bounded model-call recovery used by /answer and /guide."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from adapter.clock import SystemClock, SystemSleeper
from adapter.errors import (
    AuthError,
    ContextLengthExceeded,
    LimitExhausted,
    RateLimited,
    RetriesExhausted,
)
from adapter.limits import LimitTracker, RunLimits
from adapter.tokens import (
    TOTAL_CONTEXT_LIMIT_DEFAULT,
    count_messages,
    count_output_reservation,
    count_tool_schemas,
)
from traces.schema import ContextCounts, JSONLWriter, TokenUsage, TraceEvent


Reducer = Callable[
    [list[dict[str, Any]], int],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


@dataclass
class RecoveryResult:
    response: Any
    messages: list[dict[str, Any]]


def chat_with_recovery(
    adapter: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    response_format: dict[str, Any] | None,
    max_output_tokens: int | None,
    temperature: float | None,
    parallel_tool_calls: bool | None,
    operation_id: str,
    task: str,
    request_id: str,
    run_id: str,
    trace_path: Any,
    writer: JSONLWriter,
    tracker: LimitTracker | None = None,
    reducer: Reducer | None = None,
    source_chapter_ids: list[int] | None = None,
    context_limit: int | None = None,
) -> RecoveryResult:
    """Call the adapter with finite budgets, retry policy, and recovery.

    The adapter is the real singleton or a ``FaultInjector`` around it.
    Retry-After is honoured exactly; absent headers use bounded
    exponential backoff with deterministic jitter. R4's reduced limit
    comes only from ``ContextLengthExceeded.reported_limit``.

    A generic pre-call total-context guard applies to every model call.
    When a reducer is provided it may shrink the prompt; otherwise the
    oversize call is rejected before the adapter is invoked. Learned
    R4 limits override the default ceiling for subsequent attempts.
    """
    current_messages = list(messages)
    clock = getattr(adapter, "clock", SystemClock())
    sleeper = getattr(adapter, "sleeper", SystemSleeper())
    tracker = tracker or getattr(adapter, "tracker", None)
    if tracker is None:
        tracker = LimitTracker(limits=RunLimits(), clock=clock)
    rng = random.Random(getattr(getattr(adapter, "config", None), "jitter_seed", 0))
    retry_index = 0
    # Effective ceiling starts at the configured default (or caller
    # override). R4 may lower it via ContextLengthExceeded.reported_limit.
    effective_limit = int(context_limit or TOTAL_CONTEXT_LIMIT_DEFAULT)

    while True:
        estimated = count_messages(current_messages) + count_tool_schemas(tools)
        reserved = count_output_reservation(max_output_tokens)
        total_tokens = estimated + reserved

        # Generic pre-call total-context guard (all model calls).
        if total_tokens > effective_limit:
            if reducer is not None:
                reduced_messages, reduction_notes = reducer(
                    current_messages, effective_limit
                )
                if reduced_messages != current_messages:
                    _event(
                        writer,
                        trace_path,
                        request_id=request_id,
                        run_id=run_id,
                        task=task,
                        operation="limit_check",
                        attempt=tracker.op_attempts.get(operation_id, 0),
                        outcome="context_length_exceeded",
                        recovery_decision="reduce_context",
                        source_chapter_ids=source_chapter_ids,
                        context_counts=ContextCounts(
                            total_tokens=total_tokens,
                            output_tokens_reserved=reserved,
                        ),
                        notes={
                            "reason": "pre_call_context_guard",
                            "effective_limit": effective_limit,
                            "before_total_tokens": total_tokens,
                            **reduction_notes,
                        },
                    )
                    current_messages = reduced_messages
                    continue
            _event(
                writer,
                trace_path,
                request_id=request_id,
                run_id=run_id,
                task=task,
                operation="limit_check",
                attempt=tracker.op_attempts.get(operation_id, 0),
                outcome="context_length_exceeded",
                recovery_decision="retries_exhausted",
                source_chapter_ids=source_chapter_ids,
                context_counts=ContextCounts(
                    total_tokens=total_tokens,
                    output_tokens_reserved=reserved,
                ),
                notes={
                    "reason": "pre_call_context_guard",
                    "effective_limit": effective_limit,
                    "total_tokens": total_tokens,
                },
            )
            raise ContextLengthExceeded(
                reported_limit=effective_limit,
                message=(
                    f"pre_call_context_guard: total {total_tokens} exceeds "
                    f"effective_limit {effective_limit}"
                ),
            )

        try:
            tracker.pre_call(
                op_id=operation_id,
                estimated_input_tokens=estimated,
                reserved_output_tokens=reserved,
            )
        except (RetriesExhausted, LimitExhausted) as exc:
            _event(
                writer,
                trace_path,
                request_id=request_id,
                run_id=run_id,
                task=task,
                operation="limit_check",
                attempt=tracker.op_attempts.get(operation_id, 0),
                outcome="error",
                recovery_decision="retries_exhausted",
                source_chapter_ids=source_chapter_ids,
                context_counts=ContextCounts(
                    total_tokens=total_tokens,
                    output_tokens_reserved=reserved,
                ),
                notes={"error": str(exc), "operation_id": operation_id},
            )
            raise

        attempt = tracker.op_attempts[operation_id]
        try:
            response = adapter.chat(
                messages=current_messages,
                tools=tools,
                response_format=response_format,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                parallel_tool_calls=parallel_tool_calls,
            )
        except RateLimited as exc:
            delay = (
                exc.retry_after
                if exc.retry_after is not None
                else min(8.0, (2**retry_index) + rng.uniform(0.0, 0.5))
            )
            if tracker.would_wait_exceed_deadline(delay):
                _event(
                    writer,
                    trace_path,
                    request_id=request_id,
                    run_id=run_id,
                    task=task,
                    operation="model_call",
                    attempt=attempt,
                    outcome="error",
                    recovery_decision="retries_exhausted",
                    source_chapter_ids=source_chapter_ids,
                    context_counts=ContextCounts(
                        total_tokens=total_tokens,
                        output_tokens_reserved=reserved,
                    ),
                    notes={
                        "error": str(exc),
                        "injected": "[injected" in str(exc),
                        "requested_delay_seconds": delay,
                        "remaining_deadline_seconds": tracker.remaining_deadline(),
                    },
                )
                raise RetriesExhausted(
                    "retry wait would exceed the remaining deadline"
                ) from exc
            sleeper.sleep(delay)
            _event(
                writer,
                trace_path,
                request_id=request_id,
                run_id=run_id,
                task=task,
                operation="model_call",
                attempt=attempt,
                outcome="error",
                recovery_decision="wait_retry",
                source_chapter_ids=source_chapter_ids,
                virtual=getattr(clock, "__class__", type(clock)).__name__
                == "VirtualClock",
                context_counts=ContextCounts(
                    total_tokens=total_tokens,
                    output_tokens_reserved=reserved,
                ),
                notes={
                    "error": str(exc),
                    "injected": "[injected" in str(exc),
                    "requested_delay_seconds": delay,
                    "virtual_elapsed_seconds": clock.now(),
                },
            )
            retry_index += 1
            continue
        except AuthError as exc:
            _event(
                writer,
                trace_path,
                request_id=request_id,
                run_id=run_id,
                task=task,
                operation="model_call",
                attempt=attempt,
                outcome="error",
                recovery_decision="auth_fail",
                source_chapter_ids=source_chapter_ids,
                context_counts=ContextCounts(
                    total_tokens=total_tokens,
                    output_tokens_reserved=reserved,
                ),
                notes={"error": str(exc), "injected": "[injected" in str(exc)},
            )
            raise
        except ContextLengthExceeded as exc:
            # Learn the reduced limit from the error (R4). Never hardcode
            # the injector threshold here.
            effective_limit = int(exc.reported_limit)
            if reducer is None:
                raise
            reduced_messages, reduction_notes = reducer(
                current_messages, exc.reported_limit
            )
            if reduced_messages == current_messages:
                raise
            _event(
                writer,
                trace_path,
                request_id=request_id,
                run_id=run_id,
                task=task,
                operation="model_call",
                attempt=attempt,
                outcome="context_length_exceeded",
                recovery_decision="reduce_context",
                source_chapter_ids=source_chapter_ids,
                context_counts=ContextCounts(
                    total_tokens=total_tokens,
                    output_tokens_reserved=reserved,
                ),
                notes={
                    "error": str(exc),
                    "injected": "[injected" in str(exc),
                    "reported_limit": exc.reported_limit,
                    "before_messages": current_messages,
                    "after_messages": reduced_messages,
                    **reduction_notes,
                },
            )
            current_messages = reduced_messages
            continue

        usage = getattr(response, "usage", None) or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        reported_total: int | None = None
        if input_tokens is not None or output_tokens is not None:
            reported_total = int(input_tokens or 0) + int(output_tokens or 0)
            tracker.record_actual_tokens(
                reserved=estimated + reserved,
                actual=reported_total,
            )
        else:
            # Keep reservation when live usage is unavailable (stub).
            reported_total = None

        token_usage = TokenUsage(
            input_tokens=input_tokens if input_tokens is not None else estimated,
            output_tokens=output_tokens if output_tokens is not None else None,
            total_tokens=(
                reported_total
                if reported_total is not None
                else estimated + reserved
            ),
        )

        _event(
            writer,
            trace_path,
            request_id=request_id,
            run_id=run_id,
            task=task,
            operation="model_call",
            attempt=attempt,
            outcome="ok",
            source_chapter_ids=source_chapter_ids,
            stubbed=bool(getattr(response, "stubbed", False)),
            context_counts=ContextCounts(
                total_tokens=total_tokens,
                output_tokens_reserved=reserved,
            ),
            token_usage=token_usage,
            notes={
                "ordered_messages": current_messages,
                "raw_output": getattr(response, "content", None),
                "tool_calls": getattr(response, "tool_calls", []),
                "effective_limit": effective_limit,
                "reserved_tokens": estimated + reserved,
                "reconciled_tokens": reported_total,
            },
        )
        return RecoveryResult(response=response, messages=current_messages)


def _event(
    writer: JSONLWriter,
    path: Any,
    *,
    request_id: str,
    run_id: str,
    task: str,
    operation: str,
    attempt: int,
    outcome: str,
    recovery_decision: str = "none",
    source_chapter_ids: list[int] | None = None,
    virtual: bool = False,
    stubbed: bool = False,
    notes: dict[str, Any] | None = None,
    context_counts: ContextCounts | None = None,
    token_usage: TokenUsage | None = None,
) -> None:
    writer.append(
        path,
        TraceEvent(
            request_id=request_id,
            run_id=run_id,
            task=task,  # type: ignore[arg-type]
            operation=operation,  # type: ignore[arg-type]
            attempt=attempt,
            virtual=virtual,
            source_chapter_ids=source_chapter_ids or [],
            context_counts=context_counts or ContextCounts(),
            token_usage=token_usage or TokenUsage(),
            outcome=outcome,  # type: ignore[arg-type]
            recovery_decision=recovery_decision,  # type: ignore[arg-type]
            stubbed=stubbed,
            notes=notes or {},
        ),
    )


__all__ = ["RecoveryResult", "chat_with_recovery"]
