"""One trace schema, used by every task.

The source assessment's "Common traces" section pins the minimum fields:
``request_id``/``run_id``, ``task``, ``operation``, ``attempt``,
``elapsed_time`` (real or virtual), source/chapter IDs, context counts,
token usage, ``recovery_decision``, ``outcome``. We add three hardening
fields: ``virtual: bool`` (so an R1 trace can be told apart from a real
one by inspection), ``stubbed: bool`` (so a run against the stub
adapter is unambiguous), and ``notes`` (free-text metadata like the
chapter-guard call-site name).

The ``JSONLWriter`` is the ONLY sanctioned way to append trace events.
It performs two independent credential-redaction passes on every event
before writing:

1. Strip any dict field whose key matches a well-known credential
   header name (``authorization``, ``api-key``, ``x-api-key``,
   ``proxy-authorization``, ``cookie``, ``set-cookie``).
2. If ``OPENAI_API_KEY`` is set in the environment, refuse to write
   any string value equal to it (raises ``CredentialLeakError`` so
   the test surfaces the bug rather than silently writing the key).

Design rules:

- Events are pure data; the writer does no I/O beyond appending one
  JSON line. Callers batch by writing many events to one path.
- Paths are created lazily (parents mkdir'd on first append).
- ``recovery_decision`` and ``outcome`` are enum-like strings; we
  don't use ``Enum`` to keep JSON output stable across Python versions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Enum-like value sets (documented; not runtime-enforced) --------------

Task = Literal["t1", "t2", "t3", "t4"]

Operation = Literal[
    "model_call",
    "tool_call",
    "guard",
    "approve",
    "save",
    "repair_step",
    "limit_check",
    "retrieval",
    "validation",
    "index_build",
]

RecoveryDecision = Literal[
    "none",
    "wait_retry",
    "reduce_context",
    "schema_reprompt",
    "auth_fail",
    "retries_exhausted",
    "replay_ledger",
    "limit_exhausted",
]

Outcome = Literal[
    "ok",
    "error",
    "stubbed",
    "rejected",
    "pending_approval",
    "saved",
    "insufficient_evidence",
    "retries_exhausted",
    "context_length_exceeded",
    "auth_fail",
    "schema_invalid",
]

# --- Credential redaction --------------------------------------------------

# Header names that must never end up in a trace, regardless of casing.
_REDACT_KEYS: frozenset[str] = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "openai_api_key",  # environment variable name — belt-and-braces
    }
)


class CredentialLeakError(RuntimeError):
    """Raised when the writer detects the current OPENAI_API_KEY value
    would end up in a serialised trace. The correct fix is upstream —
    stop putting the key in the trace payload — not to widen the
    redaction rules."""


# --- The event model -------------------------------------------------------


class ContextCounts(BaseModel):
    """Token-accounting snapshot at the moment of the event."""

    model_config = ConfigDict(extra="forbid")

    evidence_tokens: int = 0
    total_tokens: int = 0
    output_tokens_reserved: int = 0


class TokenUsage(BaseModel):
    """Model-reported token usage, when available.

    Called ``available_usage`` in the source; we mirror the same fields
    the OpenAI API returns."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TraceEvent(BaseModel):
    """One line in a JSONL trace. Every field the source names, plus
    hardening fields for adversarial-review-driven invariants."""

    model_config = ConfigDict(extra="forbid")

    # Identity
    request_id: str | None = None
    run_id: str | None = None
    task: Task

    # What happened
    operation: Operation
    attempt: int = 0

    # Timing (real or virtual — read ``virtual`` to disambiguate)
    elapsed_time_ms: int = 0
    virtual: bool = False

    # Retrieval / chapter context
    source_chapter_ids: list[int] = Field(default_factory=list)
    context_counts: ContextCounts = Field(default_factory=ContextCounts)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # Recovery bookkeeping
    recovery_decision: RecoveryDecision = "none"
    outcome: Outcome = "ok"

    # Hardening / annotation
    stubbed: bool = False
    notes: dict[str, Any] = Field(default_factory=dict)


# --- Writer ----------------------------------------------------------------


class JSONLWriter:
    """Append-only JSONL trace writer with credential redaction.

    Usage::

        writer = JSONLWriter()
        writer.append(Path("traces/task1/single_chapter_r1.jsonl"), event)

    The class is stateless w.r.t. paths — the caller owns which path
    each event goes to, per the pinned trace naming convention."""

    def append(self, path: Path | str, event: TraceEvent) -> None:
        payload = event.model_dump(mode="json")
        self._redact(payload)
        line = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    # -- redaction ---------------------------------------------------------

    def _redact(self, node: Any) -> None:
        """In-place: drop known credential-header keys; raise if a
        value equals the current ``OPENAI_API_KEY``."""
        api_key = os.environ.get("OPENAI_API_KEY")

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                # Remove credential-shaped keys, case-insensitively.
                for key in list(value.keys()):
                    if isinstance(key, str) and key.lower() in _REDACT_KEYS:
                        value[key] = "[REDACTED]"
                for v in value.values():
                    visit(v)
            elif isinstance(value, list):
                for v in value:
                    visit(v)
            elif isinstance(value, str):
                if api_key and value == api_key:
                    raise CredentialLeakError(
                        "trace event contains the current OPENAI_API_KEY value"
                    )

        visit(node)


__all__ = [
    "ContextCounts",
    "CredentialLeakError",
    "JSONLWriter",
    "Operation",
    "Outcome",
    "RecoveryDecision",
    "Task",
    "TokenUsage",
    "TraceEvent",
]
