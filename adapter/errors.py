"""Structured error hierarchy used across the adapter, fault wrapper, and app.

The point of this hierarchy is that every failure mode the assessment cares
about (R1-R6, pre-generation validation, tool-arg validation, run-limit
exhaustion, schema-invalid model output) has a distinct class with the
minimum data needed for recovery — so recovery code can pattern-match on
type, not on string parsing.

Design rules:
- ``ContextLengthExceeded.reported_limit`` is the ONLY place the app learns
  the reduced context limit from. The number 4096 must never appear as a
  constant in app code (only inside ``adapter/faults.py``). This is what
  proves R4's "learn the reduced limit from the error" requirement.
- ``AuthError`` is terminal by design: it maps to HTTP 502 as a
  configuration error, never framed as ``insufficient_evidence``.
- ``RateLimited`` carries ``retry_after`` when the origin supplied one;
  falls back to bounded exponential-backoff-with-jitter otherwise
  (policy pinned in ``adapter/limits.py``/``adapter/faults.py``).
"""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for every failure raised by the adapter / fault wrapper."""


class ContextLengthExceeded(AdapterError):
    """Raised when the total prompt+tools+reserved-output exceeds the
    reported context window. ``reported_limit`` is the value the caller
    must reduce to on retry. This is the only channel through which the
    app learns any context limit smaller than the configured default."""

    def __init__(self, reported_limit: int, message: str | None = None) -> None:
        self.reported_limit = int(reported_limit)
        super().__init__(
            message
            or f"context_length_exceeded: reduce to <= {self.reported_limit} tokens"
        )


class AuthError(AdapterError):
    """HTTP 401 or equivalent. Terminal — no retry. Maps to HTTP 502
    'configuration_error' on the way out, not to insufficient_evidence."""


class RateLimited(AdapterError):
    """HTTP 429 or equivalent. ``retry_after`` is seconds when the origin
    supplied a Retry-After header; None means the wrapper must fall back
    to bounded exponential backoff with jitter."""

    def __init__(
        self, retry_after: float | None = None, message: str | None = None
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message or f"rate_limited: retry_after={retry_after}")


class RetriesExhausted(AdapterError):
    """The retry policy gave up because either the attempts-per-op cap
    was hit, the elapsed-time budget was exceeded, or the next planned
    wait would run past the remaining deadline. Never a fabricated answer."""


class LimitExhausted(AdapterError):
    """A per-run finite ceiling (total calls, elapsed time, cumulative
    tokens) has been reached before this call could start. Distinct from
    RetriesExhausted so the trace can distinguish 'we retried too many
    times on the same op' from 'this run is out of budget entirely'."""


class SchemaInvalid(AdapterError):
    """The model returned JSON that failed the response_format schema
    check. ``raw`` carries the offending output for the corrective
    prompt (R5). Never surface ``raw`` to end users."""

    def __init__(self, raw: object, message: str | None = None) -> None:
        self.raw = raw
        super().__init__(message or "schema_invalid: model output rejected")


class ToolValidationError(AdapterError):
    """A tool call from the model failed name/argument validation before
    execution. ``name`` is the requested tool; ``detail`` is a
    machine-readable description the model can use to self-correct."""

    def __init__(self, name: str, detail: str) -> None:
        self.name = name
        self.detail = detail
        super().__init__(f"tool_validation_error: {name}: {detail}")


__all__ = [
    "AdapterError",
    "AuthError",
    "ContextLengthExceeded",
    "LimitExhausted",
    "RateLimited",
    "RetriesExhausted",
    "SchemaInvalid",
    "ToolValidationError",
]
