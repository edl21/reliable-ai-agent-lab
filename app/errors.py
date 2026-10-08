"""HTTP-facing error shapes and the ``AdapterError`` → HTTP mapping.

Rules pinned by the source and the plan:

- **422** for pre-generation validation failures (malformed
  ``max_chapter``, missing fields, wrong types). These MUST be
  rejected before any model call — see Task 1's required check for
  malformed ``max_chapter``.
- **500** for ``RetriesExhausted`` / ``LimitExhausted`` — we ran out
  of budget; never dressed up as an answer.
- **502** for ``AuthError`` framed EXPLICITLY as a configuration
  error (``configuration_error`` in the ``error.code``). Source:
  "no retries, no ``insufficient_evidence`` framing" for R6.
- **409** for approval-flow tamper checks (payload hash mismatch on
  ``/approve`` for the same ``operation_id``). Rejects operation-ID
  reuse with different content.

All shapes use the same envelope so a reviewer can grep the responses
and traces uniformly.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from adapter.errors import (
    AdapterError,
    AuthError,
    ContextLengthExceeded,
    LimitExhausted,
    RateLimited,
    RetriesExhausted,
    SchemaInvalid,
    ToolValidationError,
)


class ErrorEnvelope(BaseModel):
    """Uniform error shape returned by all four routes."""

    model_config = ConfigDict(extra="forbid")

    error: "ErrorBody"


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    detail: dict[str, Any] | None = None


ErrorEnvelope.model_rebuild()


# --- pre-generation validation --------------------------------------------


class ValidationRejection(HTTPException):
    """422 — request rejected BEFORE any model call.

    Use this for malformed ``max_chapter``, missing fields, bad types.
    The check harness relies on the trace event's ``operation=validation``
    and ``outcome=rejected`` to confirm no model call happened."""

    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(
            status_code=422,
            detail={"error": {"code": code, "message": message, "detail": detail}},
        )


# --- adapter-error → HTTP mapping -----------------------------------------


def adapter_error_to_http(exc: AdapterError) -> HTTPException:
    """Convert one of our ``AdapterError`` subclasses into the matching
    ``HTTPException`` per the rules above. Used at the outer edge of
    every route so no ``AdapterError`` leaks unwrapped."""
    if isinstance(exc, AuthError):
        # R6: exactly one attempt, framed as configuration_error, NEVER
        # insufficient_evidence.
        return HTTPException(
            status_code=502,
            detail={
                "error": {
                    "code": "configuration_error",
                    "message": "authentication with the model gateway failed",
                    "detail": {"cause": str(exc)},
                }
            },
        )
    if isinstance(exc, RetriesExhausted):
        return HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "retries_exhausted",
                    "message": str(exc),
                }
            },
        )
    if isinstance(exc, LimitExhausted):
        return HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "run_limit_exhausted",
                    "message": str(exc),
                }
            },
        )
    if isinstance(exc, ContextLengthExceeded):
        # This should almost never surface to the client: the app is
        # supposed to catch it, reduce, and retry (R4). If it does
        # surface, return 500 — the run failed to recover.
        return HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "context_length_exceeded_unrecovered",
                    "message": str(exc),
                    "detail": {"reported_limit": exc.reported_limit},
                }
            },
        )
    if isinstance(exc, RateLimited):
        return HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "rate_limited",
                    "message": str(exc),
                    "detail": {"retry_after": exc.retry_after},
                }
            },
        )
    if isinstance(exc, SchemaInvalid):
        return HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "schema_invalid_unrecovered",
                    "message": "the model returned invalid JSON after correction",
                }
            },
        )
    if isinstance(exc, ToolValidationError):
        return HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "tool_validation_error",
                    "message": exc.detail,
                    "detail": {"tool": exc.name},
                }
            },
        )
    # Fallback for any AdapterError not explicitly mapped above.
    return HTTPException(
        status_code=500,
        detail={
            "error": {
                "code": "adapter_error",
                "message": str(exc),
            }
        },
    )


# --- approval-flow specific ------------------------------------------------


class OperationTampered(HTTPException):
    """409 — the caller replayed an ``operation_id`` with a payload
    whose hash differs from the stored one. A new operation is
    required. This is the tamper-check that stops an approved
    ``operation_id`` from being reused for different content."""

    def __init__(self, operation_id: str) -> None:
        super().__init__(
            status_code=409,
            detail={
                "error": {
                    "code": "operation_payload_tampered",
                    "message": (
                        "operation_id already exists with a different payload; "
                        "issue a new operation instead of reusing this one"
                    ),
                    "detail": {"operation_id": operation_id},
                }
            },
        )


__all__ = [
    "ErrorBody",
    "ErrorEnvelope",
    "OperationTampered",
    "ValidationRejection",
    "adapter_error_to_http",
]
