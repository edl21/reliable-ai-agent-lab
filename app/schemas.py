"""Pydantic schemas for /answer request and response.

The source assessment pins the response shape (verbatim):
*"status, answer (with inline citation IDs matching the citations array),
citations (chunk ID, filename, chapter, 1-based PDF page(s), exact
excerpt from normalised text), metrics (timings, token usage)."* Plus
the locked-in ``contradictions[]`` field decided during the adversarial
review to satisfy the character-claim-vs-narrative-fact rule.

We also expose the JSON Schema used with ``response_format={"type":
"json_schema", ...}`` so the model returns strictly-shaped output that
we then re-validate in Python before building the API response."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- Request --------------------------------------------------------------


class AnswerRequest(BaseModel):
    """POST /answer body. ``max_chapter`` validation happens here so a
    missing/non-integer/out-of-range value returns HTTP 422 BEFORE any
    model call — the source's required check."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=200)
    question: str = Field(..., min_length=1, max_length=2_000)
    max_chapter: int = Field(..., ge=1)

    @field_validator("request_id")
    @classmethod
    def _request_id_safe(cls, v: str) -> str:
        # Reject characters that would break trace-filename paths;
        # request IDs land in ``traces/task1/<check>_<request_id>.jsonl``.
        for ch in v:
            if not (ch.isalnum() or ch in "_-."):
                raise ValueError(
                    "request_id may only contain letters, digits, '_', '-', '.'"
                )
        return v


# --- Citation and contradiction shapes ------------------------------------


class Citation(BaseModel):
    """One row of the ``citations`` array. Field names pinned to match
    the source's verbatim requirement."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    source_filename: str
    chapter: int
    pdf_pages: list[int]
    excerpt: str


class ContradictionPosition(BaseModel):
    """One side of a claim in the ``contradictions[]`` field."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    stance: Literal["supports", "contradicts"]
    excerpt: str


class Contradiction(BaseModel):
    """An unresolved contradiction the model surfaced with citations."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    positions: list[ContradictionPosition]


# --- Response -------------------------------------------------------------


class AnswerMetrics(BaseModel):
    """Timings + token usage. All optional so a stub run without usage
    data still produces a valid response."""

    model_config = ConfigDict(extra="forbid")

    total_ms: int = 0
    retrieval_ms: int = 0
    model_ms: int = 0
    evidence_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    stubbed: bool = False


class AnswerResponse(BaseModel):
    """POST /answer response body."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    contradictions: list[Contradiction] = Field(default_factory=list)
    metrics: AnswerMetrics


# --- The JSON Schema handed to the model via response_format --------------


def model_response_json_schema() -> dict[str, Any]:
    """The strict schema the model MUST match. We re-validate its
    output against ``ModelAnswer`` in Python before building the API
    response — never pass through unvalidated model output."""
    return {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "answer", "citations", "contradictions"],
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["answered", "insufficient_evidence"],
                },
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "chunk_id",
                            "source_filename",
                            "chapter",
                            "pdf_pages",
                            "excerpt",
                        ],
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "source_filename": {"type": "string"},
                            "chapter": {"type": "integer"},
                            "pdf_pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "excerpt": {"type": "string"},
                        },
                    },
                },
                "contradictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["claim", "positions"],
                        "properties": {
                            "claim": {"type": "string"},
                            "positions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["chunk_id", "stance", "excerpt"],
                                    "properties": {
                                        "chunk_id": {"type": "string"},
                                        "stance": {
                                            "type": "string",
                                            "enum": ["supports", "contradicts"],
                                        },
                                        "excerpt": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


class ModelAnswer(BaseModel):
    """The structural shape we validate the model's raw JSON against
    before building the API response. Mirrors the JSON Schema above."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    contradictions: list[Contradiction] = Field(default_factory=list)


class GuideRequest(BaseModel):
    """POST /guide body. ``max_chapter`` is validated before the first
    model call, just like ``AnswerRequest``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=200)
    goal: str = Field(..., min_length=1, max_length=4_000)
    max_chapter: int = Field(..., ge=1)

    @field_validator("request_id")
    @classmethod
    def _request_id_safe(cls, v: str) -> str:
        for ch in v:
            if not (ch.isalnum() or ch in "_-."):
                raise ValueError(
                    "request_id may only contain letters, digits, '_', '-', '.'"
                )
        return v


class ApproveRequest(BaseModel):
    """POST /approve body. ``payload_check`` is optional so the normal
    source-shaped body remains ``{operation_id, approve}``, while the
    tamper/replay check can explicitly present the payload hash."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(..., min_length=1, max_length=200)
    approve: bool
    payload_check: str | None = Field(default=None, min_length=64, max_length=64)


class GuideResponse(BaseModel):
    """POST /guide response before approval or after a structured tool
    error. A pending draft contains the application-generated operation
    ID and canonical payload hash; no artifact exists yet."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["pending_approval", "completed", "error"]
    operation_id: str | None = None
    draft: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ApproveResponse(BaseModel):
    """POST /approve response. Replays return the same artifact ID."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    status: Literal["rejected", "saved", "already_saved"]
    artifact_id: str | None = None
    approval_token: str | None = None


__all__ = [
    "AnswerMetrics",
    "AnswerRequest",
    "AnswerResponse",
    "ApproveRequest",
    "ApproveResponse",
    "Citation",
    "Contradiction",
    "ContradictionPosition",
    "GuideRequest",
    "GuideResponse",
    "ModelAnswer",
    "model_response_json_schema",
]
