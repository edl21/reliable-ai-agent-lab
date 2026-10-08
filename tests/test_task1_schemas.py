"""Task 1 schema validation tests — pydantic request/response contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnswerMetrics,
    AnswerRequest,
    AnswerResponse,
    Citation,
    Contradiction,
    ContradictionPosition,
    ModelAnswer,
    model_response_json_schema,
)


def test_answer_request_accepts_valid_payload() -> None:
    req = AnswerRequest(request_id="r1", question="q?", max_chapter=5)
    assert req.max_chapter == 5


def test_answer_request_rejects_zero_max_chapter() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(request_id="r", question="q", max_chapter=0)


def test_answer_request_rejects_negative_max_chapter() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(request_id="r", question="q", max_chapter=-1)


def test_answer_request_rejects_non_integer_max_chapter() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(request_id="r", question="q", max_chapter="five")


def test_answer_request_rejects_missing_max_chapter() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(request_id="r", question="q")


def test_answer_request_rejects_empty_request_id() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(request_id="", question="q", max_chapter=1)


def test_answer_request_rejects_unsafe_request_id() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(
            request_id="../evil/path", question="q", max_chapter=1
        )


def test_answer_request_allows_safe_special_characters() -> None:
    req = AnswerRequest(request_id="req_2026-09-18.01", question="q", max_chapter=1)
    assert req.request_id == "req_2026-09-18.01"


def test_answer_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerRequest(
            request_id="r",
            question="q",
            max_chapter=1,
            extra_field="banned",
        )


def test_citation_shape_matches_source_requirement() -> None:
    c = Citation(
        chunk_id="abc123",
        source_filename="foo.pdf",
        chapter=5,
        pdf_pages=[10, 11],
        excerpt="verbatim excerpt",
    )
    assert c.pdf_pages == [10, 11]
    # Source verbatim: chunk_id, filename, chapter, 1-based pdf_page(s), excerpt.
    assert set(c.model_dump().keys()) == {
        "chunk_id", "source_filename", "chapter", "pdf_pages", "excerpt"
    }


def test_contradiction_position_stance_enum() -> None:
    ContradictionPosition(chunk_id="a", stance="supports", excerpt="e")
    ContradictionPosition(chunk_id="a", stance="contradicts", excerpt="e")
    with pytest.raises(ValidationError):
        ContradictionPosition(chunk_id="a", stance="maybe", excerpt="e")


def test_answer_response_requires_contradictions_field() -> None:
    """The adversarial-review fix requires contradictions[] to be part
    of the response schema."""
    r = AnswerResponse(
        request_id="r",
        status="answered",
        answer="test [abc]",
        citations=[],
        metrics=AnswerMetrics(),
    )
    assert r.contradictions == []


def test_model_answer_status_enum() -> None:
    ModelAnswer(status="answered", answer="a", citations=[])
    ModelAnswer(status="insufficient_evidence", answer="a", citations=[])
    with pytest.raises(ValidationError):
        ModelAnswer(status="unknown", answer="a", citations=[])


def test_json_schema_declares_strict_mode() -> None:
    """The response_format schema handed to the model must be strict —
    OpenAI's json_schema mode requires this so extra fields are refused."""
    schema = model_response_json_schema()
    assert schema["name"] == "answer"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_json_schema_includes_contradictions_field() -> None:
    schema = model_response_json_schema()
    assert "contradictions" in schema["schema"]["required"]
    assert schema["schema"]["properties"]["contradictions"]["type"] == "array"
