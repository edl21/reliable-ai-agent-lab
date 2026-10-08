"""Task 1 endpoint integration tests via FastAPI TestClient.

These tests exercise the full pipeline (retrieval + chapter guard +
token budgeting + stub model + post-hoc verifiers + trace emission)
against the built corpus. They are skipped if the ingest artefacts
aren't present, because we cannot rebuild the corpus in a sandboxed
test environment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


# The tests need the ingest pipeline outputs. Skip if the reviewer
# hasn't run ``python -m ingest.build`` yet.
_INGEST_READY = (
    Path("corpus/manifest.json").exists()
    and Path("corpus/chapter_map.json").exists()
    and Path("corpus/chunks.jsonl").exists()
    and Path("chroma_db").exists()
)

needs_ingest = pytest.mark.skipif(
    not _INGEST_READY,
    reason="corpus/chroma not built; run `python -m ingest.build`",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --- pre-generation validation ------------------------------------------


def test_missing_max_chapter_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        json={"request_id": "test_missing", "question": "q"},
    )
    assert resp.status_code == 422


def test_non_integer_max_chapter_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        json={"request_id": "test_nonint", "question": "q", "max_chapter": "five"},
    )
    assert resp.status_code == 422


def test_zero_max_chapter_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        json={"request_id": "test_zero", "question": "q", "max_chapter": 0},
    )
    assert resp.status_code == 422


def test_extra_fields_return_422(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        json={
            "request_id": "test_extra",
            "question": "q",
            "max_chapter": 1,
            "hack": "unwanted",
        },
    )
    assert resp.status_code == 422


def test_unsafe_request_id_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        json={
            "request_id": "../evil",
            "question": "q",
            "max_chapter": 1,
        },
    )
    assert resp.status_code == 422


# --- full pipeline (needs ingest) ---------------------------------------


@needs_ingest
def test_out_of_range_max_chapter_returns_422_and_writes_trace(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.post(
        "/answer",
        params={"trace_check_name": "unit_oor"},
        json={
            "request_id": "test_oor_01",
            "question": "q",
            "max_chapter": 999,
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"]["error"]["code"] == "max_chapter_out_of_range"
    # Trace file at pinned path.
    trace = Path("traces/task1/unit_oor_test_oor_01.jsonl")
    assert trace.exists()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    # CRITICAL: no model_call event on a pre-generation rejection.
    ops = [e["operation"] for e in events]
    assert "model_call" not in ops
    assert "validation" in ops


@needs_ingest
def test_valid_request_returns_200_and_full_trace(client: TestClient) -> None:
    resp = client.post(
        "/answer",
        params={"trace_check_name": "unit_valid"},
        json={
            "request_id": "test_valid_01",
            "question": "Who is Ishmael?",
            "max_chapter": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Response shape (source-required fields).
    assert body["request_id"] == "test_valid_01"
    assert body["status"] in {"answered", "insufficient_evidence"}
    assert isinstance(body["answer"], str)
    assert isinstance(body["citations"], list)
    assert isinstance(body["contradictions"], list)  # adversarial-review fix
    assert "metrics" in body
    # Metrics fields.
    m = body["metrics"]
    assert "evidence_tokens" in m
    assert m["evidence_tokens"] <= 8_000  # source-pinned budget
    # Trace file.
    trace = Path("traces/task1/unit_valid_test_valid_01.jsonl")
    assert trace.exists()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    ops = [e["operation"] for e in events]
    assert "retrieval" in ops
    assert "limit_check" in ops
    assert "model_call" in ops
    assert "validation" in ops
    # No cited chapter exceeds the limit (post-hoc verifier).
    for cit in body["citations"]:
        assert cit["chapter"] <= 3


@needs_ingest
def test_chapter_limit_actually_filters_retrieval(client: TestClient) -> None:
    """Two identical questions with different max_chapter — the smaller
    limit should NEVER produce citations from chapters above it."""
    resp_low = client.post(
        "/answer",
        params={"trace_check_name": "unit_filter_low"},
        json={
            "request_id": "test_filter_low",
            "question": "Where does the story end?",
            "max_chapter": 1,
        },
    )
    assert resp_low.status_code == 200

    trace_low = Path("traces/task1/unit_filter_low_test_filter_low.jsonl")
    events = [json.loads(line) for line in trace_low.read_text().splitlines()]
    retrieval_events = [e for e in events if e["operation"] == "retrieval"]
    assert retrieval_events
    # All retrieved chapter IDs must be <= 1.
    for e in retrieval_events:
        for cid in e["source_chapter_ids"]:
            assert cid <= 1, f"leak: chapter {cid} retrieved at max_chapter=1"
