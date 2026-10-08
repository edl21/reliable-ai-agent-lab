"""Authoritative citation/contradiction verification for /answer.

Covers the audit finding where live-style paraphrased excerpts and
mismatched citation metadata previously still returned HTTP 200 with
status=answered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from adapter.model import AdapterResponse
from adapter import runtime
from app.main import app
from adapter.tokens import EVIDENCE_BUDGET, count_text
from app.routes.answer import (
    _MIN_EXCERPT_CHARS,
    _assemble_evidence,
    _format_evidence_block,
    _run_post_hoc_verifiers,
)
from app.schemas import Citation, Contradiction, ContradictionPosition, ModelAnswer
from ingest.index import RetrievedChunk


def _chunk(**overrides: Any) -> RetrievedChunk:
    base = dict(
        chunk_id="abcdef0123456789",
        chapter=1,
        chapter_original_label="Chapter 1",
        chapter_title="Loomings",
        pdf_pages=[10, 11],
        source_filename="moby-dick.pdf",
        text="Ishmael decided to go to sea.",
        distance=0.1,
    )
    base.update(overrides)
    return RetrievedChunk(**base)


def test_verifier_accepts_verbatim_grounded_citation() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer=f"Ishmael decided to go to sea [{chunk.chunk_id}].",
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                chapter=chunk.chapter,
                pdf_pages=list(chunk.pdf_pages),
                excerpt="Ishmael decided to go to sea.",
            )
        ],
        contradictions=[],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    assert notes["errors"] is False
    assert notes["verifier_errors"] == []


def test_verifier_rejects_paraphrased_excerpt() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer=f"Ishmael went to sea [{chunk.chunk_id}].",
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                chapter=chunk.chapter,
                pdf_pages=list(chunk.pdf_pages),
                excerpt="Ishmael boarded a different ship.",
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "excerpt_not_substring" in kinds


def test_verifier_rejects_empty_excerpt_and_metadata_mismatch() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer=f"Claim [{chunk.chunk_id}].",
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename="",
                chapter=99,
                pdf_pages=[999],
                excerpt="",
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=3)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "citation_excerpt_empty" in kinds
    assert "citation_source_filename_mismatch" in kinds
    assert "citation_chapter_mismatch" in kinds
    assert "citation_chapter_over_limit" in kinds
    assert "citation_pdf_pages_not_allowed" in kinds


def test_verifier_rejects_bad_contradiction_position() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer="Disputed claim.",
        citations=[],
        contradictions=[
            Contradiction(
                claim="age",
                positions=[
                    ContradictionPosition(
                        chunk_id=chunk.chunk_id,
                        stance="supports",
                        excerpt="not actually in the passage",
                    )
                ],
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "contradiction_excerpt_not_substring" in kinds


def test_verifier_rejects_answered_without_citations() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer="Wholly invented fact about Ishmael with no support.",
        citations=[],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "answered_without_citations" in kinds
    assert "answered_without_inline_citations" in kinds
    assert "uncited_factual_sentences" in kinds


def test_verifier_rejects_mixed_cited_and_uncited_claims() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="answered",
        answer=(
            "Invented prelude about dragons and treasure. "
            f"Ishmael decided to go to sea [{chunk.chunk_id}]."
        ),
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                chapter=chunk.chapter,
                pdf_pages=list(chunk.pdf_pages),
                excerpt="Ishmael decided to go to sea.",
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "uncited_factual_sentences" in kinds


def test_verifier_rejects_trivial_excerpt() -> None:
    chunk = _chunk(text="Ishmael decided to go to sea.")
    answer = ModelAnswer(
        status="answered",
        answer=f"Ishmael decided to go to sea [{chunk.chunk_id}].",
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                chapter=chunk.chapter,
                pdf_pages=list(chunk.pdf_pages),
                excerpt="of",
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "citation_excerpt_too_short" in kinds
    assert _MIN_EXCERPT_CHARS >= 20


def test_assemble_evidence_counts_formatted_headers_in_budget() -> None:
    """Headers inflate evidence; body-only counting would under-report."""
    chunks = []
    for i in range(30):
        # Large bodies so the formatted block (headers + text) exceeds 8k.
        body = ("passage text token " * 200) + f" chunk-{i}"
        chunks.append(
            _chunk(
                chunk_id=f"{i:016x}",
                text=body,
                chapter=1,
            )
        )
    surviving, evidence_tokens, _total = _assemble_evidence(
        chunks,
        question="Who announced a party?",
    )
    formatted = _format_evidence_block(surviving)
    assert evidence_tokens == count_text(formatted)
    assert evidence_tokens <= EVIDENCE_BUDGET
    assert len(surviving) < len(chunks)
    # Formatted count must be strictly greater than bare body sum.
    body_only = sum(count_text(c.text) for c in surviving)
    assert evidence_tokens > body_only


def test_verifier_rejects_citations_on_insufficient_evidence() -> None:
    chunk = _chunk()
    answer = ModelAnswer(
        status="insufficient_evidence",
        answer=f"Guess [{chunk.chunk_id}].",
        citations=[
            Citation(
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                chapter=chunk.chapter,
                pdf_pages=list(chunk.pdf_pages),
                excerpt="Ishmael decided to go to sea.",
            )
        ],
    )
    notes = _run_post_hoc_verifiers(answer, [chunk], max_chapter=1)
    kinds = {e["kind"] for e in notes["verifier_errors"]}
    assert "citations_present_on_insufficient_evidence" in kinds
    assert "inline_citations_on_insufficient_evidence" in kinds


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


class _ParaphraseThenFixAdapter:
    """First call: live-style bad citations. Later calls: grounded fix."""

    def __init__(self) -> None:
        self.calls = 0
        self.tracker = None
        self.clock = None

    def chat(self, **kwargs: Any) -> AdapterResponse:
        self.calls += 1
        messages = kwargs.get("messages") or []
        evidence_ids: list[str] = []
        evidence_text = ""
        evidence_filename = "moby-dick.pdf"
        evidence_chapter = 1
        evidence_pages = [1]
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, str) and content.startswith("["):
                # not typical; fall through
                pass
            if isinstance(content, str) and "Evidence passages" in content:
                # Parse first evidence header: [chunk_id] Chapter N ... pdf_pages=[...]
                import re

                m = re.search(
                    r"\[([a-f0-9]{16})\] Chapter (\d+).*?pdf_pages=\[([^\]]*)\]\n(.+?)(?:\n\[|\n\nQuestion:)",
                    content,
                    re.S,
                )
                if m:
                    evidence_ids.append(m.group(1))
                    evidence_chapter = int(m.group(2))
                    pages_raw = m.group(3).strip()
                    evidence_pages = (
                        [int(p.strip()) for p in pages_raw.split(",") if p.strip()]
                        if pages_raw
                        else [1]
                    )
                    evidence_text = m.group(4).strip().split("\n\n")[0].strip()
                fm = re.search(r"\(([^)]+\.pdf)\)", content)
                # filename is not in header that way; keep default from retrieve
                _ = fm

        chunk_id = evidence_ids[0] if evidence_ids else "abcdef0123456789"
        if self.calls == 1:
            body = {
                "status": "answered",
                "answer": f"Ishmael went to sea suddenly [{chunk_id}].",
                "citations": [
                    {
                        "chunk_id": chunk_id,
                        "source_filename": "",
                        "chapter": evidence_chapter,
                        "pdf_pages": evidence_pages or [1],
                        "excerpt": "Ishmael departed abruptly for sea in secret.",
                    }
                ],
                "contradictions": [],
            }
        else:
            excerpt = evidence_text[:120] if evidence_text else "placeholder"
            # On corrective turns, prefer a real substring when available.
            body = {
                "status": "answered",
                "answer": f"Evidence says: {excerpt[:40]} [{chunk_id}].",
                "citations": [
                    {
                        "chunk_id": chunk_id,
                        "source_filename": evidence_filename,
                        "chapter": evidence_chapter,
                        "pdf_pages": evidence_pages or [1],
                        "excerpt": excerpt if evidence_text else "",
                    }
                ],
                "contradictions": [],
            }
            # If we still lack a real excerpt, fall back to insufficient.
            if not evidence_text:
                body = {
                    "status": "insufficient_evidence",
                    "answer": "Cannot verify.",
                    "citations": [],
                    "contradictions": [],
                }

        # Repair filename from retrieved chunk via lookup if needed on fix turn.
        if self.calls > 1 and evidence_text:
            from ingest.index import lookup_chunk_by_id

            real = lookup_chunk_by_id(chunk_id)
            if real is not None:
                body["citations"][0]["source_filename"] = real.source_filename
                body["citations"][0]["chapter"] = real.chapter
                body["citations"][0]["pdf_pages"] = list(real.pdf_pages)
                body["citations"][0]["excerpt"] = real.text[: min(120, len(real.text))]
                body["answer"] = (
                    f"Supported by passage text [{chunk_id}]."
                )

        return AdapterResponse(
            content=json.dumps(body),
            tool_calls=[],
            usage={"input_tokens": 10, "output_tokens": 20},
            raw={"stub": True},
            stubbed=True,
        )


class _AlwaysBadCitationAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.tracker = None
        self.clock = None

    def chat(self, **kwargs: Any) -> AdapterResponse:
        self.calls += 1
        body = {
            "status": "answered",
            "answer": "Invented fact [abcdef0123456789].",
            "citations": [
                {
                    "chunk_id": "abcdef0123456789",
                    "source_filename": "",
                    "chapter": 1,
                    "pdf_pages": [1],
                    "excerpt": "this excerpt is not in any evidence chunk",
                }
            ],
            "contradictions": [],
        }
        return AdapterResponse(
            content=json.dumps(body),
            tool_calls=[],
            usage={"input_tokens": 5, "output_tokens": 5},
            raw={"stub": True},
            stubbed=True,
        )


@needs_ingest
def test_paraphrased_citation_is_corrected_or_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _ParaphraseThenFixAdapter()
    monkeypatch.setattr(runtime, "_OVERRIDE", adapter)
    client = TestClient(app)
    resp = client.post(
        "/answer",
        params={"trace_check_name": "unit_paraphrase"},
        json={
            "request_id": "paraphrase_probe_01",
            "question": "Who is Ishmael?",
            "max_chapter": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"answered", "insufficient_evidence"}
    if body["status"] == "answered":
        assert body["citations"], "answered requires citations"
        for cit in body["citations"]:
            assert cit["excerpt"]
            assert cit["source_filename"]
    else:
        assert body["citations"] == []

    trace = Path("traces/task1/unit_paraphrase_paraphrase_probe_01.jsonl")
    assert trace.exists()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    validations = [e for e in events if e["operation"] == "validation"]
    assert any(
        e.get("outcome") == "error"
        and (e.get("notes") or {}).get("stage") == "grounding"
        for e in validations
    ), "expected an initial grounding validation failure"
    # No terminal successful answer with lingering verifier errors.
    final_ok = [e for e in validations if e.get("outcome") == "ok"]
    if body["status"] == "answered":
        assert final_ok
    else:
        assert any(
            e.get("outcome") in {"insufficient_evidence", "error"}
            for e in validations
        )


@needs_ingest
def test_unfixable_citations_never_return_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _AlwaysBadCitationAdapter()
    monkeypatch.setattr(runtime, "_OVERRIDE", adapter)
    client = TestClient(app)
    resp = client.post(
        "/answer",
        params={"trace_check_name": "unit_bad_cite"},
        json={
            "request_id": "bad_cite_probe_01",
            "question": "Who is Ishmael?",
            "max_chapter": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert adapter.calls >= 1

    trace = Path("traces/task1/unit_bad_cite_bad_cite_probe_01.jsonl")
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(
        e["operation"] == "validation" and e.get("outcome") == "error"
        for e in events
    )
    assert any(
        e["operation"] == "validation"
        and e.get("outcome") == "insufficient_evidence"
        for e in events
    )
