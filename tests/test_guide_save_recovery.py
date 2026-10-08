"""Guide-loop recovery: paraphrased save_guide, correction, context trim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from adapter.model import AdapterResponse
from adapter import runtime
from app.main import app
from app.routes.guide import (
    _bounded_tool_content,
    _guide_context_reducer,
    _save_guide_correction_prompt,
)
from app.tools.schemas import SaveGuideArgs
from pydantic import ValidationError


_INGEST_READY = (
    Path("corpus/manifest.json").exists()
    and Path("chroma_db").exists()
)

needs_ingest = pytest.mark.skipif(
    not _INGEST_READY,
    reason="corpus/chroma not built",
)


def test_save_guide_args_require_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        SaveGuideArgs(title="t", content="c", citations=[])


def test_save_guide_correction_prompt_mentions_verbatim_excerpt() -> None:
    prompt = _save_guide_correction_prompt(
        {"ok": False, "error": "citation excerpt is not from chunk x"}
    )
    assert "verbatim" in prompt.lower()
    assert "save_guide" in prompt


def test_bounded_tool_content_truncates_huge_payloads() -> None:
    huge = "word " * 20_000
    out = _bounded_tool_content(huge)
    assert len(out) < len(huge)
    assert "truncated" in out


def test_guide_context_reducer_truncates_old_tool_blobs() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "goal"},
        {
            "role": "tool",
            "tool_call_id": "1",
            "name": "fetch_passage",
            "content": "x" * 5_000,
        },
        {
            "role": "tool",
            "tool_call_id": "2",
            "name": "fetch_passage",
            "content": "y" * 5_000,
        },
    ]
    reduced, notes = _guide_context_reducer(messages, reported_limit=500)
    assert notes.get("trimmed") is True
    tool_lens = [
        len(m.get("content") or "")
        for m in reduced
        if m.get("role") == "tool"
    ]
    assert tool_lens
    assert min(tool_lens) < 5_000


class _ParaphraseThenFixGuideAdapter:
    """Rejectable paraphrased save, then a verbatim save from fetch content."""

    def __init__(self) -> None:
        self.calls = 0
        self.tracker = None
        self.clock = None
        self._fetched: dict[str, Any] | None = None

    def chat(self, **kwargs: Any) -> AdapterResponse:
        self.calls += 1
        messages = kwargs.get("messages") or []

        # Capture last successful fetch payload from tool messages.
        for msg in messages:
            if msg.get("role") != "tool" or msg.get("name") != "fetch_passage":
                continue
            try:
                payload = json.loads(msg.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            if payload.get("chunk_id") and payload.get("content"):
                self._fetched = payload

        if self.calls == 1:
            return AdapterResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "search_passages",
                            "arguments": json.dumps({"query": "ishmael queequeg"}),
                        },
                    }
                ],
                usage={"input_tokens": 50, "output_tokens": 10},
                stubbed=True,
            )

        if self.calls == 2:
            # Pick first search result chunk_id from prior tool message.
            chunk_id = "abcdef0123456789"
            for msg in messages:
                if msg.get("role") != "tool" or msg.get("name") != "search_passages":
                    continue
                try:
                    payload = json.loads(msg.get("content") or "{}")
                except json.JSONDecodeError:
                    continue
                results = payload.get("results") or []
                if results:
                    chunk_id = results[0]["chunk_id"]
                    break
            return AdapterResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_fetch",
                        "type": "function",
                        "function": {
                            "name": "fetch_passage",
                            "arguments": json.dumps({"chunk_id": chunk_id}),
                        },
                    }
                ],
                usage={"input_tokens": 60, "output_tokens": 10},
                stubbed=True,
            )

        if self.calls == 3:
            # Paraphrased excerpt — must be rejected by save_guide.
            fetched = self._fetched or {}
            return AdapterResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_save_bad",
                        "type": "function",
                        "function": {
                            "name": "save_guide",
                            "arguments": json.dumps(
                                {
                                    "title": "Ishmael guide",
                                    "content": "Ishmael goes to sea.",
                                    "citations": [
                                        {
                                            "chunk_id": fetched.get(
                                                "chunk_id", "abcdef0123456789"
                                            ),
                                            "source_filename": fetched.get(
                                                "source_filename", "x.pdf"
                                            ),
                                            "chapter": fetched.get("chapter", 1),
                                            "pdf_pages": fetched.get(
                                                "pdf_pages", [1]
                                            ),
                                            "excerpt": (
                                                "Ishmael departed abruptly for "
                                                "the sea in secret."
                                            ),
                                        }
                                    ],
                                }
                            ),
                        },
                    }
                ],
                usage={"input_tokens": 80, "output_tokens": 20},
                stubbed=True,
            )

        # Corrective turn: use verbatim excerpt from fetch content.
        fetched = self._fetched or {}
        content = str(fetched.get("content") or "")
        excerpt = content[:120] if content else "missing"
        return AdapterResponse(
            content=None,
            tool_calls=[
                {
                    "id": "call_save_ok",
                    "type": "function",
                    "function": {
                        "name": "save_guide",
                        "arguments": json.dumps(
                            {
                                "title": "Ishmael guide",
                                "content": "Ishmael is introduced early.",
                                "citations": [
                                    {
                                        "chunk_id": fetched.get(
                                            "chunk_id", "abcdef0123456789"
                                        ),
                                        "source_filename": fetched.get(
                                            "source_filename", "x.pdf"
                                        ),
                                        "chapter": fetched.get("chapter", 1),
                                        "pdf_pages": fetched.get("pdf_pages", [1]),
                                        "excerpt": excerpt,
                                    }
                                ],
                            }
                        ),
                    },
                }
            ],
            usage={"input_tokens": 90, "output_tokens": 20},
            stubbed=True,
        )


@needs_ingest
def test_guide_recovers_from_paraphrased_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _ParaphraseThenFixGuideAdapter()
    monkeypatch.setattr(runtime, "_OVERRIDE", adapter)
    client = TestClient(app)
    resp = client.post(
        "/guide",
        params={"trace_check_name": "unit_guide_paraphrase"},
        json={
            "request_id": "guide_paraphrase_01",
            "goal": "Write a short cited guide about Ishmael in early chapters.",
            "max_chapter": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending_approval"
    assert body.get("operation_id")
    assert body["metrics"]["rejected_save_attempts"] >= 1

    trace = Path("traces/task2/unit_guide_paraphrase_guide_paraphrase_01.jsonl")
    assert trace.exists()
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    save_events = [
        e
        for e in events
        if e.get("operation") == "tool_call"
        and (e.get("notes") or {}).get("tool_name") == "save_guide"
    ]
    assert any(e.get("outcome") == "rejected" for e in save_events)
    assert any(e.get("outcome") == "ok" for e in save_events)
