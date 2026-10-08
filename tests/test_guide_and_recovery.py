"""Guide terminal-state and recovery accounting tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from adapter.errors import ContextLengthExceeded
from adapter.limits import LimitTracker, RunLimits
from adapter.model import AdapterResponse
from adapter.recovery import chat_with_recovery
from adapter import runtime
from app.main import app
from traces.schema import JSONLWriter


class _EndWithoutSaveAdapter:
    """Search once, then stop without save_guide — reproduces the audit bug."""

    def __init__(self) -> None:
        self.calls = 0
        self.tracker = LimitTracker(limits=RunLimits(attempts_per_op=5, total_calls=20))
        self.clock = None

    def chat(self, **kwargs: Any) -> AdapterResponse:
        self.calls += 1
        if self.calls == 1:
            return AdapterResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "search_passages",
                            "arguments": json.dumps({"query": "ishmael"}),
                        },
                    }
                ],
                usage={"input_tokens": 100, "output_tokens": 20},
                stubbed=True,
            )
        return AdapterResponse(
            content="I am done without saving.",
            tool_calls=[],
            usage={"input_tokens": 120, "output_tokens": 10},
            stubbed=True,
        )


_INGEST_READY = (
    Path("corpus/manifest.json").exists()
    and Path("chroma_db").exists()
)

needs_ingest = pytest.mark.skipif(
    not _INGEST_READY,
    reason="corpus/chroma not built",
)


@needs_ingest
def test_guide_returns_error_when_model_ends_without_pending_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _EndWithoutSaveAdapter()
    monkeypatch.setattr(runtime, "_OVERRIDE", adapter)
    client = TestClient(app)
    resp = client.post(
        "/guide",
        params={"trace_check_name": "unit_guide_incomplete"},
        json={
            "request_id": "guide_incomplete_01",
            "goal": "Write a short guide about Ishmael.",
            "max_chapter": 2,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "guide_incomplete"
    assert body.get("operation_id") is None

    trace = Path("traces/task2/unit_guide_incomplete_guide_incomplete_01.jsonl")
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(
        e.get("operation") == "validation"
        and (e.get("notes") or {}).get("reason") == "ended_without_pending_draft"
        for e in events
    )
    tool_events = [e for e in events if e.get("operation") == "tool_call"]
    assert tool_events
    # Chapter IDs come from tool results, not the requested limit alone.
    for event in tool_events:
        for chapter in event.get("source_chapter_ids") or []:
            assert chapter <= 2


def test_recovery_reconciles_live_usage_and_emits_token_fields(
    tmp_path: Path,
) -> None:
    class _UsageAdapter:
        tracker = LimitTracker(limits=RunLimits())
        clock = None

        def chat(self, **kwargs: Any) -> AdapterResponse:
            return AdapterResponse(
                content='{"status":"insufficient_evidence","answer":"x","citations":[],"contradictions":[]}',
                usage={"input_tokens": 50, "output_tokens": 7},
                stubbed=False,
            )

    writer = JSONLWriter()
    path = tmp_path / "rec.jsonl"
    tracker = LimitTracker(limits=RunLimits())
    result = chat_with_recovery(
        _UsageAdapter(),
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        response_format=None,
        max_output_tokens=100,
        temperature=0.0,
        parallel_tool_calls=None,
        operation_id="usage_test",
        task="t1",
        request_id="r",
        run_id="run",
        trace_path=path,
        writer=writer,
        tracker=tracker,
    )
    assert result.response.usage["input_tokens"] == 50
    events = [json.loads(line) for line in path.read_text().splitlines()]
    model_events = [e for e in events if e["operation"] == "model_call"]
    assert model_events
    usage = model_events[-1]["token_usage"]
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 57
    assert model_events[-1]["context_counts"]["total_tokens"] > 0
    # Reservation reconciled down toward actual usage.
    assert tracker.tokens_used == 57


def test_recovery_pre_call_context_guard_rejects_without_reducer(
    tmp_path: Path,
) -> None:
    class _BoomAdapter:
        tracker = LimitTracker(limits=RunLimits())
        clock = None

        def chat(self, **kwargs: Any) -> AdapterResponse:
            raise AssertionError("adapter must not be called when over limit")

    writer = JSONLWriter()
    path = tmp_path / "guard.jsonl"
    huge = "word " * 50_000
    with pytest.raises(ContextLengthExceeded) as excinfo:
        chat_with_recovery(
            _BoomAdapter(),
            messages=[{"role": "user", "content": huge}],
            tools=None,
            response_format=None,
            max_output_tokens=100,
            temperature=0.0,
            parallel_tool_calls=None,
            operation_id="guard_test",
            task="t1",
            request_id="r",
            run_id="run",
            trace_path=path,
            writer=writer,
            tracker=LimitTracker(limits=RunLimits()),
            context_limit=100,
        )
    assert excinfo.value.reported_limit == 100
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(
        e.get("operation") == "limit_check"
        and (e.get("notes") or {}).get("reason") == "pre_call_context_guard"
        for e in events
    )


def test_recovery_pre_call_context_guard_reduces_with_reducer(
    tmp_path: Path,
) -> None:
    calls = {"n": 0}

    class _OkAfterReduce:
        tracker = LimitTracker(limits=RunLimits())
        clock = None

        def chat(self, **kwargs: Any) -> AdapterResponse:
            calls["n"] += 1
            return AdapterResponse(
                content="ok",
                usage={"input_tokens": 10, "output_tokens": 1},
                stubbed=True,
            )

    def reducer(messages: list[dict[str, Any]], reported_limit: int):
        return [{"role": "user", "content": "tiny"}], {"trimmed": True}

    writer = JSONLWriter()
    path = tmp_path / "reduce.jsonl"
    huge = "word " * 20_000
    result = chat_with_recovery(
        _OkAfterReduce(),
        messages=[{"role": "user", "content": huge}],
        tools=None,
        response_format=None,
        max_output_tokens=50,
        temperature=0.0,
        parallel_tool_calls=None,
        operation_id="reduce_test",
        task="t1",
        request_id="r",
        run_id="run",
        trace_path=path,
        writer=writer,
        tracker=LimitTracker(limits=RunLimits()),
        reducer=reducer,
        context_limit=200,
    )
    assert result.response.content == "ok"
    assert calls["n"] == 1
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(e.get("recovery_decision") == "reduce_context" for e in events)
