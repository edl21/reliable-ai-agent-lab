"""Behaviour tests for ``adapter/limits.py`` and the pass-through
skeleton of ``adapter/faults.py``. Full R1-R6 recipe tests land in
Phase 4; these tests only lock the Phase-1 contracts."""

from __future__ import annotations

import pytest

from adapter.clock import VirtualClock, VirtualSleeper
from adapter.errors import LimitExhausted, RetriesExhausted
from adapter.faults import FaultConfig, FaultInjector
from adapter.limits import LimitTracker, RunLimits
from adapter.model import AdapterResponse, StubModelAdapter


# --- LimitTracker ---------------------------------------------------------


def test_attempts_per_op_caps_retries() -> None:
    tracker = LimitTracker(limits=RunLimits(attempts_per_op=2))
    tracker.pre_call(op_id="op_a")
    tracker.pre_call(op_id="op_a")
    with pytest.raises(RetriesExhausted):
        tracker.pre_call(op_id="op_a")


def test_attempts_per_op_is_scoped_to_the_op() -> None:
    tracker = LimitTracker(limits=RunLimits(attempts_per_op=1, total_calls=10))
    tracker.pre_call(op_id="op_a")
    # A DIFFERENT op still has budget.
    tracker.pre_call(op_id="op_b")


def test_total_calls_caps_run() -> None:
    tracker = LimitTracker(limits=RunLimits(total_calls=2, attempts_per_op=99))
    tracker.pre_call(op_id="op_a")
    tracker.pre_call(op_id="op_b")
    with pytest.raises(LimitExhausted):
        tracker.pre_call(op_id="op_c")


def test_cumulative_tokens_cap_raises() -> None:
    tracker = LimitTracker(limits=RunLimits(cumulative_tokens=1000))
    tracker.pre_call(op_id="op_a", estimated_input_tokens=800, reserved_output_tokens=100)
    with pytest.raises(LimitExhausted):
        tracker.pre_call(op_id="op_b", estimated_input_tokens=200)


def test_elapsed_deadline_uses_injectable_clock() -> None:
    vc = VirtualClock()
    vs = VirtualSleeper(clock=vc)
    tracker = LimitTracker(
        limits=RunLimits(elapsed_seconds=5.0, total_calls=99), clock=vc
    )
    tracker.start_run()
    assert tracker.remaining_deadline() == 5.0
    vs.sleep(3.0)
    assert tracker.remaining_deadline() == pytest.approx(2.0)
    vs.sleep(3.0)
    with pytest.raises(LimitExhausted):
        tracker.pre_call(op_id="op_a")


def test_would_wait_exceed_deadline() -> None:
    vc = VirtualClock()
    tracker = LimitTracker(limits=RunLimits(elapsed_seconds=5.0), clock=vc)
    tracker.start_run()
    assert not tracker.would_wait_exceed_deadline(2.0)
    assert tracker.would_wait_exceed_deadline(5.0)
    assert tracker.would_wait_exceed_deadline(10.0)


def test_record_actual_tokens_reconciles_reservation() -> None:
    tracker = LimitTracker(limits=RunLimits(cumulative_tokens=1000))
    tracker.pre_call(op_id="op_a", estimated_input_tokens=100, reserved_output_tokens=100)
    # Actual usage came in at 150 total, but we reserved 200.
    tracker.record_actual_tokens(reserved=200, actual=150)
    assert tracker.tokens_used == 150


# --- FaultInjector (Phase-1 skeleton) -------------------------------------


def test_empty_fault_config_is_pass_through() -> None:
    stub = StubModelAdapter()
    injector = FaultInjector(stub)
    assert injector.config.is_empty()
    resp = injector.chat(messages=[{"role": "user", "content": "hi"}])
    assert isinstance(resp, AdapterResponse)
    assert resp.stubbed is True


def test_injector_exposes_inner_for_liveness_assertion() -> None:
    stub = StubModelAdapter()
    injector = FaultInjector(stub)
    assert injector.inner is stub


def test_injector_bumps_call_count_even_in_passthrough() -> None:
    stub = StubModelAdapter()
    cfg = FaultConfig()
    injector = FaultInjector(stub, config=cfg)
    assert cfg._call_count == 0
    injector.chat(messages=[{"role": "user", "content": "one"}])
    injector.chat(messages=[{"role": "user", "content": "two"}])
    assert cfg._call_count == 2


# --- StubModelAdapter -----------------------------------------------------


def test_stub_returns_deterministic_content_for_same_input() -> None:
    stub = StubModelAdapter()
    msgs = [{"role": "user", "content": "hello"}]
    a = stub.chat(messages=msgs)
    b = stub.chat(messages=msgs)
    assert a.content == b.content
    assert a.stubbed and b.stubbed


def test_stub_emits_schema_valid_json_when_response_format_requested() -> None:
    import json

    stub = StubModelAdapter()
    resp = stub.chat(
        messages=[{"role": "user", "content": "q"}],
        response_format={"type": "json_schema", "json_schema": {"name": "answer"}},
    )
    parsed = json.loads(resp.content)
    # We do not lock exact fields here — only that it parses and
    # advertises insufficient_evidence so no live-facts leak.
    assert parsed["status"] == "insufficient_evidence"
    assert isinstance(parsed["citations"], list)
    assert isinstance(parsed["contradictions"], list)


# --- get_model() singleton ------------------------------------------------


def test_get_model_returns_same_instance_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    import adapter.model as model_mod

    model_mod._reset_singleton_for_tests()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    a = model_mod.get_model()
    b = model_mod.get_model()
    assert a is b
    assert isinstance(a, StubModelAdapter)
    model_mod._reset_singleton_for_tests()


def test_get_model_returns_stub_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import adapter.model as model_mod

    model_mod._reset_singleton_for_tests()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    m = model_mod.get_model()
    assert isinstance(m, StubModelAdapter)
    model_mod._reset_singleton_for_tests()


def test_direct_model_adapter_construction_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adapter.model import ModelAdapter

    with pytest.raises(RuntimeError, match="Do not construct ModelAdapter directly"):
        ModelAdapter(
            base_url="http://example.invalid/v1",
            api_key="sk-fake",
            model_name="stub",
        )
