"""Direct fault-wrapper unit tests; route-level evidence is in checks/."""

from __future__ import annotations

import pytest

from adapter.clock import VirtualClock, VirtualSleeper
from adapter.errors import AuthError, ContextLengthExceeded, RateLimited
from adapter.faults import FaultConfig, FaultInjector
from adapter.model import StubModelAdapter


def _injector(config: FaultConfig) -> FaultInjector:
    clock = VirtualClock()
    return FaultInjector(
        StubModelAdapter(),
        config=config,
        clock=clock,
        sleeper=VirtualSleeper(clock),
    )


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "hello"}]


def test_r1_injects_only_first_call() -> None:
    injector = _injector(FaultConfig(r1_first_call_429=True))
    with pytest.raises(RateLimited) as exc:
        injector.chat(messages=_messages())
    assert exc.value.retry_after == 2.0
    assert injector.chat(messages=_messages()).stubbed is True


def test_r2_injects_every_call() -> None:
    injector = _injector(FaultConfig(r2_all_calls_429=True))
    with pytest.raises(RateLimited):
        injector.chat(messages=_messages())
    with pytest.raises(RateLimited):
        injector.chat(messages=_messages())


def test_r4_reports_configured_limit_from_error() -> None:
    injector = _injector(FaultConfig(r4_context_threshold=3000))
    with pytest.raises(ContextLengthExceeded) as exc:
        injector.chat(messages=_messages(), max_output_tokens=4000)
    assert exc.value.reported_limit == 3000


def test_r5_returns_literal_invalid_schema_once() -> None:
    injector = _injector(FaultConfig(r5_first_call_bad_schema=True))
    first = injector.chat(
        messages=_messages(),
        response_format={"type": "json_schema"},
    )
    second = injector.chat(
        messages=_messages(),
        response_format={"type": "json_schema"},
    )
    assert first.content == '{"status": 42}'
    assert second.content != '{"status": 42}'


def test_r6_is_auth_terminal_at_wrapper_boundary() -> None:
    injector = _injector(FaultConfig(r6_all_calls_401=True))
    with pytest.raises(AuthError):
        injector.chat(messages=_messages())
    assert injector.config._call_count == 1


def test_empty_fault_config_keeps_inner_live_path_reachable() -> None:
    injector = _injector(FaultConfig())
    assert isinstance(injector.inner, StubModelAdapter)
    assert injector.chat(messages=_messages()).stubbed is True
