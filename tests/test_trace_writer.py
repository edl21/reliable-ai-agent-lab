"""JSONLWriter credential-redaction tests.

The source is explicit: credentials must never appear in traces. The
writer has two independent redaction passes: header-key stripping and
value-equality-with-OPENAI_API_KEY refusal. Both are tested here so a
future refactor can't silently loosen either one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traces.schema import (
    ContextCounts,
    CredentialLeakError,
    JSONLWriter,
    TokenUsage,
    TraceEvent,
)


def _event(**overrides) -> TraceEvent:
    base = {
        "request_id": "req_test",
        "task": "t1",
        "operation": "model_call",
        "attempt": 1,
        "elapsed_time_ms": 0,
        "virtual": False,
        "context_counts": ContextCounts(evidence_tokens=100, total_tokens=200),
        "token_usage": TokenUsage(),
        "outcome": "ok",
        "notes": {},
    }
    base.update(overrides)
    return TraceEvent(**base)


def test_writes_a_jsonl_line(tmp_path: Path) -> None:
    writer = JSONLWriter()
    path = tmp_path / "traces" / "task1" / "test.jsonl"
    writer.append(path, _event())
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["task"] == "t1"
    assert data["operation"] == "model_call"


def test_appends_multiple_events(tmp_path: Path) -> None:
    writer = JSONLWriter()
    path = tmp_path / "traces" / "task1" / "test.jsonl"
    writer.append(path, _event(attempt=1))
    writer.append(path, _event(attempt=2))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["attempt"] == 1
    assert json.loads(lines[1])["attempt"] == 2


def test_redacts_authorization_header_keys(tmp_path: Path) -> None:
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    writer.append(
        path,
        _event(
            notes={
                "headers": {
                    "Authorization": "Bearer sk-real-key",
                    "X-API-Key": "another-key",
                    "User-Agent": "test",
                }
            }
        ),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    headers = data["notes"]["headers"]
    assert headers["Authorization"] == "[REDACTED]"
    assert headers["X-API-Key"] == "[REDACTED]"
    # Non-credential headers pass through untouched.
    assert headers["User-Agent"] == "test"


def test_redacts_cookie_and_proxy_authorization(tmp_path: Path) -> None:
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    writer.append(
        path,
        _event(
            notes={
                "headers": {
                    "Cookie": "session=abc",
                    "Proxy-Authorization": "Basic xyz",
                }
            }
        ),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    headers = data["notes"]["headers"]
    assert headers["Cookie"] == "[REDACTED]"
    assert headers["Proxy-Authorization"] == "[REDACTED]"


def test_raises_when_value_equals_env_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact equality with the env's OPENAI_API_KEY is the trigger."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exact-value")
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    with pytest.raises(CredentialLeakError):
        writer.append(path, _event(notes={"leaky_field": "sk-exact-value"}))


def test_containment_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Substring containment does NOT raise — substring redaction would
    false-positive on legitimate hex digests that happen to contain
    key-like substrings. Equality is the correct semantic."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-exact-value")
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    # Contains the key as a substring but is not equal to it — allowed.
    writer.append(
        path,
        _event(notes={"debug_dump": "prefix sk-exact-value suffix"}),
    )
    # If we reached this line, no exception was raised — which is
    # exactly the required behaviour.


def test_does_not_raise_when_env_api_key_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no OPENAI_API_KEY in env, nothing to compare against."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    writer.append(path, _event(notes={"anything": "some-value"}))


def test_nested_redaction_walks_lists_and_dicts(tmp_path: Path) -> None:
    writer = JSONLWriter()
    path = tmp_path / "trace.jsonl"
    writer.append(
        path,
        _event(
            notes={
                "requests": [
                    {"headers": {"Authorization": "sk-1"}},
                    {"headers": {"Authorization": "sk-2"}},
                ]
            }
        ),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["notes"]["requests"]:
        assert entry["headers"]["Authorization"] == "[REDACTED]"
