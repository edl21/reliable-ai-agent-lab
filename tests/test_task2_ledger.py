"""Task 2 ledger and dispatcher unit tests."""

from __future__ import annotations

from pathlib import Path

from app.ledger import ApprovalLedger
from app.tools.context import ToolContext
from app.tools.dispatch import ToolDispatcher


def _ledger(tmp_path: Path) -> ApprovalLedger:
    return ApprovalLedger(
        artifact_root=tmp_path / "artifacts",
        server_secret="test-secret",
    )


def test_pending_creation_writes_no_artifact(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    record = ledger.create_pending(
        title="Guide",
        content="Content",
        citations=[],
        chapter_limit=3,
    )
    assert record.state == "pending"
    assert not list((tmp_path / "artifacts").glob("*.json"))
    assert ledger.write_log.count(record.operation_id) == 0


def test_approval_writes_once_and_replay_is_idempotent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    record = ledger.create_pending(
        title="Guide",
        content="Content",
        citations=[],
        chapter_limit=3,
    )
    first, replayed_first = ledger.approve(
        operation_id=record.operation_id,
        approve=True,
        payload_check=record.payload_hash,
    )
    second, replayed_second = ledger.approve(
        operation_id=record.operation_id,
        approve=True,
        payload_check=record.payload_hash,
    )
    assert first.artifact_id == second.artifact_id
    assert not replayed_first
    assert replayed_second
    assert ledger.write_log.count(record.operation_id) == 1
    assert len(list((tmp_path / "artifacts").glob("*.json"))) == 1


def test_payload_hash_reuse_with_changed_content_is_rejected(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    record = ledger.create_pending(
        title="Guide",
        content="Content",
        citations=[],
        chapter_limit=3,
    )
    try:
        ledger.approve(
            operation_id=record.operation_id,
            approve=True,
            payload_check="0" * 64,
        )
    except Exception as exc:
        assert "tampered" in type(exc).__name__.lower()
    else:
        raise AssertionError("tampered payload was accepted")
    assert ledger.write_log.count(record.operation_id) == 0


def test_rejected_operation_has_no_write(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    record = ledger.create_pending(
        title="Guide",
        content="Content",
        citations=[],
        chapter_limit=3,
    )
    rejected, replayed = ledger.approve(
        operation_id=record.operation_id,
        approve=False,
    )
    assert rejected.state == "rejected"
    assert not replayed
    assert ledger.write_log.count(record.operation_id) == 0


def test_unknown_tool_is_structured_and_not_invoked() -> None:
    dispatcher = ToolDispatcher(
        context=ToolContext(request_id="test", max_chapter=3)
    )
    result = dispatcher.dispatch(
        name="delete_everything",
        arguments={},
        call_id="unknown_01",
    )
    assert not result.ok
    assert result.payload["error"]["code"] == "unknown_tool"
    assert dispatcher.invocations == []


def test_invalid_fetch_arguments_are_structured_and_not_invoked() -> None:
    dispatcher = ToolDispatcher(
        context=ToolContext(request_id="test", max_chapter=3)
    )
    result = dispatcher.dispatch(
        name="fetch_passage",
        arguments={"chunk_id": 123},
        call_id="invalid_01",
    )
    assert not result.ok
    assert result.payload["error"]["code"] == "tool_validation_error"
    assert dispatcher.invocations == []
