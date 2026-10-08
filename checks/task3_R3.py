"""R3: real approval write commits once, then response times out; retry
replays the ledger result."""

from __future__ import annotations

import sys
from pathlib import Path

from adapter.faults import FaultConfig
from adapter.runtime import clear_override
from app.ledger import ledger
from checks._task2_common import guide
from checks._task3_common import CheckResult, clear, client, events, install


def main() -> int:
    ledger.reset()
    clear_override()
    result = CheckResult("task3_R3")
    test_client = client()
    draft_response, guide_trace = guide(
        test_client,
        request_id="R3_guide_01",
        goal="Create a guide about Ishmael and Queequeg.",
        max_chapter=3,
        check_name="R3_guide",
    )
    draft = draft_response.json()
    operation_id = draft["operation_id"]
    payload_hash = draft["draft"]["payload_hash"]

    injector, _, _ = install(
        FaultConfig(
            r3_save_then_timeout=True,
            r3_target_operation_id=operation_id,
        )
    )
    first_trace = Path("traces/task3/R3_first_" + operation_id + ".jsonl")
    try:
        try:
            test_client.post(
                "/approve",
                params={"trace_check_name": "R3_first"},
                json={
                    "operation_id": operation_id,
                    "approve": True,
                    "payload_check": payload_hash,
                },
            )
        except TimeoutError:
            result.check(True, "first approval observed a timeout after commit")
        else:
            result.check(False, "first approval did not timeout")
        replay = test_client.post(
            "/approve",
            params={"trace_check_name": "R3_replay"},
            json={
                "operation_id": operation_id,
                "approve": True,
                "payload_check": payload_hash,
            },
        )
        replay_body = replay.json()
        result.check(replay.status_code == 200, "replay returned HTTP 200")
        result.check(replay_body.get("status") == "already_saved", "replay used ledger")
        result.check(bool(replay_body.get("artifact_id")), "replay returned artifact ID")
        result.check(ledger.write_log.count(operation_id) == 1, "exactly one write occurred")
        result.check(injector.config._r3_timeout_raised, "fault wrapper raised after commit")
    finally:
        clear()

    replay_trace = Path("traces/task3/R3_replay_" + operation_id + ".jsonl")
    result.check(
        any(e.get("notes", {}).get("write_count") == 1 for e in events(replay_trace)),
        "replay trace records write_count=1",
    )
    return result.finish([guide_trace, first_trace, replay_trace])


if __name__ == "__main__":
    sys.exit(main())
