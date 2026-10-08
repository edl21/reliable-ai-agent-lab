"""Task 2 required check: duplicate approval replay is idempotent."""

from __future__ import annotations

import sys

from app.ledger import ledger
from checks._task2_common import CheckResult, approve, get_client, guide


def main() -> int:
    ledger.reset()
    result = CheckResult("task2_duplicate_save")
    client = get_client()
    response, guide_trace = guide(
        client,
        request_id="task2_duplicate_01",
        goal="Create a guide about Ishmael and Queequeg.",
        max_chapter=3,
        check_name="duplicate_save",
    )
    body = response.json()
    operation_id = body["operation_id"]
    payload_hash = body["draft"]["payload_hash"]

    first, first_trace = approve(
        client,
        operation_id=operation_id,
        approve_value=True,
        check_name="duplicate_first",
        payload_check=payload_hash,
    )
    second, second_trace = approve(
        client,
        operation_id=operation_id,
        approve_value=True,
        check_name="duplicate_replay",
        payload_check=payload_hash,
    )
    first_id = first.json().get("artifact_id")
    second_id = second.json().get("artifact_id")
    result.check(first.status_code == 200, "first approval succeeded")
    result.check(second.status_code == 200, "replayed approval succeeded")
    result.check(first_id == second_id, "replay returned the same artifact ID")
    result.check(
        second.json().get("status") == "already_saved",
        "replay was classified as already_saved",
    )
    result.check(ledger.write_log.count(operation_id) == 1, "exactly one write occurred")
    return result.finish([guide_trace, first_trace, second_trace])


if __name__ == "__main__":
    sys.exit(main())
