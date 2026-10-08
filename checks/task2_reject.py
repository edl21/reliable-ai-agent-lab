"""Task 2 required check: rejected draft creates nothing."""

from __future__ import annotations

import sys

from app.ledger import ledger
from checks._task2_common import CheckResult, approve, get_client, guide


def main() -> int:
    ledger.reset()
    result = CheckResult("task2_reject")
    client = get_client()
    response, guide_trace = guide(
        client,
        request_id="task2_reject_01",
        goal="Create a guide about Ishmael and Queequeg.",
        max_chapter=3,
        check_name="reject",
    )
    body = response.json()
    operation_id = body.get("operation_id")
    result.check(response.status_code == 200, f"/guide HTTP 200 ({response.status_code})")
    result.check(body.get("status") == "pending_approval", "draft is pending")

    approval, approval_trace = approve(
        client,
        operation_id=operation_id,
        approve_value=False,
        check_name="reject_approve",
        payload_check=body["draft"]["payload_hash"],
    )
    result.check(approval.status_code == 200, "rejection request accepted")
    result.check(approval.json().get("status") == "rejected", "draft was rejected")
    result.check(ledger.write_log.count(operation_id) == 0, "nothing was written")
    result.check(
        not any(ledger.artifact_root.glob("*.json")),
        "no artifact file exists after rejection",
    )
    return result.finish([guide_trace, approval_trace])


if __name__ == "__main__":
    sys.exit(main())
