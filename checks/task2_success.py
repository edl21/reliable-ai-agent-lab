"""Task 2 required check: successful model-driven guide + approval."""

from __future__ import annotations

import sys

from app.ledger import ledger
from checks._task2_common import CheckResult, approve, get_client, guide, trace_events


def main() -> int:
    ledger.reset()
    result = CheckResult("task2_success")
    client = get_client()
    response, guide_trace = guide(
        client,
        request_id="task2_success_01",
        goal="Create a short guide explaining Ishmael and Queequeg's meeting.",
        max_chapter=5,
        check_name="success",
    )
    body = response.json()
    result.check(response.status_code == 200, f"/guide HTTP 200 ({response.status_code})")
    result.check(
        body.get("status") == "pending_approval",
        f"guide returns pending_approval ({body.get('status')!r})",
    )
    operation_id = body.get("operation_id")
    payload_hash = body.get("draft", {}).get("payload_hash")
    result.check(bool(operation_id), "application generated operation_id")
    result.check(len(payload_hash or "") == 64, "canonical payload hash returned")
    result.check(ledger.write_log.count(operation_id) == 0, "no write before approval")

    events = trace_events(guide_trace)
    tool_names = [
        e.get("notes", {}).get("tool_name")
        for e in events
        if e.get("operation") == "tool_call"
    ]
    result.check(
        tool_names == ["search_passages", "fetch_passage", "save_guide"],
        f"model selected tools through dispatcher ({tool_names})",
    )

    approval, approval_trace = approve(
        client,
        operation_id=operation_id,
        approve_value=True,
        check_name="success_approve",
        payload_check=payload_hash,
    )
    approval_body = approval.json()
    result.check(approval.status_code == 200, f"/approve HTTP 200 ({approval.status_code})")
    result.check(approval_body.get("status") == "saved", "approval saved artifact")
    artifact_id = approval_body.get("artifact_id")
    result.check(bool(artifact_id), "artifact ID was application-generated")
    result.check(ledger.write_log.count(operation_id) == 1, "exactly one write recorded")
    result.check(approval_trace.exists(), "approval trace written")

    return result.finish([guide_trace, approval_trace])


if __name__ == "__main__":
    sys.exit(main())
