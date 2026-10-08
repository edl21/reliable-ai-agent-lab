"""Task 2 required check: malformed tool args -> dispatcher error ->
model self-corrects through the real dispatcher."""

from __future__ import annotations

import sys

from app.ledger import ledger
from checks._task2_common import CheckResult, get_client, guide, trace_events


def main() -> int:
    ledger.reset()
    result = CheckResult("task2_invalid_args")
    client = get_client()
    response, trace_path = guide(
        client,
        request_id="task2_invalid_01",
        goal="Create a guide after recovering from an invalid tool argument.",
        max_chapter=3,
        check_name="invalid_args",
    )
    body = response.json()
    result.check(response.status_code == 200, f"/guide HTTP 200 ({response.status_code})")
    result.check(body.get("status") == "pending_approval", "model corrected and reached draft")

    events = trace_events(trace_path)
    tool_events = [e for e in events if e.get("operation") == "tool_call"]
    rejected = [
        e for e in tool_events
        if e.get("outcome") == "rejected"
        and e.get("notes", {}).get("tool_name") == "fetch_passage"
    ]
    successful = [
        e for e in tool_events
        if e.get("outcome") == "ok"
        and e.get("notes", {}).get("tool_name") == "fetch_passage"
    ]
    result.check(bool(rejected), "malformed fetch_passage was rejected")
    result.check(bool(successful), "model corrected fetch_passage arguments")
    result.check(
        any(e.get("notes", {}).get("tool_name") == "save_guide" for e in tool_events),
        "corrected loop reached save_guide",
    )
    result.check(
        ledger.write_log.count(body.get("operation_id")) == 0,
        "correction produced a pending draft, not an unapproved write",
    )
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
