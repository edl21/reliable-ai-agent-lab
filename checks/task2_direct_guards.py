"""Task 2 direct guard checks, independent of model behaviour."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from app.ledger import ApprovalTokenInvalid, ledger
from app.main import app
from app.tools.context import ToolContext
from app.tools.dispatch import ToolDispatcher
from app.tools.save_guide import save_guide
from app.tools.schemas import GuideCitation
from ingest.chunk import load_chunks
from traces.schema import JSONLWriter, TraceEvent
from checks._task2_common import CheckResult, approve, get_client


def main() -> int:
    from fastapi.testclient import TestClient

    result = CheckResult("task2_direct_guards")
    fetch_trace = Path("traces/task2/direct_forbidden_fetch.jsonl")
    fetch_trace.unlink(missing_ok=True)

    forbidden = next(chunk for chunk in load_chunks() if chunk.chapter > 3)
    dispatcher = ToolDispatcher(
        context=ToolContext(request_id="direct_guard", max_chapter=3)
    )
    fetch_result = dispatcher.dispatch(
        name="fetch_passage",
        arguments={"chunk_id": forbidden.chunk_id},
        call_id="direct_forbidden_fetch_01",
    )
    JSONLWriter().append(
        fetch_trace,
        TraceEvent(
            request_id="direct_guard",
            run_id=uuid.uuid4().hex,
            task="t2",
            operation="tool_call",
            outcome="rejected" if not fetch_result.ok else "error",
            source_chapter_ids=[forbidden.chapter],
            notes={
                "tool_name": "fetch_passage",
                "chunk_id": forbidden.chunk_id,
                "max_chapter": 3,
                "result": fetch_result.payload,
                "dispatcher_invocation_count": len(dispatcher.invocations),
            },
        ),
    )
    result.check(not fetch_result.ok, "forbidden chunk ID was rejected")
    result.check(
        len(dispatcher.invocations) == 0,
        "forbidden fetch did not execute the tool",
    )
    result.check(fetch_trace.exists(), "forbidden-fetch trace written")

    client = TestClient(app)
    response, approval_trace = approve(
        client,
        operation_id="op_does_not_exist",
        approve_value=True,
        check_name="direct_unapproved_approve",
    )
    result.check(response.status_code == 404, "unapproved unknown operation rejected")
    result.check(approval_trace.exists(), "unapproved-approval trace written")

    # Valid pending operation, then an unapproved direct commit attempt.
    allowed = next(chunk for chunk in load_chunks() if chunk.chapter <= 3)
    pending = save_guide(
        "Direct guard draft",
        "Content under chapter limit.",
        [
            GuideCitation(
                chunk_id=allowed.chunk_id,
                source_filename=allowed.source_filename,
                chapter=allowed.chapter,
                pdf_pages=list(allowed.pdf_pages),
                excerpt=allowed.text[: min(80, len(allowed.text))],
            )
        ],
        context=ToolContext(request_id="direct_pending", max_chapter=3),
    )
    result.check(
        pending.get("status") == "pending_approval",
        "valid pending operation created",
    )
    result.check(
        ledger.write_log.count(pending["operation_id"]) == 0,
        "pending draft produced no artifact write",
    )

    commit_trace = Path("traces/task2/direct_unapproved_commit.jsonl")
    commit_trace.unlink(missing_ok=True)
    rejected = False
    try:
        ledger._commit(
            operation_id=pending["operation_id"],
            payload_hash=pending["payload_hash"],
            approval_token=None,
        )
    except ApprovalTokenInvalid:
        rejected = True
    JSONLWriter().append(
        commit_trace,
        TraceEvent(
            request_id="direct_pending",
            run_id=uuid.uuid4().hex,
            task="t2",
            operation="approve",
            outcome="rejected",
            notes={
                "reason": "missing_approval_token",
                "operation_id": pending["operation_id"],
                "write_count": ledger.write_log.count(pending["operation_id"]),
            },
        ),
    )
    result.check(rejected, "direct commit without approval token rejected")
    result.check(
        ledger.write_log.count(pending["operation_id"]) == 0,
        "unapproved commit wrote no artifact",
    )
    result.check(commit_trace.exists(), "unapproved-commit trace written")

    return result.finish([fetch_trace, approval_trace, commit_trace])


if __name__ == "__main__":
    sys.exit(main())
