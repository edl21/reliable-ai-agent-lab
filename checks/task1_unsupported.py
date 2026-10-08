"""Task 1 required check: unsupported question → insufficient_evidence.

Source: *"Question unsupported by the whole corpus →
insufficient_evidence."*

We ask something that cannot be answered from the book
(a modern current-affairs question). Even with the maximum chapter
limit, the model should return ``status: insufficient_evidence`` with
no citations. Post-hoc verifier ensures no later-chapter facts leak."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer
from ingest.chapters import load_chapter_map


def main() -> int:
    result = CheckResult("task1_unsupported")
    client = get_client()

    status, body, trace_path = post_answer(
        client,
        request_id="task1_unsupported_01",
        question=(
            "What is the current stock price of Microsoft on the NASDAQ, "
            "and who is the current CEO?"
        ),
        max_chapter=len(load_chapter_map()),
        check_name="unsupported",
    )

    result.check(status == 200, f"HTTP 200 (got {status})")
    result.check(trace_path.exists(), f"trace written to {trace_path}")

    result.check(
        body.get("status") == "insufficient_evidence",
        f"status is insufficient_evidence (got {body.get('status')!r})",
    )
    result.check(
        len(body.get("citations", [])) == 0,
        f"zero citations on insufficient_evidence (got {len(body.get('citations', []))})",
    )

    if trace_path.exists():
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        validation_events = [e for e in events if e["operation"] == "validation"]
        # The post-hoc validation event should be ok (no leaks flagged).
        result.check(
            any(e.get("outcome") == "ok" for e in validation_events),
            "post-hoc validation passed (no later-chapter leak)",
        )

    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
