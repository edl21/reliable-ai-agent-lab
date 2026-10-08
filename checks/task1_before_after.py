"""Task 1 required check: same question before vs. after the answer
becomes available.

Source: *"Same question run before vs. after the answer becomes
available (raise max_chapter) — show both full message sets."*

We ask the same question at ``max_chapter=1`` (before the answer is
available) and then at ``max_chapter=20`` (after the relevant chapter).
Two request IDs so two trace files
are produced, each with its own ordered-message set."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer


def main() -> int:
    result = CheckResult("task1_before_after")
    client = get_client()

    question = "Which whaling ship do Ishmael and Queequeg choose?"

    # BEFORE: chapter 1 — the ship selection has not happened.
    status_b, body_b, trace_b = post_answer(
        client,
        request_id="task1_before_after_01",
        question=question,
        max_chapter=1,
        check_name="before_after_before",
    )
    result.check(status_b == 200, f"before: HTTP 200 (got {status_b})")
    result.check(trace_b.exists(), f"before: trace written to {trace_b}")

    # AFTER: chapter 20 — after the ship-selection chapters.
    status_a, body_a, trace_a = post_answer(
        client,
        request_id="task1_before_after_02",
        question=question,
        max_chapter=20,
        check_name="before_after_after",
    )
    result.check(status_a == 200, f"after: HTTP 200 (got {status_a})")
    result.check(trace_a.exists(), f"after: trace written to {trace_a}")

    # Both traces must have model_call events (BOTH message sets recorded).
    for label, trace_path in (("before", trace_b), ("after", trace_a)):
        if trace_path.exists():
            events = [
                json.loads(line) for line in trace_path.read_text().splitlines()
            ]
            model = [e for e in events if e["operation"] == "model_call"]
            result.check(
                len(model) >= 1,
                f"{label}: model_call present in trace",
            )
            if model:
                msgs = model[0].get("notes", {}).get("ordered_messages", [])
                result.check(
                    len(msgs) >= 2,
                    f"{label}: ordered_messages captured (n={len(msgs)})",
                )

    # No later-chapter leakage rule: at max_chapter=1, no chunks
    # retrieved from chapter > 1.
    if trace_b.exists():
        events = [json.loads(line) for line in trace_b.read_text().splitlines()]
        retrieval_events = [e for e in events if e["operation"] == "retrieval"]
        if retrieval_events:
            source_chapters = retrieval_events[0].get("source_chapter_ids", [])
            result.check(
                all(c <= 1 for c in source_chapters),
                f"before: no chunk retrieved from chapter > 1 ({source_chapters})",
            )

    return result.finish([trace_b, trace_a])


if __name__ == "__main__":
    sys.exit(main())
