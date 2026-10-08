"""Task 1 required check: single-chapter factual question.

Source: *"Single-chapter factual question (source-backed expected
answer)."* — one question whose answer is entirely within one chapter,
with ``max_chapter`` set to that chapter.

We assert on the pipeline shape (HTTP 200, trace file exists,
retrieval hit the right chapter). Content correctness needs a live
gateway; running against the stub still produces a full trace so
the reviewer can inspect the ordered messages, evidence chunk IDs,
and token accounting."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer


def main() -> int:
    result = CheckResult("task1_single_chapter")
    client = get_client()

    status, body, trace_path = post_answer(
        client,
        request_id="task1_single_01",
        question="Why does Ishmael decide to go to sea?",
        max_chapter=1,       # Chapter 1: Loomings
        check_name="single_chapter",
    )

    result.check(status == 200, f"HTTP 200 (got {status})")
    result.check(trace_path.exists(), f"trace written to {trace_path}")

    if trace_path.exists():
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        ops = [e["operation"] for e in events]
        result.check(
            "retrieval" in ops, "retrieval event present in trace"
        )
        result.check(
            "limit_check" in ops, "limit_check event present in trace"
        )
        result.check(
            "model_call" in ops, "model_call event present in trace"
        )
        # Retrieval hit at least one chunk from a within-limit chapter.
        retrieval_events = [e for e in events if e["operation"] == "retrieval"]
        if retrieval_events:
            chapter_ids = retrieval_events[0].get("source_chapter_ids", [])
            result.check(
                all(cid <= 1 for cid in chapter_ids),
                f"retrieved chapters within limit ({chapter_ids})",
            )
            result.check(
                len(chapter_ids) >= 1,
                f"at least one chunk retrieved ({len(chapter_ids)} chapters)",
            )

    result.check(
        body.get("request_id") == "task1_single_01",
        "response echoes request_id",
    )
    result.check(
        body.get("status") in {"answered", "insufficient_evidence"},
        f"status is a valid enum value (got {body.get('status')!r})",
    )

    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
