"""Task 1 required check: alias/indirect character reference.

This check uses an alias/indirect character reference rather than a
proper name.

We ask a question referring to Queequeg as "the tattooed harpooner".
Retrieval uses
embedding similarity, so aliases should still surface the relevant
passages even without exact name match."""

from __future__ import annotations

import json
import sys

from checks._common import CheckResult, get_client, post_answer


def main() -> int:
    result = CheckResult("task1_alias_reference")
    client = get_client()

    status, body, trace_path = post_answer(
        client,
        request_id="task1_alias_01",
        question=(
            "How does the narrator first come to share a room with the "
            "tattooed harpooner?"
        ),
        max_chapter=5,
        check_name="alias_reference",
    )

    result.check(status == 200, f"HTTP 200 (got {status})")
    result.check(trace_path.exists(), f"trace written to {trace_path}")

    if trace_path.exists():
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        retrieval_events = [e for e in events if e["operation"] == "retrieval"]
        result.check(
            len(retrieval_events) >= 1, "retrieval event present"
        )
        if retrieval_events:
            n_chunks = retrieval_events[0].get("notes", {}).get(
                "retrieved_count", 0
            )
            result.check(
                n_chunks >= 3,
                f"alias query retrieved multiple chunks ({n_chunks})",
            )
            source_chapters = retrieval_events[0].get("source_chapter_ids", [])
            result.check(
                all(c <= 5 for c in source_chapters),
                f"all retrieved chapters within limit ({source_chapters})",
            )

    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
