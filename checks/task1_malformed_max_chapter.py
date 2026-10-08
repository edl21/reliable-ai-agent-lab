"""Task 1 required check: malformed ``max_chapter`` → rejected BEFORE
any model call.

Source: *"Malformed max_chapter → rejected before generation."*

We POST four bad shapes:

1. missing field
2. non-integer type
3. zero
4. above corpus max

Each MUST return HTTP 4xx (422 in our schema) and produce NO
model_call trace event. The only trace event allowed is a
``validation`` event with ``outcome=rejected`` (for the "out of range"
case where our handler emits the event before raising)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from checks._common import CheckResult, get_client, TRACE_ROOT_T1


def main() -> int:
    result = CheckResult("task1_malformed_max_chapter")
    client = get_client()

    cases: list[tuple[str, dict[str, object] | str, str]] = [
        ("missing", {"request_id": "malformed_01", "question": "q"},
         "no max_chapter key"),
        ("non-integer", {"request_id": "malformed_02", "question": "q",
                         "max_chapter": "five"}, "non-integer max_chapter"),
        ("zero", {"request_id": "malformed_03", "question": "q",
                  "max_chapter": 0}, "max_chapter = 0"),
        ("out-of-range",
         {"request_id": "malformed_04", "question": "q", "max_chapter": 999},
         "max_chapter > corpus max"),
    ]

    trace_paths: list[Path] = []
    for label, payload, desc in cases:
        resp = client.post(
            "/answer",
            params={"trace_check_name": "malformed_max_chapter"},
            json=payload,
        )
        result.check(
            400 <= resp.status_code < 500,
            f"{label} ({desc}): HTTP 4xx (got {resp.status_code})",
        )
        # For the out-of-range case our handler emits a trace event
        # BEFORE raising; that trace should show operation=validation,
        # outcome=rejected, and NO model_call.
        if label == "out-of-range":
            trace_path = TRACE_ROOT_T1 / "malformed_max_chapter_malformed_04.jsonl"
            trace_paths.append(trace_path)
            if trace_path.exists():
                events = [
                    json.loads(line) for line in trace_path.read_text().splitlines()
                ]
                ops = [e["operation"] for e in events]
                result.check(
                    "model_call" not in ops,
                    f"{label}: NO model_call event in trace (ops={ops})",
                )
                result.check(
                    any(
                        e["operation"] == "validation"
                        and e.get("outcome") == "rejected"
                        for e in events
                    ),
                    f"{label}: validation event with outcome=rejected present",
                )

    return result.finish(trace_paths)


if __name__ == "__main__":
    sys.exit(main())
