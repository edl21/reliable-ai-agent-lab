"""R4 at 3000, proving the app learns reported_limit rather than
hardcoding 4096."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R4_alt_threshold")
    injector, _, _ = install(FaultConfig(r4_context_threshold=3000))
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R4_3000_01",
            question="Give a detailed account of Ishmael's decision to go to sea.",
            max_chapter=20,
            check_name="R4_3000",
        )
    finally:
        clear()

    result.check(response.status_code == 200, f"alternate reduction succeeded ({response.status_code})")
    trace = events(trace_path)
    reductions = [
        e for e in trace
        if e.get("recovery_decision") == "reduce_context"
    ]
    result.check(bool(reductions), "alternate context recovery was recorded")
    if reductions:
        notes = reductions[0].get("notes", {})
        result.check(notes.get("reported_limit") == 3000, "reported limit was 3000")
        result.check(
            notes.get("total_tokens_after", 999999) <= 3000,
            "reduced context fits the alternate reported limit",
        )
    result.check(injector.config._call_count >= 2, "wrapper rejected then accepted a retry")
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
