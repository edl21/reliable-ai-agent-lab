"""R4 at the source-pinned 4096 context threshold."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R4_4096")
    injector, _, _ = install(FaultConfig(r4_context_threshold=4096))
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R4_4096_01",
            question="Give a detailed account of Ishmael's decision to go to sea.",
            max_chapter=20,
            check_name="R4_4096",
        )
    finally:
        clear()

    result.check(response.status_code == 200, f"reduced request succeeded ({response.status_code})")
    trace = events(trace_path)
    reductions = [
        e for e in trace
        if e.get("recovery_decision") == "reduce_context"
    ]
    result.check(bool(reductions), "context-length recovery was recorded")
    if reductions:
        notes = reductions[0].get("notes", {})
        result.check(notes.get("reported_limit") == 4096, "reported limit was 4096")
        result.check(
            notes.get("before_messages") != notes.get("after_messages"),
            "model messages were reduced on the fly",
        )
        result.check(
            notes.get("total_tokens_after", 999999) <= 4096,
            "reduced total context fits the reported limit",
        )
        result.check(
            all(
                int(chapter) <= 20
                for chapter in reductions[0].get("source_chapter_ids", [])
            ),
            "chapter limit was preserved",
        )
    result.check(injector.config._call_count >= 2, "wrapper rejected then accepted a retry")
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
