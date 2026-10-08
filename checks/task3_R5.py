"""R5: invalid first JSON schema output, then corrective success."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R5")
    injector, _, _ = install(FaultConfig(r5_first_call_bad_schema=True))
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R5_01",
            question="Who is Ishmael?",
            max_chapter=3,
            check_name="R5",
        )
    finally:
        clear()

    result.check(response.status_code == 200, f"corrective call succeeded ({response.status_code})")
    trace = events(trace_path)
    invalid = [
        e for e in trace
        if e.get("outcome") == "schema_invalid"
        and e.get("notes", {}).get("raw_output") == '{"status": 42}'
    ]
    accepted = [
        e for e in trace
        if e.get("operation") == "model_call"
        and e.get("outcome") == "ok"
        and e.get("notes", {}).get("raw_output")
    ]
    result.check(bool(invalid), "literal invalid output was rejected")
    result.check(bool(accepted), "a validated output was accepted")
    result.check(injector.config._call_count == 2, "exactly two generation calls occurred")
    result.check(
        any(e.get("recovery_decision") == "schema_reprompt" for e in trace),
        "trace records schema corrective action",
    )
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
