"""R1: first generation 429 + Retry-After 2, then success."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R1")
    injector, clock, sleeper = install(
        FaultConfig(r1_first_call_429=True, r1_retry_after_seconds=2.0)
    )
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R1_01",
            question="Who is Ishmael?",
            max_chapter=3,
            check_name="R1",
        )
    finally:
        clear()

    result.check(response.status_code == 200, f"request completed ({response.status_code})")
    result.check(clock.now() >= 2.0, f"virtual elapsed time >= 2s ({clock.now()}s)")
    result.check(sleeper.history == [2.0], f"Retry-After honoured ({sleeper.history})")
    result.check(injector.config._call_count == 2, "two adapter calls occurred")
    trace = events(trace_path)
    result.check(
        any(e.get("recovery_decision") == "wait_retry" for e in trace),
        "trace records wait_retry",
    )
    result.check(
        any(e.get("outcome") == "ok" for e in trace),
        "trace records eventual success",
    )
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
