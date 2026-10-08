"""R2: every generation 429; bounded retries and no fabricated answer."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R2")
    injector, clock, sleeper = install(
        FaultConfig(r2_all_calls_429=True, r2_retry_after_seconds=2.0)
    )
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R2_01",
            question="Who is Ishmael?",
            max_chapter=3,
            check_name="R2",
        )
    finally:
        clear()

    body = response.json()
    result.check(response.status_code == 500, f"structured failure HTTP 500 ({response.status_code})")
    result.check(
        body.get("detail", {}).get("error", {}).get("code") == "retries_exhausted",
        "error classified as retries_exhausted",
    )
    result.check("answer" not in body, "no fabricated answer returned")
    result.check(injector.config._call_count == 3, "attempt budget stopped at three adapter calls")
    result.check(clock.now() == 6.0, "virtual waits totalled 6 seconds")
    result.check(sleeper.history == [2.0, 2.0, 2.0], "all Retry-After delays honoured")
    result.check(
        any(e.get("recovery_decision") == "retries_exhausted" for e in events(trace_path)),
        "trace records exhausted retries",
    )
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
