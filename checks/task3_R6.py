"""R6: every generation 401; one attempt and configuration error."""

from __future__ import annotations

import sys

from adapter.faults import FaultConfig
from checks._task3_common import CheckResult, clear, client, events, install, post_answer


def main() -> int:
    result = CheckResult("task3_R6")
    injector, _, _ = install(FaultConfig(r6_all_calls_401=True))
    try:
        response, trace_path = post_answer(
            client(),
            request_id="R6_01",
            question="Who is Ishmael?",
            max_chapter=3,
            check_name="R6",
        )
    finally:
        clear()

    body = response.json()
    result.check(response.status_code == 502, f"configuration failure HTTP 502 ({response.status_code})")
    result.check(
        body.get("detail", {}).get("error", {}).get("code")
        == "configuration_error",
        "error classified as configuration_error",
    )
    result.check("insufficient_evidence" not in str(body), "not framed as insufficient_evidence")
    result.check(injector.config._call_count == 1, "auth failure used exactly one attempt")
    result.check(
        any(e.get("recovery_decision") == "auth_fail" for e in events(trace_path)),
        "trace records terminal auth failure",
    )
    return result.finish([trace_path])


if __name__ == "__main__":
    sys.exit(main())
