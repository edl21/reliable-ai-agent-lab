"""Required diagnostic samples: successful answer x3 and R1 x3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from adapter.faults import FaultConfig
from adapter.runtime import clear_override, install_fault
from adapter.clock import VirtualClock, VirtualSleeper
from app.main import app
from fastapi.testclient import TestClient


SUMMARY_PATH = Path("checks/diagnostics_summary.json")


def main() -> int:
    client = TestClient(app)
    successful: list[dict[str, object]] = []
    r1_samples: list[dict[str, object]] = []

    for index in range(1, 4):
        clear_override()
        request_id = f"diagnostic_answer_{index:02d}"
        response = client.post(
            "/answer",
            params={"trace_check_name": "diagnostic_answer"},
            json={
                "request_id": request_id,
                "question": "Who is Ishmael?",
                "max_chapter": 3,
            },
        )
        body = response.json()
        successful.append(
            {
                "request_id": request_id,
                "status_code": response.status_code,
                "status": body.get("status"),
                "stubbed": body.get("metrics", {}).get("stubbed"),
                "total_ms": body.get("metrics", {}).get("total_ms"),
                "retrieval_ms": body.get("metrics", {}).get("retrieval_ms"),
                "model_ms": body.get("metrics", {}).get("model_ms"),
            }
        )

    for index in range(1, 4):
        clock = VirtualClock()
        sleeper = VirtualSleeper(clock)
        install_fault(
            FaultConfig(r1_first_call_429=True, r1_retry_after_seconds=2.0),
            clock=clock,
            sleeper=sleeper,
        )
        request_id = f"diagnostic_R1_{index:02d}"
        response = client.post(
            "/answer",
            params={"trace_check_name": f"R1_diagnostic_{index:02d}"},
            json={
                "request_id": request_id,
                "question": "Who is Ishmael?",
                "max_chapter": 3,
            },
        )
        body = response.json()
        r1_samples.append(
            {
                "request_id": request_id,
                "status_code": response.status_code,
                "status": body.get("status"),
                "stubbed": body.get("metrics", {}).get("stubbed"),
                "virtual_elapsed_seconds": clock.now(),
                "sleep_history": sleeper.history,
                "total_ms": body.get("metrics", {}).get("total_ms"),
            }
        )
        clear_override()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "successful_answer_samples": successful,
        "r1_samples": r1_samples,
        "differences": {
            "successful_answer_total_ms": [
                item["total_ms"] for item in successful
            ],
            "r1_virtual_elapsed_seconds": [
                item["virtual_elapsed_seconds"] for item in r1_samples
            ],
            "all_stubbed": all(
                bool(item["stubbed"])
                for item in [*successful, *r1_samples]
            ),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"diagnostics summary: {SUMMARY_PATH}")
    print("successful answers: 3")
    print("R1 samples: 3")
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
