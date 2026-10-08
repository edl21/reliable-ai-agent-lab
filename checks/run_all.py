"""Run every required-check script in isolated subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = REPO_ROOT / "checks" / "results_summary.json"

CHECK_MODULES = [
    # Task 1
    "checks.task1_single_chapter",
    "checks.task1_two_chapter",
    "checks.task1_before_after",
    "checks.task1_unsupported",
    "checks.task1_malformed_max_chapter",
    "checks.task1_alias_reference",
    "checks.task1_contradiction",
    # Task 2
    "checks.task2_success",
    "checks.task2_reject",
    "checks.task2_invalid_args",
    "checks.task2_duplicate_save",
    "checks.task2_injection",
    "checks.task2_direct_guards",
    "checks.task2_inert_tool_call_syntax",
    # Task 3
    "checks.task3_R1",
    "checks.task3_R2",
    "checks.task3_R3",
    "checks.task3_R4_4096",
    "checks.task3_R4_alt_threshold",
    "checks.task3_R5",
    "checks.task3_R6",
    # Task 4
    "checks.task4_repair",
]


def main() -> int:
    results: list[dict[str, object]] = []
    overall = 0
    for module in CHECK_MODULES:
        print(f"\n### {module} ###")
        completed = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT,
            env=_safe_environment(),
            shell=False,
            capture_output=True,
            text=True,
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        passed = completed.returncode == 0
        results.append(
            {
                "module": module,
                "return_code": completed.returncode,
                "passed": passed,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            }
        )
        if not passed:
            overall = 1
            print(f"FAILED: {module}", file=sys.stderr)
            break

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": overall == 0,
        "checks_run": len(results),
        "checks_total": len(CHECK_MODULES),
        "results": results,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {SUMMARY_PATH}")
    print(f"RESULT: {'PASSED' if overall == 0 else 'FAILED'}")
    return overall


def _safe_environment() -> dict[str, str]:
    """Preserve normal Python execution but remove interview credentials."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENAI_")
    }
    return env


if __name__ == "__main__":
    raise SystemExit(main())
