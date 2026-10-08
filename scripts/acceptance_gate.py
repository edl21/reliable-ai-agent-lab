"""Layered acceptance gate: stub controls, then optional live workflows.

Usage:
  # Tier 1 (deterministic; strips OPENAI_* via checks.run_all)
  python scripts/acceptance_gate.py

  # Tier 1 + live Tier 2 (requires gateway env)
  python scripts/acceptance_gate.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUMMARY_PATH = REPO_ROOT / "checks" / "acceptance_summary.json"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> dict:
    print(f"\n### {' '.join(cmd)} ###")
    run_env = env or os.environ.copy()
    # Ensure child processes can import adapter/app/repair packages.
    existing = run_env.get("PYTHONPATH", "")
    root = str(REPO_ROOT)
    run_env["PYTHONPATH"] = (
        root if not existing else f"{root}{os.pathsep}{existing}"
    )
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=run_env,
        shell=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return {
        "cmd": cmd,
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-1500:],
        "stderr_tail": completed.stderr[-1500:],
    }


def _stub_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENAI_")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run live answer/guide/repair workflow gates",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest tier (not recommended)",
    )
    args = parser.parse_args()

    results: list[dict] = []
    overall = 0

    # Preconditions: corpus present for integration tests.
    ingest_ready = (
        (REPO_ROOT / "corpus" / "chunks.jsonl").exists()
        and (REPO_ROOT / "chroma_db").exists()
    )
    if not ingest_ready:
        print(
            "WARNING: corpus/chroma missing. Run `python -m ingest.build` "
            "before relying on integration coverage.",
            file=sys.stderr,
        )

    results.append(
        _run([sys.executable, "-m", "checks.run_all"], env=_stub_env())
    )
    if not results[-1]["passed"]:
        overall = 1

    if overall == 0 and not args.skip_pytest:
        results.append(
            _run([sys.executable, "-m", "pytest", "-q"], env=_stub_env())
        )
        if not results[-1]["passed"]:
            overall = 1

    if overall == 0:
        results.append(
            _run(
                [sys.executable, "scripts/validate_clean_clone_evidence.py"],
                env=_stub_env(),
            )
        )
        if not results[-1]["passed"]:
            overall = 1

    live_results: list[dict] = []
    if overall == 0 and args.live:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "LIVE gate requested but OPENAI_API_KEY is unset.",
                file=sys.stderr,
            )
            overall = 1
        else:
            for script in (
                "scripts/_live_critical_batch.py",
                "scripts/_live_guide_batch.py",
                "scripts/_live_repair_batch.py",
            ):
                live_results.append(_run([sys.executable, script]))
                if not live_results[-1]["passed"]:
                    overall = 1
                    break

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": overall == 0,
        "live_requested": bool(args.live),
        "ingest_ready": ingest_ready,
        "tier1": results,
        "tier2_live": live_results,
        "notes": (
            "Tier 1 proves control-flow under the deterministic stub. "
            "Tier 2 proves complete live workflows when --live is set."
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {SUMMARY_PATH}")
    print(f"RESULT: {'PASSED' if overall == 0 else 'FAILED'}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
