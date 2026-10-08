"""Shared helpers for check scripts.

Each check script:
1. Prints its check name and configuration.
2. Uses a FastAPI ``TestClient`` for in-process route execution.
3. Writes its trace to the pinned filename convention.
4. Prints PASS / FAIL with a one-line reason AND the trace path.
5. Exits non-zero on failure.

The ``TestClient`` is constructed ONCE per script — every request in the
same script reuses it, which matches how a reviewer would run the
scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app


TRACE_ROOT_T1 = Path("traces/task1")
TRACE_ROOT_T2 = Path("traces/task2")
TRACE_ROOT_T3 = Path("traces/task3")


class CheckResult:
    """Simple pass/fail carrier."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passes: list[str] = []
        self.fails: list[str] = []

    def check(self, condition: bool, msg: str) -> None:
        if condition:
            self.passes.append(msg)
        else:
            self.fails.append(msg)

    def finish(self, trace_paths: list[Path]) -> int:
        """Print summary and return the exit code (0 = pass)."""
        print()
        print("=" * 72)
        print(f"Check: {self.name}")
        print("=" * 72)
        for m in self.passes:
            print(f"  PASS  {m}")
        for m in self.fails:
            print(f"  FAIL  {m}")
        print()
        for p in trace_paths:
            marker = "OK" if p.exists() else "MISSING"
            print(f"  trace [{marker}]: {p}")
        print()
        if self.fails:
            print(f"RESULT: FAILED ({len(self.fails)} of "
                  f"{len(self.passes) + len(self.fails)})")
            return 1
        print(f"RESULT: PASSED ({len(self.passes)} checks)")
        return 0


def get_client() -> TestClient:
    """Fresh TestClient per script — the app is a module-level singleton
    so this is cheap."""
    return TestClient(app)


def post_answer(
    client: TestClient,
    *,
    request_id: str,
    question: str,
    max_chapter: int,
    check_name: str,
) -> tuple[int, dict[str, Any], Path]:
    """POST /answer with a trace_check_name query param and return
    (status_code, response_json, expected_trace_path)."""
    resp = client.post(
        "/answer",
        params={"trace_check_name": check_name},
        json={
            "request_id": request_id,
            "question": question,
            "max_chapter": max_chapter,
        },
    )
    trace_path = TRACE_ROOT_T1 / f"{check_name}_{request_id}.jsonl"
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    return resp.status_code, body, trace_path


def exit_with(exit_code: int) -> None:
    sys.exit(exit_code)


__all__ = [
    "CheckResult",
    "TRACE_ROOT_T1",
    "TRACE_ROOT_T2",
    "TRACE_ROOT_T3",
    "exit_with",
    "get_client",
    "post_answer",
]
