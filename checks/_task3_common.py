"""Shared helpers for R1-R6 fault checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from adapter.clock import VirtualClock, VirtualSleeper
from adapter.faults import FaultConfig, FaultInjector
from adapter.runtime import clear_override, install_fault
from app.main import app

TRACE_ROOT = Path("traces/task3")


def client() -> TestClient:
    return TestClient(app)


def install(
    config: FaultConfig,
) -> tuple[FaultInjector, VirtualClock, VirtualSleeper]:
    clock = VirtualClock()
    sleeper = VirtualSleeper(clock=clock)
    injector = install_fault(config, clock=clock, sleeper=sleeper)
    return injector, clock, sleeper


def clear() -> None:
    clear_override()


def post_answer(
    test_client: TestClient,
    *,
    request_id: str,
    question: str,
    max_chapter: int,
    check_name: str,
) -> tuple[Any, Path]:
    response = test_client.post(
        "/answer",
        params={"trace_check_name": check_name},
        json={
            "request_id": request_id,
            "question": question,
            "max_chapter": max_chapter,
        },
    )
    return response, TRACE_ROOT / f"{check_name}_{request_id}.jsonl"


def events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


class CheckResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passes: list[str] = []
        self.fails: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passes if condition else self.fails).append(message)

    def finish(self, trace_paths: list[Path]) -> int:
        print("=" * 72)
        print(f"Check: {self.name}")
        print("=" * 72)
        for message in self.passes:
            print(f"  PASS  {message}")
        for message in self.fails:
            print(f"  FAIL  {message}")
        for path in trace_paths:
            print(f"  trace [{'OK' if path.exists() else 'MISSING'}]: {path}")
        if self.fails:
            print(f"RESULT: FAILED ({len(self.fails)} failures)")
            return 1
        print(f"RESULT: PASSED ({len(self.passes)} checks)")
        return 0
