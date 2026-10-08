"""Shared Task 2 check helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app

TRACE_ROOT = Path("traces/task2")


def get_client() -> TestClient:
    return TestClient(app)


def trace_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def guide(
    client: TestClient,
    *,
    request_id: str,
    goal: str,
    max_chapter: int,
    check_name: str,
) -> tuple[Any, Path]:
    response = client.post(
        "/guide",
        params={"trace_check_name": check_name},
        json={
            "request_id": request_id,
            "goal": goal,
            "max_chapter": max_chapter,
        },
    )
    trace_root = Path("traces/task3") if check_name.startswith("R") else TRACE_ROOT
    return response, trace_root / f"{check_name}_{request_id}.jsonl"


def approve(
    client: TestClient,
    *,
    operation_id: str,
    approve_value: bool,
    check_name: str,
    payload_check: str | None = None,
) -> tuple[Any, Path]:
    body: dict[str, Any] = {
        "operation_id": operation_id,
        "approve": approve_value,
    }
    if payload_check is not None:
        body["payload_check"] = payload_check
    response = client.post(
        "/approve",
        params={"trace_check_name": check_name},
        json=body,
    )
    trace_root = Path("traces/task3") if check_name.startswith("R") else TRACE_ROOT
    return response, trace_root / f"{check_name}_{operation_id}.jsonl"


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
