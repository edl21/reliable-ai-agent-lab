"""Local workbench for exploring the agent reliability capabilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates

from ingest.chapters import load_chapter_map
from ingest.fetch import load_manifest


router = APIRouter()
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOTS = (REPO_ROOT / "traces", REPO_ROOT / "repair" / "traces")
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))

SUITES: dict[str, list[str]] = {
    "all": ["checks.run_all"],
    "diagnostics": ["checks.diagnostics"],
    "grounding": [
        "checks.task1_single_chapter",
        "checks.task1_two_chapter",
        "checks.task1_before_after",
        "checks.task1_unsupported",
        "checks.task1_malformed_max_chapter",
        "checks.task1_alias_reference",
        "checks.task1_contradiction",
    ],
    "approval": [
        "checks.task2_success",
        "checks.task2_reject",
        "checks.task2_invalid_args",
        "checks.task2_duplicate_save",
        "checks.task2_injection",
        "checks.task2_direct_guards",
        "checks.task2_inert_tool_call_syntax",
    ],
    "recovery": [
        "checks.task3_R1",
        "checks.task3_R2",
        "checks.task3_R3",
        "checks.task3_R4_4096",
        "checks.task3_R4_alt_threshold",
        "checks.task3_R5",
        "checks.task3_R6",
    ],
    "repair": ["checks.task4_repair"],
}


@router.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/dashboard/data")
def dashboard_data() -> dict[str, Any]:
    try:
        chapters = [chapter.to_dict() for chapter in load_chapter_map()]
    except Exception:
        chapters = []
    try:
        manifest = load_manifest().to_dict()
    except Exception:
        manifest = None
    traces = sorted(
        str(path.relative_to(REPO_ROOT))
        for root in TRACE_ROOTS
        if root.exists()
        for path in root.rglob("*.jsonl")
    )
    return {
        "chapters": chapters,
        "traces": traces,
        "trace_count": len(traces),
        "chapter_count": len(chapters),
        "corpus": manifest,
        "model_mode": "live" if os.environ.get("OPENAI_API_KEY") else "stub",
        "results_summary": _read_json(REPO_ROOT / "checks" / "results_summary.json"),
        "diagnostics": _read_json(
            REPO_ROOT / "checks" / "diagnostics_summary.json"
        ),
    }


@router.get("/dashboard/trace")
def dashboard_trace(path: str = Query(...)) -> dict[str, Any]:
    """Return one validated JSONL trace for the expandable viewer."""
    candidate = (REPO_ROOT / path).resolve()
    if candidate.suffix != ".jsonl" or not _under_allowed_root(candidate):
        raise HTTPException(status_code=404, detail="trace not found")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="trace not found")
    events = [
        json.loads(line)
        for line in candidate.read_text(encoding="utf-8").splitlines()
    ]
    return {"path": path, "events": events}


@router.post("/dashboard/run/{suite}")
def run_dashboard_suite(suite: str) -> dict[str, Any]:
    """Run only fixed allowlisted local suites; never accepts a command."""
    modules = SUITES.get(suite)
    if modules is None:
        raise HTTPException(status_code=404, detail="unknown dashboard suite")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("OPENAI_")
    }
    outputs: list[str] = []
    errors: list[str] = []
    return_code = 0
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO_ROOT,
            env=env,
            shell=False,
            timeout=240,
            capture_output=True,
            text=True,
        )
        outputs.append(f"### {module} ###\n{completed.stdout}")
        errors.append(completed.stderr)
        if completed.returncode != 0:
            return_code = completed.returncode
            break
    return {
        "suite": suite,
        "return_code": return_code,
        "stdout": "\n".join(outputs),
        "stderr": "\n".join(errors),
    }


def _under_allowed_root(path: Path) -> bool:
    return any(root.resolve() in path.parents for root in TRACE_ROOTS if root.exists())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


__all__ = ["router"]
