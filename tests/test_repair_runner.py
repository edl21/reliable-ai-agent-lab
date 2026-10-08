"""End-to-end RepairRunner tests with injected models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from adapter.model import AdapterResponse
from repair.allowlist import validate_source
from repair.loop import (
    ALLOWLIST_CONSTRAINTS,
    BEHAVIOR_SPEC,
    REPAIR_SYSTEM_PROMPT,
    SUPPLIED_BROKEN_SOURCE,
    RepairRunner,
    _allowlist_rewrite_hint,
)
from repair.sandbox import sha256_file


SAFE_SOURCE = """\
def select_chunks(chunks, max_chapter, max_words):
    ordered = sorted(enumerate(chunks), key=lambda item: (-item[1]["score"], item[0]))
    selected = []
    seen = set()
    used = 0
    for _, chunk in ordered:
        if chunk["chapter"] > max_chapter or chunk["id"] in seen:
            continue
        words = len(chunk["text"].split())
        if used + words > max_words:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        used += words
    return selected
"""


class _SequenceModel:
    """Return canned repair patches in order."""

    def __init__(self, patches: list[str]) -> None:
        self.patches = list(patches)
        self.calls = 0
        self.tracker = None
        self.clock = None

    def chat(self, **kwargs: Any) -> AdapterResponse:
        self.calls += 1
        if not self.patches:
            content = "def select_chunks(chunks, max_chapter, max_words):\n    return []\n"
        else:
            content = self.patches.pop(0)
        return AdapterResponse(
            content=content,
            tool_calls=[],
            usage={"input_tokens": 10, "output_tokens": 40},
            stubbed=True,
        )


def _prep_repair_dir(tmp_path: Path) -> Path:
    repair_dir = tmp_path / "repair"
    repair_dir.mkdir()
    shutil.copy("repair/checks.py", repair_dir / "checks.py")
    shutil.copy("repair/hashes.json", repair_dir / "hashes.json")
    (repair_dir / "select_chunks.py").write_text(
        SUPPLIED_BROKEN_SOURCE, encoding="utf-8"
    )
    (repair_dir / "traces").mkdir()
    (repair_dir / "snapshots").mkdir()
    return repair_dir


def test_supplied_broken_fails_allowlist_on_sort() -> None:
    result = validate_source(SUPPLIED_BROKEN_SOURCE)
    assert not result.allowed
    assert "sort" in (result.reason or "").lower()


def test_repair_prompts_forbid_get_and_inplace_sort() -> None:
    blob = " ".join(
        [BEHAVIOR_SPEC, ALLOWLIST_CONSTRAINTS, REPAIR_SYSTEM_PROMPT]
    ).lower()
    assert ".get" in blob or "never use .get" in blob
    assert "chunk['score']" in blob or 'chunk["score"]' in blob
    assert "sorted(enumerate" in blob
    assert "list.sort" in blob or "chunks.sort" in blob


def test_allowlist_rewrite_hint_for_get() -> None:
    hint = _allowlist_rewrite_hint("disallowed call: get")
    assert "bracket" in hint.lower()
    assert "chunk[" in hint


def test_repair_runner_succeeds_with_mock_model(tmp_path: Path) -> None:
    repair_dir = _prep_repair_dir(tmp_path)
    model = _SequenceModel([SAFE_SOURCE])
    result = RepairRunner(
        repair_dir=repair_dir,
        model=model,
        reset_to_supplied=True,
    ).run()
    assert result.success is True
    assert result.attempts == 1
    events = [
        json.loads(line) for line in result.trace_path.read_text().splitlines()
    ]
    states = [(e.get("notes") or {}).get("state") for e in events]
    assert "S0_initial_run" in states
    assert "Sy_success" in states


def test_repair_runner_exhausts_after_three_bad_patches(tmp_path: Path) -> None:
    repair_dir = _prep_repair_dir(tmp_path)
    # Keep the mutating sort — allowlist rejects each attempt.
    bad = SUPPLIED_BROKEN_SOURCE
    model = _SequenceModel([bad, bad, bad])
    result = RepairRunner(
        repair_dir=repair_dir,
        model=model,
        reset_to_supplied=True,
        max_edits=3,
    ).run()
    assert result.success is False
    assert result.attempts == 3
    events = [
        json.loads(line) for line in result.trace_path.read_text().splitlines()
    ]
    assert any(
        (e.get("notes") or {}).get("state") == "Sx_attempt_exhausted"
        for e in events
    )
    assert any(e.get("recovery_decision") == "limit_exhausted" for e in events)
    # Allowlist rejection still preserves prior sandbox failure text.
    rejected = [
        e
        for e in events
        if (e.get("notes") or {}).get("state") == "S2_validate_patch"
        and e.get("outcome") == "rejected"
    ]
    assert rejected
    # Next prompt feedback path is recorded via last_error on exhaustion.
    exhaust = next(
        e
        for e in events
        if (e.get("notes") or {}).get("state") == "Sx_attempt_exhausted"
    )
    last_error = (exhaust.get("notes") or {}).get("last_error") or ""
    assert "allowlist_rejected" in last_error
    assert "Previous sandbox failure" in last_error or "FAIL" in last_error


def test_repair_runner_rejects_partial_fix_then_accepts(tmp_path: Path) -> None:
    """Partial fix (still mutates via sort) then full fix."""
    repair_dir = _prep_repair_dir(tmp_path)
    partial = """\
def select_chunks(chunks, max_chapter, max_words):
    chunks.sort(key=lambda c: c["score"], reverse=True)
    selected = []
    seen = set()
    used = 0
    for chunk in chunks:
        if chunk["chapter"] > max_chapter or chunk["id"] in seen:
            continue
        words = len(chunk["text"].split())
        if used + words > max_words:
            continue
        selected.append(chunk)
        seen.add(chunk["id"])
        used += words
    return selected
"""
    model = _SequenceModel([partial, SAFE_SOURCE])
    result = RepairRunner(
        repair_dir=repair_dir,
        model=model,
        reset_to_supplied=True,
    ).run()
    assert result.success is True
    assert result.attempts == 2


def test_reset_baseline_hash_matches(tmp_path: Path) -> None:
    repair_dir = _prep_repair_dir(tmp_path)
    runner = RepairRunner(repair_dir=repair_dir, reset_to_supplied=True)
    runner.utility_path.write_text(SUPPLIED_BROKEN_SOURCE, encoding="utf-8")
    expected = json.loads((repair_dir / "hashes.json").read_text())[
        "select_chunks.py_original_sha256"
    ]
    assert sha256_file(runner.utility_path) == expected
