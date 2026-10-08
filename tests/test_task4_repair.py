"""Task 4 allowlist and sandbox tests."""

from __future__ import annotations

from pathlib import Path

from repair.allowlist import validate_source
from repair.sandbox import SandboxRunner


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


def test_safe_general_repair_passes_allowlist() -> None:
    result = validate_source(SAFE_SOURCE)
    assert result.allowed, result.reason


def test_import_is_rejected() -> None:
    result = validate_source(
        "import os\n\ndef select_chunks(chunks, max_chapter, max_words):\n    return []\n"
    )
    assert not result.allowed
    assert result.reason


def test_dynamic_execution_is_rejected() -> None:
    result = validate_source(
        "def select_chunks(chunks, max_chapter, max_words):\n"
        "    return eval('[]')\n"
    )
    assert not result.allowed
    assert "disallowed call" in result.reason


def test_file_access_is_rejected() -> None:
    result = validate_source(
        "def select_chunks(chunks, max_chapter, max_words):\n"
        "    return open('secret').read()\n"
    )
    assert not result.allowed
    assert "disallowed call" in result.reason


def test_sandbox_runs_only_immutable_runner() -> None:
    result = SandboxRunner(
        repair_dir=Path("repair"),
        timeout_seconds=10,
    ).run()
    assert result.success
    assert result.checks_hash_before == result.checks_hash_after
