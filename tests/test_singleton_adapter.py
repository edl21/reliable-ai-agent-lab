"""Grep-based invariant: the only file allowed to construct an
``openai.OpenAI(`` client is ``adapter/model.py``.

Rationale (from the adversarial review): if any helper in another file
ever does ``openai.OpenAI(...)`` directly, that call bypasses
``get_model()`` and therefore bypasses the fault wrapper. R1's
virtual-elapsed assertion, R2's attempt-count assertion, and R6's
one-attempt assertion would all silently rot. This test kills that
class of bug at CI time."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories we walk. Skip caches, VCS, virtualenv, and the tests
# directory itself (test doubles are allowed to construct clients).
SEARCH_DIRS = ("adapter", "app", "ingest", "repair", "checks", "traces")

# The one legal call site.
ALLOWED_PATH = Path("adapter/model.py")

# We look for both ``openai.OpenAI(`` and ``openai.Client(`` — both are
# legitimate ways to construct the client from the SDK.
FORBIDDEN = re.compile(r"\bopenai\.(OpenAI|Client)\s*\(")


def _iter_python_files() -> list[Path]:
    out: list[Path] = []
    for d in SEARCH_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            out.append(p)
    return out


def test_only_model_py_constructs_openai_client() -> None:
    offenders: list[str] = []
    for path in _iter_python_files():
        rel = path.relative_to(REPO_ROOT)
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN.search(text):
            if rel != ALLOWED_PATH:
                offenders.append(str(rel))

    assert not offenders, (
        "openai.OpenAI(/Client( found outside adapter/model.py — every "
        "LLM call must go through adapter.get_model() so the fault "
        f"wrapper stays authoritative. Offenders: {offenders}"
    )


def test_model_py_actually_constructs_client() -> None:
    """Sanity: the allowed file must actually contain the call. If a
    refactor moves it elsewhere we want this test to fail loudly."""
    text = (REPO_ROOT / ALLOWED_PATH).read_text(encoding="utf-8")
    assert FORBIDDEN.search(text), (
        "adapter/model.py is expected to construct openai.OpenAI(...); "
        "if that moved, update this test and the singleton invariant."
    )
