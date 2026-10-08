"""Enumeration test: every retrieval-like function must *call* the chapter guard.

Walks ``app/`` and ``adapter/`` for function defs whose names match a
retrieval-like pattern and asserts each body contains an AST ``Call`` to
``enforce_chapter_limit`` or ``enforce_chapter_limit_strict`` — not merely
a textual mention. A comment or string containing the name does not count.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEARCH_ROOTS = ("app", "adapter", "ingest")

RETRIEVAL_NAME = re.compile(
    r"^(retrieve|fetch|search|rerank|rewrite|summari[sz]e|cache_(?:read|write)|reduce)"
    r"(_[a-zA-Z0-9_]+)?$"
)

# The guard module itself and pure schema/error modules are excluded.
EXCLUDE_FILES: frozenset[Path] = frozenset(
    {
        Path("ingest/chapter_guard.py"),
        # PDF download helper — not model-facing retrieval.
        Path("ingest/fetch.py"),
    }
)

GUARD_NAMES = frozenset(
    {"enforce_chapter_limit", "enforce_chapter_limit_strict"}
)


def _iter_python_files() -> list[Path]:
    out: list[Path] = []
    for root_name in SEARCH_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel in EXCLUDE_FILES:
                continue
            out.append(path)
    return out


def _call_names(fn_node: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _retrieval_functions(source: str) -> list[tuple[str, ast.AST]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if RETRIEVAL_NAME.match(node.name):
                out.append((node.name, node))
    return out


def test_every_retrieval_function_calls_the_guard() -> None:
    offenders: list[str] = []
    scanned = 0
    for path in _iter_python_files():
        rel = path.relative_to(REPO_ROOT)
        src = path.read_text(encoding="utf-8")
        for fn_name, fn_node in _retrieval_functions(src):
            scanned += 1
            if not (GUARD_NAMES & _call_names(fn_node)):
                offenders.append(f"{rel}::{fn_name}")

    assert scanned > 0, "expected at least one retrieval-like function"
    assert not offenders, (
        "Retrieval-like functions do not *call* enforce_chapter_limit via AST. "
        "A textual mention alone is insufficient. Offenders: "
        f"{offenders}"
    )
