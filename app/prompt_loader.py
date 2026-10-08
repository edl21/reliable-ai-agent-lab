"""Load versioned model prompts from repository files."""

from __future__ import annotations

from pathlib import Path


_PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Read one trusted, application-owned prompt file as UTF-8."""
    path = _PROMPT_ROOT / name
    if path.parent != _PROMPT_ROOT:
        raise ValueError(f"invalid prompt name: {name!r}")
    return path.read_text(encoding="utf-8").rstrip()


__all__ = ["load_prompt"]
