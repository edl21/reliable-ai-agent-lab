"""Repository-level checks for the standalone project identity."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_ROOTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "artifacts",
    "chroma_db",
    "traces",
}
SKIP_GENERATED = {
    Path("checks/acceptance_summary.json"),
    Path("checks/diagnostics_summary.json"),
    Path("checks/results_summary.json"),
}


def test_previous_organization_name_is_absent() -> None:
    forbidden = "data" + "com"
    matches: list[str] = []

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in SKIP_ROOTS or "__pycache__" in relative.parts:
            continue
        if relative.parts[:2] == ("repair", "traces"):
            continue
        if relative in SKIP_GENERATED:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if forbidden in text:
            matches.append(str(relative))

    assert not matches, f"old organization name remains in: {matches}"


def test_public_project_name_is_visible_in_primary_surfaces() -> None:
    expected = "Reliable AI Agent Lab"
    assert expected in (ROOT / "README.md").read_text(encoding="utf-8")
    assert expected in (ROOT / "templates" / "dashboard.html").read_text(
        encoding="utf-8"
    )
