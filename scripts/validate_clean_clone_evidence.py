"""Validate that a clean checkout contains the tracked evidence bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evidence" / "manifest.json"


def main() -> int:
    if not MANIFEST.exists():
        print("FAIL: evidence/manifest.json missing")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(manifest) < 22:
        print(f"FAIL: expected >=22 manifest entries, got {len(manifest)}")
        return 1
    missing = []
    empty = []
    for name, entry in sorted(manifest.items()):
        path = ROOT / entry["evidence_path"]
        if not path.exists():
            missing.append(name)
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty.append(name)
            continue
        for line in lines:
            json.loads(line)  # must be valid JSONL
    if missing or empty:
        print("FAIL missing:", missing)
        print("FAIL empty:", empty)
        return 1
    print(f"OK: {len(manifest)} evidence traces present and non-empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
