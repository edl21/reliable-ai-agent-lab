"""Refresh the tracked sanitized evidence bundle from working traces/.

Copies complete JSONL events for every README-required check into
``evidence/traces/``, redacts credential-shaped strings, and writes
``evidence/manifest.json`` mapping check name → relative event path.

Does not modify ``repair/`` fixtures or ``repair/checks.py``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "traces"
EVIDENCE = ROOT / "evidence" / "traces"
MANIFEST = ROOT / "evidence" / "manifest.json"

# Check name → working-tree relative path (after checks.run_all).
REQUIRED: dict[str, str] = {
    "task1_single_chapter": "traces/task1/single_chapter_task1_single_01.jsonl",
    "task1_two_chapter": "traces/task1/two_chapter_task1_two_01.jsonl",
    "task1_before_after": "traces/task1/before_after_before_task1_before_after_01.jsonl",
    "task1_alias_reference": "traces/task1/alias_reference_task1_alias_01.jsonl",
    "task1_unsupported": "traces/task1/unsupported_task1_unsupported_01.jsonl",
    "task1_malformed_max_chapter": "traces/task1/malformed_max_chapter_malformed_04.jsonl",
    "task1_contradiction": "traces/task1/contradiction_task1_contradiction_01.jsonl",
    "task2_success": "traces/task2/success_task2_success_01.jsonl",
    "task2_reject": "traces/task2/reject_task2_reject_01.jsonl",
    "task2_duplicate_save": "traces/task2/duplicate_save_task2_duplicate_01.jsonl",
    "task2_invalid_args": "traces/task2/invalid_args_task2_invalid_01.jsonl",
    "task2_injection": "traces/task2/injection_task2_injection_01.jsonl",
    "task2_inert_tool_call_syntax": "traces/task2/inert_tool_call_syntax_task2_inert_01.jsonl",
    "task2_direct_guards": "traces/task2/direct_forbidden_fetch.jsonl",
    "task3_R1": "traces/task3/R1_R1_01.jsonl",
    "task3_R2": "traces/task3/R2_R2_01.jsonl",
    "task3_R3": "traces/task3/R3_guide_R3_guide_01.jsonl",
    "task3_R4_4096": "traces/task3/R4_4096_R4_4096_01.jsonl",
    "task3_R4_3000": "traces/task3/R4_3000_R4_3000_01.jsonl",
    "task3_R5": "traces/task3/R5_R5_01.jsonl",
    "task3_R6": "traces/task3/R6_R6_01.jsonl",
    "task4_repair": "repair/traces",  # special: newest file
}

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{10,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*)",
    re.IGNORECASE,
)


def _sanitize_line(line: str) -> str:
    sanitized = _SECRET_RE.sub("[REDACTED]", line)
    return sanitized.replace(str(ROOT), "<repo>")


def _newest_repair_trace() -> Path | None:
    repair_dir = ROOT / "repair" / "traces"
    if not repair_dir.exists():
        return None
    files = sorted(repair_dir.glob("repair_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    # Clear previous bundle copies (keep directory).
    for old in EVIDENCE.rglob("*.jsonl"):
        old.unlink()

    manifest: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for check_name, rel in REQUIRED.items():
        if check_name == "task4_repair":
            src = _newest_repair_trace()
            if src is None:
                missing.append(check_name)
                continue
            dest_rel = f"evidence/traces/task4/{src.name}"
        else:
            src = ROOT / rel
            if not src.exists():
                missing.append(check_name)
                continue
            dest_rel = f"evidence/traces/{'/'.join(Path(rel).parts[1:])}"

        dest = ROOT / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = src.read_text(encoding="utf-8").splitlines()
        sanitized = [_sanitize_line(line) for line in lines if line.strip()]
        dest.write_text("\n".join(sanitized) + ("\n" if sanitized else ""), encoding="utf-8")
        events = [json.loads(line) for line in sanitized]
        manifest[check_name] = {
            "source": str(src.relative_to(ROOT)),
            "evidence_path": dest_rel,
            "event_count": len(events),
            "operations": sorted({e.get("operation") for e in events}),
            "complete": len(events) > 0,
        }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST} ({len(manifest)} entries)")
    if missing:
        print("MISSING:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
