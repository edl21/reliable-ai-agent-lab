"""Live Task 4 repair gate: start from broken baseline, require acceptance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import adapter.model as model_mod
from adapter import runtime
from repair.loop import RepairRunner


def main() -> int:
    model_mod._SINGLETON = None
    runtime.clear_override()

    result = RepairRunner(reset_to_supplied=True, max_edits=3).run()
    events = [
        json.loads(line) for line in result.trace_path.read_text().splitlines()
    ]
    stubbed_flags = [
        (e.get("notes") or {}).get("stubbed")
        for e in events
        if (e.get("notes") or {}).get("state") == "S2_validate_patch"
    ]
    row = {
        "success": result.success,
        "attempts": result.attempts,
        "trace_path": str(result.trace_path),
        "final_error_head": (result.final_error or "")[:300],
        "stubbed_any": any(flag is True for flag in stubbed_flags),
        "stubbed_all_false": bool(stubbed_flags)
        and all(flag is False for flag in stubbed_flags),
    }
    print(json.dumps(row, indent=2))
    print("--- GATE ---")
    gate = result.success and row["stubbed_all_false"]
    print("live_repair_gate", gate)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
