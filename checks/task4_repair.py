"""Task 4 required check: bounded model-feedback repair loop."""

from __future__ import annotations

import sys

from repair.loop import RepairRunner


def main() -> int:
    result = RepairRunner(reset_to_supplied=True).run()
    print(f"trace: {result.trace_path}")
    print(f"attempts: {result.attempts}")
    if result.success:
        print("RESULT: PASSED")
        return 0
    print("RESULT: FAILED")
    print(f"last_error: {result.final_error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
