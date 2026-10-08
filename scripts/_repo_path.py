"""Ensure the repository root is importable when scripts are run directly."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ROOT = str(_REPO_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
