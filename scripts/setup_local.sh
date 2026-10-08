#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY'
import sys

if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(
        "Reliable AI Agent Lab requires Python 3.11, 3.12, or 3.13 "
        f"(found {sys.version.split()[0]})."
    )
PY

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating .venv with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv .venv
fi

echo "Installing pinned dependencies..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ ! -f corpus/chunks.jsonl || ! -d chroma_db ]]; then
  echo "Building the public-domain corpus and local vector index..."
  .venv/bin/python -m ingest.build
else
  echo "Corpus and vector index already exist; skipping rebuild."
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created optional .env from .env.example (stub mode is enabled by default)."
fi

echo
echo "Setup complete. Start the workbench with:"
echo "  ./scripts/run_local.sh"
