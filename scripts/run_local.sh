#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -f corpus/chunks.jsonl || ! -d chroma_db ]]; then
  "$ROOT/scripts/setup_local.sh"
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PORT="${PORT:-8000}"
echo "Starting Reliable AI Agent Lab at http://127.0.0.1:${PORT}/dashboard"
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
