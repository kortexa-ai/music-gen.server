#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[music-gen.server] UV_MISSING" >&2
  exit 2
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-4009}

MUSIC_ENV=${MUSIC_ENV:-production}
PASS_MODE_FLAG=""
RELOAD_FLAG=""

for arg in "$@"; do
  case "$arg" in
    --dev)
      MUSIC_ENV=development
      PASS_MODE_FLAG="--dev"
      RELOAD_FLAG="--reload"
      ;;
    --prod)
      MUSIC_ENV=production
      PASS_MODE_FLAG="--prod"
      ;;
  esac
done

echo "Starting Kortexa Music Generation server ($MUSIC_ENV)..."
# Use .venv/bin directly to avoid uv run re-resolving and downgrading CUDA torch to CPU
exec .venv/bin/kortexa-music-gen --host "$HOST" --port "$PORT" $PASS_MODE_FLAG $RELOAD_FLAG
