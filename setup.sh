#!/bin/bash

set -euo pipefail

if ! command -v uv &> /dev/null; then
    cat <<'MSG'
Error: uv is not installed.

Install uv via one of:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  brew install uv
  pip install uv

Documentation: https://docs.astral.sh/uv/latest/installation/
MSG
    exit 1
fi

uv venv
source .venv/bin/activate

uv pip install -e .

deactivate

echo ""
echo "Setup complete. Models will download automatically on first use."
echo ""
echo "Run: ./run.sh            # Starts server on port 4009 (prod)"
echo "     ./run.sh --dev      # Development mode with auto-reload"
echo "     uv run kortexa-music-gen [--dev|--prod]"
