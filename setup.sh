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

echo ""
echo "Downloading ACE-Step models into checkpoints/..."
python -c "
from pathlib import Path
from acestep.model_downloader import ensure_main_model, ensure_dit_model
cp = Path('checkpoints')
cp.mkdir(exist_ok=True)
ok, msg = ensure_main_model(cp, prefer_source='huggingface')
print(msg)
ok2, msg2 = ensure_dit_model('acestep-v15-turbo', cp, prefer_source='huggingface')
print(msg2)
"

deactivate

echo ""
echo "Setup complete."
echo ""
echo "Run: ./run.sh            # Starts server on port 4009 (prod)"
echo "     ./run.sh --dev      # Development mode with auto-reload"
echo "     uv run kortexa-music-gen [--dev|--prod]"
