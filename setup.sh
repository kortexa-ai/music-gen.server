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

OS="$(uname -s)"
HAS_CUDA=false
INSTALL_TARGET="-e ."
if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected."
    INSTALL_TARGET="-e '.[cuda-compat]'"
    HAS_CUDA=true
fi

uv venv
source .venv/bin/activate

eval uv pip install $INSTALL_TARGET
if [[ "$HAS_CUDA" == true ]]; then
    # PyPI default torch on aarch64 is CPU-only; overwrite with CUDA torch from nightly index
    echo "Installing PyTorch with CUDA support..."
    uv pip install --reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

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
