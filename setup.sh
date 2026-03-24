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
ARCH="$(uname -m)"
HAS_CUDA=false
if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected."
    HAS_CUDA=true
fi

uv venv

if [[ "$HAS_CUDA" == true && "$ARCH" == "aarch64" ]]; then
    # DGX Spark (aarch64): PyPI stable torch is CPU-only, use nightly + manual install
    source .venv/bin/activate
    uv pip install -e .
    echo "Installing PyTorch nightly with CUDA support (aarch64)..."
    uv pip install --reinstall --pre torch torchvision torchaudio torchao --index-url https://download.pytorch.org/whl/nightly/cu128
else
    # x86_64 CUDA / macOS: lockfile-managed via uv sync
    uv sync
fi

echo ""
echo "Downloading ACE-Step models into checkpoints/..."
uv run python -c "
from pathlib import Path
from acestep.model_downloader import ensure_main_model, ensure_dit_model
cp = Path('checkpoints')
cp.mkdir(exist_ok=True)
ok, msg = ensure_main_model(cp, prefer_source='huggingface')
print(msg)
ok2, msg2 = ensure_dit_model('acestep-v15-turbo', cp, prefer_source='huggingface')
print(msg2)
"

# Patch acestep: CUDA graph capture fails on Blackwell, force eager mode
if [[ "$HAS_CUDA" == true ]]; then
    ACESTEP_LLM=$(find .venv/lib -name "llm_inference.py" -path "*/acestep/*" 2>/dev/null | head -1)
    [[ -n "$ACESTEP_LLM" ]] && sed -i 's/enforce_eager_for_vllm = bool(is_rocm or is_jetson)/enforce_eager_for_vllm = True/' "$ACESTEP_LLM"
fi

# deactivate venv if we activated it (aarch64 path)
if [[ "$(type -t deactivate)" == "function" ]]; then
    deactivate
fi

echo ""
echo "Setup complete."
echo ""
echo "Run: ./run.sh            # Starts server on port 4009 (prod)"
echo "     ./run.sh --dev      # Development mode with auto-reload"
echo "     uv run kortexa-music-gen [--dev|--prod]"
