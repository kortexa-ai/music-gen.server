# Kortexa Music Generation Server

FastAPI service for ACE-Step 1.5 music generation. It provides text-to-music,
streaming, cover, and repaint endpoints on port 4009.

## Setup and run

`./setup.sh` installs the locked platform dependencies and downloads the
ACE-Step model checkpoints. It is intentionally heavyweight. After setup, run
the managed production service with `./run.sh` or use `./run.sh --dev` for
local auto-reload.

The lightweight health check is `GET /health`. Generation is available through
`POST /generate`, `POST /generate/stream`, `POST /cover`, and `POST /repaint`.

On Smarty, the systemd unit pins CUDA to the RTX PRO 6000 by UUID. Install or
refresh the unit with `ktxsvc install music-gen.server`. Before starting it,
check the GPU workloads and coordinate with other agents; leave room for active
experiments and production services. For a manual run on Smarty, use:

```bash
CUDA_VISIBLE_DEVICES=GPU-a71210ca-e14a-755a-88bb-77f53a2102f6 ./run.sh
```

## Dependency updates and startup warnings

Use `uv lock --upgrade` and `uv sync --locked` to refresh dependencies. ACE-Step
currently pins Linux PyTorch to 2.10.0 with CUDA 12.8 and constrains other ML
packages. Keep those upstream compatibility constraints when updating; the newest
PyPI release of every package cannot be installed together.
The lockfile targets Linux x86_64 and macOS on Apple Silicon. The bundled
`nano-vllm` source follows the same ACE-Step Git repository as the main package.

`bitsandbytes not installed. Using standard AdamW.` comes from ACE-Step's
training module at import time. It means fine-tuning uses the standard optimizer
instead of the optional 8-bit optimizer. This server performs inference, so the
warning does not affect generation quality, speed, or inference memory use.
No workaround is needed. Revisit the dependency if training is added and optimizer
memory becomes a constraint.

## Development checks

```bash
uv sync --locked
uv run pytest -q
uv run black --check .
uv run ruff check .
uv run mypy src scripts
uv build
```
