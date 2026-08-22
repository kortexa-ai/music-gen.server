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

## Development checks

```bash
uv sync --locked
uv run pytest -q
uv run black --check .
uv run ruff check .
uv run mypy src scripts
uv build
```
