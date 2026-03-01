#!/usr/bin/env python3
"""CLI entrypoint for the Kortexa Music Generation server."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kortexa.music_gen.config import settings
from kortexa.music_gen.pipelines import preload_if_requested
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Kortexa Music Generation server")
    parser.add_argument("--host", default=settings.host, help="Host to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--root-path", default="", help="ASGI root path behind reverse proxies")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dev", action="store_true", help="Use development environment")
    group.add_argument("--prod", action="store_true", help="Use production environment")
    args = parser.parse_args()

    mode = "development" if args.dev else "production"
    os.environ["MUSIC_ENV"] = mode

    if settings.preload_models:
        preload_if_requested()

    log_config = copy.deepcopy(UVICORN_LOGGING_CONFIG)
    loggers = log_config.setdefault("loggers", {})
    loggers["kortexa"] = {"handlers": ["default"], "level": "INFO", "propagate": False}

    print("♪ Kortexa Music Generation Server")
    print(f"Environment: {mode}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Device: {settings.device} ({settings.dtype})")
    print(f"DiT: {settings.dit_config}")
    print(f"LM: {settings.lm_model_path} (enabled={settings.enable_lm}, backend={settings.lm_backend})")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "kortexa.music_gen.server:app",
        host=args.host,
        port=args.port,
        root_path=args.root_path,
        reload=args.reload,
        log_level="info",
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
