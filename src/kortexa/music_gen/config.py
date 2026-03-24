"""Configuration helpers for the music generation server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import torch


@dataclass
class Settings:
    """Runtime settings resolved from environment variables."""

    # ACE-Step model configuration
    dit_config: str = os.getenv("DIT_CONFIG", "acestep-v15-turbo")
    lm_model_path: str = os.getenv("LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
    enable_lm: bool = os.getenv("ENABLE_LM", "1") not in {"0", "false", "False", ""}

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "4009"))

    # Audio defaults
    default_duration: float = float(os.getenv("DEFAULT_DURATION", "30"))
    max_duration: float = float(os.getenv("MAX_DURATION", "600"))
    default_inference_steps: int = int(os.getenv("NUM_INFERENCE_STEPS", "8"))
    default_guidance_scale: float = float(os.getenv("GUIDANCE_SCALE", "7.0"))
    default_audio_format: str = os.getenv("AUDIO_FORMAT", "flac")
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "2"))

    dev_mode: bool = os.getenv("MUSIC_ENV", "production") == "development"

    @property
    def preload_models(self) -> bool:
        """Preload models on startup. Disabled in dev mode to avoid reloader memory issues."""
        env_val = os.getenv("PRELOAD_MODELS")
        if env_val is not None:
            return env_val not in {"0", "false", "False", ""}
        # Default: preload in prod, skip in dev (reloader causes double memory usage)
        return not self.dev_mode

    @cached_property
    def device(self) -> str:
        """Determine the best available device unless overridden."""
        env_device = os.getenv("DEVICE")
        if env_device:
            return env_device

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if getattr(torch.backends, "xpu", None) and torch.backends.xpu.is_available():
            return "xpu"
        return "cpu"

    @cached_property
    def dtype(self) -> torch.dtype:
        """Resolve torch dtype based on precision hint and device."""
        precision = os.getenv("MODEL_PRECISION")
        if precision:
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            key = precision.lower()
            if key in mapping:
                return mapping[key]

        if self.device in {"cuda", "mps", "xpu"}:
            return torch.bfloat16

        return torch.float32

    @cached_property
    def lm_backend(self) -> str:
        """Auto-detect best LM backend: pt on CUDA (faster, lower VRAM), mlx on MPS."""
        env_val = os.getenv("LM_BACKEND")
        if env_val:
            return env_val

        if self.device == "mps":
            return "mlx"
        return "pt"


settings = Settings()
