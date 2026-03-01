"""Model loading and inference helpers for ACE-Step."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class MusicGenManager:
    """Lazy loader and wrapper around ACE-Step handlers."""

    def __init__(self) -> None:
        self._dit = None
        self._llm = None
        self._lm_available: bool = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_dit(self):
        """Load the AceStepHandler (DiT model)."""
        from acestep.handler import AceStepHandler

        logger.info("Loading DiT handler: %s", settings.dit_config)
        handler = AceStepHandler()
        handler.device = settings.device
        handler.dtype = settings.dtype
        # ACE-Step handler initialization loads model weights on first use;
        # we set config so it knows which checkpoint to load.
        return handler

    def _load_llm(self):
        """Load the LLMHandler. Returns None on failure (graceful degradation)."""
        if not settings.enable_lm:
            logger.info("LM disabled by configuration")
            return None

        try:
            from acestep.llm_inference import LLMHandler

            logger.info("Loading LLM: %s (backend=%s)", settings.lm_model_path, settings.lm_backend)
            handler = LLMHandler()
            handler.initialize(
                checkpoint_dir=settings.dit_config,
                lm_model_path=settings.lm_model_path,
                backend=settings.lm_backend,
                device="auto",
                dtype=settings.dtype,
            )
            return handler
        except Exception:
            logger.warning("LLM initialization failed, running DiT-only", exc_info=True)
            return None

    def get_dit(self):
        with self._lock:
            if self._dit is None:
                self._dit = self._load_dit()
            return self._dit

    def get_llm(self):
        with self._lock:
            if self._llm is None and not self._lm_available and settings.enable_lm:
                self._llm = self._load_llm()
                self._lm_available = self._llm is not None
            return self._llm

    @property
    def lm_available(self) -> bool:
        return self._lm_available

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(
        self,
        *,
        params,
        config,
        save_dir: str,
    ):
        """Run ACE-Step generate_music synchronously."""
        from acestep.inference import generate_music

        dit = self.get_dit()
        llm = self.get_llm()

        result = generate_music(
            dit_handler=dit,
            llm_handler=llm,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        return result

    async def generate_async(self, **kwargs):
        return await asyncio.to_thread(self.generate, **kwargs)


pipeline_manager = MusicGenManager()


def preload_if_requested() -> None:
    if settings.preload_models:
        start = time.perf_counter()
        pipeline_manager.get_dit()
        pipeline_manager.get_llm()
        duration = time.perf_counter() - start
        logger.info("Preloaded models in %.2fs", duration)
