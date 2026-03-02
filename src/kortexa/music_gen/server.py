"""FastAPI application exposing ACE-Step music generation endpoints."""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import settings
from .pipelines import pipeline_manager, preload_if_requested
from .schemas import AudioResponse, GenerateRequest, InferenceMetadata

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kortexa Music Generation Server",
    version="0.1.0",
    description="Music generation using ACE-Step 1.5 diffusion models.",
)


@app.on_event("startup")
async def _startup() -> None:
    if settings.preload_models:
        logger.info("Preloading ACE-Step models per configuration")
        preload_if_requested()


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "device": settings.device,
        "dtype": str(settings.dtype),
        "dit_config": settings.dit_config,
        "lm_model": settings.lm_model_path,
        "lm_enabled": settings.enable_lm,
        "lm_backend": settings.lm_backend,
        "lm_available": pipeline_manager.lm_available,
    }


@contextmanager
def _temp_dir():
    """Create a temp directory and clean it up on exit."""
    d = tempfile.mkdtemp(prefix="music_gen_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _collect_audio_files(directory: str, audio_format: str) -> list[str]:
    """Read all audio files from directory and return as base64 strings."""
    encoded = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        # Accept files matching the requested format, or any audio file
        if fname.endswith(f".{audio_format}") or fname.endswith((".flac", ".mp3", ".wav", ".opus", ".aac")):
            with open(fpath, "rb") as f:
                encoded.append(base64.b64encode(f.read()).decode("utf-8"))
    return encoded


def _build_params(
    *,
    task_type: str,
    caption: str,
    lyrics: str = "[Instrumental]",
    instrumental: bool = False,
    vocal_language: str = "en",
    duration: Optional[float] = None,
    bpm: Optional[int] = None,
    keyscale: Optional[str] = None,
    timesignature: Optional[str] = None,
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    seed: int = -1,
    thinking: bool = True,
    # cover-specific
    reference_audio: Optional[str] = None,
    audio_cover_strength: float = 0.5,
    # repaint-specific
    src_audio: Optional[str] = None,
    repainting_start: float = 0.0,
    repainting_end: float = 0.0,
):
    """Build ACE-Step GenerationParams."""
    from acestep.inference import GenerationParams

    return GenerationParams(
        caption=caption,
        lyrics=lyrics,
        instrumental=instrumental,
        vocal_language=vocal_language,
        duration=duration,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        thinking=thinking,
        task_type=task_type,
        # Turbo defaults
        shift=3.0,
        infer_method="ode",
        # cover/repaint fields
        reference_audio=reference_audio or None,
        audio_cover_strength=audio_cover_strength,
        src_audio=src_audio or None,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
    )


def _build_config(*, batch_size: int = 1, audio_format: str = "flac", seed: int = -1):
    """Build ACE-Step GenerationConfig."""
    from acestep.inference import GenerationConfig

    return GenerationConfig(
        batch_size=batch_size,
        audio_format=audio_format,
        use_random_seed=(seed == -1),
        seeds=None if seed == -1 else [seed],
    )


def _build_metadata(
    *,
    request_type: str,
    caption: str,
    duration: Optional[float],
    steps: int,
    guidance_scale: float,
    seed: int,
    elapsed: float,
    num_audios: int,
    audio_format: str,
) -> InferenceMetadata:
    return InferenceMetadata(
        request_type=request_type,
        dit_config=settings.dit_config,
        device=settings.device,
        dtype=str(settings.dtype),
        caption=caption,
        duration=duration,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        elapsed=elapsed,
        num_audios=num_audios,
        audio_format=audio_format,
        lm_enabled=pipeline_manager.lm_available,
    )


# ----------------------------------------------------------------------
# POST /generate — text-to-music
# ----------------------------------------------------------------------
@app.post(
    "/generate",
    response_model=AudioResponse,
    responses={400: {"description": "Bad Request"}, 500: {"description": "Inference failed"}},
)
async def generate(req: GenerateRequest) -> AudioResponse:
    logger.info(
        "generate: caption='%s' duration=%s steps=%d guidance=%.1f",
        req.caption, req.duration, req.inference_steps, req.guidance_scale,
    )

    params = _build_params(
        task_type="text2music",
        caption=req.caption,
        lyrics=req.lyrics,
        instrumental=req.instrumental,
        vocal_language=req.vocal_language,
        duration=req.duration,
        bpm=req.bpm,
        keyscale=req.keyscale,
        timesignature=req.timesignature,
        inference_steps=req.inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        thinking=req.thinking,
    )
    config = _build_config(
        batch_size=req.batch_size,
        audio_format=req.audio_format,
        seed=req.seed,
    )

    start = time.perf_counter()
    with _temp_dir() as save_dir:
        try:
            result = await pipeline_manager.generate_async(
                params=params, config=config, save_dir=save_dir,
            )
        except Exception as exc:
            logger.exception("Generation failure")
            raise HTTPException(status_code=500, detail="Music generation failed") from exc

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or "Generation failed")

        encoded = _collect_audio_files(save_dir, req.audio_format)

    elapsed = time.perf_counter() - start
    logger.info("Generated %d audios in %.2fs", len(encoded), elapsed)

    return AudioResponse(
        audios=encoded,
        metadata=_build_metadata(
            request_type="text2music",
            caption=req.caption,
            duration=req.duration,
            steps=req.inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            elapsed=elapsed,
            num_audios=len(encoded),
            audio_format=req.audio_format,
        ),
    )


# ----------------------------------------------------------------------
# POST /cover — style transfer with reference audio
# ----------------------------------------------------------------------
@app.post(
    "/cover",
    response_model=AudioResponse,
    responses={400: {"description": "Bad Request"}, 500: {"description": "Inference failed"}},
)
async def cover(
    reference_audio: UploadFile = File(...),
    caption: str = Form(..., min_length=1, max_length=512),
    lyrics: str = Form("[Instrumental]"),
    instrumental: bool = Form(False),
    vocal_language: str = Form("en"),
    duration: Optional[float] = Form(default=settings.default_duration),
    bpm: Optional[int] = Form(None),
    keyscale: Optional[str] = Form(None),
    timesignature: Optional[str] = Form(None),
    inference_steps: int = Form(default=settings.default_inference_steps),
    guidance_scale: float = Form(default=settings.default_guidance_scale),
    seed: int = Form(-1),
    batch_size: int = Form(1),
    audio_format: str = Form(default=settings.default_audio_format),
    audio_cover_strength: float = Form(0.5),
    thinking: bool = Form(True),
) -> AudioResponse:
    logger.info("cover: caption='%s' strength=%.2f", caption, audio_cover_strength)

    if batch_size > settings.max_batch_size:
        raise HTTPException(status_code=400, detail=f"batch_size must be <= {settings.max_batch_size}")

    start = time.perf_counter()
    with _temp_dir() as save_dir:
        # Save uploaded reference audio
        ref_path = os.path.join(save_dir, "reference_audio" + _ext(reference_audio.filename))
        data = await reference_audio.read()
        await reference_audio.close()
        with open(ref_path, "wb") as f:
            f.write(data)

        params = _build_params(
            task_type="cover",
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=vocal_language,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            thinking=thinking,
            reference_audio=ref_path,
            audio_cover_strength=audio_cover_strength,
        )
        config = _build_config(batch_size=batch_size, audio_format=audio_format, seed=seed)

        try:
            result = await pipeline_manager.generate_async(
                params=params, config=config, save_dir=save_dir,
            )
        except Exception as exc:
            logger.exception("Cover failure")
            raise HTTPException(status_code=500, detail="Cover generation failed") from exc

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or "Cover failed")

        encoded = _collect_audio_files(save_dir, audio_format)

    elapsed = time.perf_counter() - start
    logger.info("Cover generated %d audios in %.2fs", len(encoded), elapsed)

    return AudioResponse(
        audios=encoded,
        metadata=_build_metadata(
            request_type="cover",
            caption=caption,
            duration=duration,
            steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            elapsed=elapsed,
            num_audios=len(encoded),
            audio_format=audio_format,
        ),
    )


# ----------------------------------------------------------------------
# POST /repaint — selective region editing
# ----------------------------------------------------------------------
@app.post(
    "/repaint",
    response_model=AudioResponse,
    responses={400: {"description": "Bad Request"}, 500: {"description": "Inference failed"}},
)
async def repaint(
    src_audio: UploadFile = File(...),
    caption: str = Form(..., min_length=1, max_length=512),
    repainting_start: float = Form(...),
    repainting_end: float = Form(...),
    lyrics: str = Form("[Instrumental]"),
    instrumental: bool = Form(False),
    vocal_language: str = Form("en"),
    duration: Optional[float] = Form(default=settings.default_duration),
    bpm: Optional[int] = Form(None),
    keyscale: Optional[str] = Form(None),
    timesignature: Optional[str] = Form(None),
    inference_steps: int = Form(default=settings.default_inference_steps),
    guidance_scale: float = Form(default=settings.default_guidance_scale),
    seed: int = Form(-1),
    batch_size: int = Form(1),
    audio_format: str = Form(default=settings.default_audio_format),
    thinking: bool = Form(True),
) -> AudioResponse:
    logger.info("repaint: caption='%s' range=%.1f-%.1f", caption, repainting_start, repainting_end)

    if repainting_end <= repainting_start:
        raise HTTPException(status_code=400, detail="repainting_end must be > repainting_start")
    if batch_size > settings.max_batch_size:
        raise HTTPException(status_code=400, detail=f"batch_size must be <= {settings.max_batch_size}")

    start = time.perf_counter()
    with _temp_dir() as save_dir:
        # Save uploaded source audio
        src_path = os.path.join(save_dir, "src_audio" + _ext(src_audio.filename))
        data = await src_audio.read()
        await src_audio.close()
        with open(src_path, "wb") as f:
            f.write(data)

        params = _build_params(
            task_type="repaint",
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=vocal_language,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            thinking=thinking,
            src_audio=src_path,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
        )
        config = _build_config(batch_size=batch_size, audio_format=audio_format, seed=seed)

        try:
            result = await pipeline_manager.generate_async(
                params=params, config=config, save_dir=save_dir,
            )
        except Exception as exc:
            logger.exception("Repaint failure")
            raise HTTPException(status_code=500, detail="Repaint generation failed") from exc

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or "Repaint failed")

        encoded = _collect_audio_files(save_dir, audio_format)

    elapsed = time.perf_counter() - start
    logger.info("Repaint generated %d audios in %.2fs", len(encoded), elapsed)

    return AudioResponse(
        audios=encoded,
        metadata=_build_metadata(
            request_type="repaint",
            caption=caption,
            duration=duration,
            steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            elapsed=elapsed,
            num_audios=len(encoded),
            audio_format=audio_format,
        ),
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _ext(filename: Optional[str]) -> str:
    """Extract file extension from upload filename, defaulting to empty."""
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1]
    return ""


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
