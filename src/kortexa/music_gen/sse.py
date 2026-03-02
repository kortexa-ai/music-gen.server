"""SSE streaming endpoint for music generation with real-time progress."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time

from fastapi.responses import StreamingResponse

from .pipelines import pipeline_manager
from .schemas import GenerateRequest
from .server import _build_config, _build_metadata, _build_params, _collect_audio_files

logger = logging.getLogger(__name__)

# ~256KB of raw audio = ~341KB base64, but we chunk the base64 string directly
CHUNK_SIZE = 256 * 1024


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_generate(req: GenerateRequest, include_audio: bool = True):
    """Run generation in a background thread, streaming progress via SSE."""

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

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

    save_dir = tempfile.mkdtemp(prefix="music_gen_sse_")

    def progress_cb(value: float, desc: str = "") -> None:
        """Called from worker thread — bridges to async queue."""
        try:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("progress", {"value": round(float(value), 4), "stage": desc or "Generating..."}),
            )
        except Exception:
            pass  # queue full or loop closed — non-fatal

    async def event_stream():
        start = time.perf_counter()
        gen_task = None
        try:
            # Kick off generation in thread pool
            gen_task = asyncio.ensure_future(
                pipeline_manager.generate_async(
                    params=params, config=config, save_dir=save_dir, progress=progress_cb,
                )
            )

            # Drain progress events while generation runs
            while not gen_task.done():
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield _sse_event(evt[0], evt[1])
                except asyncio.TimeoutError:
                    continue

            # Flush any remaining queued events
            while not queue.empty():
                evt = queue.get_nowait()
                yield _sse_event(evt[0], evt[1])

            result = gen_task.result()
            if not result.success:
                yield _sse_event("error", {"detail": result.error or "Generation failed"})
                return

            elapsed = time.perf_counter() - start

            # Stream audio chunks
            num_audios = 0
            if include_audio:
                encoded_audios = _collect_audio_files(save_dir, req.audio_format)
                num_audios = len(encoded_audios)
                for audio_idx, audio_b64 in enumerate(encoded_audios):
                    total_chunks = (len(audio_b64) + CHUNK_SIZE - 1) // CHUNK_SIZE
                    for chunk_idx in range(total_chunks):
                        chunk_data = audio_b64[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
                        yield _sse_event("audio_chunk", {
                            "audio_index": audio_idx,
                            "chunk_index": chunk_idx,
                            "total_chunks": total_chunks,
                            "data": chunk_data,
                        })

            # Metadata event
            meta = _build_metadata(
                request_type="text2music",
                caption=req.caption,
                duration=req.duration,
                steps=req.inference_steps,
                guidance_scale=req.guidance_scale,
                seed=req.seed,
                elapsed=elapsed,
                num_audios=num_audios,
                audio_format=req.audio_format,
            )
            yield _sse_event("metadata", meta.model_dump())

            # Done
            yield _sse_event("done", {
                "elapsed": round(elapsed, 2),
                "num_audios": num_audios,
            })

        except asyncio.CancelledError:
            if gen_task and not gen_task.done():
                gen_task.cancel()
            logger.info("SSE stream cancelled by client")
        except Exception as exc:
            logger.exception("SSE stream error")
            yield _sse_event("error", {"detail": str(exc)})
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
