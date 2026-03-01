"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import settings


class GenerateRequest(BaseModel):
    """Request body for text-to-music generation."""

    caption: str = Field(..., min_length=1, max_length=512)
    lyrics: str = Field("[Instrumental]", max_length=4096)
    instrumental: bool = Field(False)
    vocal_language: str = Field("en")
    duration: Optional[float] = Field(default=settings.default_duration, ge=10, le=600)
    bpm: Optional[int] = Field(None, ge=30, le=300)
    keyscale: Optional[str] = Field(None)
    timesignature: Optional[str] = Field(None)
    inference_steps: int = Field(default=settings.default_inference_steps, ge=1, le=100)
    guidance_scale: float = Field(default=settings.default_guidance_scale, ge=0.0, le=20.0)
    seed: int = Field(-1, ge=-1, le=2**32 - 1)
    batch_size: int = Field(1, ge=1)
    audio_format: str = Field(default=settings.default_audio_format)
    thinking: bool = Field(True)

    @field_validator("batch_size")
    @classmethod
    def _check_batch(cls, value: int) -> int:
        if value > settings.max_batch_size:
            raise ValueError(f"batch_size must be <= {settings.max_batch_size}")
        return value

    @field_validator("audio_format")
    @classmethod
    def _check_format(cls, value: str) -> str:
        allowed = {"mp3", "wav", "flac", "wav32", "opus", "aac"}
        if value not in allowed:
            raise ValueError(f"audio_format must be one of {allowed}")
        return value

    @field_validator("timesignature")
    @classmethod
    def _check_timesig(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in {"2/4", "3/4", "4/4", "6/8"}:
            raise ValueError("timesignature must be one of: 2/4, 3/4, 4/4, 6/8")
        return value


class InferenceMetadata(BaseModel):
    """Response metadata shared across endpoints."""

    request_type: Literal["text2music", "cover", "repaint"]
    dit_config: str
    device: str
    dtype: str
    caption: str
    duration: Optional[float]
    steps: int
    guidance_scale: float
    seed: int
    elapsed: float
    num_audios: int
    audio_format: str
    lm_enabled: bool


class AudioResponse(BaseModel):
    """Base64 encoded audio results and metadata."""

    audios: List[str]
    metadata: InferenceMetadata


class ErrorResponse(BaseModel):
    detail: str
