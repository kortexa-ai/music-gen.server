"""Tests for API helpers that do not require model checkpoints."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from kortexa.music_gen.schemas import GenerateRequest
from kortexa.music_gen.server import _collect_audio_files, _ext, health


def test_collect_audio_files_sorts_and_ignores_non_audio(tmp_path) -> None:
    (tmp_path / "b.wav").write_bytes(b"second")
    (tmp_path / "a.flac").write_bytes(b"first")
    (tmp_path / "notes.txt").write_text("not audio")

    assert _collect_audio_files(str(tmp_path), "flac") == [
        base64.b64encode(b"first").decode(),
        base64.b64encode(b"second").decode(),
    ]


def test_upload_extension() -> None:
    assert _ext("demo.track.wav") == ".wav"
    assert _ext("README") == ""
    assert _ext(None) == ""


def test_request_rejects_unsupported_audio_format() -> None:
    with pytest.raises(ValidationError, match="audio_format"):
        GenerateRequest(caption="test", audio_format="midi")


@pytest.mark.asyncio
async def test_health_does_not_load_models() -> None:
    response = await health()

    assert response["status"] == "ok"
    assert response["dit_config"]
