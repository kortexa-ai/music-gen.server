"""Tests for API helpers that do not require model checkpoints."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from kortexa.music_gen.schemas import GenerateRequest
from kortexa.music_gen.server import _collect_audio_files, _ext, app, health, pipeline_manager


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


@pytest.mark.parametrize(
    ("endpoint", "upload_field", "extra_fields"),
    [
        ("/cover", "reference_audio", {}),
        ("/repaint", "src_audio", {"repainting_start": "2", "repainting_end": "4"}),
    ],
)
def test_audio_upload_uses_source_audio_and_returns_only_generated_files(
    monkeypatch, endpoint: str, upload_field: str, extra_fields: dict[str, str]
) -> None:
    monkeypatch.setenv("PRELOAD_MODELS", "0")

    async def fake_generate(*, params: Any, save_dir: str, **_kwargs: Any) -> SimpleNamespace:
        source = Path(params.src_audio)
        assert source.read_bytes() == b"uploaded audio"
        assert source.parent != Path(save_dir)
        assert params.reference_audio is None
        assert params.keyscale == ""
        assert params.timesignature == ""
        Path(save_dir, "generated.flac").write_bytes(b"generated audio")
        return SimpleNamespace(success=True)

    monkeypatch.setattr(pipeline_manager, "generate_async", fake_generate)
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            data={"caption": "test", "duration": "10", **extra_fields},
            files={upload_field: ("upload.flac", b"uploaded audio", "audio/flac")},
        )

    assert response.status_code == 200, response.text
    assert response.json()["audios"] == [base64.b64encode(b"generated audio").decode()]
    assert response.json()["metadata"]["num_audios"] == 1
