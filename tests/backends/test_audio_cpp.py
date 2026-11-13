# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared audio.cpp runtime and verified TTS packages."""

import base64
import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Self, cast
from unittest import mock

import numpy as np
import pytest

from celune.backends.tts.audio_cpp import (
    AudioCppRuntime,
    load_audio_cpp_tts_model,
    materialize_audio_cpp_gguf,
)
from celune.backends.tts.audio_cpp_models import (
    AUDIOCPP_GGUF_REVISION,
    audio_cpp_model_artifact,
)
from celune.backends.tts.base import CeluneBackend
from celune.exceptions import ModelConsentRequired
from celune.registry.tts import (
    DOTS_TTS_MODEL_ID,
    POCKET_TTS_MODEL_ID,
    QWEN3_TTS_0_6B_MODEL_ID,
    QWEN3_TTS_1_7B_MODEL_ID,
    VOXCPM2_MODEL_ID,
)
from celune.registry.types import ModelMetadata


@pytest.mark.parametrize(
    ("backend_id", "model_id", "quantize", "language", "filename"),
    (
        (
            "mini",
            POCKET_TTS_MODEL_ID,
            True,
            "en",
            "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
        ),
        (
            "mini",
            POCKET_TTS_MODEL_ID,
            False,
            "fr",
            "PocketTTS-GGUF/french/pocket-tts-french-bf16.gguf",
        ),
        (
            "qwen3",
            QWEN3_TTS_0_6B_MODEL_ID,
            True,
            None,
            "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf",
        ),
        (
            "qwen3",
            QWEN3_TTS_1_7B_MODEL_ID,
            False,
            None,
            "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-bf16.gguf",
        ),
        (
            "dotstts",
            DOTS_TTS_MODEL_ID,
            True,
            None,
            "DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf",
        ),
        (
            "voxcpm2",
            VOXCPM2_MODEL_ID,
            False,
            None,
            "VoxCPM2-GGUF/voxcpm2-bf16.gguf",
        ),
    ),
)
def test_audio_cpp_model_artifacts_are_pinned(
    backend_id: str,
    model_id: str,
    quantize: bool,
    language: Optional[str],
    filename: str,
) -> None:
    """Resolve only exact model and language packages with integrity metadata."""
    artifact = audio_cpp_model_artifact(
        backend_id,
        model_id,
        quantize=quantize,
        language=language,
    )

    assert artifact is not None
    assert artifact.filename == filename
    assert artifact.size > 0
    assert len(artifact.sha256) == 64
    assert artifact.model_id == model_id
    assert AUDIOCPP_GGUF_REVISION == "73ce4162f6cce370dd533c6782b772135c1b2a8e"


def test_audio_cpp_resolver_rejects_unverified_models_and_languages() -> None:
    """Leave unreviewed model variants and Pocket languages on native inference."""
    assert (
        audio_cpp_model_artifact(
            "qwen3",
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            quantize=True,
        )
        is None
    )
    assert (
        audio_cpp_model_artifact(
            "mini",
            POCKET_TTS_MODEL_ID,
            quantize=True,
            language="ja",
        )
        is None
    )


def test_audio_cpp_fallback_preserves_policy_and_cancellation_errors() -> None:
    """Do not hide consent, permission, or cancellation behind native fallback."""
    metadata = ModelMetadata(
        model_id="example/model",
        display_name="Example",
        category="tts",
        revision="0123456789abcdef",
        license_id="apache-2.0",
        license_url="https://example.com/license",
        non_commercial=False,
        research_only=False,
        repository_url="https://example.com/model",
    )
    errors = (
        ModelConsentRequired(metadata),
        PermissionError("runtime access denied"),
        InterruptedError("runtime initialization cancelled"),
    )

    for error in errors:
        for requires_audio_cpp in (False, True):
            log_callback = mock.Mock()
            backend = cast(
                CeluneBackend[object],
                SimpleNamespace(
                    name="mini",
                    requires_audio_cpp=requires_audio_cpp,
                    effective_inference_mode="audio_cpp",
                    requested_inference_mode="gguf",
                    _audio_cpp_fallback_logged=False,
                    log=log_callback,
                ),
            )
            with pytest.raises(type(error)) as raised:
                CeluneBackend.fallback_audio_cpp_to_native(backend, error)

            assert raised.value is error
            assert backend.effective_inference_mode == "audio_cpp"
            log_callback.assert_not_called()


def test_required_audio_cpp_failure_keeps_the_startup_cause() -> None:
    """Include the required audio.cpp initialization failure in the startup error."""
    failure = RuntimeError("CUDA runtime binary is unavailable")
    backend = cast(
        CeluneBackend[object],
        SimpleNamespace(
            name="ming3",
            requires_audio_cpp=True,
            effective_inference_mode="audio_cpp",
            requested_inference_mode="native",
            _audio_cpp_fallback_logged=False,
            log=mock.Mock(),
        ),
    )

    with pytest.raises(
        RuntimeError, match="CUDA runtime binary is unavailable"
    ) as raised:
        CeluneBackend.fallback_audio_cpp_to_native(backend, failure)

    assert raised.value.__cause__ is failure
    assert "ming3" in str(raised.value)


def test_audio_cpp_loader_uses_artifact_filename_and_pinned_metadata(
    tmp_path: Path,
) -> None:
    """Download the exact GGUF file selected by the artifact resolver."""
    artifact = audio_cpp_model_artifact(
        "dotstts",
        DOTS_TTS_MODEL_ID,
        quantize=True,
    )
    assert artifact is not None
    server_path = tmp_path / "audiocpp_server.exe"
    cached_path = tmp_path / "blobs" / "verified-hash"
    model_path = tmp_path / "runtime" / artifact.filename
    runtime = cast(
        AudioCppRuntime,
        SimpleNamespace(server_model_id="celune-dotstts"),
    )
    with (
        mock.patch(
            "celune.backends.tts.audio_cpp.ensure_audio_cpp_server",
            return_value=server_path,
        ),
        mock.patch(
            "celune.backends.tts.audio_cpp.download_verified_gguf",
            return_value=cached_path,
        ) as download,
        mock.patch(
            "celune.backends.tts.audio_cpp.materialize_audio_cpp_gguf",
            return_value=model_path,
        ) as materialize,
        mock.patch(
            "celune.backends.tts.audio_cpp.start_audio_cpp_server",
            return_value=runtime,
        ) as start,
    ):
        loaded = load_audio_cpp_tts_model(
            artifact,
            backend_id="dotstts",
            sample_rate=48_000,
            torch_root=tmp_path,
            device=0,
            log=mock.Mock(),
            report_progress=mock.Mock(),
            default_request_options={"num_inference_steps": 4},
        )

    assert loaded is runtime
    download.assert_called_once_with(
        artifact.filename,
        size=artifact.size,
        sha256=artifact.sha256,
        report_progress=mock.ANY,
    )
    materialize.assert_called_once_with(cached_path, artifact, "dotstts")
    model_config = start.call_args.kwargs["model_config"]
    assert model_config["path"] == model_path.as_posix()
    assert model_config["family"] == "dots_tts"
    assert model_config["mode"] == "streaming"
    assert model_config["default_request_options"] == {"num_inference_steps": 4}


def test_audio_cpp_materializes_hub_blob_with_gguf_extension(tmp_path: Path) -> None:
    """Keep the GGUF suffix after resolving Hugging Face's extensionless blob path."""
    artifact = audio_cpp_model_artifact(
        "mini",
        POCKET_TTS_MODEL_ID,
        quantize=True,
        language="en",
    )
    assert artifact is not None
    payload = b"verified GGUF test payload"
    blob_path = tmp_path / "blobs" / hashlib.sha256(payload).hexdigest()
    blob_path.parent.mkdir()
    blob_path.write_bytes(payload)
    test_artifact = replace(
        artifact,
        filename="PocketTTS-GGUF/english/test.gguf",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    cache_root = tmp_path / "appdata"

    with mock.patch(
        "celune.backends.tts.audio_cpp.app_data_dir",
        return_value=cache_root,
    ):
        path = materialize_audio_cpp_gguf(blob_path, test_artifact, "mini")

    assert path.name == "test.gguf"
    assert path.is_file()
    assert not path.is_symlink()
    assert path.read_bytes() == payload


class _SSEResponse:
    """Small iterable response for one audio.cpp SSE request."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._lines)


def test_audio_cpp_stream_converts_sse_pcm_to_normalized_float32(
    tmp_path: Path,
) -> None:
    """Yield SSE PCM deltas as normalized float32 audio with the event rate."""
    pcm = np.tile(np.asarray([32767, -32768], dtype="<i2"), 1320).tobytes()
    event = {
        "type": "speech.audio.delta",
        "audio": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": 24_000,
    }
    response = _SSEResponse(
        [
            f"data: {json.dumps(event)}\n".encode(),
            b"data: [DONE]\n",
        ]
    )
    runtime = AudioCppRuntime(
        process=cast(subprocess.Popen[bytes], SimpleNamespace(poll=lambda: None)),
        base_url="http://127.0.0.1:1234",
        log_path=tmp_path / "server.log",
        model_path=tmp_path / "model.gguf",
        precision="Q8_0",
    )
    with mock.patch(
        "celune.backends.tts.audio_cpp.urllib.request.urlopen",
        return_value=response,
    ):
        chunks = list(
            runtime.stream_speech(
                {"input": "A small streaming PCM check."},
            )
        )

    assert len(chunks) == 1
    audio, sample_rate, metadata = chunks[0]
    assert sample_rate == 24_000
    assert metadata is None
    assert audio.dtype == np.float32
    assert np.all(audio >= -1.0)
    assert np.all(audio <= 1.0)
    np.testing.assert_allclose(audio[:2], [32767 / 32768, -1.0])
