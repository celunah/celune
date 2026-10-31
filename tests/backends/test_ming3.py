# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming 3.0 package selection and native streaming integration."""

import io
import json
import base64
import threading
from pathlib import Path
from unittest import mock
from typing import Self, cast
from types import SimpleNamespace
from collections.abc import Iterator

import pytest
import numpy as np

from celune.cedts import worker
from celune.registry.tts import (
    MING3_TTS_BF16_FILE,
    MING3_TTS_GGUF_REPO,
    MING3_TTS_MODEL_ID,
    MING3_TTS_Q8_FILE,
)
from celune.typing.worker import WorkerRequest
from celune.typing.backends import BackendRuntime
from celune.backends.tts.ming3 import Ming3, MingRuntime


def _backend(quantize: bool) -> Ming3:
    """Construct the backend fields used by isolated unit tests."""
    backend = Ming3.__new__(Ming3)
    backend.quantization_requested = quantize
    backend._verified_model_artifacts = set()
    return backend


@pytest.mark.parametrize(
    ("quantize", "precision", "filename"),
    (
        (True, "q8_0", MING3_TTS_Q8_FILE),
        (False, "bf16", MING3_TTS_BF16_FILE),
    ),
)
def test_ming3_selects_package_from_celune_quantization(
    quantize: bool,
    precision: str,
    filename: str,
) -> None:
    """Use Celune's quantization toggle to choose the GGUF package."""
    backend = _backend(quantize)

    assert backend.package_precision == precision
    assert backend._package()[0] == filename
    assert backend.model_contract(MING3_TTS_MODEL_ID).model_id == MING3_TTS_GGUF_REPO
    assert backend.model_contract(MING3_TTS_MODEL_ID).variant == precision


@pytest.mark.parametrize(
    ("language", "expected"),
    (("Auto", "en"), ("it", "en"), ("ja-JP", "ja")),
)
def test_ming3_resolves_auto_and_unsupported_languages_to_english(
    language: str,
    expected: str,
) -> None:
    """Use a supported detected language or fall back to English."""
    assert _backend(True).resolve_generation_language(language) == expected


def test_ming3_runtime_architecture_falls_back_to_python_platform() -> None:
    """Resolve the bundled platform when the isolated worker omits machine env."""
    with (
        mock.patch("celune.backends.tts.ming3.platform.machine", return_value=""),
        mock.patch(
            "celune.backends.tts.ming3.sysconfig.get_platform",
            return_value="win-amd64",
        ),
    ):
        assets = Ming3._runtime_assets()

    assert assets[0][0] == "audio-v0.9.3-bin-windows-x64-cuda13.3.zip"


def test_ming3_preserves_gguf_filename_with_a_hard_link(tmp_path: Path) -> None:
    """Keep the GGUF extension without duplicating the cached model bytes."""
    source_path = tmp_path / "hub" / "blobs" / "model-hash"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"GGUF data")
    backend_data = tmp_path / "celune"

    with mock.patch(
        "celune.backends.tts.ming3.app_data_dir",
        return_value=backend_data,
    ):
        runtime_path = Ming3._model_runtime_path(source_path, "ming3-q8.gguf")

    assert runtime_path.suffix == ".gguf"
    assert runtime_path.read_bytes() == source_path.read_bytes()
    assert runtime_path.samefile(source_path)


def test_ming3_load_failure_does_not_switch_to_bf16() -> None:
    """Keep the selected Q8 precision when a worker model load fails."""

    class FailingQuantizedBackend:
        """Backend stand-in that must retain its configured precision."""

        quantization_requested = True
        allow_quantization_fallback = False
        model = None

        def __init__(self) -> None:
            self.load_count = 0
            self.quantization_disabled = False

        def load_model(self, model_id: str) -> None:
            """Fail the selected precision's only load attempt."""
            self.load_count += 1
            raise RuntimeError(f"failed to load {model_id}")

        def disable_runtime_quantization(self) -> None:
            """Record an unwanted switch to the unquantized package."""
            self.quantization_disabled = True

    backend = FailingQuantizedBackend()
    request = cast(
        WorkerRequest,
        {"operation": "load_model", "arguments": {"model_id": "ming3"}},
    )

    with pytest.raises(RuntimeError, match="failed to load ming3"):
        worker._run_request(
            cast(BackendRuntime, backend),
            request,
            {},
            1,
            io.BytesIO(),
        )

    assert backend.load_count == 1
    assert not backend.quantization_disabled


class _StreamingResponse:
    """Provide deterministic SSE lines and expose whether the stream finished."""

    def __init__(self, lines: list[bytes]) -> None:
        self.lines = lines
        self.completed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield self.lines[0]
        self.completed = True
        yield from self.lines[1:]


def test_ming3_yields_normalized_audio_before_stream_completion(
    tmp_path: Path,
) -> None:
    """Yield decoded SSE PCM before the upstream stream finishes."""
    reference = tmp_path / "reference.wav"
    reference.touch()
    pcm = np.tile(np.array([-32768, 0, 32767], dtype="<i2"), 1470)
    event = {
        "type": "speech.audio.delta",
        "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
    }
    response = _StreamingResponse(
        [
            b"data: " + json.dumps(event).encode("utf-8") + b"\n",
            b'data: {"type":"speech.audio.done"}\n',
            b"data: [DONE]\n",
        ]
    )
    backend = _backend(True)
    backend.current_seed = 17
    loader = SimpleNamespace(
        materialize=mock.Mock(return_value=reference),
        bundle=SimpleNamespace(voices={"balanced": {"reference_text": "Hello there."}}),
    )
    process = mock.Mock()
    process.poll.return_value = None
    runtime = MingRuntime(
        process=process,
        base_url="http://127.0.0.1:3456",
        log_path=tmp_path / "ming.log",
        model_path=tmp_path / "model.gguf",
        precision="q8_0",
        request_lock=threading.Lock(),
    )

    with (
        mock.patch.object(
            Ming3,
            "_require_compatible_bundle",
            return_value=(loader, ("balanced",)),
        ),
        mock.patch.object(Ming3, "_truncate_reference", return_value=reference),
        mock.patch.object(Ming3, "_apply_seed"),
        mock.patch(
            "celune.backends.tts.ming3.urllib.request.urlopen", return_value=response
        ),
    ):
        stream = backend.generate_stream(
            runtime,
            text="A short spoken sentence.",
            voice="balanced",
            language="en-US",
        )
        audio, sample_rate, metadata = next(stream)
        assert not response.completed
        remainder = list(stream)

    assert not remainder
    assert audio.dtype == np.float32
    assert audio.size == 4410
    assert audio[0] == -1.0
    assert audio[1] == 0.0
    assert audio[2] == pytest.approx(32767 / 32768)
    assert sample_rate == 44_100
    assert metadata is None


def test_ming3_rejects_empty_streamed_audio(tmp_path: Path) -> None:
    """Treat an empty upstream SSE audio response as a generation failure."""
    reference = tmp_path / "reference.wav"
    reference.touch()
    response = _StreamingResponse(
        [
            b'data: {"type":"speech.audio.delta","audio":""}\n',
            b"data: [DONE]\n",
        ]
    )
    backend = _backend(True)
    backend.current_seed = 17
    loader = SimpleNamespace(
        materialize=mock.Mock(return_value=reference),
        bundle=SimpleNamespace(voices={"balanced": {"reference_text": "Hello there."}}),
    )
    process = mock.Mock()
    process.poll.return_value = None
    runtime = MingRuntime(
        process=process,
        base_url="http://127.0.0.1:3456",
        log_path=tmp_path / "ming.log",
        model_path=tmp_path / "model.gguf",
        precision="q8_0",
        request_lock=threading.Lock(),
    )

    with (
        mock.patch.object(
            Ming3,
            "_require_compatible_bundle",
            return_value=(loader, ("balanced",)),
        ),
        mock.patch.object(Ming3, "_truncate_reference", return_value=reference),
        mock.patch.object(Ming3, "_apply_seed"),
        mock.patch(
            "celune.backends.tts.ming3.urllib.request.urlopen", return_value=response
        ),
        pytest.raises(RuntimeError, match="Ming returned empty audio"),
    ):
        list(
            backend.generate_stream(
                runtime,
                text="A short spoken sentence.",
                voice="balanced",
                language="en-US",
            )
        )


def test_ming3_checks_package_hash_and_plausible_duration(tmp_path: Path) -> None:
    """Verify GGUF integrity and cap expected audio duration from input length."""
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"verified model package")
    digest = Ming3._archive_sha256(model_path)
    backend = _backend(True)

    backend._verify_model_artifact(model_path, model_path.stat().st_size, digest)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        backend._verify_model_artifact(
            model_path,
            model_path.stat().st_size,
            "0" * 64,
        )

    assert Ming3._maximum_audio_seconds("Hi there.") == 10.0
    assert Ming3._maximum_audio_seconds("sentence " * 100) == 120.0
