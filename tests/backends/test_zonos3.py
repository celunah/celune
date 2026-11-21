# SPDX-License-Identifier: Apache-2.0
"""Tests for ZONOS3-Mini backend selection and progressive stream adaptation."""

import base64
import threading
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from collections.abc import Callable

import pytest
import numpy as np

from celune.typing.backends import BackendArgumentValue
from celune.backends.tts.zonos3 import Zonos3, Zonos3Runtime
from celune.registry.tts import TTS_MODELS, ZONOS3_TTS_MODEL_ID, ZONOS3_TTS_REVISION


def _backend() -> Zonos3:
    """Build only the backend state used by these isolated tests."""
    backend = object.__new__(Zonos3)
    backend.current_seed = 19
    backend.model_name = ZONOS3_TTS_MODEL_ID
    return backend


@pytest.mark.parametrize(
    ("language", "expected"),
    (
        ("en", "en_us"),
        ("en-GB", "en_gb"),
        ("fr-FR", "fr_fr"),
        ("pt", "pt_br"),
        ("zh-CN", "cmn"),
        ("pl", "pl"),
        ("unknown", "en_us"),
    ),
)
def test_zonos3_resolves_celune_language_codes(
    language: str,
    expected: str,
) -> None:
    """Map Celune language identifiers to ZONOS3's request codes."""
    assert _backend().resolve_generation_language(language) == expected


def test_zonos3_registers_apache_model_snapshot() -> None:
    """Expose the pinned Apache-2.0 weight snapshot through Celune consent."""
    metadata = next(item for item in TTS_MODELS if item.model_id == ZONOS3_TTS_MODEL_ID)

    assert metadata.revision == ZONOS3_TTS_REVISION
    assert metadata.license_id == "apache-2.0"
    assert not metadata.requires_consent


def test_zonos3_forwards_progressive_audio_before_synthesis_finishes(
    tmp_path: Path,
) -> None:
    """Yield normalized callback chunks while the upstream producer is still active."""
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference wav")
    loader = SimpleNamespace(materialize=mock.Mock(return_value=reference))
    runtime = Zonos3Runtime(None, None, tmp_path)
    first_chunk = threading.Event()
    second_chunk = threading.Event()
    allow_second_chunk = threading.Event()
    allow_finish = threading.Event()
    synthesis_finished = threading.Event()

    def synthesize(
        received_runtime: Zonos3Runtime,
        request: dict[str, BackendArgumentValue],
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        """Emit two chunks with a gate between streaming and completion."""
        assert received_runtime is runtime
        assert request["language"] == "en_us"
        assert request["speaker_mode"] == "embedding"
        assert request["speaker_audio_base64"] == base64.b64encode(
            reference.read_bytes()
        ).decode("ascii")
        on_audio(np.array([0.0, 0.5], dtype=np.float32))
        first_chunk.set()
        assert allow_second_chunk.wait(timeout=5)
        on_audio(np.array([2.0, -2.0], dtype=np.float32))
        second_chunk.set()
        assert allow_finish.wait(timeout=5)
        synthesis_finished.set()

    def release_second_chunk() -> None:
        """Let the upstream callback continue after its first chunk is queued."""
        assert first_chunk.wait(timeout=5)
        allow_second_chunk.set()

    backend = _backend()
    with (
        mock.patch.object(
            backend,
            "_require_compatible_bundle",
            return_value=(loader, ("balanced",)),
        ),
        mock.patch.object(backend, "_truncate_reference", return_value=reference),
        mock.patch.object(backend, "_apply_seed"),
        mock.patch.object(Zonos3, "_run_synthesis", side_effect=synthesize),
    ):
        stream = backend.generate_stream(
            runtime,
            text="A short sentence.",
            voice="balanced",
            language="en-US",
            max_new_tokens=1024,
        )

        release = threading.Thread(target=release_second_chunk)
        release.start()
        first_audio, sample_rate, first_metadata = next(stream)
        assert not synthesis_finished.is_set()
        allow_finish.set()
        remaining = list(stream)
        release.join(timeout=1)

    assert second_chunk.is_set()
    assert first_audio.dtype == np.float32
    assert first_audio.tolist() == [0.0, 0.5]
    assert sample_rate == 44_100
    assert first_metadata == {
        "backend": "zonos3",
        "chunk_index": 0,
        "is_final": False,
    }
    assert len(remaining) == 1
    final_audio, final_sample_rate, final_metadata = remaining[0]
    assert final_audio.tolist() == [1.0, -1.0]
    assert final_sample_rate == 44_100
    assert final_metadata is not None
    assert final_metadata["is_final"]
    assert synthesis_finished.is_set()
