# SPDX-License-Identifier: Apache-2.0
"""Tests for ZONOS3-Mini backend selection and progressive stream adaptation."""

import base64
import threading
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
from collections.abc import Callable

import pytest
import torch
import numpy as np

from celune.typing.backends import BackendArgumentValue
from celune.registry.tts import TTS_MODELS, ZONOS3_TTS_MODEL_ID, ZONOS3_TTS_REVISION
from celune.backends.tts.zonos3 import (
    Zonos3,
    Zonos3Runtime,
    _load_codec_decoder_with_fallback,
    _resolve_codec_device,
    _snapshot_asset_path,
)


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


def test_zonos3_defaults_codec_to_the_model_device_and_respects_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer GPU codec decoding while preserving explicit device selection."""
    model_device = torch.device("cuda:0")
    monkeypatch.delenv("ZONOS3_AUDIO_CODEC_DEVICE", raising=False)

    assert _resolve_codec_device(model_device) == (model_device, True)

    monkeypatch.setenv("ZONOS3_AUDIO_CODEC_DEVICE", "cpu")

    assert _resolve_codec_device(model_device) == (torch.device("cpu"), False)


def test_zonos3_retries_automatic_cuda_decoder_load_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover from decoder VRAM pressure without failing the TTS worker."""
    attempted_devices: list[torch.device] = []
    log = mock.Mock()

    def load_decoder(device: torch.device) -> torch.nn.Module:
        """Simulate CUDA allocation failure and successful CPU fallback."""
        attempted_devices.append(device)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("codec did not fit")
        return torch.nn.Identity()

    monkeypatch.setattr(torch.cuda, "empty_cache", mock.Mock())
    decoder_model, decoder_device = _load_codec_decoder_with_fallback(
        load_decoder,
        torch.device("cuda:0"),
        automatic=True,
        log=log,
    )

    assert attempted_devices == [torch.device("cuda:0"), torch.device("cpu")]
    assert isinstance(decoder_model, torch.nn.Identity)
    assert decoder_device == torch.device("cpu")
    log.assert_called_once()


def test_zonos3_keeps_snapshot_file_paths_unresolved_for_hub_blob_links(
    tmp_path: Path,
) -> None:
    """Preserve lexical snapshot paths because Hugging Face files link to blobs."""
    snapshot = tmp_path / "snapshots" / "revision"
    asset_uri = f"hf://{ZONOS3_TTS_MODEL_ID}/autoencoder/generator.pt"

    assert _snapshot_asset_path(snapshot, asset_uri) == (
        snapshot.resolve() / "autoencoder" / "generator.pt"
    )


def test_zonos3_rejects_release_asset_uri_path_traversal(tmp_path: Path) -> None:
    """Reject upstream release URIs that escape the pinned snapshot."""
    asset_uri = f"hf://{ZONOS3_TTS_MODEL_ID}/../outside/model.pt"

    with pytest.raises(ValueError, match="escapes the pinned snapshot"):
        _snapshot_asset_path(tmp_path, asset_uri)


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


def test_zonos3_stream_close_waits_for_synthesis_worker(tmp_path: Path) -> None:
    """Do not finish stream cancellation while synthesis still uses the model."""
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference wav")
    loader = SimpleNamespace(materialize=mock.Mock(return_value=reference))
    runtime = Zonos3Runtime(None, None, tmp_path)
    producer_waiting = threading.Event()
    allow_producer_finish = threading.Event()
    producer_finished = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    close_errors: list[Exception] = []

    def synthesize(
        _runtime: Zonos3Runtime,
        _request: dict[str, BackendArgumentValue],
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        """Emit enough audio to yield once, then wait for cancellation cleanup."""
        on_audio(np.array([0.0, 0.5], dtype=np.float32))
        on_audio(np.array([0.25, 0.75], dtype=np.float32))
        producer_waiting.set()
        try:
            assert allow_producer_finish.wait(timeout=10)
        finally:
            producer_finished.set()

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
        )
        first_audio, sample_rate, _metadata = next(stream)
        close = getattr(stream, "close", None)

        assert first_audio.tolist() == [0.0, 0.5]
        assert sample_rate == 44_100
        assert producer_waiting.wait(timeout=1)
        assert callable(close)

        def close_stream() -> None:
            """Close the consumer stream and capture cleanup failures."""
            close_started.set()
            try:
                close()
            except Exception as error:
                close_errors.append(error)
            finally:
                close_finished.set()

        closer = threading.Thread(target=close_stream)
        closer.start()
        try:
            assert close_started.wait(timeout=1)
            assert not close_finished.wait(timeout=2.2)
        finally:
            allow_producer_finish.set()
            closer.join(timeout=5)

    assert not closer.is_alive()
    assert not close_errors
    assert producer_finished.is_set()
    assert close_finished.is_set()
