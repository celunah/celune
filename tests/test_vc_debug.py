# SPDX-License-Identifier: Apache-2.0
"""Tests for temporary live voice-conversion diagnostic captures."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from celune.vc_debug import VCDebugCapture, VCDebugSaveResult


def test_vc_debug_capture_exports_original_levels_as_48k_pcm24_flac(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Keep raw input and converted levels in separate diagnostic files."""
    monkeypatch.setattr("celune.vc_debug.app_data_dir", lambda **_kwargs: tmp_path)
    capture = VCDebugCapture()
    microphone_time = np.arange(2400, dtype=np.float32) / 24000
    converted_time = np.arange(2205, dtype=np.float32) / 22050
    microphone = np.asarray(
        np.stack(
            (
                0.2 * np.sin(2 * np.pi * 440 * microphone_time),
                0.2 * np.sin(2 * np.pi * 440 * microphone_time),
            ),
            axis=1,
        ),
        dtype=np.float32,
    )
    converted = np.asarray(
        0.1 * np.sin(2 * np.pi * 440 * converted_time),
        dtype=np.float32,
    )
    results: list[VCDebugSaveResult] = []
    completed = threading.Event()

    capture.record_microphone(microphone, 24000)
    token = capture.begin_conversion()
    capture.record_conversion(token, converted, 22050)

    def on_complete(result: VCDebugSaveResult) -> None:
        results.append(result)
        completed.set()

    capture.stop(on_complete)

    assert completed.wait(5)
    result = results[0]
    assert result.directory.parent == tmp_path
    assert result.microphone_path is not None
    assert result.converted_path is not None

    microphone_info = sf.info(result.microphone_path)
    converted_info = sf.info(result.converted_path)
    assert (microphone_info.samplerate, microphone_info.channels) == (48000, 2)
    assert (converted_info.samplerate, converted_info.channels) == (48000, 1)
    assert microphone_info.subtype == "PCM_24"
    assert converted_info.subtype == "PCM_24"

    saved_microphone, _ = sf.read(
        result.microphone_path,
        dtype="float32",
        always_2d=True,
    )
    saved_converted, _ = sf.read(
        result.converted_path,
        dtype="float32",
        always_2d=True,
    )
    microphone_rms = np.sqrt(np.mean(saved_microphone[100:-100] ** 2))
    converted_rms = np.sqrt(np.mean(saved_converted[100:-100] ** 2))
    assert np.isclose(microphone_rms, 0.2 / np.sqrt(2), atol=0.001)
    assert np.isclose(converted_rms, 0.1 / np.sqrt(2), atol=0.001)
    assert result.errors == ()


def test_vc_debug_stop_waits_for_inflight_converted_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Include the last conversion result when capture is stopped mid-call."""
    monkeypatch.setattr("celune.vc_debug.app_data_dir", lambda **_kwargs: tmp_path)
    capture = VCDebugCapture()
    results: list[VCDebugSaveResult] = []
    completed = threading.Event()
    token = capture.begin_conversion()

    def on_complete(result: VCDebugSaveResult) -> None:
        results.append(result)
        completed.set()

    capture.stop(on_complete)
    assert not completed.is_set()

    capture.record_microphone(np.zeros(100, dtype=np.float32), 48000)
    capture.record_conversion(token, np.full(100, 0.25, dtype=np.float32), 48000)

    assert completed.wait(5)
    assert results[0].converted_path is not None
    assert results[0].errors == ()


def test_vc_debug_capture_without_audio_removes_empty_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Avoid leaving an empty temporary directory when no mic data arrived."""
    monkeypatch.setattr("celune.vc_debug.app_data_dir", lambda **_kwargs: tmp_path)
    capture = VCDebugCapture()
    result: list[VCDebugSaveResult] = []
    completed = threading.Event()

    def on_complete(saved: VCDebugSaveResult) -> None:
        result.append(saved)
        completed.set()

    capture.stop(on_complete)

    assert completed.wait(5)
    assert not result[0].directory.exists()
    assert result[0].microphone_path is None
    assert result[0].converted_path is None


def test_vc_debug_shutdown_can_discard_a_stalled_conversion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Let application shutdown finish even if conversion never returns."""
    monkeypatch.setattr("celune.vc_debug.app_data_dir", lambda **_kwargs: tmp_path)
    capture = VCDebugCapture()
    results: list[VCDebugSaveResult] = []
    completed = threading.Event()
    capture.begin_conversion()

    def on_complete(result: VCDebugSaveResult) -> None:
        results.append(result)
        completed.set()

    capture.stop(on_complete)
    capture.stop(discard_pending=True)

    assert completed.wait(5)
    assert results[0].converted_path is None
    assert results[0].errors == ()
