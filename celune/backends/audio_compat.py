# SPDX-License-Identifier: MIT
"""Small audio compatibility helpers for isolated backend runtimes."""

import sys
from types import ModuleType
from typing import Optional, cast

import numpy as np
import soundfile as sf

from ..audio_resampling import resample_audio as _resample_array


def _resample(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample mono audio with a high-quality polyphase filter."""
    if original_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    return np.asarray(
        _resample_array(audio, original_rate, target_rate),
        dtype=np.float32,
    )


def _load(
    path: str,
    sr: Optional[int] = 22050,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load audio and optionally resample it to the requested rate."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = cast(np.ndarray, np.asarray(audio, dtype=np.float32))
    if mono and audio.ndim > 1:
        audio = cast(np.ndarray, np.mean(audio, axis=1, dtype=np.float32))
    if sr is not None and sample_rate != sr:
        audio = _resample(audio, sample_rate, sr)
        sample_rate = sr
    return audio, sample_rate


def _trim(
    audio: np.ndarray,
    top_db: float = 60.0,
    _frame_length: int = 2048,
    _hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim samples below a peak-relative decibel threshold."""
    if audio.size == 0:
        return audio, np.array([0, 0], dtype=np.int64)

    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio, np.array([0, audio.size], dtype=np.int64)

    threshold = peak * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if active.size == 0:
        return audio, np.array([0, audio.size], dtype=np.int64)

    start = int(active[0])
    end = int(active[-1]) + 1
    return audio[start:end], np.array([start, end], dtype=np.int64)


def _mel(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    htk: bool = False,
    norm: Optional[str] = "slaney",
    dtype: type[np.float32] = np.float32,
) -> np.ndarray:
    """Build the Slaney mel filterbank used by the affected backends."""
    if fmax is None:
        fmax = sr / 2.0

    if htk:
        min_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
        max_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
        mel_frequencies = 700.0 * (
            10.0
            ** (np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64) / 2595.0)
            - 1.0
        )
    else:
        frequency_step = 200.0 / 3.0
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / frequency_step
        log_step = np.log(6.4) / 27.0
        min_mel = fmin / frequency_step
        max_mel = (
            min_log_mel + np.log(fmax / min_log_hz) / log_step
            if fmax >= min_log_hz
            else fmax / frequency_step
        )
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2, dtype=np.float64)
        mel_frequencies = np.where(
            mel_points >= min_log_mel,
            min_log_hz * np.exp(log_step * (mel_points - 3.0)),
            frequency_step * mel_points,
        )

    fft_frequencies = np.linspace(
        0.0,
        sr / 2.0,
        1 + n_fft // 2,
        dtype=np.float64,
    )
    ramps = np.subtract.outer(mel_frequencies, fft_frequencies)
    frequency_differences = np.diff(mel_frequencies)
    lower = -ramps[:-2] / frequency_differences[:-1, None]
    upper = ramps[2:] / frequency_differences[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        weights *= (2.0 / (mel_frequencies[2 : n_mels + 2] - mel_frequencies[:n_mels]))[
            :, None
        ]
    return np.asarray(weights, dtype=dtype)


def install_librosa_compat() -> None:
    """Register the librosa API used by affected backend packages."""
    if "librosa" in sys.modules:
        return

    effects = ModuleType("librosa.effects")
    effects.trim = _trim  # type: ignore[attr-defined]
    filters = ModuleType("librosa.filters")
    filters.mel = _mel  # type: ignore[attr-defined]
    module = ModuleType("librosa")
    module.effects = effects  # type: ignore[attr-defined]
    module.filters = filters  # type: ignore[attr-defined]
    module.load = _load  # type: ignore[attr-defined]
    module.resample = _resample  # type: ignore[attr-defined]
    sys.modules["librosa.effects"] = effects
    sys.modules["librosa.filters"] = filters
    sys.modules["librosa"] = module
