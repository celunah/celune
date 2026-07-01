# SPDX-License-Identifier: MIT
"""Optional AI helpers for live voice-conversion input."""

import importlib
from typing import Optional, Protocol, cast
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy import signal

from .config import config_bool
from .typing.common import JSONSerializable

_LIVE_VAD_TARGET_SAMPLE_RATE = 16000
_LIVE_VAD_FRAME_SAMPLES = 512
_LIVE_VAD_THRESHOLD = 0.50
_LIVE_VAD_NEGATIVE_THRESHOLD = 0.35


class _StreamingSpeechModel(Protocol):
    """Protocol for stateful streaming speech detectors."""

    def __call__(self, audio: object, sample_rate: int) -> object:
        """Return one speech probability tensor-like object."""

    def reset_states(self) -> None:
        """Reset the detector's internal streaming state."""


def _normalize_live_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return one mono float32 waveform for live input helpers."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim != 2:
        raise ValueError(f"expected 1D or 2D live audio, got {normalized.shape}")
    if normalized.shape[1] == 1:
        return normalized[:, 0]
    return np.asarray(
        np.mean(normalized, axis=1, dtype=np.float32),
        dtype=np.float32,
    )


def _resample_audio(
    audio: npt.NDArray[np.float32],
    source_sample_rate: int,
    target_sample_rate: int,
) -> npt.NDArray[np.float32]:
    """Resample one mono waveform when the sample rate differs."""
    if source_sample_rate == target_sample_rate:
        return np.asarray(audio, dtype=np.float32)

    gcd = int(np.gcd(source_sample_rate, target_sample_rate))
    up = target_sample_rate // gcd
    down = source_sample_rate // gcd
    return np.asarray(
        signal.resample_poly(audio, up, down),
        dtype=np.float32,
    )


def _torch_probability(output: object) -> float:
    """Convert one model output tensor-like object into a scalar probability."""
    detached = getattr(output, "detach", None)
    if callable(detached):
        output = detached()

    cpu = getattr(output, "cpu", None)
    if callable(cpu):
        output = cpu()

    if hasattr(output, "numpy"):
        output = output.numpy()

    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        return 0.0
    return float(values[-1])


class LiveVoiceActivityDetector:
    """Stateful Silero-based speech detector for live VC capture."""

    def __init__(
        self,
        model: _StreamingSpeechModel,
        *,
        threshold: float = _LIVE_VAD_THRESHOLD,
        negative_threshold: float = _LIVE_VAD_NEGATIVE_THRESHOLD,
        target_sample_rate: int = _LIVE_VAD_TARGET_SAMPLE_RATE,
        frame_samples: int = _LIVE_VAD_FRAME_SAMPLES,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self.target_sample_rate = target_sample_rate
        self.frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_active = False

        reset_states = getattr(self.model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def reset(self) -> None:
        """Reset buffered audio and the streaming detector state."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_active = False
        reset_states = getattr(self.model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def has_voice(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> bool:
        """Return whether one live microphone callback likely contains speech.

        Args:
            audio: Value for `audio`.
            sample_rate: Value for `sample_rate`.

        Returns:
            Result of this function.
        """
        mono_audio = _normalize_live_audio(audio)
        resampled = _resample_audio(
            mono_audio,
            sample_rate,
            self.target_sample_rate,
        )
        if self._pending.size > 0:
            resampled = np.concatenate((self._pending, resampled))

        complete_frames = (len(resampled) // self.frame_samples) * self.frame_samples
        if complete_frames <= 0:
            self._pending = resampled
            return self._speech_active

        had_speech = self._speech_active
        torch = __import__("torch")
        for start in range(0, complete_frames, self.frame_samples):
            frame = np.asarray(
                resampled[start : start + self.frame_samples],
                dtype=np.float32,
            )
            probability = _torch_probability(
                self.model(
                    torch.from_numpy(frame),
                    self.target_sample_rate,
                )
            )
            if self._speech_active:
                if probability < self.negative_threshold:
                    self._speech_active = False
            elif probability >= self.threshold:
                self._speech_active = True

        self._pending = np.asarray(resampled[complete_frames:], dtype=np.float32)
        return had_speech or self._speech_active


def create_live_voice_activity_detector(
    config: Optional[Mapping[str, JSONSerializable]],
) -> Optional[LiveVoiceActivityDetector]:
    """Return the optional AI VAD used by live VC capture.

    Args:
        config: Value for `config`.

    Returns:
        Result of this function.
    """
    if not config_bool(
        config,
        "CELUNE_VC_LIVE_AI_VAD",
        "voice_conversion_live_ai_vad",
        True,
    ):
        return None

    try:
        silero_vad = importlib.import_module("silero_vad")
    except Exception:
        return None

    try:
        load_silero_vad = getattr(silero_vad, "load_silero_vad", None)
        if not callable(load_silero_vad):
            return None
        return LiveVoiceActivityDetector(cast(_StreamingSpeechModel, load_silero_vad()))
    except (RuntimeError, AssertionError, ValueError, ImportError):
        return None
