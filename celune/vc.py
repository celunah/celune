# SPDX-License-Identifier: MIT
"""Shared runtime helpers for Celune voice conversion."""

import importlib
from collections.abc import Mapping
from typing import Optional, cast

import numpy as np
import torch
from scipy import signal

from .config import config_bool
from .typing.aliases import AudioChunk
from .typing.backends import _StreamingSpeechModel
from .typing.common import JSONSerializable

VC_PITCH_SHIFT_MIN = -3
VC_PITCH_SHIFT_MAX = 3
VC_VAD_RMS_THRESHOLD = 0.005
VC_VAD_HANGOVER_SECONDS = 0.3
VC_VAD_PREROLL_SECONDS = 0.18
VC_LIVE_CHUNK_SECONDS = 0.18
VC_LIVE_CHUNK_OVERLAP_SECONDS = 0.0

_LIVE_VAD_TARGET_SAMPLE_RATE = 16000
_LIVE_VAD_FRAME_SAMPLES = 512
_LIVE_VAD_THRESHOLD = 0.50
_LIVE_VAD_NEGATIVE_THRESHOLD = 0.35

__all__ = [
    "VC_LIVE_CHUNK_OVERLAP_SECONDS",
    "VC_LIVE_CHUNK_SECONDS",
    "VC_PITCH_SHIFT_MAX",
    "VC_PITCH_SHIFT_MIN",
    "VC_VAD_HANGOVER_SECONDS",
    "VC_VAD_PREROLL_SECONDS",
    "VC_VAD_RMS_THRESHOLD",
    "LiveVoiceActivityDetector",
    "clamp_vc_pitch_shift",
    "create_live_voice_activity_detector",
    "vc_input_has_voice",
    "vc_input_rms",
    "vc_live_chunk_frames",
    "vc_live_chunk_overlap_frames",
    "vc_vad_hangover_frames",
    "vc_vad_preroll_frames",
]


def clamp_vc_pitch_shift(value: int) -> int:
    """Clamp one VC pitch shift to Celune's supported semitone range.

    Args:
        value: Requested semitone offset for voice conversion.

    Returns:
        int: The requested offset clamped to Celune's supported VC range.
    """
    return max(VC_PITCH_SHIFT_MIN, min(VC_PITCH_SHIFT_MAX, value))


def vc_input_rms(audio: AudioChunk) -> float:
    """Return RMS energy for one microphone callback buffer.

    Args:
        audio: One live microphone callback buffer.

    Returns:
        float: The RMS energy of the provided buffer.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def vc_vad_hangover_frames(sample_rate: int) -> int:
    """Return tolerated trailing silent frames before one VC flush.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many silent frames to retain before ending one speech chunk.
    """
    return max(1, int(sample_rate * VC_VAD_HANGOVER_SECONDS))


def vc_vad_preroll_frames(sample_rate: int) -> int:
    """Return retained pre-speech frames before one VC onset.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many pre-speech frames to retain for chunk onset recovery.
    """
    return max(1, int(sample_rate * VC_VAD_PREROLL_SECONDS))


def vc_input_has_voice(audio: AudioChunk) -> bool:
    """Return whether one live callback buffer likely contains voice.

    Args:
        audio: One live microphone callback buffer.

    Returns:
        bool: Whether the buffer meets the fallback RMS speech threshold.
    """
    return vc_input_rms(audio) >= VC_VAD_RMS_THRESHOLD


def vc_live_chunk_frames(sample_rate: int) -> int:
    """Return the active-speech chunk size for low-latency live VC.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many frames to accumulate before flushing one mid-speech chunk.
    """
    return max(1, int(sample_rate * VC_LIVE_CHUNK_SECONDS))


def vc_live_chunk_overlap_frames(sample_rate: int) -> int:
    """Return the retained tail overlap between adjacent live VC chunks.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: Zero when no overlap is configured because Seed-VC's native live path
            owns crossfade alignment; otherwise the configured overlap in frames.
    """
    if VC_LIVE_CHUNK_OVERLAP_SECONDS <= 0:
        return 0
    return max(1, int(sample_rate * VC_LIVE_CHUNK_OVERLAP_SECONDS))


def _normalize_live_audio(audio: AudioChunk) -> AudioChunk:
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
    audio: AudioChunk,
    source_sample_rate: int,
    target_sample_rate: int,
) -> AudioChunk:
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


def _torch_probability(output: torch.Tensor) -> float:
    """Convert one model output tensor into a scalar probability."""
    values = output.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
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
        audio: AudioChunk,
        sample_rate: int,
    ) -> bool:
        """Return whether one live microphone callback likely contains speech.

        Args:
            audio: Live microphone audio to inspect.
            sample_rate: Sample rate of ``audio``.

        Returns:
            bool: Whether speech appears active in the current callback window.
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
        config: Optional Celune configuration mapping.

    Returns:
        Optional[LiveVoiceActivityDetector]: The live VC detector when enabled and available.
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
    except ImportError:
        return None

    try:
        load_silero_vad = getattr(silero_vad, "load_silero_vad", None)
        if not callable(load_silero_vad):
            return None
        return LiveVoiceActivityDetector(cast(_StreamingSpeechModel, load_silero_vad()))
    except (RuntimeError, AssertionError, ValueError, ImportError):
        return None
