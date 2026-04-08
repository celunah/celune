# pylint: disable=R0902, R0913, R0914, R0917
"""Celune audio processing functions."""

import math

import numpy as np
from scipy.signal import resample_poly
from pedalboard import Pedalboard, Reverb

from celune.exceptions import AudioMismatchError, BadAudioError

def _resample_audio(
    audio: np.ndarray, source_sr: int, target_sr: int = 48000
) -> np.ndarray:
    """Resample the given audio to the given sample rate."""
    if source_sr <= 0 or target_sr <= 0:
        raise BadAudioError("cannot resample to negative sample rate")

    audio = _make_stereo(audio)

    if source_sr == target_sr:
        return audio

    factor = math.gcd(source_sr, target_sr)
    up = target_sr // factor
    down = source_sr // factor

    return np.ascontiguousarray(
        resample_poly(audio, up=up, down=down, axis=0),
        dtype=np.float32
    )

def _make_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono input to stereo input."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    elif audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] != 2:
            raise AudioMismatchError(f"expected mono or stereo time-first audio, got {audio.shape}")
    else:
        raise AudioMismatchError(f"expected 1D or 2D audio, got {audio.shape}")

    return np.ascontiguousarray(audio, dtype=np.float32)


def _to_48khz(audio: np.ndarray, source_sr: int) -> np.ndarray:
    """Cast an audio chunk to 48 kHz stereo format."""
    return _resample_audio(audio, source_sr, 48000)


def _soften_onset(
    audio: np.ndarray, sr: int, duration: float = 0.2, start_gain: float = 0.5
) -> np.ndarray:
    """Soften the leading audio. This makes any leading breath-like artifacts sound more natural."""
    samples = int(sr * duration)
    samples = min(samples, len(audio))

    ramp = np.linspace(start_gain, 1.0, samples, dtype=np.float32)

    audio[:samples, 0] *= ramp
    audio[:samples, 1] *= ramp

    return audio

class StreamingPedalboardReverb:
    """Stateful reverb based on `pedalboard`."""
    def __init__(self):
        self.strength = 0.0
        self._first_chunk = True

        self.reverb = Reverb(
            room_size=0.5,
            damping=0.75,
            width=0.85,
            wet_level=0.0,
            dry_level=1.0,
        )

        self.board = Pedalboard([self.reverb])

    def _update_params(self):
        """Update reverb strength."""
        s = np.clip(self.strength, 0.0, 1.0)

        wet = 0.16 * (s ** 2)

        self.reverb.wet_level = wet
        self.reverb.dry_level = 1.0

    def process(self, audio: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Apply reverb effect."""
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise AudioMismatchError("expected stereo audio shaped (samples, 2)")

        self._update_params()

        chunk = audio.T.astype(np.float32, copy=False)

        out = self.board.process(
            chunk,
            sample_rate=sr,
            reset=self._first_chunk,
        )

        self._first_chunk = False
        return np.ascontiguousarray(out.T.astype(np.float32, copy=False))

    def flush(self, tail_seconds: float = 1.5, sr: int = 48000) -> np.ndarray:
        """Extract the remaining reverb by pushing silence."""
        silence = np.zeros((int(tail_seconds * sr), 2), dtype=np.float32)
        return self.process(silence, sr)

    def reset(self) -> None:
        """Reset reverb state."""
        self._first_chunk = True
