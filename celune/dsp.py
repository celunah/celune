# pylint: disable=R0902, R0913, R0914, R0917
"""Celune audio processing functions."""

import math
from typing import Iterable

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly
from pedalboard import Pedalboard, Reverb

from celune.exceptions import AudioMismatchError, BadAudioError


def _resample_audio(
    audio: npt.NDArray[np.float32], source_sr: int, target_sr: int = 48000
) -> npt.NDArray[np.float32]:
    """Resample the given audio to the given sample rate.

    Args:
        audio: The input audio array.
        source_sr: The sample rate of the input audio.
        target_sr: The desired output sample rate.

    Returns:
        npt.NDArray[np.float32]: The resampled stereo audio array.
    """
    if source_sr == 0:
        raise BadAudioError("cannot resample from zero sample rate")
    if target_sr == 0:
        raise BadAudioError("cannot resample to zero sample rate")
    if source_sr < 0:
        raise BadAudioError("cannot resample from negative sample rate")
    if target_sr < 0:
        raise BadAudioError("cannot resample to negative sample rate")

    audio = _make_stereo(audio)

    if source_sr == target_sr:
        return audio

    factor = math.gcd(source_sr, target_sr)
    up = target_sr // factor
    down = source_sr // factor

    return np.ascontiguousarray(
        resample_poly(audio, up=up, down=down, axis=0), dtype=np.float32
    )


def _make_stereo(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert mono input to stereo input.

    Args:
        audio: The input mono or stereo audio array.

    Returns:
        npt.NDArray[np.float32]: A contiguous stereo audio array.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    elif audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] != 2:
            raise AudioMismatchError(
                f"expected mono or stereo time-first audio, got {audio.shape}"
            )
    else:
        raise AudioMismatchError(f"expected 1D or 2D audio, got {audio.shape}")

    return np.ascontiguousarray(audio, dtype=np.float32)


def _to_48khz(audio: npt.NDArray[np.float32], source_sr: int) -> npt.NDArray[np.float32]:
    """Cast a speech chunk to 48 kHz stereo format.

    Args:
        audio: The input audio array.
        source_sr: The input sample rate.

    Returns:
        npt.NDArray[np.float32]: The audio resampled to 48 kHz stereo.
    """
    return _resample_audio(audio, source_sr, 48000)


def _soften(
    audio: npt.NDArray[np.float32], sr: int, duration: float = 0.2, start_gain: float = 0.5, end: bool = False
) -> npt.NDArray[np.float32]:
    """Soften the leading or trailing audio.

    Args:
        audio: The stereo audio array to modify in place.
        sr: The sample rate of the audio.
        duration: The fade-in duration in seconds.
        start_gain: The gain applied at the first sample before ramping to full
            volume.

    Returns:
        npt.NDArray[np.float32]: The softened audio array.
    """
    samples = int(sr * duration)
    samples = min(samples, len(audio))

    ramp = np.linspace(start_gain, 1.0, samples, dtype=np.float32)

    if not end:
        audio[:samples, 0] *= ramp
        audio[:samples, 1] *= ramp
    else:
        audio[-samples:, 0] *= ramp
        audio[-samples:, 1] *= ramp

    return audio


def _split(audio: npt.NDArray[np.float32], sr: int, chunk_size: float) -> Iterable[npt.NDArray[np.float32]]:
    """Chop up input audio into chunks.

    Args:
        audio: The stereo audio array to split.
        sr: The sample rate of the audio.
        chunk_size: Celune's chunk size multiplier used to derive frame counts.

    Returns:
        Iterable[npt.NDArray[np.float32]]: An iterator of smaller audio chunks.
    """
    duration = chunk_size * 0.08
    frames = max(1, int(sr * duration))

    for i in range(0, len(audio), frames):
        yield audio[i : i + frames]


class StreamingPedalboardReverb:
    """Stateful reverb based on `pedalboard`."""

    def __init__(self):
        self.strength = 0.0
        self._first_chunk = True

        # default Celune reverb, with strength control
        self.reverb = Reverb(
            room_size=0.5,
            damping=0.75,
            width=0.85,
            wet_level=0.0,
            dry_level=1.0,
        )

        self.board = Pedalboard([self.reverb])

    def _update_params(self):
        """Update reverb strength.

        Returns:
            None: This method applies the current strength to the pedalboard
                parameters.
        """
        s = np.clip(self.strength, 0.0, 1.0)

        wet = 0.16 * (s**2)

        self.reverb.wet_level = wet
        self.reverb.dry_level = 1.0

    def process(self, audio: npt.NDArray[np.float32], sr: int = 48000) -> npt.NDArray[np.float32]:
        """Apply reverb effect.

        Args:
            audio: Stereo audio shaped ``(samples, 2)``.
            sr: The sample rate of the input audio.

        Returns:
            npt.NDArray[np.float32]: The processed stereo audio chunk.
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise AudioMismatchError(f"expected stereo audio shaped (samples, 2), got {audio.shape}")

        self._update_params()

        chunk = audio.T.astype(np.float32, copy=False)

        out = self.board.process(
            chunk,
            sample_rate=sr,
            reset=self._first_chunk,
        )

        self._first_chunk = False
        return np.ascontiguousarray(out.T.astype(np.float32, copy=False))

    def flush(
        self, sr: int = 48000, threshold: float = 1e-4, max_secs: float = 3.0
    ) -> npt.NDArray[np.float32]:
        """Extract the remaining reverb by pushing silence.

        Args:
            sr: The sample rate used for the generated silence.
            threshold: The RMS threshold below which the tail is considered done.
            max_secs: The maximum amount of tail audio to extract.

        Returns:
            npt.NDArray[np.float32]: The remaining reverb tail as stereo audio.
        """
        chunk_size = int(0.1 * sr)
        max_chunks = int(max_secs / 0.1)

        outputs = []

        silence = np.zeros((chunk_size, 2), dtype=np.float32)

        for _ in range(max_chunks):
            out = self.process(silence, sr)

            rms = np.sqrt(np.mean(out**2))

            if rms < threshold:
                break

            outputs.append(out)

        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.zeros((0, 2), dtype=np.float32)

    def reset(self) -> None:
        """Reset reverb state.

        Returns:
            None: This method marks the next processed chunk as a fresh stream.
        """
        self._first_chunk = True
