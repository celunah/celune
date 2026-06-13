# SPDX-License-Identifier: MIT
"""Celune audio processing functions."""

import math
from typing import Iterable, Callable
from importlib.resources import as_file, files

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly
from pedalboard import Pedalboard, PitchShift, Reverb

from .constants import UtteranceLoudnessTier, BASE_SR
from .exceptions import AudioMismatchError, BadAudioError


_SIGNAL_CACHE: dict[str, npt.NDArray[np.float32]] = {}


def _resample_audio(
    audio: npt.NDArray[np.float32], source_sr: int, target_sr: int = BASE_SR
) -> npt.NDArray[np.float32]:
    """Resample the given audio to the given sample rate."""
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
    """Convert mono input to stereo input."""
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


def _to_48khz(
    audio: npt.NDArray[np.float32], source_sr: int
) -> npt.NDArray[np.float32]:
    """Cast a speech chunk to 48 kHz stereo format."""
    return _resample_audio(audio, source_sr, BASE_SR)


def _pitch_shift_ui_signal(
    audio: npt.NDArray[np.float32], n_steps: float
) -> npt.NDArray[np.float32]:
    """Shift pitch while preserving tempo for short deterministic UI signals."""
    shifted = Pedalboard([PitchShift(semitones=n_steps)])(audio, BASE_SR)
    return np.ascontiguousarray(shifted, dtype=np.float32)


def _freeze_signal(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return one shared read-only buffer for a cached UI signal."""
    frozen = np.ascontiguousarray(audio, dtype=np.float32)
    frozen.setflags(write=False)
    return frozen


def _cached_signal(
    name: str, factory: Callable[[], npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    """Return a cached signal waveform."""
    if name not in _SIGNAL_CACHE:
        _SIGNAL_CACHE[name] = _freeze_signal(factory())

    return _SIGNAL_CACHE[name]


def _load_readiness_signal() -> npt.NDArray[np.float32]:
    """Load Celune's startup readiness sound."""
    readiness_wav = files("celune").joinpath("assets", "chord.wav")

    # we did not find the Celune chord, return silence instead
    if not readiness_wav.is_file():
        return _to_48khz(np.zeros((BASE_SR, 2), dtype=np.float32), BASE_SR)

    with as_file(readiness_wav) as path:
        audio, sr = sf.read(path, dtype="float32")

    return _to_48khz(np.asarray(audio, dtype=np.float32), sr)


def readiness_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's readiness sound.

    Returns:
        npt.NDArray[np.float32]: The readiness sound formatted as a NumPy array, or silent array if not found.
    """

    return _cached_signal("readiness", _load_readiness_signal)


def sleeping_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's sleeping sound.

    Returns:
        npt.NDArray[np.float32]: The sleeping sound formatted as a NumPy array, or a silent array if the readiness
            sound wasn't found.
    """

    return _cached_signal(
        "sleeping",
        lambda: _pitch_shift_ui_signal(
            readiness_signal(),
            n_steps=-1,
        ),
    )


def working_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's working sound.

    Returns:
        npt.NDArray[np.float32]: The working sound formatted as a NumPy array, or a silent array if the readiness
            sound wasn't found.
    """

    return _cached_signal(
        "working",
        lambda: _pitch_shift_ui_signal(
            readiness_signal(),
            n_steps=4,
        ),
    )


def error_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's error sound.

    Returns:
        npt.NDArray[np.float32]: The error sound formatted as a NumPy array, or a silent array if the readiness
            sound wasn't found.
    """

    def factory() -> npt.NDArray[np.float32]:
        base = readiness_signal()
        high = _pitch_shift_ui_signal(base, n_steps=6)
        tritone = base + high

        base_peak = np.max(np.abs(base))
        peak = np.max(np.abs(tritone))
        if peak > 0 and base_peak > 0:
            tritone = tritone * (base_peak / peak)

        return tritone

    return _cached_signal("error", factory)


def _soften(
    audio: npt.NDArray[np.float32],
    sr: int,
    duration: float = 0.2,
    start_gain: float = 0.5,
    end: bool = False,
) -> npt.NDArray[np.float32]:
    """Soften the leading or trailing audio."""
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


def _split(
    audio: npt.NDArray[np.float32], sr: int, chunk_size: float
) -> Iterable[npt.NDArray[np.float32]]:
    """Chop up input audio into chunks."""
    duration = chunk_size * 0.08
    frames = max(1, int(sr * duration))

    for i in range(0, len(audio), frames):
        yield audio[i : i + frames]


def is_silent_utterance(audio: npt.NDArray[np.float32]) -> tuple[bool, int]:
    """Validate if this utterance is silent or not.

    Args:
        audio: NumPy array containing target audio.

    Returns:
        tuple[bool, int]: Whether this utterance is silent and how silent it is.
    """
    rms = np.sqrt(np.mean(np.square(audio)))

    if rms <= 0.001:  # likely only contains surface noise
        return True, UtteranceLoudnessTier.SILENT
    if rms <= 0.01:  # some speech occurred, but it is suspicious
        return True, UtteranceLoudnessTier.SUSPICIOUS

    # Celune spoke normally
    return False, UtteranceLoudnessTier.NORMAL


class StreamingPedalboardReverb:
    """Stateful reverb based on `pedalboard`."""

    def __init__(self) -> None:
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
        """Update reverb strength."""
        s = np.clip(self.strength, 0.0, 1.0)

        wet = 0.16 * (s**2)

        self.reverb.wet_level = wet
        self.reverb.dry_level = 1.0

    def process(
        self, audio: npt.NDArray[np.float32], sr: int = BASE_SR
    ) -> npt.NDArray[np.float32]:
        """Apply reverb effect.

        Args:
            audio: Stereo audio shaped ``(samples, 2)``.
            sr: The sample rate of the input audio.

        Returns:
            npt.NDArray[np.float32]: The processed stereo audio chunk.

        Raises:
            AudioMismatchError: ``audio`` is not stereo audio shaped ``(samples, 2)``.
        """
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise AudioMismatchError(
                f"expected stereo audio shaped (samples, 2), got {audio.shape}"
            )

        self._update_params()

        chunk = audio.transpose().astype(np.float32, copy=False)

        out = self.board.process(
            chunk,
            sample_rate=sr,
            reset=self._first_chunk,
        )

        self._first_chunk = False
        return np.ascontiguousarray(out.transpose().astype(np.float32, copy=False))

    def flush(
        self, sr: int = BASE_SR, threshold: float = 1e-4, max_secs: float = 3.0
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
        """Reset reverb state."""
        self._first_chunk = True
