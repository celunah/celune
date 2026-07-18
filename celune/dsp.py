# SPDX-License-Identifier: MIT
"""Celune audio processing functions."""

import math
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, resample_poly, sosfilt
from pedalboard import Pedalboard, PitchShift, Reverb

from .constants import UtteranceLoudnessTier, BASE_SR
from .exceptions import AudioMismatchError, BadAudioError


_SIGNAL_CACHE: dict[str, npt.NDArray[np.float32]] = {}
_READINESS_FREQUENCIES = (261.63, 329.63, 369.99, 440.0, 493.88, 739.99)


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


def pitch_shift_audio(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    n_steps: float,
) -> npt.NDArray[np.float32]:
    """Shift audio pitch while preserving tempo at the given sample rate.

    Args:
        audio: Input waveform to shift, provided as a float32 NumPy array.
        sample_rate: Sample rate associated with ``audio``.
        n_steps: Number of semitones to shift the signal up or down.

    Returns:
        npt.NDArray[np.float32]: A contiguous float32 waveform with pitch shifted and tempo preserved.
    """
    if n_steps == 0:
        return np.ascontiguousarray(audio, dtype=np.float32)

    shifted = Pedalboard([PitchShift(semitones=n_steps)])(
        np.asarray(audio, dtype=np.float32),
        sample_rate,
    )
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


def _transpose_frequencies(
    frequencies: Iterable[float], semitones: float
) -> tuple[float, ...]:
    """Transpose frequencies by an equal-tempered semitone interval."""
    multiplier = 2 ** (semitones / 12)
    return tuple(frequency * multiplier for frequency in frequencies)


def pad_note(
    frequencies: Iterable[float],
    duration: float = 3.0,
    sample_rate: int = BASE_SR,
    target_rms_dbfs: float = -36.0,
    attack_seconds: float = 3.0,
    release_seconds: float = 3.0,
    detune_cents: float = 1.0,
    leading_silence_seconds: float = 1.0,
    trailing_silence_seconds: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Generate a softly blended, normalized stereo chord pad.

    Args:
        frequencies: Frequencies in hertz for the simultaneously played notes.
        duration: Duration of the audible pad in seconds, excluding silence.
        sample_rate: Output sample rate in hertz.
        target_rms_dbfs: RMS level for the audible pad in dBFS.
        attack_seconds: Fade-in duration in seconds.
        release_seconds: Fade-out duration in seconds.
        detune_cents: Maximum detuning of the quiet unison voices in cents.
        leading_silence_seconds: Silence added before the audible pad.
        trailing_silence_seconds: Silence added after the audible pad.

    Returns:
        npt.NDArray[np.float32]: Stereo float32 audio with shape ``(samples, 2)``.

    Raises:
        BadAudioError: If the frequencies, duration, or sample rate are invalid.
    """
    note_frequencies = tuple(frequencies)

    if not note_frequencies or any(frequency <= 0 for frequency in note_frequencies):
        raise BadAudioError("all notes must be positive frequency")
    if duration <= 0:
        raise BadAudioError("duration must be positive")
    if sample_rate <= 0:
        raise BadAudioError("sample rate must be positive")
    if (
        min(
            attack_seconds,
            release_seconds,
            leading_silence_seconds,
            trailing_silence_seconds,
        )
        < 0
    ):
        raise BadAudioError("timing values must be positive")

    sample_count = max(1, int(duration * sample_rate))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    signal = np.zeros(sample_count, dtype=np.float64)

    for frequency in note_frequencies:
        for cents, gain in (
            (-detune_cents, 0.1),
            (0.0, 0.6),
            (detune_cents, 0.1),
        ):
            detuned_frequency = frequency * 2 ** (cents / 1200)
            signal += gain * np.sin(2 * np.pi * detuned_frequency * time)

    signal /= len(note_frequencies)

    attack_samples = min(int(attack_seconds * sample_rate), sample_count)
    release_samples = min(int(release_seconds * sample_rate), sample_count)
    envelope = np.ones(sample_count, dtype=np.float64)

    if attack_samples:
        envelope[:attack_samples] *= (
            np.sin(np.linspace(0, np.pi / 2, attack_samples)) ** 2
        )
    if release_samples:
        envelope[-release_samples:] *= (
            np.cos(np.linspace(0, np.pi / 2, release_samples)) ** 2
        )

    signal *= envelope

    cutoff = 206.0
    sos = butter(
        3,
        cutoff,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = np.asarray(sosfilt(sos, signal), dtype=np.float64)

    stereo = np.column_stack((filtered * 0.96, filtered * 1.04))
    target_rms = 10 ** (target_rms_dbfs / 20)
    rms = np.sqrt(np.mean(np.square(stereo), dtype=np.float64))

    if rms > 0:
        stereo *= target_rms / rms

    peak = np.max(np.abs(stereo))
    if peak > 0.95:
        stereo *= 0.95 / peak

    leading_silence = np.zeros(
        (int(leading_silence_seconds * sample_rate), 2), dtype=np.float64
    )
    trailing_silence = np.zeros(
        (int(trailing_silence_seconds * sample_rate), 2), dtype=np.float64
    )

    return np.ascontiguousarray(
        np.vstack((leading_silence, stereo, trailing_silence)),
        dtype=np.float32,
    )


def _load_readiness_signal() -> npt.NDArray[np.float32]:
    """Generate Celune's startup readiness sound."""
    return pad_note(_READINESS_FREQUENCIES)


def readiness_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's readiness sound.

    Returns:
        npt.NDArray[np.float32]: The readiness sound formatted as a NumPy array, or silent array if not found.
    """

    return _cached_signal("readiness", _load_readiness_signal)


def sleeping_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's sleeping sound.

    Returns:
        npt.NDArray[np.float32]: The sleeping sound formatted as a NumPy array, or a silent array if the readiness sound
        wasn't found.
    """

    return _cached_signal(
        "sleeping",
        lambda: pad_note(
            _transpose_frequencies(_READINESS_FREQUENCIES, -1),
        ),
    )


def working_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's working sound.

    Returns:
        npt.NDArray[np.float32]: The working sound formatted as a NumPy array, or a silent array if the readiness sound
        wasn't found.
    """

    return _cached_signal(
        "working",
        lambda: pad_note(
            _transpose_frequencies(_READINESS_FREQUENCIES, 4),
        ),
    )


def error_signal() -> npt.NDArray[np.float32]:
    """Dynamically generate Celune's error sound.

    Returns:
        npt.NDArray[np.float32]: The error sound formatted as a NumPy array, or a silent array if the readiness sound
        wasn't found.
    """

    def factory() -> npt.NDArray[np.float32]:
        stacked_frequencies = _READINESS_FREQUENCIES + _transpose_frequencies(
            _READINESS_FREQUENCIES,
            6,
        )
        return pad_note(stacked_frequencies)

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


resample_audio = _resample_audio
make_stereo = _make_stereo
to_48khz = _to_48khz
soften = _soften
split = _split


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
        dry = 1.0 - wet

        self.reverb.wet_level = wet
        self.reverb.dry_level = dry

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

    update_params = _update_params
