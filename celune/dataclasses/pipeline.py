# SPDX-License-Identifier: MIT
"""Speech pipeline dataclasses."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ..constants import N_A_NUMERIC
from ..typing.aliases import AudioChunk


@dataclass(frozen=True)
class SpeechRequest:
    """Queued speech input and output persistence preference."""

    text: str
    display_text: str
    language: str = "Auto"
    save: bool = True
    stream_queue: Optional[queue.Queue[Optional[Union[AudioChunk, Exception]]]] = None  # noqa
    normalize: bool = False
    silent_retry_count: int = 0
    generation: int = 0


@dataclass(frozen=True)
class AudioInputRequest:
    """Engine-level audio input accepted for future non-TTS modes."""

    audio: npt.NDArray[np.float32]
    sample_rate: int
    label: str = "audio input"
    pitch_shift: Optional[int] = None
    f0_condition: Optional[bool] = None
    log_playback: bool = True
    reset_ready_announcement: bool = True


@dataclass(frozen=True)
class VoiceConversionRequest:
    """Audio input plus target voice metadata for voice conversion backends."""

    source_audio: npt.NDArray[np.float32]
    sample_rate: int
    target_voice: Optional[str] = None
    target_character: Optional[str] = None
    target_references: tuple[Path, ...] = ()
    label: str = "audio input"
    pitch_shift: Optional[int] = None
    f0_condition: Optional[bool] = None


@dataclass(frozen=True)
class AudioOutput:
    """Decoded playable audio returned by speech or conversion pipelines."""

    audio: npt.NDArray[np.float32]
    sample_rate: int
    label: str = "audio output"


@dataclass(frozen=True)
class SpeechDone:
    """Playback completion marker for one generated utterance."""

    saved_path: Optional[str] = None
    analysis_audio: Optional[AudioChunk] = None


@dataclass(frozen=True)
class PlaybackChunk:
    """One playback-source chunk routed through the shared DSP mixer."""

    source_id: int
    audio: npt.NDArray[np.float32]
    sample_rate: int
    timing: Optional[SpeechTiming] = None
    generation: int = 0


@dataclass(frozen=True)
class PlaybackSourceDone:
    """Completion marker for one playback source in the shared DSP mixer."""

    source_id: int
    release_pipeline: bool = False
    notify_idle: bool = True
    saved_path: Optional[str] = None
    analysis_audio: Optional[AudioChunk] = None
    generation: int = 0


@dataclass
class SpeechTiming:
    """Timing data for a generated speech utterance."""

    start_time: float
    first_chunk_time: Optional[float] = None
    first_playback_time: Optional[float] = None

    def mark_first_chunk(self, chunk_time: Optional[float] = None) -> None:
        """Record when the backend produces its first audio chunk.

        Args:
            chunk_time: Optional backend-provided monotonic timestamp to use.
        """
        if self.first_chunk_time is None:
            self.first_chunk_time = (
                chunk_time if isinstance(chunk_time, float) else time.monotonic()
            )

    def mark_first_playback(self) -> None:
        """Record when the first audio chunk is sent to the output stream."""
        if self.first_playback_time is None:
            self.first_playback_time = time.monotonic()

    def ttfc_ms(self) -> float:
        """Return time to first generated chunk in milliseconds.

        Returns:
            float: Elapsed milliseconds until the first generated chunk.
        """
        if self.first_chunk_time is None:
            return N_A_NUMERIC
        return (self.first_chunk_time - self.start_time) * 1000

    def ttfp_seconds(self) -> float:
        """Return time to first playback in seconds.

        Returns:
            float: Elapsed seconds until the first audible playback chunk.
        """
        if self.first_playback_time is None:
            return N_A_NUMERIC
        return self.first_playback_time - self.start_time
