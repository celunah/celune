"""Speech pipeline dataclasses."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ..constants import N_A_NUMERIC


@dataclass(frozen=True)
class SpeechRequest:
    """Queued speech input and output persistence preference."""

    text: str
    display_text: str
    language: str = "Auto"
    save: bool = True
    stream_queue: Optional[
        "queue.Queue[Optional[Union[npt.NDArray[np.float32], Exception]]]"
    ] = None
    normalize: bool = False


@dataclass(frozen=True)
class SpeechDone:
    """Playback completion marker for one generated utterance."""

    saved_path: Optional[str] = None
    analysis_audio: Optional[npt.NDArray[np.float32]] = None


@dataclass(frozen=True)
class PlaybackChunk:
    """One playback-source chunk routed through the shared DSP mixer."""

    source_id: int
    audio: npt.NDArray[np.float32]
    sample_rate: int
    timing: Optional["SpeechTiming"] = None


@dataclass(frozen=True)
class PlaybackSourceDone:
    """Completion marker for one playback source in the shared DSP mixer."""

    source_id: int
    release_pipeline: bool = False
    saved_path: Optional[str] = None
    analysis_audio: Optional[npt.NDArray[np.float32]] = None


@dataclass
class SpeechTiming:
    """Timing data for a generated speech utterance."""

    start_time: float
    first_chunk_time: Optional[float] = None
    first_playback_time: Optional[float] = None

    def mark_first_chunk(self) -> None:
        """Record when the backend yields its first audio chunk."""
        if self.first_chunk_time is None:
            self.first_chunk_time = time.monotonic()

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
