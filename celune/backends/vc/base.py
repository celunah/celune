# SPDX-License-Identifier: Apache-2.0
"""Unified voice-conversion backend abstractions for Celune."""

from abc import ABC, abstractmethod
from typing import Optional
from collections.abc import Callable

from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from ...typing.aliases import LogLevel

__all__ = ["CeluneVCBackend"]


class CeluneVCBackend(ABC):
    """Base class for Celune voice-conversion backends."""

    name: str = "unknown"
    is_fake: bool = False
    pitch_shift: int
    f0_condition: bool
    log_level: LogLevel = "info"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        self.log = log
        self._progress_callback: Optional[
            Callable[[Optional[float], Optional[float]], None]
        ] = None

    def __str__(self) -> str:
        """Return the backend name for callers using str(CeluneVCBackend(...))."""
        return self.name

    def bind_progress(
        self,
        progress: Optional[Callable[[Optional[float], Optional[float]], None]],
    ) -> None:
        """Bind the active Celune progress callback to this backend instance.

        Args:
            progress: Callback receiving current progress and an optional total.
        """
        self._progress_callback = progress

    def report_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Forward backend-owned loading or download progress to Celune.

        Args:
            progress: Current progress, or ``None`` for an indeterminate update.
            total: Total progress, or ``None`` when the total is unavailable.
        """
        callback = getattr(self, "_progress_callback", None)
        if callback is not None:
            callback(progress, total)

    def preload_models(self) -> None:
        """Ensure any optional backend assets are ready before conversion."""

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Release optional backend runtime state.

        Args:
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """

    @abstractmethod
    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert one source performance into the target voice.

        Args:
            request: The voice-conversion request to process.
        """
