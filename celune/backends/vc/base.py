# SPDX-License-Identifier: Apache-2.0
"""Unified voice-conversion backend abstractions for Celune."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest

__all__ = ["CeluneVCBackend"]


class CeluneVCBackend(ABC):
    """Base class for Celune voice-conversion backends."""

    name: str = "unknown"
    is_fake: bool = False
    pitch_shift: int
    f0_condition: bool

    def __init__(self, log: Callable[[str, str], None]) -> None:
        self.log = log

    def __str__(self) -> str:
        """Return the backend name for callers using str(CeluneVCBackend(...))."""
        return self.name

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
