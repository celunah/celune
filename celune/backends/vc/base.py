# SPDX-License-Identifier: MIT
"""Unified voice-conversion backend abstractions for Celune."""

from abc import ABC, abstractmethod
from typing import Callable

from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest

__all__ = ["CeluneVCBackend"]


class CeluneVCBackend(ABC):
    """Base class for Celune voice-conversion backends."""

    name: str = "unknown"
    is_fake: bool = False

    def __init__(self, log: Callable[[str, str], None]) -> None:
        self.log = log

    def __str__(self) -> str:
        """Return the backend name for callers using str(CeluneVCBackend(...))."""
        return self.name

    def preload_models(self) -> None:
        """Ensure any optional backend assets are ready before conversion."""
        return None

    def unload_model(self) -> None:
        """Release optional backend runtime state."""
        return None

    @abstractmethod
    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert one source performance into the target voice.

        Args:
            request: The voice-conversion request to process.
        """
