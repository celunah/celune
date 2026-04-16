"""Unified backend abstractions for Celune."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class CeluneBackend(ABC):
    """Base class for Celune speech backends."""

    name = "unknown"
    voice_models: Optional[dict[str, str]] = None
    reference_wavs: Optional[dict[str, str]] = None
    default_voice: Optional[str] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        if model_name is not None:
            self.model_name = model_name
        elif self.voice_models and self.default_voice is not None:
            self.model_name = self.voice_models[self.default_voice]
        else:
            self.model_name = None

        self.model = None

    @property
    def default_model_id(self) -> str:
        """Return the default model identifier for this backend."""
        if self.voice_models and self.default_voice is not None:
            return self.voice_models[self.default_voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} does not define a default model")

    @property
    def all_model_ids(self) -> list[str]:
        """Return every known model identifier for this backend."""
        if self.voice_models:
            return list(dict.fromkeys(self.voice_models.values()))

        if self.model_name is not None:
            return [self.model_name]

        return []

    @property
    def voices(self) -> list[str]:
        """Return the available voice names for this backend."""
        if self.voice_models:
            return list(self.voice_models)
        return []

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice name to a backend-specific model identifier."""
        if self.voice_models:
            return self.voice_models[voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} cannot resolve a model for voice '{voice}'")

    def load_default_model(self, log: Callable[[str, str], None]):
        """Load the configured default model for this backend."""
        if self.model_name is None:
            raise ValueError(f"{self.name} does not have a configured model to load")
        self.model = self.load_model(self.model_name, log)
        return self.model

    def unload_model(self) -> None:
        """Release references held by the backend to its loaded model."""
        self.model = None

    @abstractmethod
    def preload_models(self, log: Callable[[str, str], None]) -> None:
        """Ensure all required models are available locally."""

    @abstractmethod
    def load_model(self, model_id: str, log: Callable[[str, str], None]):
        """Load a model by backend-specific identifier."""

    @abstractmethod
    def generate_stream(self, model, **kwargs):
        """Yield audio chunks from a loaded backend model."""
