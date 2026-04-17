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
        """Return the default model identifier for this backend.

        Returns:
            str: The backend-specific model identifier used by default.
        """
        if self.voice_models and self.default_voice is not None:
            return self.voice_models[self.default_voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} does not define a default model")

    @property
    def all_model_ids(self) -> list[str]:
        """Return every known model identifier for this backend.

        Returns:
            list[str]: The unique model identifiers exposed by the backend.
        """
        if self.voice_models:
            return list(dict.fromkeys(self.voice_models.values()))

        if self.model_name is not None:
            return [self.model_name]

        return []

    @property
    def voices(self) -> list[str]:
        """Return the available voice names for this backend.

        Returns:
            list[str]: The selectable voice names supported by the backend.
        """
        if self.voice_models:
            return list(self.voice_models)
        return []

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice name to a backend-specific model identifier.

        Args:
            voice: The voice name to resolve.

        Returns:
            str: The model identifier associated with the requested voice.
        """
        if self.voice_models:
            return self.voice_models[voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} cannot resolve a model for voice '{voice}'")

    def load_default_model(self, log: Callable[[str, str], None]):
        """Load the configured default model for this backend.

        Args:
            log: Logging callback used to report load progress and status.

        Returns:
            Any: The loaded backend model instance.
        """
        if self.model_name is None:
            raise ValueError(f"{self.name} does not have a configured model to load")
        self.model = self.load_model(self.model_name, log)
        return self.model

    def unload_model(self) -> None:
        """Release references held by the backend to its loaded model.

        Returns:
            None: This method clears the backend's cached model reference.
        """
        self.model = None

    @abstractmethod
    def preload_models(self, log: Callable[[str, str], None]) -> None:
        """Ensure all required models are available locally.

        Args:
            log: Logging callback used to report download or cache activity.

        Returns:
            None: Implementations prepare model assets for later loading.
        """

    @abstractmethod
    def load_model(self, model_id: str, log: Callable[[str, str], None]):
        """Load a model by backend-specific identifier.

        Args:
            model_id: The backend-specific model identifier to load.
            log: Logging callback used to report load progress and status.

        Returns:
            Any: The loaded backend model instance.
        """

    @abstractmethod
    def generate_stream(self, model, **kwargs):
        """Yield audio chunks from a loaded backend model.

        Args:
            model: The backend model instance to use for generation.
            **kwargs: Backend-specific generation parameters.

        Returns:
            Iterable[Any]: An iterator of backend-specific audio chunk payloads.
        """
