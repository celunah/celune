"""Unified backend abstractions for Celune."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Generator

import numpy as np
import numpy.typing as npt
from transformers import PreTrainedModel


class CeluneBackend(ABC):
    """Base class for Celune speech backends."""

    name: str = "unknown"
    voice_models: Optional[dict[str, str]] = None
    reference_wavs: Optional[dict[str, str]] = None
    default_voice: Optional[str] = None

    def __init__(
        self, log: Callable[[str, str], None], model_name: Optional[str] = None
    ) -> None:
        if model_name is not None:
            self.model_name = model_name
        elif self.voice_models and self.default_voice is not None:
            self.model_name = self.voice_models[self.default_voice]
        else:
            self.model_name = None

        self.model = None
        self.log = log

    @staticmethod
    @abstractmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Determine if the given model is available and return its path if found.

        Args:
            model: The model name to check availability of.
        Returns:
            tuple[bool, Optional[str]]: Whether the given model is available and relevant path.
        """

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

    def load_default_model(self) -> PreTrainedModel:
        """Load the configured default model for this backend.

        Returns:
            Any: The loaded backend model instance.
        """
        if self.model_name is None:
            raise ValueError(f"{self.name} does not have a configured model to load")
        self.model = self.load_model(self.model_name)
        return self.model

    def unload_model(self) -> None:
        """Release references held by the backend to its loaded model.

        Returns:
            None: This method clears the backend's cached model reference.
        """
        self.model = None

    @abstractmethod
    def preload_models(self) -> None:
        """Ensure all required models are available locally.

        Returns:
            None: Implementations prepare model assets for later loading.
        """

    @abstractmethod
    def load_model(self, model_id: str, optimize: bool = True) -> PreTrainedModel:
        """Load a model by backend-specific identifier.

        Args:
            model_id: The backend-specific model identifier to load.
            optimize: (VoxCPM2 only) Whether to attempt optimizing VoxCPM2.

        Returns:
            Any: The loaded backend model instance.
        """

    @abstractmethod
    def generate_stream(
        self, model: PreTrainedModel, **kwargs
    ) -> Generator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Yield audio chunks from a loaded backend model.

        Args:
            model: The backend model instance to use for generation.
            **kwargs: Backend-specific generation parameters.

        Returns:
            Iterable[Any]: An iterator of backend-specific audio chunk payloads.
        """
