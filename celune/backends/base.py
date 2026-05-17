# SPDX-License-Identifier: MIT
"""Unified backend abstractions for Celune."""

from __future__ import annotations

import os
import glob
import random
import secrets
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Callable, Optional

import torch
import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from ..constants import N_A_NUMERIC
from ..utils import discard
from ..exceptions import BackendError
from ..cevoice import default_loader


def cached_hf_snapshot_path(
    model: str, expected_files: list[str]
) -> tuple[bool, Optional[str]]:
    """Return whether a cached Hugging Face cache path for a model is available and usable.

    Args:
        model: The model ID to return a cache path for.
        expected_files: The files that are expected to already exist in the cache path, if found.

    Returns:
        tuple[bool, Optional[str]]: Whether there is a usable cache path for the model, and its location.
    """
    model_dir = os.path.join(HF_HUB_CACHE, f"models--{model.replace('/', '--')}")
    refs_main = os.path.join(model_dir, "refs", "main")
    snapshot_dir = os.path.join(model_dir, "snapshots")

    if not os.path.exists(refs_main):
        return False, None

    with open(refs_main, encoding="utf-8") as f:
        commit = f.read().strip()

    snapshot_path = os.path.join(snapshot_dir, commit)
    if not os.path.isdir(snapshot_path):
        return False, None

    if all(
        glob.glob(os.path.join(snapshot_path, pattern)) for pattern in expected_files
    ):
        return True, snapshot_path

    return False, None


class CeluneBackend(ABC):
    """Base class for Celune speech backends."""

    name: str = "unknown"
    chunk_rate: float = N_A_NUMERIC
    supported_languages: tuple = ()
    voice_models: Optional[dict[str, str]] = None
    default_voice: Optional[str] = None
    uses_voice_bundles: bool = False

    def __init__(
        self, log: Callable[[str, str], None], model_name: Optional[str] = None
    ) -> None:
        self.model_name: Optional[str]
        if model_name is not None:
            self.model_name = model_name
        elif self.voice_models and self.default_voice is not None:
            self.model_name = self.voice_models[self.default_voice]
        else:
            self.model_name = None

        self.model: Optional[Any] = None
        self.log = log
        self.current_seed: Optional[int] = None
        self.random_seed = True

    @staticmethod
    def _reference_wave_path(name: str) -> Path:
        loader = default_loader()
        if loader is not None:
            return loader.materialize(name, "wav")
        return Path(__file__).resolve().parents[1] / "refs" / f"{name}.wav"

    def _validate_refs(self) -> None:
        loader = default_loader()
        if loader is not None:
            for name in loader.bundle.voice_order:
                loader.materialize(name, "wav")
            return

        if not self.voice_models:
            return

        for name in self.voice_models:
            full_path = self._reference_wave_path(name)
            try:
                with open(full_path, "rb") as f:
                    checksum = hashlib.file_digest(f, "sha256").hexdigest()
            except FileNotFoundError as e:
                raise BackendError(f"reference audio for '{name}' not found") from e
            except PermissionError as e:
                raise BackendError(f"cannot access reference audio for '{name}'") from e

            checksum_path = f"{os.path.splitext(full_path)[0]}.sha256"
            if os.path.exists(checksum_path):
                with open(checksum_path, "r", encoding="utf-8") as f:
                    expected = f.read().strip()

                if checksum != expected:
                    self.log(
                        f"Checksum mismatch for '{name}', output may be affected.",
                        "warning",
                    )
            else:
                self.log(
                    f"Checksum not found for '{name}', skipping checksum verification.",
                    "warning",
                )

    def _apply_seed(self) -> None:
        """Seed all generation RNGs for the next backend operation."""
        if self.random_seed:
            self.current_seed = secrets.randbits(32)

        if self.current_seed is None:
            return

        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        torch.cuda.manual_seed_all(self.current_seed)
        torch.manual_seed(self.current_seed)

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

        Raises:
            ValueError: No default model can be resolved for this backend.
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

        Raises:
            KeyError: The voice name is not defined by this backend.
            ValueError: The backend cannot resolve model IDs by voice.
        """
        if self.voice_models:
            return self.voice_models[voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} cannot resolve a model for voice '{voice}'")

    def generation_progress_total(self, text: Optional[str] = None) -> Optional[int]:
        """Return the backend's maximum streaming generation steps, if known.

        Args:
            text: Optional text for backends whose generation budget depends on
                input token length.

        Returns:
            Optional[int]: Maximum generated codec/token steps for one text chunk,
                or ``None`` when the backend does not expose a stable limit.
        """
        discard(text)

    @staticmethod
    def generation_progress_steps(timing: Optional[dict]) -> int:
        """Return how many generation steps a streamed chunk represents.

        Args:
            timing: Optional backend timing metadata yielded with the audio chunk.

        Returns:
            int: Number of generated codec/token steps represented by the chunk.
        """
        if not timing:
            return 1

        steps = timing.get("chunk_steps")
        if isinstance(steps, int) and steps > 0:
            return steps

        return 1

    def load_default_model(self) -> Any:
        """Load the configured default model for this backend.

        Returns:
            Any: The loaded backend model instance.

        Raises:
            ValueError: The backend does not have a configured model to load.
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

    def preload_models(self) -> None:
        """Ensure all required models are available locally.

        Returns:
            None: Implementations prepare model assets for later loading.
        """
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                self.log(f"Downloading {model_id}...", "info")
                snapshot_download(repo_id=model_id)
            else:
                self.log(f"{model_id} is already available.", "info")

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> Any:
        """Load a model by backend-specific identifier.

        Args:
            model_id: The backend-specific model identifier to load.
            **kwargs: Backend-specific load options (e.g., VoxCPM2's `load_denoiser` or `optimize`).

        Returns:
            Any: The loaded backend model instance.
        """

    @abstractmethod
    def generate_stream(
        self, model: Any, **kwargs
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Yield audio chunks from a loaded backend model.

        Args:
            model: The backend model instance to use for generation.
            **kwargs: Backend-specific generation parameters.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of
                Celune compatible audio chunks.
        """
