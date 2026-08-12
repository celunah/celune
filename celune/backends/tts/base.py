# SPDX-License-Identifier: Apache-2.0
"""Unified backend abstractions for Celune."""

import contextlib
import gc
import glob
import hashlib
import os
import random
import secrets
import threading
import unittest.mock
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator, Mapping
from pathlib import Path
from typing import (
    Optional,
    Union,
    cast,
)

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from ...cevoice import CEVoiceLoader, default_loader
from ...constants import N_A_NUMERIC
from ...i18n import string
from ...paths import huggingface_hub_cache_dir, temp_data_dir
from ...typing.aliases import AudioChunk, RuntimeValue
from ...typing.backends import BackendModel
from ...utils import discard

__all__ = [
    "BackendModel",
    "CeluneBackend",
    "cached_hf_snapshot_path",
    "local_hf_offline_mode",
]


_HF_HUB_OFFLINE_LOCK = threading.Lock()
_MAX_REFERENCE_SECONDS = 10.0
_RUNTIME_PRIMITIVE_TYPES = (str, bytes, bytearray, int, float, bool, type(None))


def _to_numpy_audio(chunk: Union[torch.Tensor, np.ndarray]) -> AudioChunk:
    """Convert one streamed tensor or array to normalized one-dimensional audio."""
    if isinstance(chunk, torch.Tensor):
        audio = (
            chunk.detach().float().cpu().numpy()
            if chunk.is_floating_point()
            else chunk.detach().cpu().numpy()
        )
    else:
        audio = np.asarray(chunk)

    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            scale = (limits.max + 1) / 2
            normalized = (audio.astype(np.float32) - scale) / scale
        else:
            scale = max(abs(limits.min), abs(limits.max))
            normalized = audio.astype(np.float32) / scale
    else:
        normalized = np.asarray(audio, dtype=np.float32)
    return np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32)


def _call_runtime_hook_if_present(value: RuntimeValue, name: str) -> bool:
    """Call one release hook only when it already exists on the runtime object."""
    if hasattr(value, "__dict__"):
        with contextlib.suppress(TypeError):
            existing = cast(dict[str, RuntimeValue], value.__dict__).get(name)
            if callable(existing):
                with contextlib.suppress(Exception):
                    existing()
                return True

    if isinstance(value, unittest.mock.NonCallableMock):
        return False

    hook = getattr(value, name, None)
    if callable(hook):
        with contextlib.suppress(Exception):
            hook()
        return True
    return False


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
    model_dir = os.path.join(
        str(huggingface_hub_cache_dir()),
        f"models--{model.replace('/', '--')}",
    )
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


@contextlib.contextmanager
def local_hf_offline_mode(enabled: bool = True) -> Generator[None, None, None]:
    """Temporarily set ``HF_HUB_OFFLINE`` while serializing process-global access.

    Args:
        enabled: Whether to enable Hugging Face offline mode for the guarded block.

    Returns:
        None: Control back to the guarded caller while the environment mutation is active.
    """
    if not enabled:
        yield
        return

    with _HF_HUB_OFFLINE_LOCK:
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            yield
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline


def _release_runtime_container_members(value: RuntimeValue, seen: set[int]) -> None:
    """Recursively release nested runtime members held in common containers."""
    if isinstance(value, dict):
        for nested in list(value.values()):
            _release_runtime_references(nested, seen)
        with contextlib.suppress(Exception):
            value.clear()
        return

    if isinstance(value, list):
        for nested in list(value):
            _release_runtime_references(nested, seen)
        with contextlib.suppress(Exception):
            value.clear()
        return

    if isinstance(value, set):
        for nested in list(value):
            _release_runtime_references(nested, seen)
        with contextlib.suppress(Exception):
            value.clear()
        return

    if isinstance(value, tuple):
        for nested in value:
            _release_runtime_references(nested, seen)


def _release_runtime_object_members(value: RuntimeValue, seen: set[int]) -> None:
    """Recursively release nested runtime members held on one object instance."""
    if not hasattr(value, "__dict__"):
        return

    with contextlib.suppress(TypeError):
        members = list(cast(dict[str, RuntimeValue], value.__dict__).items())
        for attr_name, attr_value in members:
            if attr_value is value:
                continue
            _release_runtime_references(attr_value, seen)
            if attr_name in {"close", "unload"}:
                continue
            with contextlib.suppress(Exception):
                setattr(value, attr_name, None)


def _release_runtime_references(value: RuntimeValue, seen: set[int]) -> None:
    """Recursively release nested references on an about-to-be-discarded runtime object."""
    if isinstance(value, _RUNTIME_PRIMITIVE_TYPES):
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if _call_runtime_hook_if_present(value, "close"):
        return
    if _call_runtime_hook_if_present(value, "unload"):
        return

    if isinstance(value, unittest.mock.NonCallableMock):
        return

    _release_runtime_container_members(value, seen)
    _release_runtime_object_members(value, seen)


class CeluneBackend[ModelT](ABC):
    """Base class for Celune speech backends."""

    name: str = "unknown"
    chunk_rate: float = N_A_NUMERIC
    supported_languages: tuple = ()
    voice_models: Optional[Mapping[str, str]] = None
    default_voice: Optional[str] = None
    uses_voice_bundles: bool = False
    max_new_tokens: int = 512
    is_fake: bool = False

    def __init__(
        self,
        log: Callable[[str, str], None],
        model_name: Optional[str] = None,
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        self.model_name: Optional[str]
        if model_name is not None:
            self.model_name = model_name
        elif self.voice_models and self.default_voice is not None:
            self.model_name = self.voice_models[self.default_voice]
        else:
            self.model_name = None

        self.model: Optional[ModelT] = None
        self.log = log
        self._fatal_callback = fatal
        self.current_seed: Optional[int] = None
        self.random_seed = True
        self._truncated_reference_paths: set[Path] = set()

    def __str__(self) -> str:
        """Return the backend name for callers using str(CeluneBackend(...))."""
        return self.name

    def bind_fatal(self, fatal: Optional[Callable[[], None]]) -> None:
        """Bind the active Celune fatal callback to this backend instance.

        Args:
            fatal: Callback invoked when the backend must transition Celune into a fatal state.
        """
        self._fatal_callback = fatal

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for this backend module."""
        return default_loader()

    def _trigger_fatal_bundle_error(self) -> None:
        """Report one incompatible CEVOICE/CECHAR pack and enter fatal state."""
        message = string("celune.compatible_bundle_required", backend=self.name)
        self.log(message, "error")
        if self._fatal_callback is not None:
            self._fatal_callback()

    def _require_compatible_bundle(
        self,
    ) -> Optional[tuple[CEVoiceLoader, tuple[str, ...]]]:
        """Return the active CEVOICE/CECHAR loader and its usable voice names."""
        loader = self._get_default_loader()
        if loader is None:
            self._trigger_fatal_bundle_error()
            return None

        voice_names = tuple(
            voice
            for voice in loader.bundle.voice_order
            if (
                isinstance(voice, str)
                and voice.strip()
                and voice in loader.bundle.voices
                and isinstance(loader.bundle.voices[voice].get("reference_text"), str)
                and bool(str(loader.bundle.voices[voice]["reference_text"]).strip())
            )
        )
        if not voice_names:
            self._trigger_fatal_bundle_error()
            return None

        return loader, voice_names

    def _truncate_reference(self, reference_wav: Path) -> Path:
        """Return a reference WAV truncated to Celune's backend-safe duration."""
        info = sf.info(reference_wav)
        if info.duration <= _MAX_REFERENCE_SECONDS:
            return reference_wav

        sample_rate = int(info.samplerate)
        frame_limit = int(sample_rate * _MAX_REFERENCE_SECONDS)
        audio, _ = sf.read(reference_wav, frames=frame_limit, dtype="float32")
        temp_dir = temp_data_dir(create=True)
        digest = hashlib.sha1(str(reference_wav.resolve()).encode("utf-8")).hexdigest()[
            :12
        ]
        truncated_path = temp_dir / f"{reference_wav.stem}-{digest}-10s.wav"
        sf.write(truncated_path, audio, sample_rate)
        self._truncated_reference_paths.add(truncated_path)
        return truncated_path

    def _validate_refs(self) -> None:
        """Validate reference audio files found in the current CEVOICE/CECHAR pack."""
        loader = self._get_default_loader()
        if loader is None:
            return
        for name in loader.bundle.voice_order:
            loader.materialize(name, "wav")
            voice_entry = loader.bundle.voices.get(name, {})
            assets = (
                voice_entry.get("assets", {}) if isinstance(voice_entry, dict) else {}
            )
            if isinstance(assets, dict) and "pt" in assets:
                loader.materialize(name, "pt")

    truncate_reference = _truncate_reference
    validate_refs = _validate_refs

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

    @abstractmethod
    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Determine if the given model is available and return its path if found.

        Args:
            model: The model name to check availability of.
            lang: The language identifier for differentiating models by language.

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
            ValueError: The backend cannot resolve model IDs by voice.
            KeyError: The voice name is not defined by this backend.
        """
        if self.voice_models:
            return self.voice_models[voice]

        if self.model_name is not None:
            return self.model_name

        raise ValueError(f"{self.name} cannot resolve a model for voice '{voice}'")

    def resolve_generation_language(self, lang: Optional[str]) -> Optional[str]:
        """Normalize a requested generation language for this backend.

        Args:
            lang: The requested language identifier, if any.

        Returns:
            Optional[str]: The backend-specific normalized language identifier.
        """
        return lang

    def should_reload_for_language(self, lang: Optional[str]) -> bool:
        """Return whether generation should reload the backend model for ``lang``.

        Args:
            lang: The normalized language identifier for the upcoming generation.

        Returns:
            bool: ``True`` when the backend requires a model reload before generation.
        """
        discard(lang)
        return False

    def load_default_model(self) -> ModelT:
        """Load the configured default model for this backend.

        Returns:
            ModelT: The loaded backend model instance.

        Raises:
            ValueError: The backend does not have a configured model to load.
        """
        if self.model_name is None:
            raise ValueError(f"{self.name} does not have a configured model to load")

        self.model = self.load_model(self.model_name)
        return self.model

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Release references held by the backend to its loaded model.

        Args:
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """
        model = self.model
        self.model = None
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            else:
                unload = getattr(model, "unload", None)
                if callable(unload):
                    with contextlib.suppress(Exception):
                        unload()
            seen = {id(model)}
            _release_runtime_container_members(model, seen)
            _release_runtime_object_members(model, seen)

        gc.collect()
        if release_cuda_cache and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

        for truncated_path in self._truncated_reference_paths:
            with contextlib.suppress(OSError):
                truncated_path.unlink(missing_ok=True)
        self._truncated_reference_paths.clear()

    def preload_models(self) -> None:
        """Ensure all required models are available locally."""
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                self.log(f"Downloading {model_id}...", "info")
                snapshot_download(repo_id=model_id)
            else:
                self.log(f"{model_id} is already available.", "info")

    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> ModelT:
        """Load a model by backend-specific identifier.

        Args:
            model_id: The backend-specific model identifier to load.
            kwargs: Backend-specific load options (e.g., VoxCPM2's `load_denoiser` or `optimize`).

        Returns:
            ModelT: The loaded backend model instance.
        """

    @abstractmethod
    def generate_stream(
        self, model: ModelT, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Yield audio chunks from a loaded backend model.

        Args:
            model: The backend model instance to use for generation.
            kwargs: Backend-specific generation parameters.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of audio chunks.
        """
