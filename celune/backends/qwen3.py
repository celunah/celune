# SPDX-License-Identifier: MIT
"""Qwen3 backend implementation for Celune."""

from __future__ import annotations

import os
import contextlib
from collections.abc import Iterator
from typing import Callable, Optional, Final, Mapping

import numpy as np
import numpy.typing as npt
from faster_qwen3_tts import FasterQwen3TTS, __version__ as qwen3_ver

from ..cevoice import default_loader
from .base import CeluneBackend, cached_hf_snapshot_path, BackendModel


class Qwen3(CeluneBackend):
    """Celune Qwen3-TTS backend."""

    name: Final[str] = "qwen3"

    uses_voice_bundles: bool = True
    chunk_rate: Final[float] = 12.5
    max_new_tokens: Final[int] = 2048

    # setting this parameter will lock in identity, but expression may be reduced
    x_vector_only: bool = True
    supported_languages: Final[tuple[str, ...]] = (
        "zh-cn",
        "en",
        "ja",
        "ko",
        "de",
        "fr",
        "ru",
        "pt",
        "es",
        "it",
    )
    clone_model: Final[str] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    reference_texts: Final[Mapping[str, str]] = {
        "balanced": (
            "My name is Celune, pronounced Celune. It is a pleasure to meet you."
        ),
        "calm": "My name is... Celune... It is so... quiet.",
        "bold": "My name is Celune! Let's do this, we have to get it done!",
        "upbeat": (
            "Hehehe... Hi, I'm Celune. Look, I have something to tell... "
            "might as well make it fun. Shall we?"
        ),
    }
    default_voice: Final[str] = "balanced"

    def __init__(
        self,
        log: Callable[[str, str], None],
        x_vector_only: bool = False,
        clone_model_id: Optional[str] = None,
    ) -> None:
        super().__init__(log=log)
        self.x_vector_only = x_vector_only
        self.model_name = clone_model_id or self.clone_model
        self._validate_refs()
        self.clone_model_id = clone_model_id or self.clone_model

    @property
    def default_model_id(self) -> str:
        """Return the model loaded by default for Qwen3 cloning.

        Returns:
            str: The default Qwen3 model identifier.
        """
        return self.clone_model_id

    @property
    def all_model_ids(self) -> list[str]:
        """Return every model required by Qwen3 cloning.

        Returns:
            list[str]: The model identifiers needed by the backend.
        """
        return [self.clone_model_id]

    @property
    def voices(self) -> list[str]:
        """Return the built-in Qwen3 voice names.

        Returns:
            list[str]: Voice names supported by Celune's Qwen3 references.
        """
        return list(self.reference_texts)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a Celune voice to the shared Qwen3 clone model.

        Args:
            voice: The Celune voice name to resolve.

        Returns:
            str: The model identifier for the requested voice.

        Raises:
            ValueError: The requested voice is unknown.
        """
        loader = default_loader()
        if loader is not None:
            return self.clone_model_id
        if voice not in self.reference_texts:
            raise ValueError(f"{self.name} cannot resolve a model for voice '{voice}'")
        return self.clone_model_id

    def generation_progress_total(self, text: Optional[str] = None) -> int:
        """Return the Qwen3 streaming generation context length.

        Args:
            text: The text to check context usage of with this value.

        Returns:
            int: The max context length.
        """
        return self.max_new_tokens

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.

        Returns:
            tuple[bool, Optional[str]]: A cache availability flag and the resolved snapshot path when present.
        """
        return cached_hf_snapshot_path(
            model,
            [
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "tokenizer_config.json",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> Optional[BackendModel]:
        """Load the given voice model.

        Args:
            model_id: The Qwen3 model repository ID to load.
            kwargs: Additional keyword arguments to use.

        Returns:
            FasterQwen3TTS: The loaded Qwen3 TTS model instance.
        """
        available, path = self.model_is_available_locally(model_id)

        if available and path is not None:
            previous_offline = os.environ.get("HF_HUB_OFFLINE")
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.model = FasterQwen3TTS.from_pretrained(path)
            finally:
                if previous_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = previous_offline
            return self.model

        self.log("Downloading TTS model...", "info")
        self.model = FasterQwen3TTS.from_pretrained(model_id)
        return self.model

    def generate_stream(
        self, model: FasterQwen3TTS, **kwargs
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded Qwen3 model instance.
            kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of Qwen3 streaming audio chunks.

        Raises:
            ValueError: The requested voice is unsupported, or input text is empty.
        """
        if not kwargs.get("text", None):
            raise ValueError("expected text to say")

        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        self._apply_seed()

        # if faster_qwen3_tts >= 0.2.5 use instructions, else remove this arg
        if tuple(int(num) for num in qwen3_ver.split(".")[:3]) < (0, 2, 5):
            kwargs.pop("instruct", None)

        voice = kwargs.pop("voice", self.default_voice)

        try:
            loader = default_loader()
            if loader is not None:
                ref_wav = loader.materialize(voice, "wav")
                configured_ref_text = loader.bundle.voices[voice].get("reference_text")
            else:
                ref_wav = self._reference_wave_path(voice)
                configured_ref_text = None
            ref_text = (
                configured_ref_text
                if isinstance(configured_ref_text, str)
                else self.reference_texts.get(
                    voice, self.reference_texts[self.default_voice]
                )
            )
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        stream = None
        try:
            stream = model.generate_voice_clone_streaming(
                ref_audio=ref_wav,
                ref_text=ref_text,
                non_streaming_mode=False,  # VERY IMPORTANT ON >=0.2.5
                xvec_only=self.x_vector_only,
                **kwargs,
            )

            for chunk in stream:  # pylint: disable=R1737
                yield chunk
        finally:
            if stream is not None and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()
