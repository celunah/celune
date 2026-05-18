# SPDX-License-Identifier: MIT
"""Qwen3 backend implementation for Celune."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from typing import Callable, Literal, Optional, Final, Mapping

import numpy as np
import numpy.typing as npt

# imported with a name because __version__ is reserved to Celune
# it's not in __all__, but that's not Celune's job, so we have to ignore the warning
from faster_qwen3_tts import FasterQwen3TTS, __version__ as qwen3_ver

from .base import CeluneBackend, cached_hf_snapshot_path
from ..cevoice import default_loader


class Qwen3(CeluneBackend):
    """Celune Qwen3-TTS backend."""

    name: Final[str] = "qwen3"

    # this is a default value, Celune sets this properly during backend initialization
    uses_voice_bundles: bool = False
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
    supported_modes: Final[tuple[str, ...]] = (
        "native",  # deprecated operation mode, will be removed soon
        "clone",  # now uses CEVOICE voice packs
    )

    # these models are deprecated as of Celune 3.5.0
    voice_models: Final[Mapping[str, str]] = {
        "balanced": "lunahr/Celune-1.7B-Neutral",
        "calm": "lunahr/Celune-1.7B-Calm",
        "bold": "lunahr/Celune-1.7B-Energetic",
        "upbeat": "lunahr/Celune-1.7B-Upbeat",
    }

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
        mode: Literal["native", "clone"] = "clone",
        x_vector_only: bool = False,
    ) -> None:
        if mode not in self.supported_modes:
            raise ValueError(
                f"unsupported qwen3 mode '{mode}' "
                f"(available: {', '.join(self.supported_modes)})"
            )

        super().__init__(log=log)
        self.mode = mode
        self.x_vector_only = x_vector_only
        self.uses_voice_bundles = self.mode == "clone"
        if self.mode == "native":
            warnings.warn(
                "Qwen3 native mode is deprecated and will be removed soon. "
                "Please load a CEVOICE pack for optimal operation.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.mode == "clone":
            self.model_name = self.clone_model
            self._validate_refs()

    @property
    def default_model_id(self) -> str:
        """Return the model loaded by default for the active Qwen3 mode.

        Returns:
            str: The default Qwen3 model identifier.
        """
        if self.mode == "clone":
            return self.clone_model
        return super().default_model_id

    @property
    def all_model_ids(self) -> list[str]:
        """Return every model required by the active Qwen3 mode.

        Returns:
            list[str]: The model identifiers needed by the selected mode.
        """
        if self.mode == "clone":
            return [self.clone_model]
        return super().all_model_ids

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a Celune voice to the model required by the active Qwen3 mode.

        Args:
            voice: The Celune voice name to resolve.

        Returns:
            str: The model identifier for the requested voice.

        Raises:
            ValueError: Clone mode cannot resolve the requested voice.
        """
        if self.mode == "clone":
            if voice not in self.voice_models:
                raise ValueError(
                    f"{self.name} cannot resolve a model for voice '{voice}'"
                )
            return self.clone_model

        return super().model_id_for_voice(voice)

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
            tuple[bool, Optional[str]]: A flag indicating cache availability and
            the resolved snapshot path when present.
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

    def load_model(self, model_id: str, **kwargs) -> FasterQwen3TTS:
        """Load the given voice model.

        Args:
            model_id: The Qwen3 model repository ID to load.
            **kwargs: Additional keyword arguments to use.

        Returns:
            FasterQwen3TTS: The loaded Qwen3 TTS model instance.
        """
        available, path = self.model_is_available_locally(model_id)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = FasterQwen3TTS.from_pretrained(path)
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
            **kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of Qwen3 streaming audio chunks.

        Raises:
            ValueError: The current Qwen3 mode and/or requested voice is unsupported, or input text is empty.
        """
        if not kwargs.get("text", None):
            raise ValueError("expected text to say")

        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        self._apply_seed()

        # if faster_qwen3_tts >= 0.2.5 use instructions, else remove this arg
        major, minor, patch = (int(num) for num in qwen3_ver.split("."))
        if not (major >= 0 and minor >= 2 and patch >= 5):
            kwargs.pop("instruct", None)

        if self.mode == "native":
            # we are not using the voice param here, as the model defines only one
            # and you have to reload the model to apply voice settings
            kwargs.pop("voice", None)
            # Celune natively works with Qwen-formatted chunks
            yield from model.generate_custom_voice_streaming(speaker="celune", **kwargs)
        elif self.mode == "clone":
            # we are using the voice param here as it tells Celune which reference to use
            voice = kwargs.pop("voice", self.default_voice)

            try:
                loader = default_loader()
                if loader is not None:
                    ref_wav = loader.materialize(voice, "wav")
                    configured_ref_text = loader.bundle.voices[voice].get(
                        "reference_text"
                    )
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

            yield from model.generate_voice_clone_streaming(
                ref_audio=ref_wav,
                ref_text=ref_text,
                non_streaming_mode=False,  # VERY IMPORTANT ON >=0.2.5
                xvec_only=self.x_vector_only,
                **kwargs,
            )
        else:
            raise ValueError(f"unsupported qwen3 mode '{self.mode}'")
