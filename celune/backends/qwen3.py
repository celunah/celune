"""Qwen3 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Callable, Generator, Literal, Optional

import numpy as np
import numpy.typing as npt
from faster_qwen3_tts import FasterQwen3TTS
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from .base import CeluneBackend


class Qwen3(CeluneBackend):
    """Celune Qwen3-TTS backend."""

    name: str = "qwen3"
    clone_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    supported_modes: tuple[str, ...] = ("native", "clone")
    voice_models: dict[str, str] = {
        "balanced": "lunahr/Celune-1.7B-Neutral",
        "calm": "lunahr/Celune-1.7B-Calm",
        "bold": "lunahr/Celune-1.7B-Energetic",
        "upbeat": "lunahr/Celune-1.7B-Upbeat",
    }
    reference_wavs: dict[str, str] = {
        "balanced": "refs/balanced.wav",
        "calm": "refs/calm.wav",
        "bold": "refs/bold.wav",
        "upbeat": "refs/upbeat.wav",
    }
    reference_texts: dict[str, str] = {
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
    default_voice: str = "balanced"

    def __init__(
        self,
        log: Callable[[str, str], None],
        mode: Literal["native", "clone"] = "native",
    ) -> None:
        if mode not in self.supported_modes:
            raise ValueError(
                f"unsupported qwen3 mode '{mode}' "
                f"(available: {', '.join(self.supported_modes)})"
            )

        super().__init__(log=log)
        self.mode = mode
        if self.mode == "clone":
            self.model_name = self.clone_model

    @property
    def default_model_id(self) -> str:
        """Return the model loaded by default for the active Qwen3 mode."""
        if self.mode == "clone":
            return self.clone_model
        return super().default_model_id

    @property
    def all_model_ids(self) -> list[str]:
        """Return every model required by the active Qwen3 mode."""
        if self.mode == "clone":
            return [self.clone_model]
        return super().all_model_ids

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a Celune voice to the model required by the active Qwen3 mode."""
        if self.mode == "clone":
            if voice not in self.voice_models:
                raise ValueError(
                    f"{self.name} cannot resolve a model for voice '{voice}'"
                )
            return self.clone_model

        return super().model_id_for_voice(voice)

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.

        Returns:
            tuple[bool, Optional[str]]: A flag indicating cache availability and
            the resolved snapshot path when present.
        """
        base = HF_HUB_CACHE
        model_dir = os.path.join(base, f"models--{model.replace('/', '--')}")

        refs_main = os.path.join(model_dir, "refs", "main")
        snapshots_dir = os.path.join(model_dir, "snapshots")

        expected_files = [
            "config.json",
            "generation_config.json",
            "model*.safetensors",
            "tokenizer_config.json",
        ]

        if not os.path.exists(refs_main):
            return False, None

        with open(refs_main, encoding="utf-8") as f:
            commit = f.read().strip()

        snapshot_path = os.path.join(snapshots_dir, commit)

        if not os.path.isdir(snapshot_path):
            return False, None

        if all(
            glob.glob(os.path.join(snapshot_path, pattern))
            for pattern in expected_files
        ):
            return True, snapshot_path

        return False, None

    def preload_models(self) -> None:
        """Ensure all known Qwen3 voice models are cached locally.

        Returns:
            None: This method downloads any missing voice models.
        """
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                self.log(f"Downloading {model_id}...", "info")
                snapshot_download(repo_id=model_id)
            else:
                self.log(f"{model_id} is already available.", "info")

    def load_model(self, model_id: str, load_denoiser: bool = True) -> FasterQwen3TTS:
        """Load the given voice model.

        Args:
            model_id: The Qwen3 model repository ID to load.
            load_denoiser: Unused.

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
    ) -> Generator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded Qwen3 model instance.
            **kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterable[Any]: An iterator of Qwen3 streaming audio chunks.
        """
        if self.mode == "native":
            kwargs.pop("voice", None)  # Celune has native voices in this backend
            # Celune natively works with Qwen-formatted chunks
            yield from model.generate_custom_voice_streaming(speaker="celune", **kwargs)
        elif self.mode == "clone":
            voice = kwargs.pop("voice", self.default_voice)

            try:
                ref_wav = (
                    Path(__file__).resolve().parents[1] / self.reference_wavs[voice]
                )
                ref_text = self.reference_texts[voice]
            except KeyError as e:
                raise ValueError(
                    f"unknown voice '{voice}' for backend '{self.name}'"
                ) from e

            yield from model.generate_voice_clone_streaming(
                ref_audio=ref_wav,
                ref_text=ref_text,
                xvec_only=False,
                **kwargs,
            )
        else:
            raise ValueError(f"unsupported qwen3 mode '{self.mode}'")
