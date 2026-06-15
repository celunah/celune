# SPDX-License-Identifier: MIT
"""VoxCPM2 backend implementation for Celune."""

import os
import contextlib
from collections.abc import Iterator
from typing import Callable, Optional, Mapping, Generator

import torch
import numpy as np
import numpy.typing as npt
from voxcpm import VoxCPM

from . import get_version
from ..constants import BASE_SR
from ..utils import custom_assert
from ..exceptions import BackendError
from ..cevoice import default_loader, CEVoiceLoader
from .base import CeluneBackend, cached_hf_snapshot_path, local_hf_offline_mode


class VoxCPM2(CeluneBackend[VoxCPM]):
    """Celune VoxCPM2 backend."""

    name: str = "voxcpm2"
    uses_voice_bundles: bool = True
    chunk_rate: float = 6.25
    max_new_tokens: int = 512
    supported_languages: tuple[str, ...] = (
        "ar",
        "my",
        "zh-cn",
        "da",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "he",
        "hi",
        "id",
        "it",
        "ja",
        "km",
        "ko",
        "lo",
        "ms",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "sw",
        "sv",
        "tl",
        "th",
        "tr",
        "vi",
    )

    voice_models: Optional[Mapping[str, str]] = {
        "balanced": "openbmb/VoxCPM2",
        "calm": "openbmb/VoxCPM2",
        "bold": "openbmb/VoxCPM2",
        "upbeat": "openbmb/VoxCPM2",
    }

    # fallback values for packs that omit cfg_scale
    voice_cfg: Mapping[str, float] = {
        "balanced": 2.4,
        "calm": 3.0,
        "bold": 2.4,
        "upbeat": 2.4,
    }
    default_voice: Optional[str] = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self.log = log
        self.optimize_enabled = False
        self._validate_refs()

    @staticmethod
    def _require_compatible_bundle() -> tuple[CEVoiceLoader, tuple[str, ...]]:
        """Return the active CEVOICE/CECHAR loader and its usable voice names."""
        loader = default_loader()
        custom_assert(
            loader is not None,
            BackendError(
                "backend 'voxcpm2' requires a compatible CEVOICE/CECHAR package "
                "with at least one valid voice identifier"
            ),
        )
        assert loader is not None

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
        custom_assert(
            bool(voice_names),
            BackendError(
                "backend 'voxcpm2' requires a compatible CEVOICE/CECHAR package "
                "with at least one valid voice identifier"
            ),
        )
        assert bool(voice_names)

        return loader, voice_names

    def _validate_refs(self) -> None:
        """Validate VoxCPM2 reference audio files from the active CEVOICE/CECHAR pack."""
        loader, voice_names = self._require_compatible_bundle()
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack.

        Returns:
            list[str]: The list of available voices to use from current CEVOICE/CECHAR pack.
        """
        _, voice_names = self._require_compatible_bundle()
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice from the active pack to the shared VoxCPM2 model.

        Args:
            voice: The voice name to resolve.

        Returns:
            str: A resolved model name for this voice.
        """
        _, voice_names = self._require_compatible_bundle()
        custom_assert(
            voice in voice_names,
            ValueError(f"{self.name} cannot resolve a model for voice '{voice}'"),
        )
        assert voice in voice_names

        return self.default_model_id

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress unnecessary backend output."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    yield

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.
            lang: The language identifier for differentiating models by language.

        Returns:
            tuple[bool, Optional[str]]: A flag indicating cache availability and the resolved snapshot path when
            present.
        """
        return cached_hf_snapshot_path(
            model,
            [
                "config.json",
                "model*.safetensors",
                "tokenizer_config.json",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> VoxCPM:
        """Load the given voice model.

        Args:
            model_id: The VoxCPM2 model repository ID to load.
            kwargs: Additional keyword arguments to use while loading VoxCPM2.

        Returns:
            VoxCPM: The loaded VoxCPM2 model instance.
        """
        available, path = self.model_is_available_locally(model_id)

        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        if available and path is not None:
            with local_hf_offline_mode():
                with self._suppress_backend_output():
                    self.model = VoxCPM.from_pretrained(
                        path,
                        load_denoiser=kwargs.get("load_denoiser", False),
                        optimize=kwargs.get("optimize", False),
                    )

            return self.model

        self.log("Downloading TTS model...", "info")
        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=kwargs.get("load_denoiser", False),
                optimize=kwargs.get("optimize", False),
            )
        return self.model

    def generate_stream(
        self, model: VoxCPM, **kwargs
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded VoxCPM model instance.
            kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of ``(audio, sample_rate,
            timing)`` tuples suitable for Celune's playback pipeline.

        Raises:
            ValueError: The requested voice is unknown or input text is empty.
            NotImplementedError: If streaming support is unavailable.
        """
        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        chunk_size = kwargs.pop("chunk_size", 1)

        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        kwargs.pop("repetition_penalty", None)

        try:
            loader, _ = self._require_compatible_bundle()
            ref_wav = self._truncate_reference(loader.materialize(voice, "wav"))
            configured_cfg = loader.bundle.voices[voice].get("cfg_scale")
            cfg = (
                float(configured_cfg)
                if isinstance(configured_cfg, (int, float))
                and not isinstance(configured_cfg, bool)
                else self.voice_cfg.get(
                    voice, self.voice_cfg[self.default_voice or "balanced"]
                )
            )
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        text = kwargs.pop("text", None)
        if not text:
            raise ValueError("expected text to say")

        if instruct:
            text = f"({instruct}) {text}"

        self._apply_seed()

        if not hasattr(model, "generate_streaming"):
            version = get_version("voxcpm")
            raise NotImplementedError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )

        chunks_per_batch = max(1, round(chunk_size / (1 / self.chunk_rate)))

        stream = None
        try:
            with self._suppress_backend_output():
                stream = model.generate_streaming(
                    text,
                    reference_wav_path=ref_wav,
                    inference_timesteps=4,
                    cfg_value=cfg,
                    max_len=self.max_new_tokens,
                    **kwargs,
                )

                batch: list[npt.NDArray[np.float32]] = []
                pending_audio: Optional[npt.NDArray[np.float32]] = None
                pending_steps = 0
                chunk_index = 0
                total_steps = 0

                for chunk in stream:
                    batch.append(chunk)

                    if len(batch) < chunks_per_batch:
                        continue

                    if pending_audio is not None:
                        total_steps += pending_steps
                        yield (
                            pending_audio,
                            BASE_SR,
                            {
                                "backend": self.name,
                                "chunk_index": chunk_index,
                                "chunk_steps": pending_steps,
                                "total_steps_so_far": total_steps,
                                "is_final": False,
                            },
                        )
                        chunk_index += 1

                    pending_audio = np.concatenate(batch)
                    pending_steps = len(batch)
                    batch.clear()

                if batch:
                    if pending_audio is not None:
                        total_steps += pending_steps
                        yield (
                            pending_audio,
                            BASE_SR,
                            {
                                "backend": self.name,
                                "chunk_index": chunk_index,
                                "chunk_steps": len(batch),
                                "total_steps_so_far": total_steps,
                                "is_final": False,
                            },
                        )
                        chunk_index += 1

                    total_steps += len(batch)
                    yield (
                        np.concatenate(batch),
                        BASE_SR,
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": len(batch),
                            "total_steps_so_far": total_steps,
                            "is_final": True,
                            "missing_eos": total_steps >= self.max_new_tokens,
                        },
                    )
                elif pending_audio is not None:
                    total_steps += pending_steps
                    yield (
                        pending_audio,
                        BASE_SR,
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": pending_steps,
                            "total_steps_so_far": total_steps,
                            "is_final": True,
                            "missing_eos": total_steps >= self.max_new_tokens,
                        },
                    )

        finally:
            if stream is not None and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()
