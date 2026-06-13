# SPDX-License-Identifier: MIT
"""dots.tts MeanFlow backend implementation for Celune."""

import os
import contextlib
from collections.abc import Iterator
from typing import Callable, Optional, Mapping, Generator

import torch
import numpy as np
import numpy.typing as npt
from dots_tts.runtime import DotsTtsRuntime

try:
    from loguru import logger as loguru_logger
except ModuleNotFoundError:
    loguru_logger = None

from ..utils import custom_assert
from ..exceptions import BackendError
from ..cevoice import default_loader, CEVoiceLoader
from .base import CeluneBackend, cached_hf_snapshot_path


class DotsTtsMF(CeluneBackend[DotsTtsRuntime]):
    """Celune dots.tts MeanFlow backend."""

    name: str = "dotstts"
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
        "balanced": "rednote-hilab/dots.tts-mf",
        "calm": "rednote-hilab/dots.tts-mf",
        "bold": "rednote-hilab/dots.tts-mf",
        "upbeat": "rednote-hilab/dots.tts-mf",
    }
    default_voice: Optional[str] = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self._validate_refs()

    @staticmethod
    def _require_compatible_bundle() -> tuple[CEVoiceLoader, tuple[str, ...]]:
        """Return the active CEVOICE/CECHAR loader and its usable voice names."""
        loader = default_loader()
        custom_assert(
            loader is not None,
            BackendError(
                "backend 'dotstts' requires a compatible CEVOICE/CECHAR package "
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
                "backend 'dotstts' requires a compatible CEVOICE/CECHAR package "
                "with at least one valid voice identifier"
            ),
        )
        assert bool(voice_names)

        return loader, voice_names

    def _validate_refs(self) -> None:
        """Validate dots.tts reference audio files from the active CEVOICE/CECHAR pack."""
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
        """Resolve a voice from the active pack to the shared dots.tts model.

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

    def resolve_generation_language(self, lang: Optional[str]) -> Optional[str]:
        """Normalize generation language tags to dots.tts-friendly values.

        Args:
            lang: The requested language identifier, if any.

        Returns:
            Optional[str]: The normalized backend-facing language identifier.
        """
        if lang is None:
            return None

        normalized = lang.strip().lower()
        if not normalized or normalized == "auto":
            return None
        if normalized.startswith("zh"):
            return "zh"
        return normalized

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress unnecessary backend output."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            disabled_loguru = False
            if loguru_logger is not None:
                with contextlib.suppress(Exception):
                    loguru_logger.disable("dots_tts")
                    disabled_loguru = True

            try:
                with contextlib.redirect_stdout(devnull):
                    with contextlib.redirect_stderr(devnull):
                        yield
            finally:
                if disabled_loguru and loguru_logger is not None:
                    with contextlib.suppress(Exception):
                        loguru_logger.enable("dots_tts")

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.
            lang: The language identifier for differentiating models by language.

        Returns:
            tuple[bool, Optional[str]]: A cache availability flag and the resolved snapshot path when present.
        """
        del lang
        return cached_hf_snapshot_path(
            model,
            [
                "config.json",
                "*.safetensors",
                "tokenizer_config.json",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> DotsTtsRuntime:
        """Load the given dots.tts model.

        Args:
            model_id: The dots.tts model repository ID to load.
            kwargs: Additional keyword arguments to use while loading dots.tts.

        Returns:
            DotsTtsRuntime: The loaded dots.tts runtime instance.
        """
        available, path = self.model_is_available_locally(model_id)
        precision = kwargs.get("precision", "bfloat16")
        optimize = bool(kwargs.get("optimize", False))
        max_generate_length = int(
            kwargs.get("max_generate_length", self.max_new_tokens)
        )

        target = path if available and path is not None else model_id
        if target == model_id:
            self.log("Downloading TTS model...", "info")

        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            if available and path is not None:
                os.environ["HF_HUB_OFFLINE"] = "1"
            with self._suppress_backend_output():
                self.model = DotsTtsRuntime.from_pretrained(
                    target,
                    precision=precision,
                    optimize=optimize,
                    max_generate_length=max_generate_length,
                )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline

        return self.model

    @staticmethod
    def _to_numpy_audio(chunk: torch.Tensor) -> npt.NDArray[np.float32]:
        """Convert one streamed torch chunk to a Celune-compatible audio array."""
        audio = chunk.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio

    def generate_stream(
        self, model: DotsTtsRuntime, **kwargs
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune-compatible audio chunks.

        Args:
            model: The loaded dots.tts runtime instance.
            kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of dots.tts streaming audio
            chunks.

        Raises:
            ValueError: The requested voice is unsupported, or input text is empty.
        """
        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        language = self.resolve_generation_language(kwargs.pop("language", None))
        chunk_size = max(1, int(kwargs.pop("chunk_size", 1)))
        text = kwargs.pop("text", None)

        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        kwargs.pop("repetition_penalty", None)

        if not text:
            raise ValueError("expected text to say")

        if instruct:
            text = f"({instruct}) {text}"

        try:
            loader, _ = self._require_compatible_bundle()
            ref_wav = loader.materialize(voice, "wav")
            configured_ref_text = loader.bundle.voices[voice].get("reference_text")
            ref_text = (
                configured_ref_text if isinstance(configured_ref_text, str) else ""
            )
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        self._apply_seed()

        stream = None
        try:
            with self._suppress_backend_output():
                stream = model.generate_stream(
                    text=text,
                    prompt_audio_path=str(ref_wav),
                    prompt_text=ref_text,
                    language=language,
                    speaker_scale=float(kwargs.pop("speaker_scale", 1.5)),
                    ode_method="euler",
                    num_steps=4,
                    normalize_text=False,
                    **kwargs,
                )

                batch: list[npt.NDArray[np.float32]] = []
                chunk_index = 0
                total_steps = 0
                pending_audio: Optional[npt.NDArray[np.float32]] = None
                pending_steps = 0

                for chunk in stream:
                    batch.append(self._to_numpy_audio(chunk))
                    if len(batch) < chunk_size:
                        continue

                    if pending_audio is not None:
                        total_steps += pending_steps
                        yield (
                            pending_audio,
                            int(getattr(model, "sample_rate", 48000)),
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
                            int(getattr(model, "sample_rate", 48000)),
                            {
                                "backend": self.name,
                                "chunk_index": chunk_index,
                                "chunk_steps": pending_steps,
                                "total_steps_so_far": total_steps,
                                "is_final": False,
                            },
                        )
                        chunk_index += 1

                    total_steps += len(batch)
                    yield (
                        np.concatenate(batch),
                        int(getattr(model, "sample_rate", 48000)),
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": len(batch),
                            "total_steps_so_far": total_steps,
                            "is_final": True,
                        },
                    )
                elif pending_audio is not None:
                    total_steps += pending_steps
                    yield (
                        pending_audio,
                        int(getattr(model, "sample_rate", 48000)),
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": pending_steps,
                            "total_steps_so_far": total_steps,
                            "is_final": True,
                        },
                    )
        finally:
            if stream is not None and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()
