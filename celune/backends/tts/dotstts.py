# SPDX-License-Identifier: MIT
"""dots.tts MeanFlow backend implementation for Celune."""

import os
import time
import contextlib
from typing import Optional, cast
from collections.abc import Mapping, Callable, Iterator, Generator

import loguru
import numpy as np
from transformers import AutoTokenizer
from dots_tts.runtime import DotsTtsRuntime

from ...utils import discard, custom_assert
from ...typing.backends import _LoguruLogger
from ...cevoice import CEVoiceLoader, default_loader
from ...typing.aliases import AudioChunk, AudioChunks
from .base import (
    _to_numpy_audio as normalize_streamed_audio,
)
from .base import (
    CeluneBackend,
    local_hf_offline_mode,
    cached_hf_snapshot_path,
)


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

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(log=log, fatal=fatal)
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for the dots.tts backend module."""
        return default_loader()

    def _validate_refs(self) -> None:
        """Validate dots.tts reference audio files from the active CEVOICE/CECHAR pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack.

        Returns:
            list[str]: The list of available voices to use from current CEVOICE/CECHAR pack.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice from the active pack to the shared dots.tts model.

        Args:
            voice: The voice name to resolve.

        Returns:
            str: A resolved model name for this voice.
        """
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return self.default_model_id
        _, voice_names = compatible_bundle
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
            bound_logger = cast(
                Optional[_LoguruLogger],
                getattr(loguru, "logger", None),
            )
            with contextlib.suppress(Exception):
                if bound_logger is not None:
                    bound_logger.disable("dots_tts")
                    disabled_loguru = True

            try:
                with (
                    contextlib.redirect_stdout(devnull),
                    contextlib.redirect_stderr(devnull),
                ):
                    yield
            finally:
                if disabled_loguru and bound_logger is not None:
                    with contextlib.suppress(Exception):
                        bound_logger.enable("dots_tts")

    suppress_backend_output = _suppress_backend_output

    @staticmethod
    def _fix_checkpoint_tokenizer(model: DotsTtsRuntime) -> None:
        """Reload dots.tts's tokenizer with Transformers' Mistral regex fix."""
        tokenizer = AutoTokenizer.from_pretrained(
            str(model.pretrained_path),
            local_files_only=True,
            fix_mistral_regex=True,
        )
        model.model.tokenizer = tokenizer
        model.model.core.tokenizer = tokenizer

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
        discard(lang)
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

        with (
            local_hf_offline_mode(available and path is not None),
            self._suppress_backend_output(),
        ):
            self.model = DotsTtsRuntime.from_pretrained(
                target,
                precision=precision,
                optimize=optimize,
                max_generate_length=max_generate_length,
            )
            self._fix_checkpoint_tokenizer(self.model)

        return self.model

    _to_numpy_audio = staticmethod(normalize_streamed_audio)

    def generate_stream(
        self, model: DotsTtsRuntime, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune-compatible audio chunks.

        Args:
            model: The loaded dots.tts runtime instance.
            kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of dots.tts streaming audio chunks.

        Raises:
            ValueError: The requested voice is unsupported, or input text is empty.
        """
        voice = kwargs.pop("voice", self.default_voice)
        kwargs.pop("instruct", None)
        language = self.resolve_generation_language(kwargs.pop("language", None))
        chunk_size = max(1, int(kwargs.pop("chunk_size", 1)))
        text = kwargs.pop("text", None)

        kwargs.pop("temperature", None)
        kwargs.pop("top_k", None)
        kwargs.pop("top_p", None)
        kwargs.pop("repetition_penalty", None)

        if not text:
            raise ValueError("expected text to say")

        try:
            compatible_bundle = self._require_compatible_bundle()
            if compatible_bundle is None:
                return
            loader, voice_names = compatible_bundle
            if voice not in loader.bundle.voices:
                voice = voice_names[0]
            ref_wav = self._truncate_reference(loader.materialize(voice, "wav"))
            configured_ref_text = loader.bundle.voices[voice].get("reference_text")
            ref_text = (
                configured_ref_text.strip()
                if isinstance(configured_ref_text, str)
                else ""
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
                    speaker_scale=1.5,
                    ode_method="euler",
                    num_steps=4,
                    normalize_text=False,
                    **kwargs,
                )

                batch: AudioChunks = []
                chunk_index = 0
                total_steps = 0
                pending_audio: Optional[AudioChunk] = None
                pending_steps = 0
                first_chunk_time: Optional[float] = None

                for chunk in stream:
                    audio = self._to_numpy_audio(chunk)
                    if audio.size == 0:
                        continue
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    batch.append(audio)
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
                                "first_chunk_time": first_chunk_time,
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
                                "first_chunk_time": first_chunk_time,
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
                            "first_chunk_time": first_chunk_time,
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
                            "first_chunk_time": first_chunk_time,
                            "is_final": True,
                        },
                    )
        finally:
            if stream is not None and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()
