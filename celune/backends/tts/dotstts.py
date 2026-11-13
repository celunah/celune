# SPDX-License-Identifier: Apache-2.0
"""dots.tts MeanFlow backend implementation for Celune."""

import os
import time
import contextlib
from pathlib import Path
from typing import Optional, Union, cast
from collections.abc import Mapping, Callable, Iterator, Generator

import numpy as np
import loguru
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from dots_tts.runtime import DotsTtsRuntime

from .base import (
    CeluneBackend,
    local_hf_offline_mode,
    cached_hf_snapshot_path,
)
from .base import (
    to_numpy_audio as normalize_streamed_audio,
)
from ...i18n import string
from ...paths import huggingface_progress, huggingface_hub_cache_dir
from ...utils import discard, custom_assert
from ...cevoice import CEVoiceLoader, default_loader
from ...registry.tts import TTS_DEFAULT_VOICE, DOTS_TTS_VOICE_MODELS
from ...typing.aliases import AudioChunk, AudioChunks
from ...typing.backends import LoguruLogger
from ...registry.consent import require_model_consent
from ...typing.backends import BackendArgumentValue
from .audio_cpp import (
    AudioCppRuntime,
    ensure_audio_cpp_server,
    load_audio_cpp_tts_model,
)
from .audio_cpp_models import audio_cpp_model_artifact


class DotsTtsMF(CeluneBackend[Union[DotsTtsRuntime, AudioCppRuntime]]):
    """Celune dots.tts MeanFlow backend."""

    name: str = "dotstts"
    supports_audio_cpp = True
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

    voice_models: Optional[Mapping[str, str]] = DOTS_TTS_VOICE_MODELS
    default_voice: Optional[str] = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
    ) -> None:
        super().__init__(log=log, fatal=fatal, quantize=quantize)
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

    def preload_models(self) -> None:
        """Prepare audio.cpp when selected, or preload native DotTTS weights."""
        if self.effective_inference_mode != "audio_cpp":
            super().preload_models()
            return

        for model_id in self.all_model_ids:
            require_model_consent(model_id)
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("audio.cpp requires an available CUDA device")
            ensure_audio_cpp_server(
                report_progress=self.report_progress,
                log=self.log,
                backend_name=self.name,
            )
        except Exception as error:
            self.fallback_audio_cpp_to_native(error)
        if self.effective_inference_mode == "native":
            super().preload_models()

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
                Optional[LoguruLogger],
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

    def load_model(
        self, model_id: str, **kwargs
    ) -> Union[DotsTtsRuntime, AudioCppRuntime]:
        """Load the given dots.tts model.

        Args:
            model_id: The dots.tts model repository ID to load.
            kwargs: Additional keyword arguments to use while loading dots.tts.

        Returns:
            Union[DotsTtsRuntime, AudioCppRuntime]: The loaded native or GGUF runtime.
        """
        metadata = require_model_consent(model_id)
        if self.effective_inference_mode == "audio_cpp":
            artifact = audio_cpp_model_artifact(
                self.name,
                model_id,
                quantize=self.quantization_requested,
            )
            precision = kwargs.get("precision", "bfloat16")
            optimize = bool(kwargs.get("optimize", False))
            max_generate_length = int(
                kwargs.get("max_generate_length", self.max_new_tokens)
            )
            if artifact is None:
                self.fallback_audio_cpp_to_native(
                    RuntimeError(
                        f"audio.cpp has no verified DotTTS MeanFlow package for '{model_id}'"
                    )
                )
            elif optimize or precision not in {"bf16", "bfloat16"}:
                self.fallback_audio_cpp_to_native(
                    RuntimeError(
                        "audio.cpp cannot honor the configured DotTTS precision or optimization"
                    )
                )
            else:
                try:
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "audio.cpp requires an available CUDA device"
                        )
                    runtime = load_audio_cpp_tts_model(
                        artifact,
                        backend_id=self.name,
                        sample_rate=48_000,
                        torch_root=Path(torch.__file__).resolve().parent,
                        device=torch.cuda.current_device(),
                        log=self.log,
                        report_progress=self.report_progress,
                        default_request_options={
                            "speaker_scale": 1.5,
                            "sampler_mode": "euler",
                            "num_inference_steps": 4,
                            "max_tokens": max_generate_length,
                        },
                    )
                except Exception as error:
                    self.fallback_audio_cpp_to_native(error)
                else:
                    self.quantization_attempted = self.quantization_requested
                    self.quantization_active = artifact.quantization == "Q8_0"
                    self.max_new_tokens = max_generate_length
                    self.model = runtime
                    return runtime

        available, path = self.model_is_available_locally(model_id)
        if not available or path is None:
            self.log(string("tts.model_download_start"), "info")
            with huggingface_progress(self.report_progress):
                path = snapshot_download(
                    repo_id=model_id,
                    revision=metadata.revision,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )
        precision = kwargs.get("precision", "bfloat16")
        optimize = bool(kwargs.get("optimize", False))
        max_generate_length = int(
            kwargs.get("max_generate_length", self.max_new_tokens)
        )

        target = path
        quantize_on_cpu = getattr(self, "quantization_mode", None) is not None
        with (
            local_hf_offline_mode(),
            huggingface_progress(self.report_progress),
            self._suppress_backend_output(),
        ):
            if quantize_on_cpu:
                self.model = self._load_quantized_runtime(
                    target,
                    model_id=model_id,
                    precision=precision,
                    optimize=optimize,
                    max_generate_length=max_generate_length,
                )
            else:
                self.model = DotsTtsRuntime.from_pretrained(
                    target,
                    precision=precision,
                    optimize=optimize,
                    max_generate_length=max_generate_length,
                )
            self._fix_checkpoint_tokenizer(self.model)

        if not quantize_on_cpu:
            self.model = self.apply_runtime_quantization(self.model, model_id)
        return self.model

    def _load_quantized_runtime(
        self,
        target: str,
        *,
        model_id: str,
        precision: str,
        optimize: bool,
        max_generate_length: int,
    ) -> DotsTtsRuntime:
        """Quantize dots.tts on CPU before its runtime moves the model to CUDA."""
        import torch
        from dots_tts.models.dots_tts.model import DotsTtsModel

        # noinspection PyProtectedMember
        pretrained_path = DotsTtsRuntime._resolve_pretrained_path(target)
        loaded_model = cast(
            DotsTtsModel,
            DotsTtsModel.from_pretrained(pretrained_path),
        )
        loaded_model.core.to(dtype=torch.bfloat16)
        loaded_model = cast(
            DotsTtsModel,
            self.apply_runtime_quantization(
                cast(DotsTtsRuntime, loaded_model),
                model_id,
            ),
        )
        return DotsTtsRuntime(
            model=loaded_model,
            pretrained_path=pretrained_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
        )

    to_numpy_audio = staticmethod(normalize_streamed_audio)

    def generate_stream(
        self, model: Union[DotsTtsRuntime, AudioCppRuntime], **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune-compatible audio chunks.

        Args:
            model: The loaded native or audio.cpp DotTTS runtime instance.
            kwargs: Streaming generation keyword arguments to use.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of dots.tts streaming audio chunks.

        Raises:
            ValueError: The requested voice is unsupported, or input text is empty.
        """
        if isinstance(model, AudioCppRuntime):
            yield from self._generate_audio_cpp_stream(model, kwargs)
            return

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

            # with longer references like the natural pack, this caused reference leakage
            ref_wav = loader.materialize(voice, "wav")
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
                    audio = self.to_numpy_audio(chunk)
                    if audio.size == 0:
                        continue
                    if pending_audio is not None:
                        total_steps += pending_steps
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic()
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
                        pending_audio = None
                        pending_steps = 0

                    batch.append(audio)
                    if len(batch) < chunk_size:
                        continue

                    pending_audio = np.concatenate(batch)
                    pending_steps = len(batch)
                    batch.clear()

                if batch:
                    if pending_audio is not None:
                        total_steps += pending_steps
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic()
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
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
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
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
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
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                with contextlib.suppress(Exception):
                    close_stream()

    def _generate_audio_cpp_stream(
        self,
        model: AudioCppRuntime,
        kwargs: dict[str, BackendArgumentValue],
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Stream DotTTS MeanFlow GGUF audio with Celune's reference voice."""
        voice_value = kwargs.pop("voice", self.default_voice)
        voice = voice_value if isinstance(voice_value, str) else self.default_voice
        kwargs.pop("instruct", None)
        language_value = kwargs.pop("language", None)
        language = self.resolve_generation_language(
            language_value if isinstance(language_value, str) else None
        )
        kwargs.pop("chunk_size", None)
        text_value = kwargs.pop("text", None)
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError("expected text to say")

        # Native dots.tts ignores these sampling controls as well.
        for option in ("temperature", "top_k", "top_p", "repetition_penalty"):
            kwargs.pop(option, None)
        max_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        kwargs.pop("seed", None)

        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if voice not in loader.bundle.voices:
            voice = voice_names[0]
        try:
            reference_path = loader.materialize(voice, "wav")
            configured_ref_text = loader.bundle.voices[voice].get("reference_text")
        except KeyError as error:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from error
        reference_text = (
            configured_ref_text.strip() if isinstance(configured_ref_text, str) else ""
        )

        self._apply_seed()
        options: dict[str, object] = {"max_tokens": max_tokens}
        if self.current_seed is not None:
            options["seed"] = self.current_seed
        request_body: dict[str, object] = {
            "model": model.server_model_id,
            "input": text_value,
            "voice_ref": reference_path.resolve().as_posix(),
            "reference_text": reference_text,
            "response_format": "pcm",
            "stream_format": "sse",
            "options": options,
        }
        if language is not None:
            request_body["language"] = language
        yield from model.stream_speech(request_body, default_sample_rate=48_000)
