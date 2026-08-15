# SPDX-License-Identifier: MIT
"""VoxCPM2 backend implementation for Celune."""

import contextlib
import os
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from typing import Optional

import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from voxcpm import VoxCPM

from ...cevoice import CEVoiceLoader, default_loader
from ...constants import BASE_SR
from ...i18n import string
from ...typing.aliases import AudioChunk, AudioChunks
from ...utils import custom_assert
from . import get_version
from .base import (
    CeluneBackend,
    _to_numpy_audio as normalize_streamed_audio,
    cached_hf_snapshot_path,
    local_hf_offline_mode,
)


class _VoxCPMTextTokenizer:
    """Adapt the checkpoint tokenizer to VoxCPM2's list-returning interface."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def __call__(self, text: str) -> list[int]:
        """Tokenize text without adding tokenizer-wrapper special tokens."""
        return [
            int(token_id)
            for token_id in self._tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        ]


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

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(log=log, fatal=fatal)
        self.log = log
        self.optimize_enabled = False
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for the VoxCPM2 backend module."""
        return default_loader()

    def _validate_refs(self) -> None:
        """Validate VoxCPM2 reference audio files from the active CEVOICE/CECHAR pack."""
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
        """Resolve a voice from the active pack to the shared VoxCPM2 model.

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

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress unnecessary backend output."""
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            yield

    suppress_backend_output = _suppress_backend_output

    @staticmethod
    def _install_checkpoint_tokenizer(
        model: Optional[VoxCPM],
        snapshot_path: Optional[str],
    ) -> None:
        """Install VoxCPM2's checkpoint tokenizer after the package loader runs."""
        if model is None or snapshot_path is None:
            return

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        runtime = getattr(model, "tts_model", None)
        if runtime is None or not hasattr(runtime, "text_tokenizer"):
            raise RuntimeError(string("voxcpm2.tokenizer_unavailable"))
        runtime.text_tokenizer = _VoxCPMTextTokenizer(tokenizer)

    _to_numpy_audio = staticmethod(normalize_streamed_audio)

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

        # NOTE:
        # this may cause errors in internal ops when switching backends
        # where one backend ran with deterministic algorithms, others without them,
        # which can cause errors such as:
        #
        #   RuntimeError: _unsafe_index found unexpected index type Float
        #
        # while switching in order from: voxcpm2 -> dotstts -> qwen3,
        # which is then trapped in Celune's warmup failure except block, and may potentially
        # leave the runtime in a buggy state, or even trigger fatal errors
        #
        # please do not modulate deterministic algorithm state in PyTorch on a per-backend basis

        # import torch
        # torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)

        if available and path is not None:
            with local_hf_offline_mode(), self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
                self._install_checkpoint_tokenizer(self.model, path)

            return self.model

        self.log("Downloading TTS model...", "info")
        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=kwargs.get("load_denoiser", False),
                optimize=kwargs.get("optimize", False),
            )
            _, path = self.model_is_available_locally(model_id)
            self._install_checkpoint_tokenizer(self.model, path)
        return self.model

    def generate_stream(
        self, model: VoxCPM, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded VoxCPM model instance.
            kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of ``(audio, sample_rate, timing)`` tuples
            suitable for Celune's playback pipeline.

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
            compatible_bundle = self._require_compatible_bundle()
            if compatible_bundle is None:
                return
            loader, _ = compatible_bundle
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
        sample_rate = int(getattr(model, "sample_rate", BASE_SR))

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

                batch: AudioChunks = []
                pending_audio: Optional[AudioChunk] = None
                pending_steps = 0
                chunk_index = 0
                total_steps = 0
                first_chunk_time: Optional[float] = None

                for chunk in stream:
                    audio = self._to_numpy_audio(chunk)
                    if audio.size == 0:
                        continue
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                    batch.append(audio)

                    if len(batch) < chunks_per_batch:
                        continue

                    if pending_audio is not None:
                        total_steps += pending_steps
                        yield (
                            pending_audio,
                            sample_rate,
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
                            sample_rate,
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
                        sample_rate,
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": len(batch),
                            "total_steps_so_far": total_steps,
                            "first_chunk_time": first_chunk_time,
                            "is_final": True,
                            "missing_eos": total_steps >= self.max_new_tokens,
                        },
                    )
                elif pending_audio is not None:
                    total_steps += pending_steps
                    yield (
                        pending_audio,
                        sample_rate,
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": pending_steps,
                            "total_steps_so_far": total_steps,
                            "first_chunk_time": first_chunk_time,
                            "is_final": True,
                            "missing_eos": total_steps >= self.max_new_tokens,
                        },
                    )

        finally:
            if stream is not None and hasattr(stream, "close"):
                with contextlib.suppress(Exception):
                    stream.close()
