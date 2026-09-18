# SPDX-License-Identifier: Apache-2.0
"""LuxTTS backend implementation for Celune."""

import os
import time
import math
import contextlib
from types import MethodType
from typing import ClassVar, Protocol, Union, Optional, cast
from collections.abc import Mapping, Callable, Iterator, Generator

import numpy as np
import torch
from huggingface_hub import snapshot_download

from ...cevoice import CEVoiceLoader, default_loader
from ...i18n import string
from ...paths import huggingface_hub_cache_dir, huggingface_progress
from ...typing.aliases import AudioChunk
from ...typing.backends import BackendModel
from ...utils import custom_assert
from .base import (
    CeluneBackend,
    _to_numpy_audio,
    local_hf_offline_mode,
    cached_hf_snapshot_path,
)

__all__ = ["LuxTTS"]

_LUXTTS_MODEL_ID = "YatharthS/LuxTTS"
_LUXTTS_TRANSCRIBER_MODEL_ID = "openai/whisper-tiny"
_LUXTTS_TRANSCRIBER_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
_LUXTTS_SAMPLE_RATE = 48000
_LUXTTS_PROMPT_DURATION_SECONDS = 5
_LUXTTS_CPU_THREADS = 2
_LUXTTS_VOCODER_MIN_FRAMES = 7
_LUXTTS_VOCODER_TAIL_FRAMES = 15
_LuxPromptValue = Union[torch.Tensor, float, int]
_LuxPrompt = dict[str, _LuxPromptValue]


class _LuxTTSVocoder(Protocol):
    """Subset of the LuxTTS vocoder used by the stability guard."""

    def decode(self, features_input: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Decode acoustic features into a waveform."""
        raise NotImplementedError("protocol not defined")


class _LuxTTSOnnxModel(Protocol):
    """Subset of the LuxTTS CPU model used by the duration correction."""

    def run_text_encoder(
        self,
        tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_features_len: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build text conditioning for the requested token sequence."""
        raise NotImplementedError("protocol not defined")


class _LuxTTSModel(BackendModel, Protocol):
    """Subset of LuxTTS required by Celune's adapter."""

    model: _LuxTTSOnnxModel
    vocos: _LuxTTSVocoder

    def encode_prompt(
        self,
        prompt_audio: str,
        duration: int = _LUXTTS_PROMPT_DURATION_SECONDS,
        rms: float = 0.01,
    ) -> _LuxPrompt:
        """Encode a reference recording into a LuxTTS prompt."""
        raise NotImplementedError("protocol not defined")

    def generate_speech(
        self,
        text: str,
        encoded_prompt: _LuxPrompt,
        *,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ) -> torch.Tensor:
        """Generate one waveform from text and an encoded LuxTTS prompt."""
        raise NotImplementedError("protocol not defined")


def _load_runtime_class() -> Callable[..., _LuxTTSModel]:
    """Load LuxTTS only inside its isolated backend environment."""
    from zipvoice.luxvoice import LuxTTS as RuntimeLuxTTS

    return cast(Callable[..., _LuxTTSModel], RuntimeLuxTTS)


def _pad_vocoder_features(features: torch.Tensor) -> torch.Tensor:
    """Add the context frames required by LuxTTS's Vocos decoder.

    LuxTTS removes the reference prompt before decoding. Very short text can
    therefore leave fewer frames than Vocos's first convolution accepts, and
    the last generated frames otherwise lack the decoder's look-ahead context.

    Args:
        features: Acoustic features shaped as ``(batch, channels, frames)``.

    Returns:
        torch.Tensor: Features with a valid minimum length and decoder tail.

    Raises:
        ValueError: The model produced no acoustic frames for the input.
    """
    frame_count = features.shape[-1]
    if frame_count == 0:
        raise ValueError("LuxTTS produced no acoustic frames for the input")

    padding_frames = max(_LUXTTS_VOCODER_MIN_FRAMES - frame_count, 0)
    padding_frames += _LUXTTS_VOCODER_TAIL_FRAMES
    tail = features[..., -1:].expand(*features.shape[:-1], padding_frames)
    return torch.cat((features, tail), dim=-1)


def _install_cpu_duration_correction(model: _LuxTTSModel) -> None:
    """Correct the CPU ONNX duration ratio before prompt frames are removed."""
    onnx_model = getattr(model, "model", None)
    if onnx_model is None:
        return

    original_run_text_encoder = getattr(type(onnx_model), "run_text_encoder", None)
    if not callable(original_run_text_encoder):
        return
    run_text_encoder = cast(
        Callable[..., tuple[torch.Tensor, torch.Tensor]],
        original_run_text_encoder,
    )

    def corrected_run_text_encoder(
        onnx_instance: _LuxTTSOnnxModel,
        tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_features_len: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the GPU path's prompt-plus-generated duration calculation."""
        prompt_token_count = int(prompt_tokens.shape[1])
        target_token_count = int(tokens.shape[1])
        prompt_frame_count = int(prompt_features_len.item())
        if (
            prompt_token_count <= 0
            or target_token_count <= 0
            or prompt_frame_count <= 0
        ):
            return run_text_encoder(
                onnx_instance,
                tokens,
                prompt_tokens,
                prompt_features_len,
                speed,
            )

        requested_speed = float(speed.item())
        generated_frame_count = math.ceil(
            prompt_frame_count
            / prompt_token_count
            * target_token_count
            / requested_speed
        )
        desired_frame_count = prompt_frame_count + generated_frame_count
        onnx_frame_ratio = (
            prompt_frame_count
            / prompt_token_count
            * (prompt_token_count + target_token_count - 1)
        )
        corrected_speed = onnx_frame_ratio / desired_frame_count
        corrected_speed_tensor = speed.new_tensor(corrected_speed)
        return run_text_encoder(
            onnx_instance,
            tokens,
            prompt_tokens,
            prompt_features_len,
            corrected_speed_tensor,
        )

    onnx_model.run_text_encoder = MethodType(
        corrected_run_text_encoder,
        onnx_model,
    )


def _install_vocoder_decode_guard(model: _LuxTTSModel) -> None:
    """Protect the isolated LuxTTS Vocos decoder from empty short outputs."""
    vocoder = getattr(model, "vocos", None)
    if vocoder is None:
        return

    original_decode = getattr(type(vocoder), "decode", None)
    if not callable(original_decode):
        return
    decode = cast(Callable[..., torch.Tensor], original_decode)

    def guarded_decode(
        vocoder_instance: _LuxTTSVocoder,
        features_input: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Decode features after applying LuxTTS's required context padding."""
        return decode(
            vocoder_instance,
            _pad_vocoder_features(features_input),
            **kwargs,
        )

    vocoder.decode = MethodType(guarded_decode, vocoder)


class LuxTTS(CeluneBackend[_LuxTTSModel]):
    """Celune's CPU-capable LuxTTS voice-cloning backend."""

    name: str = "luxtts"
    uses_voice_bundles: bool = True
    chunk_rate: float = 1.0
    supported_languages: tuple[str, ...] = ("en",)
    voice_models: ClassVar[Optional[Mapping[str, str]]] = {
        "balanced": _LUXTTS_MODEL_ID,
        "calm": _LUXTTS_MODEL_ID,
        "bold": _LUXTTS_MODEL_ID,
        "upbeat": _LUXTTS_MODEL_ID,
    }
    default_voice: Optional[str] = "balanced"

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        threads: int = _LUXTTS_CPU_THREADS,
    ) -> None:
        super().__init__(log=log, fatal=fatal)
        self._threads = max(1, int(threads))
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for LuxTTS."""
        return default_loader()

    def _validate_refs(self) -> None:
        """Validate LuxTTS reference WAV files in the active voice pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for name in voice_names:
            loader.materialize(name, "wav")

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active CEVOICE/CECHAR pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve an active voice to the shared LuxTTS model."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return _LUXTTS_MODEL_ID
        _, voice_names = compatible_bundle
        custom_assert(
            voice in voice_names,
            ValueError(f"{self.name} cannot resolve a model for voice '{voice}'"),
        )
        assert voice in voice_names
        return _LUXTTS_MODEL_ID

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
        """Suppress LuxTTS and ONNX diagnostic output from Celune's UI log."""
        with (
            open(os.devnull, "w", encoding="utf-8") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            yield

    suppress_backend_output = _suppress_backend_output

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check for a complete CPU LuxTTS snapshot in Celune's Hub cache."""
        del lang
        return cached_hf_snapshot_path(
            model,
            [
                "tokens.txt",
                "text_encoder.onnx",
                "fm_decoder.onnx",
                "config.json",
                "vocoder/config.yaml",
                "vocoder/vocos.bin",
            ],
        )

    def load_model(self, model_id: str, **kwargs) -> _LuxTTSModel:
        """Load LuxTTS on CPU from a cached or newly downloaded snapshot."""
        del kwargs
        available, snapshot_path = self.model_is_available_locally(model_id)
        target = snapshot_path if available and snapshot_path is not None else model_id
        if not available:
            self.log(string("tts.model_download_start"), "info")

        runtime_class = _load_runtime_class()
        with (
            local_hf_offline_mode(available),
            huggingface_progress(self.report_progress),
            self._suppress_backend_output(),
        ):
            model = runtime_class(
                target,
                device="cpu",
                threads=self._threads,
            )
        _install_cpu_duration_correction(model)
        _install_vocoder_decode_guard(model)
        return model

    def prepare_model_loading(self) -> None:
        """Import LuxTTS before the worker accepts request-thread operations."""
        with self._suppress_backend_output():
            _load_runtime_class()

    def preload_models(self) -> None:
        """Ensure LuxTTS and its hard-coded Whisper transcriber are cached."""
        super().preload_models()
        available, _ = cached_hf_snapshot_path(
            _LUXTTS_TRANSCRIBER_MODEL_ID,
            _LUXTTS_TRANSCRIBER_FILES,
        )
        if available:
            self.log(
                string("tts.model_available", model_id=_LUXTTS_TRANSCRIBER_MODEL_ID),
                "info",
            )
            return

        self.log(
            string("tts.model_downloading", model_id=_LUXTTS_TRANSCRIBER_MODEL_ID),
            "info",
        )
        with huggingface_progress(self.report_progress):
            snapshot_download(
                repo_id=_LUXTTS_TRANSCRIBER_MODEL_ID,
                cache_dir=str(huggingface_hub_cache_dir(create=True)),
            )

    def generate_stream(
        self, model: _LuxTTSModel, **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate one normalized, complete LuxTTS waveform for Celune."""
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(string("tts.text_required"))

        voice = kwargs.pop("voice", self.default_voice)
        for name in (
            "instruct",
            "language",
            "chunk_size",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "max_new_tokens",
        ):
            kwargs.pop(name, None)

        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if not isinstance(voice, str) or voice not in voice_names:
            raise ValueError(f"unknown voice '{voice}' for backend '{self.name}'")

        ref_wav = self._truncate_reference(loader.materialize(voice, "wav"))
        self._apply_seed()
        started = time.monotonic()
        with self._suppress_backend_output():
            encoded_prompt = model.encode_prompt(
                str(ref_wav),
                duration=_LUXTTS_PROMPT_DURATION_SECONDS,
                rms=0.01,
            )
            waveform = model.generate_speech(
                text,
                encoded_prompt,
                num_steps=4,
                guidance_scale=3.0,
                t_shift=0.5,
                speed=1.0,
                return_smooth=False,
            )

        audio = np.ascontiguousarray(
            np.clip(_to_numpy_audio(waveform), -1.0, 1.0), dtype=np.float32
        )
        if audio.size == 0:
            raise ValueError("LuxTTS produced no acoustic frames for the input")
        yield (
            audio,
            _LUXTTS_SAMPLE_RATE,
            {
                "backend": self.name,
                "chunk_index": 0,
                "chunk_steps": 1,
                "total_steps_so_far": 1,
                "first_chunk_time": started,
                "is_final": True,
            },
        )
