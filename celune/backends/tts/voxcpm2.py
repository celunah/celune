# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 backend implementation for Celune."""

import os
import time
import contextlib
from types import MethodType
from typing import Union, Optional, Protocol, cast
from pathlib import Path
from collections.abc import Mapping, Callable, Iterator, Generator

import numpy as np
import torch
from voxcpm import VoxCPM
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import get_version
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
from ...utils import discard, available, custom_assert
from ...cevoice import CEVoiceLoader, default_loader
from .audio_cpp import (
    AudioCppRuntime,
    ensure_audio_cpp_server,
    load_audio_cpp_tts_model,
)
from ...modeling import QuantizedKVCacheMode, quantized_kv_cache_mode
from ...constants import BASE_SR
from ...registry.tts import TTS_DEFAULT_VOICE, VOXCPM2_VOICE_MODELS
from ...typing.aliases import AudioChunk, AudioChunks
from .audio_cpp_models import audio_cpp_model_artifact
from ...typing.backends import BackendArgumentValue
from ...registry.consent import require_model_consent

_VOXCPM_RUNTIME_KV_CACHE_LENGTH = 2048


class _VoxCPMCache(Protocol):
    """Storage surface exposed by VoxCPM's static cache."""

    max_length: int
    kv_cache: torch.Tensor


class _VoxCPMAttention(Protocol):
    """Attention projections used by VoxCPM's one-token decode path."""

    num_heads: int
    num_key_value_heads: int
    head_dim: int
    q_proj: Callable[..., torch.Tensor]
    k_proj: Callable[..., torch.Tensor]
    v_proj: Callable[..., torch.Tensor]
    o_proj: Callable[..., torch.Tensor]
    forward_step: Callable[..., torch.Tensor]


class _VoxCPMDecoderLayer(Protocol):
    """One VoxCPM decoder layer with an attention module."""

    self_attn: _VoxCPMAttention


class _VoxCPMLanguageModel(Protocol):
    """Language model holding one VoxCPM static cache."""

    kv_cache: _VoxCPMCache
    layers: list[_VoxCPMDecoderLayer]


class _VoxCPMRuntime(Protocol):
    """VoxCPM runtime containing its base and residual language models."""

    base_lm: _VoxCPMLanguageModel
    residual_lm: _VoxCPMLanguageModel


class VoxCPMModel(Protocol):
    """Loaded VoxCPM model surface used by the cache adapter."""

    tts_model: _VoxCPMRuntime


class _QuantizedStaticKVLayer:
    """Layer handle that keeps VoxCPM keys and values in compact storage."""

    def __init__(self, cache: "QuantizedStaticKVCache", layer_index: int) -> None:
        self.cache = cache
        self.layer_index = layer_index


class QuantizedStaticKVCache:
    """Quantized fixed-capacity KV storage for VoxCPM's incremental decoder."""

    def __init__(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int,
        mode: QuantizedKVCacheMode,
        residual_length: int = 128,
    ) -> None:
        if len(shape) != 6 or shape[0] != 2:
            raise ValueError("VoxCPM cache shape is not supported")
        if residual_length <= 0 or max_length <= 0:
            raise ValueError("KV cache length must be positive")
        quantized_dtype = torch.int8 if mode == "int8" else _fp8_dtype()
        if quantized_dtype is None:
            raise RuntimeError("FP8 KV cache storage is unavailable")

        self.max_length = max_length
        self.num_layers = shape[1]
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.residual_length = min(residual_length, max_length)
        self.current_length = 0
        self.kv_cache = torch.empty(
            (2, shape[1], shape[2], shape[3], max_length, shape[5]),
            dtype=quantized_dtype,
            device=device,
        )
        self.scales = torch.empty(
            (2, shape[1], shape[2], shape[3], max_length, 1),
            dtype=torch.float16,
            device=device,
        )
        self.residual = torch.empty(
            (
                2,
                shape[1],
                shape[2],
                shape[3],
                self.residual_length,
                shape[5],
            ),
            dtype=dtype,
            device=device,
        )
        self._layer_handles = [
            _QuantizedStaticKVLayer(self, layer_index)
            for layer_index in range(self.num_layers)
        ]

    def get_layer_cache(self, layer_idx: int) -> _QuantizedStaticKVLayer:
        """Return the compact-cache handle for one attention layer."""
        return self._layer_handles[layer_idx]

    def step(self) -> int:
        """Reserve and return the next token position."""
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")
        position = self.current_length
        self.current_length += 1
        return position

    def fill_caches(
        self,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Populate compact storage from VoxCPM's full-precision prefill output."""
        if len(kv_caches) != self.num_layers:
            raise ValueError("VoxCPM cache layer count is not supported")
        sequence_length = kv_caches[0][0].shape[-2]
        if sequence_length > self.max_length:
            raise ValueError("VoxCPM prefill exceeds the KV cache capacity")
        prefix_length = max(0, sequence_length - self.residual_length)
        for layer_index, (keys, values) in enumerate(kv_caches):
            self._fill_tensor(layer_index, 0, keys, prefix_length)
            self._fill_tensor(layer_index, 1, values, prefix_length)
            tail_length = sequence_length - prefix_length
            if tail_length:
                positions = torch.arange(
                    prefix_length,
                    sequence_length,
                    device=self.device,
                )
                slots = positions.remainder(self.residual_length)
                self.residual[0, layer_index].index_copy_(
                    -2,
                    slots,
                    keys[..., prefix_length:, :].to(self.dtype),
                )
                self.residual[1, layer_index].index_copy_(
                    -2,
                    slots,
                    values[..., prefix_length:, :].to(self.dtype),
                )
        self.current_length = sequence_length

    def reset(self) -> None:
        """Discard the current request's cache positions."""
        self.current_length = 0

    def _fill_tensor(
        self,
        layer_index: int,
        key_value_index: int,
        states: torch.Tensor,
        prefix_length: int,
    ) -> None:
        """Store one prefill tensor's compact prefix and per-token scales."""
        if not prefix_length:
            return
        quantized, scales = self._quantize(states[..., :prefix_length, :])
        self.kv_cache[key_value_index, layer_index, ..., :prefix_length, :].copy_(
            quantized
        )
        self.scales[key_value_index, layer_index, ..., :prefix_length, :].copy_(scales)

    def store_step(
        self,
        layer_index: int,
        position: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store one newly decoded key/value pair."""
        if position < 0 or position >= self.max_length:
            raise ValueError("KV cache position is out of range")
        if position != self.current_length - 1:
            raise ValueError("KV cache position does not match the active step")

        if position >= self.residual_length:
            evicted_position = position - self.residual_length
            residual_slot = evicted_position % self.residual_length
            for index in range(2):
                evicted = self.residual[
                    index,
                    layer_index,
                    ...,
                    residual_slot,
                    :,
                ].unsqueeze(-2)
                quantized, scales = self._quantize(evicted)
                self.kv_cache[
                    index,
                    layer_index,
                    ...,
                    evicted_position,
                    :,
                ].copy_(quantized.squeeze(-2))
                self.scales[
                    index,
                    layer_index,
                    ...,
                    evicted_position,
                    :,
                ].copy_(scales.squeeze(-2))

        residual_slot = position % self.residual_length
        self.residual[0, layer_index, ..., residual_slot, :].copy_(
            keys.squeeze(-2).to(self.dtype)
        )
        self.residual[1, layer_index, ..., residual_slot, :].copy_(
            values.squeeze(-2).to(self.dtype)
        )

    def attention_states(
        self,
        layer_index: int,
        sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized attention tensors through the current position."""
        prefix_length = max(0, sequence_length - self.residual_length)
        attention_states: list[torch.Tensor] = []
        for index in range(2):
            if prefix_length:
                values = self.kv_cache[
                    index, layer_index, ..., :prefix_length, :
                ].float()
                scales = self.scales[index, layer_index, ..., :prefix_length, :].float()
                prefix = (values * scales).to(self.dtype)
            else:
                prefix = self.residual[index, layer_index, ..., :0, :]
            if sequence_length > prefix_length:
                positions = torch.arange(
                    prefix_length,
                    sequence_length,
                    device=self.device,
                )
                tail = self.residual[index, layer_index].index_select(
                    -2,
                    positions.remainder(self.residual_length),
                )
                attention_states.append(torch.cat((prefix, tail), dim=-2))
            else:
                attention_states.append(prefix)
        return attention_states[0], attention_states[1]

    def _quantize(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize each cached token with a symmetric per-head scale."""
        max_value = (
            torch.iinfo(torch.int8).max
            if self.mode == "int8"
            else torch.finfo(self.kv_cache.dtype).max
        )
        maximum = states.float().abs().amax(dim=-1, keepdim=True)
        scales = torch.where(maximum > 0, maximum / max_value, 1.0)
        normalized = states.float() / scales
        if self.mode == "int8":
            quantized = normalized.round().clamp(-127, 127).to(torch.int8)
        else:
            quantized = normalized.to(self.kv_cache.dtype)
        return quantized, scales.to(torch.float16)


def _fp8_dtype() -> Optional[torch.dtype]:
    """Return the FP8 dtype supported by the active PyTorch build."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _voxcpm_quantized_forward_step(
    attention: _VoxCPMAttention,
    hidden_states: torch.Tensor,
    position_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
    position_id: Union[int, torch.Tensor],
    cache_layer: _QuantizedStaticKVLayer,
    rotary: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Run one VoxCPM attention step against quantized prefix storage."""
    discard(position_id)
    batch_size = hidden_states.size(0)
    query_states = attention.q_proj(hidden_states)
    key_states = attention.k_proj(hidden_states)
    value_states = attention.v_proj(hidden_states)
    query_states = query_states.view(
        batch_size,
        1,
        attention.num_heads,
        attention.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size,
        1,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size,
        1,
        attention.num_key_value_heads,
        attention.head_dim,
    ).transpose(1, 2)

    if position_emb is not None:
        query_states, key_states = rotary(
            query_states,
            key_states,
            position_emb[0],
            position_emb[1],
        )

    position = cache_layer.cache.current_length - 1
    cache_layer.cache.store_step(
        cache_layer.layer_index,
        position,
        key_states,
        value_states,
    )
    key_cache, value_cache = cache_layer.cache.attention_states(
        cache_layer.layer_index,
        position + 1,
    )
    attn_mask = (
        torch.arange(key_cache.size(2), device=key_cache.device) <= position
    ).view(1, 1, 1, -1)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        attn_mask=attn_mask,
        enable_gqa=True,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        batch_size,
        attention.num_heads * attention.head_dim,
    )
    return attention.o_proj(attn_output)


def install_voxcpm_quantized_cache(model: VoxCPMModel, enabled: bool) -> bool:
    """Install compact caches on both VoxCPM autoregressive language models.

    Args:
        model: Loaded VoxCPM model whose cache should be adapted.
        enabled: Whether CUDA cache quantization is enabled.

    Returns:
        bool: Whether both supported static caches were replaced.
    """
    mode = quantized_kv_cache_mode(enabled)
    if mode is None:
        return False

    runtime = model.tts_model
    language_models = (runtime.base_lm, runtime.residual_lm)
    prepared: list[
        tuple[
            _VoxCPMLanguageModel,
            QuantizedStaticKVCache,
            list[
                tuple[
                    _VoxCPMAttention,
                    Callable[..., torch.Tensor],
                    Callable[..., tuple[torch.Tensor, torch.Tensor]],
                ]
            ],
        ]
    ] = []
    for language_model in language_models:
        original_cache = language_model.kv_cache
        storage = original_cache.kv_cache
        if len(storage.shape) != 6 or storage.shape[0] != 2:
            return False
        layers = language_model.layers
        if len(layers) != storage.shape[1]:
            return False
        attention_methods: list[
            tuple[
                _VoxCPMAttention,
                Callable[..., torch.Tensor],
                Callable[..., tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = []
        for layer in layers:
            attention = layer.self_attn
            original_method = getattr(attention, "forward_step", None)
            method_function = getattr(original_method, "__func__", None)
            if not callable(original_method) or not callable(method_function):
                return False
            globals_dict = getattr(method_function, "__globals__", None)
            if not isinstance(globals_dict, dict):
                return False
            rotary = globals_dict.get("apply_rotary_pos_emb")
            if not callable(rotary):
                return False
            attention_methods.append(
                (
                    attention,
                    cast(Callable[..., torch.Tensor], original_method),
                    cast(
                        Callable[..., tuple[torch.Tensor, torch.Tensor]],
                        rotary,
                    ),
                )
            )

        compact_cache = QuantizedStaticKVCache(
            shape=storage.shape,
            device=storage.device,
            dtype=storage.dtype,
            max_length=original_cache.max_length,
            mode=mode,
        )
        prepared.append((language_model, compact_cache, attention_methods))

    for language_model, compact_cache, attention_methods in prepared:
        language_model.kv_cache = compact_cache
        for attention, original_method, rotary in attention_methods:

            def forward_step(
                module: _VoxCPMAttention,
                hidden_states: torch.Tensor,
                position_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
                position_id: Union[int, torch.Tensor],
                kv_cache: Union[
                    tuple[torch.Tensor, torch.Tensor],
                    _QuantizedStaticKVLayer,
                ],
                *,
                _original_method: Callable[..., torch.Tensor] = original_method,
                _rotary: Callable[..., tuple[torch.Tensor, torch.Tensor]] = rotary,
            ) -> torch.Tensor:
                if not isinstance(kv_cache, _QuantizedStaticKVLayer):
                    return _original_method(
                        hidden_states=hidden_states,
                        position_emb=position_emb,
                        position_id=position_id,
                        kv_cache=kv_cache,
                    )
                return _voxcpm_quantized_forward_step(
                    module,
                    hidden_states,
                    position_emb,
                    position_id,
                    kv_cache,
                    _rotary,
                )

            attention.forward_step = MethodType(forward_step, attention)
    return True


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


class VoxCPM2(CeluneBackend[Union[VoxCPM, AudioCppRuntime]]):
    """Celune VoxCPM2 backend."""

    name: str = "voxcpm2"
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

    voice_models: Optional[Mapping[str, str]] = VOXCPM2_VOICE_MODELS

    # fallback values for packs that omit cfg_scale
    voice_cfg: Mapping[str, float] = {
        "balanced": 2.4,
        "calm": 3.0,
        "bold": 2.4,
        "upbeat": 2.4,
    }
    default_voice: Optional[str] = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
        quantize_kv_cache: bool = True,
    ) -> None:
        super().__init__(
            log=log,
            fatal=fatal,
            quantize=quantize,
            quantize_kv_cache=quantize_kv_cache,
        )
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

    def preload_models(self) -> None:
        """Prepare audio.cpp when selected, or preload native VoxCPM2 weights."""
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
            trust_remote_code=True,
        )
        runtime = getattr(model, "tts_model", None)
        if runtime is None or not available("text_tokenizer", obj=runtime):
            raise RuntimeError("VoxCPM2 did not expose a compatible text tokenizer")
        runtime.text_tokenizer = _VoxCPMTextTokenizer(tokenizer)

    to_numpy_audio = staticmethod(normalize_streamed_audio)

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

    def load_model(self, model_id: str, **kwargs) -> Union[VoxCPM, AudioCppRuntime]:
        """Load the given voice model.

        Args:
            model_id: The VoxCPM2 model repository ID to load.
            kwargs: Additional keyword arguments to use while loading VoxCPM2.

        Returns:
            Union[VoxCPM, AudioCppRuntime]: The loaded native or GGUF model.
        """
        metadata = require_model_consent(model_id)
        load_denoiser = kwargs.get("load_denoiser", False)
        if not isinstance(load_denoiser, bool):
            raise TypeError("load_denoiser must be a boolean")
        optimize = bool(kwargs.get("optimize", False))

        if self.effective_inference_mode == "audio_cpp":
            artifact = audio_cpp_model_artifact(
                self.name,
                model_id,
                quantize=self.quantization_requested,
            )
            if artifact is None:
                self.fallback_audio_cpp_to_native(
                    RuntimeError(
                        f"audio.cpp has no verified VoxCPM2 package for '{model_id}'"
                    )
                )
            elif load_denoiser or optimize:
                self.fallback_audio_cpp_to_native(
                    RuntimeError(
                        "audio.cpp cannot honor the configured VoxCPM2 load options"
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
                        sample_rate=BASE_SR,
                        torch_root=Path(torch.__file__).resolve().parent,
                        device=torch.cuda.current_device(),
                        log=self.log,
                        report_progress=self.report_progress,
                        default_request_options={"retry_badcase": False},
                    )
                except Exception as error:
                    self.fallback_audio_cpp_to_native(error)
                else:
                    self.quantization_attempted = self.quantization_requested
                    self.quantization_active = artifact.quantization == "Q8_0"
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
        quantize_on_cpu = getattr(self, "quantization_mode", None) is not None
        load_optimize = optimize and not quantize_on_cpu

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

        with (
            local_hf_offline_mode(),
            huggingface_progress(self.report_progress),
            self._suppress_backend_output(),
        ):
            if quantize_on_cpu:
                native_model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=load_denoiser,
                    optimize=load_optimize,
                    device="cpu",
                )
            else:
                native_model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=load_denoiser,
                    optimize=load_optimize,
                )
            self.model = native_model
            self._install_checkpoint_tokenizer(native_model, path)
        self._resize_runtime_caches(native_model)
        native_model = cast(
            VoxCPM,
            self.apply_runtime_quantization(native_model, model_id),
        )
        self.model = native_model
        if quantize_on_cpu:
            self._move_quantized_runtime_to_cuda(native_model)
            if optimize:
                native_model.tts_model.optimize()
                native_model.tts_model.generate(
                    target_text="Hello, this is the first test sentence.",
                    max_len=10,
                )
        install_voxcpm_quantized_cache(
            cast(VoxCPMModel, native_model),
            self.quantize_kv_cache,
        )
        return native_model

    @staticmethod
    def _runtime_cache_length(max_length: int) -> int:
        """Bound VoxCPM's static cache to the supported generation envelope."""
        return min(max_length, _VOXCPM_RUNTIME_KV_CACHE_LENGTH)

    @classmethod
    def _resize_runtime_caches(cls, model: VoxCPM) -> None:
        """Release VoxCPM's oversized load-time caches before inference."""
        runtime = getattr(model, "tts_model", None)
        if runtime is None:
            return
        for language_model in (runtime.base_lm, runtime.residual_lm):
            cache = language_model.kv_cache
            if cache is None:
                continue
            max_length = getattr(cache, "max_length", None)
            if not isinstance(max_length, int) or max_length <= 0:
                continue
            target_length = cls._runtime_cache_length(max_length)
            if target_length == max_length:
                continue
            storage = cache.kv_cache
            language_model.setup_cache(
                1,
                target_length,
                storage.device,
                storage.dtype,
            )

    @classmethod
    def _move_quantized_runtime_to_cuda(cls, model: VoxCPM) -> None:
        """Move a CPU-quantized VoxCPM runtime to CUDA and rebuild its caches."""
        runtime = model.tts_model
        runtime.to(torch.device("cuda"))
        runtime.device = "cuda"
        runtime.config.device = "cuda"
        for language_model in (runtime.base_lm, runtime.residual_lm):
            cache = language_model.kv_cache
            if cache is None:
                continue
            language_model.setup_cache(
                1,
                cls._runtime_cache_length(cache.max_length),
                torch.device("cuda"),
                torch.bfloat16,
            )

    def generate_stream(
        self, model: Union[VoxCPM, AudioCppRuntime], **kwargs
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded native or audio.cpp VoxCPM2 model.
            kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterator[tuple[AudioChunk, int, Optional[dict]]]: An iterator of ``(audio, sample_rate, timing)`` tuples
            suitable for Celune's playback pipeline.

        Raises:
            ValueError: The requested voice is unknown or input text is empty.
            NotImplementedError: If streaming support is unavailable.
        """
        if isinstance(model, AudioCppRuntime):
            yield from self._generate_audio_cpp_stream(model, kwargs)
            return

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

        if not available("generate_streaming", obj=model):
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
                    audio = self.to_numpy_audio(chunk)
                    if audio.size == 0:
                        continue
                    if pending_audio is not None:
                        total_steps += pending_steps
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic()
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
                        pending_audio = None
                        pending_steps = 0

                    batch.append(audio)

                    if len(batch) < chunks_per_batch:
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
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
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
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
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
            if stream is not None and available("close", obj=stream):
                with contextlib.suppress(Exception):
                    stream.close()

    def _generate_audio_cpp_stream(
        self,
        model: AudioCppRuntime,
        kwargs: dict[str, BackendArgumentValue],
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Stream VoxCPM2 GGUF audio using Celune's voice reference and controls."""
        voice_value = kwargs.pop("voice", self.default_voice)
        voice = (
            voice_value
            if isinstance(voice_value, str)
            else self.default_voice or TTS_DEFAULT_VOICE
        )
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        kwargs.pop("chunk_size", None)
        for option in ("temperature", "top_k", "top_p", "repetition_penalty"):
            kwargs.pop(option, None)
        kwargs.pop("max_new_tokens", None)
        kwargs.pop("seed", None)
        text_value = kwargs.pop("text", None)
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError("expected text to say")

        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, _ = compatible_bundle
        try:
            reference_path = self._truncate_reference(loader.materialize(voice, "wav"))
            voice_metadata = loader.bundle.voices[voice]
            configured_cfg = voice_metadata.get("cfg_scale")
            configured_ref_text = voice_metadata.get("reference_text")
        except KeyError as error:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from error
        cfg = (
            float(configured_cfg)
            if isinstance(configured_cfg, (int, float))
            and not isinstance(configured_cfg, bool)
            else self.voice_cfg.get(
                voice, self.voice_cfg[self.default_voice or "balanced"]
            )
        )
        reference_text = (
            configured_ref_text.strip() if isinstance(configured_ref_text, str) else ""
        )
        if instruct:
            text_value = f"({instruct}) {text_value}"

        self._apply_seed()
        options: dict[str, object] = {
            "retry_badcase": False,
            "guidance_scale": cfg,
            "num_inference_steps": 4,
            "max_tokens": self.max_new_tokens,
        }
        if self.current_seed is not None:
            options["seed"] = self.current_seed
        request_body: dict[str, object] = {
            "model": model.server_model_id,
            "input": text_value,
            "voice_ref": reference_path.resolve().as_posix(),
            "response_format": "pcm",
            "stream_format": "sse",
            "options": options,
        }
        if reference_text:
            request_body["reference_text"] = reference_text
        yield from model.stream_speech(request_body, default_sample_rate=BASE_SR)
