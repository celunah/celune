# SPDX-License-Identifier: Apache-2.0
"""Model loading and attention cache helpers for Celune."""

from __future__ import annotations

# noinspection PyProtectedMember
from collections.abc import Mapping, Callable
from typing import TYPE_CHECKING, Any, Union, Literal, Optional, cast

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .i18n import string
from .vram import resolve_vram_preset
from .backends.tts import CeluneBackend
from .paths import huggingface_progress
from .typing.common import JSONSerializable
from .registry.other import NORMALIZER_MODEL_ID
from .registry.consent import require_model_consent

try:
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
except ImportError:
    LinearAttentionCacheLayerMixin = CacheLayerMixin

if TYPE_CHECKING:
    from transformers.configuration_utils import PreTrainedConfig

NORMALIZER_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<NORM>")
NORMALIZER_DEVICE = "cpu"


def normalizer_device(
    config: Optional[Mapping[str, JSONSerializable]] = None,
) -> str:
    """Return the runtime device used for CeluneNorm loading.

    Args:
        config: Celune's current configuration.

    Returns:
        str: The selected normalizer device based on current VRAM preset.
    """
    if config is None:
        return NORMALIZER_DEVICE
    return resolve_vram_preset(config).normalizer_device


def load_normalizer_components(
    log: Callable[[str, str], None],
    backend: Union[CeluneBackend, type[CeluneBackend]],
    config: Optional[Mapping[str, JSONSerializable]] = None,
    progress_callback: Optional[
        Callable[[Optional[float], Optional[float]], None]
    ] = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load CeluneNorm and return its tokenizer and model.

    Args:
        log: Logging callback used to report cache and loading progress.
        backend: Backend type or instance used to resolve model cache helpers.
        config: Celune configuration used to resolve the target device.
        progress_callback: Callback receiving Hugging Face transfer progress.

    Returns:
        tuple[PreTrainedTokenizerBase, PreTrainedModel]: The loaded tokenizer and causal language model.
    """
    if callable(backend):
        backend_obj = backend(log=log)
    else:
        backend_obj = backend

    metadata = require_model_consent(NORMALIZER_MODEL_ID)
    available, path = backend_obj.model_is_available_locally(NORMALIZER_MODEL_ID)
    model_ref = path if available and path is not None else NORMALIZER_MODEL_ID
    model_options: dict[str, str] = {}
    if model_ref == NORMALIZER_MODEL_ID:
        model_options["revision"] = metadata.revision

    if available:
        log(string("modeling.normalizer_cached"), "info")

    with huggingface_progress(progress_callback):
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref,
            extra_special_tokens=list(NORMALIZER_SPECIAL_TOKENS),
            **model_options,
        )

        if tokenizer is None:
            raise RuntimeError("celunenorm tokenizer not available")

        device = normalizer_device(config)
        supported_dispatch = {"auto", "balanced", "balanced_low_0", "sequential"}
        device_map = device if device in supported_dispatch else {"": device}
        llm = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=torch.bfloat16,
            device_map=device_map,
            **model_options,
        )
    return tokenizer, llm


QuantizedKVCacheMode = Literal["int8", "fp8"]

_DEFAULT_RESIDUAL_LENGTH = 128


def _fp8_dtype() -> Optional[torch.dtype]:
    """Return the FP8 dtype supported by the active PyTorch build."""
    dtype = getattr(torch, "float8_e4m3fn", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def quantized_kv_cache_mode(enabled: bool) -> Optional[QuantizedKVCacheMode]:
    """Select the cache storage format for the active CUDA device.

    Args:
        enabled: Whether quantized cache storage was requested.

    Returns:
        Optional[QuantizedKVCacheMode]: ``int8`` on Ampere, ``fp8`` on Ada and
            newer GPUs, or ``None`` when the runtime cannot use a quantized cache.
    """
    if not enabled or not torch.cuda.is_available():
        return None

    try:
        major, minor = torch.cuda.get_device_capability()
    except RuntimeError:
        return None

    if major < 8:
        return None
    if major > 8 or minor >= 9:
        return "fp8" if _fp8_dtype() is not None else None
    return "int8"


def _cache_layer_types(config: PreTrainedConfig) -> Optional[list[str]]:
    """Return the cache layer types supported by the compact cache."""
    decoder_config = config.get_text_config(decoder=True)
    layer_count = getattr(decoder_config, "num_hidden_layers", None)
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        return None

    layer_types = getattr(decoder_config, "layer_types", None)
    if layer_types is None:
        if getattr(decoder_config, "sliding_window", None) is not None:
            return None
        if getattr(decoder_config, "attention_chunk_size", None) is not None:
            return None
        return ["full_attention"] * layer_count

    if not isinstance(layer_types, (list, tuple)):
        return None
    normalized = [str(layer_type) for layer_type in layer_types]
    if len(normalized) != layer_count or any(
        layer_type != "full_attention" for layer_type in normalized
    ):
        return None
    return normalized


class QuantizedKVCacheLayer(CacheLayerMixin):
    """One attention layer with a quantized prefix and compute-dtype tail."""

    is_sliding = False

    def __init__(
        self,
        mode: QuantizedKVCacheMode,
        residual_length: int,
    ) -> None:
        super().__init__()
        if residual_length <= 0:
            raise ValueError("KV cache residual length must be positive")
        self.mode = mode
        self.residual_length = residual_length
        self.quantized_dtype = torch.int8 if mode == "int8" else _fp8_dtype()
        if self.quantized_dtype is None:
            raise RuntimeError("FP8 KV cache storage is unavailable")
        self.cumulative_length = 0
        self.batch_size = -1
        self.num_heads = 0
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self._key_head_dim = 0
        self._value_head_dim = 0
        self._scale_dtype = torch.float16
        self._quantized_keys: Optional[torch.Tensor] = None
        self._quantized_values: Optional[torch.Tensor] = None
        self._key_scales: Optional[torch.Tensor] = None
        self._value_scales: Optional[torch.Tensor] = None

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize metadata and empty storage from the first attention states."""
        if value_states is None:
            value_states = key_states
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = key_states.shape[0]
        self.num_heads = key_states.shape[1]
        self._key_head_dim = key_states.shape[-1]
        self._value_head_dim = value_states.shape[-1]
        self.keys = key_states[..., :0, :].contiguous()
        self.values = value_states[..., :0, :].contiguous()
        self._quantized_keys = self._empty_quantized(key_states)
        self._quantized_values = self._empty_quantized(value_states)
        self._key_scales = self._empty_scales(key_states)
        self._value_scales = self._empty_scales(value_states)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new states and return the dequantized attention view."""
        del args, kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if key_states.dtype != self.dtype:
            key_states = key_states.to(self.dtype)
        if value_states.dtype != self.dtype:
            value_states = value_states.to(self.dtype)

        self.cumulative_length += key_states.shape[-2]
        residual_keys, residual_values = self._residual_states()
        combined_keys = torch.cat((residual_keys, key_states), dim=-2)
        combined_values = torch.cat((residual_values, value_states), dim=-2)
        overflow = max(0, combined_keys.shape[-2] - self.residual_length)
        if overflow:
            self._append_quantized(
                combined_keys[..., :overflow, :],
                combined_values[..., :overflow, :],
            )
            self.keys = combined_keys[..., overflow:, :].contiguous()
            self.values = combined_values[..., overflow:, :].contiguous()
        else:
            self.keys = combined_keys
            self.values = combined_values

        return self.dequantized_cache()

    def get_mask_sizes(
        self,
        query_length: Union[int, torch.Tensor],
    ) -> tuple[int, int]:
        """Return the attention mask dimensions for the current cache length."""
        query_length = (
            query_length.shape[0]
            if isinstance(query_length, torch.Tensor)
            else query_length
        )
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        """Return the number of tokens represented by this layer."""
        return self.cumulative_length

    def get_max_length(self) -> int:
        """Return ``-1`` because the cache grows with the generation request."""
        return -1

    def get_max_cache_shape(self) -> int:
        """Return ``-1`` because this cache has no fixed maximum length."""
        return -1

    def crop(self, max_length: int) -> None:
        """Crop the cache while retaining its compact representation."""
        if max_length <= 0:
            max_length = max(0, self.cumulative_length - abs(max_length))
        if self.cumulative_length <= max_length:
            return

        quantized_length = self._quantized_length()
        residual_keys, residual_values = self._residual_states()
        if max_length <= quantized_length:
            assert self._quantized_keys is not None
            assert self._quantized_values is not None
            assert self._key_scales is not None
            assert self._value_scales is not None
            self._quantized_keys = self._quantized_keys[
                ..., :max_length, :
            ].contiguous()
            self._quantized_values = self._quantized_values[
                ..., :max_length, :
            ].contiguous()
            self._key_scales = self._key_scales[..., :max_length, :].contiguous()
            self._value_scales = self._value_scales[..., :max_length, :].contiguous()
            self.keys = residual_keys[..., :0, :].contiguous()
            self.values = residual_values[..., :0, :].contiguous()
        else:
            residual_length = max_length - quantized_length
            self.keys = residual_keys[..., :residual_length, :].contiguous()
            self.values = residual_values[..., :residual_length, :].contiguous()
        self.cumulative_length = max_length

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat cached sequences along the batch dimension."""
        if not self.is_initialized:
            return
        residual_keys, residual_values = self._residual_states()
        self.keys = residual_keys.repeat_interleave(repeats, dim=0)
        self.values = residual_values.repeat_interleave(repeats, dim=0)
        if self._quantized_keys is not None:
            self._quantized_keys = self._quantized_keys.repeat_interleave(
                repeats, dim=0
            )
        if self._quantized_values is not None:
            self._quantized_values = self._quantized_values.repeat_interleave(
                repeats, dim=0
            )
        if self._key_scales is not None:
            self._key_scales = self._key_scales.repeat_interleave(repeats, dim=0)
        if self._value_scales is not None:
            self._value_scales = self._value_scales.repeat_interleave(repeats, dim=0)
        self.batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select cached sequences along the batch dimension."""
        if not self.is_initialized:
            return
        residual_keys, residual_values = self._residual_states()
        self.keys = residual_keys[indices, ...]
        self.values = residual_values[indices, ...]
        if self._quantized_keys is not None:
            self._quantized_keys = self._quantized_keys[indices, ...]
        if self._quantized_values is not None:
            self._quantized_values = self._quantized_values[indices, ...]
        if self._key_scales is not None:
            self._key_scales = self._key_scales[indices, ...]
        if self._value_scales is not None:
            self._value_scales = self._value_scales[indices, ...]
        self.batch_size = indices.shape[0]

    def reset(self) -> None:
        """Discard cached tokens while retaining initialized metadata."""
        if not self.is_initialized:
            return
        residual_keys, residual_values = self._residual_states()
        self.keys = residual_keys[..., :0, :].contiguous()
        self.values = residual_values[..., :0, :].contiguous()
        if self._quantized_keys is not None:
            self._quantized_keys = self._quantized_keys.new_empty(
                (*self._quantized_keys.shape[:-2], 0, self._quantized_keys.shape[-1])
            )
        if self._quantized_values is not None:
            self._quantized_values = self._quantized_values.new_empty(
                (
                    *self._quantized_values.shape[:-2],
                    0,
                    self._quantized_values.shape[-1],
                )
            )
        if self._key_scales is not None:
            self._key_scales = self._key_scales.new_empty(
                (*self._key_scales.shape[:-2], 0, self._key_scales.shape[-1])
            )
        if self._value_scales is not None:
            self._value_scales = self._value_scales.new_empty(
                (
                    *self._value_scales.shape[:-2],
                    0,
                    self._value_scales.shape[-1],
                )
            )
        self.cumulative_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cached sequences for beam-style generation."""
        self.batch_select_indices(beam_idx)

    def offload(self) -> None:
        """Move every cache representation to CPU."""
        if not self.is_initialized:
            return
        residual_keys, residual_values = self._residual_states()
        self.keys = residual_keys.to("cpu", non_blocking=True)
        self.values = residual_values.to("cpu", non_blocking=True)
        if self._quantized_keys is not None:
            self._quantized_keys = self._quantized_keys.to("cpu", non_blocking=True)
        if self._quantized_values is not None:
            self._quantized_values = self._quantized_values.to("cpu", non_blocking=True)
        if self._key_scales is not None:
            self._key_scales = self._key_scales.to("cpu", non_blocking=True)
        if self._value_scales is not None:
            self._value_scales = self._value_scales.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        """Move the cache representations back to their model device."""
        if not self.is_initialized:
            return
        residual_keys, residual_values = self._residual_states()
        if residual_keys.device == self.device:
            return
        self.keys = residual_keys.to(self.device, non_blocking=True)
        self.values = residual_values.to(self.device, non_blocking=True)
        if self._quantized_keys is not None:
            self._quantized_keys = self._quantized_keys.to(
                self.device, non_blocking=True
            )
        if self._quantized_values is not None:
            self._quantized_values = self._quantized_values.to(
                self.device, non_blocking=True
            )
        if self._key_scales is not None:
            self._key_scales = self._key_scales.to(self.device, non_blocking=True)
        if self._value_scales is not None:
            self._value_scales = self._value_scales.to(self.device, non_blocking=True)

    def _empty_quantized(self, states: torch.Tensor) -> torch.Tensor:
        """Return empty quantized storage matching one state tensor."""
        return torch.empty(
            (*states.shape[:-2], 0, states.shape[-1]),
            dtype=cast(torch.dtype, self.quantized_dtype),
            device=states.device,
        )

    def _empty_scales(self, states: torch.Tensor) -> torch.Tensor:
        """Return empty per-token scales matching one state tensor."""
        return torch.empty(
            (*states.shape[:-2], 0, 1),
            dtype=self._scale_dtype,
            device=states.device,
        )

    def _append_quantized(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Quantize and append a promoted residual segment."""
        key_values = self._quantize(key_states)
        value_values = self._quantize(value_states)
        assert self._quantized_keys is not None
        assert self._quantized_values is not None
        assert self._key_scales is not None
        assert self._value_scales is not None
        quantized_keys, key_scales = key_values
        quantized_values, value_scales = value_values
        self._quantized_keys = torch.cat((self._quantized_keys, quantized_keys), dim=-2)
        self._quantized_values = torch.cat(
            (self._quantized_values, quantized_values), dim=-2
        )
        self._key_scales = torch.cat((self._key_scales, key_scales), dim=-2)
        self._value_scales = torch.cat((self._value_scales, value_scales), dim=-2)

    def _quantize(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize states with one symmetric scale per cached token."""
        max_value = (
            torch.iinfo(torch.int8).max
            if self.mode == "int8"
            else torch.finfo(cast(torch.dtype, self.quantized_dtype)).max
        )
        maximum = states.float().abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            maximum > 0,
            maximum / max_value,
            torch.ones_like(maximum),
        )
        normalized = states.float() / scale
        if self.mode == "int8":
            quantized = normalized.round().clamp(-127, 127).to(torch.int8)
        else:
            quantized = normalized.to(cast(torch.dtype, self.quantized_dtype))
        return quantized, scale.to(self._scale_dtype)

    def dequantized_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the complete cache in the model's compute dtype."""
        residual_keys, residual_values = self._residual_states()
        prefix_keys = self._dequantize(
            self._quantized_keys,
            self._key_scales,
            self._key_head_dim,
        )
        prefix_values = self._dequantize(
            self._quantized_values,
            self._value_scales,
            self._value_head_dim,
        )
        if prefix_keys.shape[-2] == 0:
            return residual_keys, residual_values
        return (
            torch.cat((prefix_keys, residual_keys), dim=-2),
            torch.cat((prefix_values, residual_values), dim=-2),
        )

    def _dequantize(
        self,
        quantized: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        head_dim: int,
    ) -> torch.Tensor:
        """Dequantize one compact prefix for the attention operation."""
        if quantized is None or scales is None or quantized.shape[-2] == 0:
            residual_keys, _ = self._residual_states()
            return torch.empty(
                (*residual_keys.shape[:-2], 0, head_dim),
                dtype=self.dtype,
                device=self.device,
            )
        return (quantized.float() * scales.float()).to(self.dtype)

    def _residual_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the initialized full-precision residual tensors."""
        if self.keys is None or self.values is None:
            raise RuntimeError("KV cache layer is not initialized")
        return self.keys, self.values

    def _quantized_length(self) -> int:
        """Return the number of tokens in the quantized prefix."""
        if self._quantized_keys is None:
            return 0
        return self._quantized_keys.shape[-2]


class QuantizedKVCache(Cache):
    """A Transformers cache using INT8 or FP8 prefix storage."""

    def __init__(
        self,
        config: PreTrainedConfig,
        mode: QuantizedKVCacheMode,
        residual_length: int = _DEFAULT_RESIDUAL_LENGTH,
    ) -> None:
        layer_types = _cache_layer_types(config)
        if layer_types is None:
            raise ValueError("model cache layout is not supported")
        layers = cast(
            list[CacheLayerMixin],
            [QuantizedKVCacheLayer(mode, residual_length) for _ in layer_types],
        )
        super().__init__(
            layers=cast(
                list[Union[CacheLayerMixin, LinearAttentionCacheLayerMixin]],
                layers,
            )
        )

    def __iter__(self):
        """Expose dequantized tuples for compatibility consumers."""
        for layer in self.layers:
            if not isinstance(layer, QuantizedKVCacheLayer):
                raise TypeError("unexpected quantized cache layer")
            if not layer.is_initialized:
                yield layer.keys, layer.values
                continue
            keys, values = layer.dequantized_cache()
            yield keys, values

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the complete dequantized state for a compatibility consumer."""
        if layer_idx >= len(self.layers):
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, "
                f"attempted to access layer with index {layer_idx}"
            )
        layer = self.layers[layer_idx]
        if not isinstance(layer, QuantizedKVCacheLayer):
            raise TypeError("unexpected quantized cache layer")
        if not layer.is_initialized:
            return cast(tuple[torch.Tensor, torch.Tensor], (layer.keys, layer.values))
        return layer.dequantized_cache()

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert each layer to full-precision legacy key/value tensors."""
        return cast(
            tuple[tuple[torch.Tensor, torch.Tensor], ...],
            tuple(self),
        )


class _GraphQuantizedKVWorkspace:
    """Reusable dense attention buffers shared by graph-cache layers."""

    def __init__(self, max_cache_len: int) -> None:
        self.max_cache_len = max_cache_len
        self.positions: Optional[torch.Tensor] = None
        self.ring_positions: Optional[torch.Tensor] = None
        self.key_states: Optional[torch.Tensor] = None
        self.value_states: Optional[torch.Tensor] = None

    def initialize(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        residual_length: int,
    ) -> None:
        """Allocate fixed-address attention buffers from the first layer shape."""
        if self.key_states is not None:
            return
        self.key_states = torch.zeros(
            (*key_states.shape[:-2], self.max_cache_len, key_states.shape[-1]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.value_states = torch.zeros(
            (*value_states.shape[:-2], self.max_cache_len, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.positions = torch.arange(self.max_cache_len, device=key_states.device)
        self.ring_positions = self.positions.remainder(residual_length)


class GraphQuantizedKVCacheLayer(QuantizedKVCacheLayer):
    """Fixed-capacity quantized cache layer compatible with CUDA graph replay."""

    is_compileable = True

    def __init__(
        self,
        mode: QuantizedKVCacheMode,
        max_cache_len: int,
        residual_length: int,
        workspace: _GraphQuantizedKVWorkspace,
    ) -> None:
        residual_length = min(residual_length, max_cache_len)
        super().__init__(mode=mode, residual_length=residual_length)
        self.max_cache_len = max_cache_len
        self.prefix_capacity = max_cache_len - residual_length
        self.workspace = workspace
        self.cumulative_length = 0
        self.max_batch_size = -1

    def lazy_initialization(
        self,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Allocate quantized storage and initialize shared attention buffers."""
        if value_states is None:
            value_states = key_states
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.batch_size = key_states.shape[0]
        self.max_batch_size = self.batch_size
        self.num_heads = key_states.shape[1]
        self._key_head_dim = key_states.shape[-1]
        self._value_head_dim = value_states.shape[-1]
        self.keys = torch.zeros(
            (*key_states.shape[:-2], self.residual_length, key_states.shape[-1]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.zeros(
            (*value_states.shape[:-2], self.residual_length, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        quantized_dtype = cast(torch.dtype, self.quantized_dtype)
        self._quantized_keys = torch.zeros(
            (*key_states.shape[:-2], self.prefix_capacity, key_states.shape[-1]),
            dtype=quantized_dtype,
            device=key_states.device,
        )
        self._quantized_values = torch.zeros(
            (*value_states.shape[:-2], self.prefix_capacity, value_states.shape[-1]),
            dtype=quantized_dtype,
            device=value_states.device,
        )
        self._key_scales = torch.ones(
            (*key_states.shape[:-2], self.prefix_capacity, 1),
            dtype=self._scale_dtype,
            device=key_states.device,
        )
        self._value_scales = torch.ones(
            (*value_states.shape[:-2], self.prefix_capacity, 1),
            dtype=self._scale_dtype,
            device=value_states.device,
        )
        self.workspace.initialize(key_states, value_states, self.residual_length)
        self.is_initialized = True

    def _index_copy_quantized(
        self,
        destination: torch.Tensor,
        positions: torch.Tensor,
        source: torch.Tensor,
    ) -> None:
        """Copy quantized values without relying on CUDA FP8 index-copy support."""
        if self.mode == "fp8":
            destination.view(torch.uint8).index_copy_(
                2,
                positions,
                source.view(torch.uint8),
            )
            return
        destination.index_copy_(2, positions, source)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update fixed storage and return a graph-stable attention view."""
        cache_kwargs_value: object = args[0] if args else kwargs.get("cache_kwargs")
        cache_kwargs = (
            cast(dict[str, Any], cache_kwargs_value)
            if isinstance(cache_kwargs_value, dict)
            else None
        )
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        keys = cast(torch.Tensor, self.keys)
        values = cast(torch.Tensor, self.values)
        if key_states.dtype != self.dtype:
            key_states = key_states.to(self.dtype)
        if value_states.dtype != self.dtype:
            value_states = value_states.to(self.dtype)

        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if key_states.shape[-2] != 1:
            return self._update_prefill(key_states, value_states, cache_position)
        if cache_position is None:
            raise ValueError("graph cache decode requires cache_position")

        position = cache_position.reshape(-1)
        if self.prefix_capacity:
            promote_position = (position - self.residual_length).clamp(min=0)
            promote_index = promote_position.remainder(self.residual_length)
            promoted_keys = keys.index_select(2, promote_index)
            promoted_values = values.index_select(2, promote_index)
            quantized_keys, key_scales = self._quantize(promoted_keys)
            quantized_values, value_scales = self._quantize(promoted_values)
            promote_valid = (position >= self.residual_length).view(1, 1, -1, 1)
            quantized_keys = torch.where(
                promote_valid, quantized_keys, torch.zeros_like(quantized_keys)
            )
            quantized_values = torch.where(
                promote_valid, quantized_values, torch.zeros_like(quantized_values)
            )
            key_scales = torch.where(
                promote_valid, key_scales, torch.ones_like(key_scales)
            )
            value_scales = torch.where(
                promote_valid, value_scales, torch.ones_like(value_scales)
            )
            quantized_position = promote_position.clamp(max=self.prefix_capacity - 1)
            quantized_keys_storage = cast(torch.Tensor, self._quantized_keys)
            quantized_values_storage = cast(torch.Tensor, self._quantized_values)
            key_scales_storage = cast(torch.Tensor, self._key_scales)
            value_scales_storage = cast(torch.Tensor, self._value_scales)
            self._index_copy_quantized(
                quantized_keys_storage,
                quantized_position,
                quantized_keys,
            )
            self._index_copy_quantized(
                quantized_values_storage,
                quantized_position,
                quantized_values,
            )
            key_scales_storage.index_copy_(2, quantized_position, key_scales)
            value_scales_storage.index_copy_(2, quantized_position, value_scales)

        tail_position = position.remainder(self.residual_length)
        keys.index_copy_(2, tail_position, key_states)
        values.index_copy_(2, tail_position, value_states)
        self.cumulative_length += key_states.shape[-2]
        return self._attention_view(position[-1])

    def get_mask_sizes(
        self,
        query_length: Union[int, torch.Tensor],
    ) -> tuple[int, int]:
        """Return the fixed attention length used by the graph cache."""
        del query_length
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        """Return the most recently populated sequence length."""
        return self.cumulative_length

    def get_max_length(self) -> int:
        """Return the fixed graph cache capacity."""
        return self.max_cache_len

    def get_max_cache_shape(self) -> int:
        """Return the fixed graph cache capacity."""
        return self.max_cache_len

    def reset(self) -> None:
        """Clear cache storage without reallocating graph-captured tensors."""
        if not self.is_initialized:
            return
        cast(torch.Tensor, self.keys).zero_()
        cast(torch.Tensor, self.values).zero_()
        if self._quantized_keys is not None:
            self._quantized_keys.zero_()
        if self._quantized_values is not None:
            self._quantized_values.zero_()
        if self._key_scales is not None:
            self._key_scales.fill_(1)
        if self._value_scales is not None:
            self._value_scales.fill_(1)
        self.cumulative_length = 0

    def copy_from(self, source: QuantizedKVCacheLayer) -> int:
        """Copy a dynamically quantized prompt into fixed graph storage."""
        if not source.is_initialized:
            return 0
        if source.mode != self.mode:
            raise ValueError("graph and dynamic cache modes must match")
        source_keys, source_values = source._residual_states()
        if not self.is_initialized:
            self.lazy_initialization(source_keys, source_values)
        target_keys = cast(torch.Tensor, self.keys)
        target_values = cast(torch.Tensor, self.values)
        sequence_length = source.get_seq_length()
        if sequence_length > self.max_cache_len:
            raise ValueError("prompt exceeds the graph cache capacity")

        prefix_length = source._quantized_length()
        if prefix_length > self.prefix_capacity:
            raise ValueError("quantized prompt prefix exceeds graph cache capacity")
        if prefix_length:
            assert self._quantized_keys is not None
            assert self._quantized_values is not None
            assert self._key_scales is not None
            assert self._value_scales is not None
            assert source._quantized_keys is not None
            assert source._quantized_values is not None
            assert source._key_scales is not None
            assert source._value_scales is not None
            self._quantized_keys[..., :prefix_length, :].copy_(source._quantized_keys)
            self._quantized_values[..., :prefix_length, :].copy_(
                source._quantized_values
            )
            self._key_scales[..., :prefix_length, :].copy_(source._key_scales)
            self._value_scales[..., :prefix_length, :].copy_(source._value_scales)

        tail_length = source_keys.shape[-2]
        if tail_length:
            positions = torch.arange(
                prefix_length,
                prefix_length + tail_length,
                device=self.device,
            ).remainder(self.residual_length)
            target_keys.index_copy_(2, positions, source_keys)
            target_values.index_copy_(2, positions, source_values)
        self.cumulative_length = sequence_length
        return sequence_length

    def _update_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Populate fixed storage from a multi-token prefill operation."""
        sequence_length = key_states.shape[-2]
        if sequence_length > self.max_cache_len:
            raise ValueError("prompt exceeds the graph cache capacity")
        self.reset()
        keys = cast(torch.Tensor, self.keys)
        values = cast(torch.Tensor, self.values)
        prefix_length = max(0, sequence_length - self.residual_length)
        if prefix_length:
            quantized_keys, key_scales = self._quantize(
                key_states[..., :prefix_length, :]
            )
            quantized_values, value_scales = self._quantize(
                value_states[..., :prefix_length, :]
            )
            assert self._quantized_keys is not None
            assert self._quantized_values is not None
            assert self._key_scales is not None
            assert self._value_scales is not None
            self._quantized_keys[..., :prefix_length, :].copy_(quantized_keys)
            self._quantized_values[..., :prefix_length, :].copy_(quantized_values)
            self._key_scales[..., :prefix_length, :].copy_(key_scales)
            self._value_scales[..., :prefix_length, :].copy_(value_scales)

        tail_keys = key_states[..., prefix_length:, :]
        tail_values = value_states[..., prefix_length:, :]
        if tail_keys.shape[-2]:
            if cache_position is None:
                positions = torch.arange(
                    prefix_length, sequence_length, device=self.device
                )
            else:
                positions = cache_position[-tail_keys.shape[-2] :]
            positions = positions.remainder(self.residual_length)
            keys.index_copy_(2, positions, tail_keys)
            values.index_copy_(2, positions, tail_values)
        self.cumulative_length = sequence_length
        return self._attention_view(
            torch.tensor(sequence_length - 1, device=self.device)
        )

    def _attention_view(
        self,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize into one reusable dense workspace for attention."""
        if (
            self.workspace.positions is None
            or self.workspace.ring_positions is None
            or self.workspace.key_states is None
            or self.workspace.value_states is None
        ):
            raise RuntimeError("graph cache workspace is not initialized")
        keys = cast(torch.Tensor, self.keys)
        values = cast(torch.Tensor, self.values)
        key_states = keys.index_select(2, self.workspace.ring_positions)
        value_states = values.index_select(2, self.workspace.ring_positions)
        tail_start = position - self.residual_length + 1
        valid_tail = (self.workspace.positions >= tail_start) & (
            self.workspace.positions <= position
        )
        key_states.masked_fill_(~valid_tail.view(1, 1, -1, 1), 0)
        value_states.masked_fill_(~valid_tail.view(1, 1, -1, 1), 0)
        self.workspace.key_states.copy_(key_states)
        self.workspace.value_states.copy_(value_states)

        if self.prefix_capacity:
            assert self._quantized_keys is not None
            assert self._quantized_values is not None
            assert self._key_scales is not None
            assert self._value_scales is not None
            prefix_keys = self._dequantize(
                self._quantized_keys,
                self._key_scales,
                self._key_head_dim,
            )
            prefix_values = self._dequantize(
                self._quantized_values,
                self._value_scales,
                self._value_head_dim,
            )
            self.workspace.key_states[..., : self.prefix_capacity, :].copy_(prefix_keys)
            self.workspace.value_states[..., : self.prefix_capacity, :].copy_(
                prefix_values
            )
        return self.workspace.key_states, self.workspace.value_states


class GraphQuantizedKVCache(Cache):
    """Fixed-capacity quantized cache used by Qwen3 CUDA graph decoding."""

    def __init__(
        self,
        config: PreTrainedConfig,
        mode: QuantizedKVCacheMode,
        max_cache_len: int,
        residual_length: int = _DEFAULT_RESIDUAL_LENGTH,
    ) -> None:
        layer_types = _cache_layer_types(config)
        if layer_types is None:
            raise ValueError("model cache layout is not supported")
        workspace = _GraphQuantizedKVWorkspace(max_cache_len)
        layers = cast(
            list[Union[CacheLayerMixin, LinearAttentionCacheLayerMixin]],
            [
                GraphQuantizedKVCacheLayer(
                    mode=mode,
                    max_cache_len=max_cache_len,
                    residual_length=residual_length,
                    workspace=workspace,
                )
                for _ in layer_types
            ],
        )
        super().__init__(layers=layers)
        self.workspace = workspace

    def copy_from(self, source: QuantizedKVCache) -> int:
        """Transfer every dynamically quantized prompt layer into graph storage."""
        if len(source.layers) != len(self.layers):
            raise ValueError("dynamic and graph cache layer counts must match")
        self.reset()
        sequence_length = 0
        for target, original in zip(self.layers, source.layers):
            if not isinstance(target, GraphQuantizedKVCacheLayer):
                raise TypeError("unexpected graph cache layer")
            if not isinstance(original, QuantizedKVCacheLayer):
                raise TypeError("unexpected dynamic quantized cache layer")
            sequence_length = max(sequence_length, target.copy_from(original))
        return sequence_length

    def reset(self) -> None:
        """Reset quantized storage and the shared attention workspace."""
        super().reset()
        if self.workspace.key_states is not None:
            self.workspace.key_states.zero_()
        if self.workspace.value_states is not None:
            self.workspace.value_states.zero_()


def create_graph_quantized_kv_cache(
    config: PreTrainedConfig,
    device: torch.device,
    max_cache_len: int,
    enabled: bool,
) -> Optional[GraphQuantizedKVCache]:
    """Create graph-compatible KV storage for supported full-attention CUDA models."""
    mode = quantized_kv_cache_mode(enabled)
    if mode is None or device.type != "cuda":
        return None
    if max_cache_len <= 0:
        return None
    try:
        return GraphQuantizedKVCache(
            config=config,
            mode=mode,
            max_cache_len=max_cache_len,
        )
    except ValueError:
        return None


def create_quantized_kv_cache(
    config: PreTrainedConfig,
    device: torch.device,
    enabled: bool,
) -> Optional[QuantizedKVCache]:
    """Create a cache only when the current model and GPU support it.

    Args:
        config: Configuration of the decoder that will consume the cache.
        device: Device on which attention states will be produced.
        enabled: Whether cache quantization is enabled in configuration.

    Returns:
        Optional[QuantizedKVCache]: A compact cache, or ``None`` for CPU,
            unsupported architectures, or unsupported CUDA capability.
    """
    if device.type != "cuda":
        return None
    mode = quantized_kv_cache_mode(enabled)
    if mode is None:
        return None
    return QuantizedKVCache(config=config, mode=mode)
