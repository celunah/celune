# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded key/value caches for Persona generation."""

from __future__ import annotations

from typing import Union, Literal, Optional, cast

import torch
from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    LinearAttentionCacheLayerMixin,
)
from transformers.configuration_utils import PreTrainedConfig

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


class _QuantizedKVCacheLayer(CacheLayerMixin):
    """One attention layer with a quantized prefix and BF16 residual tail."""

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
        value_states: torch.Tensor,
    ) -> None:
        """Initialize metadata and empty storage from the first attention states."""
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

        return self._dequantized_cache()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return the attention mask dimensions for the current cache length."""
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        """Return the number of tokens represented by this layer."""
        return self.cumulative_length

    def get_max_length(self) -> int:
        """Return ``-1`` because the cache grows with the generation request."""
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

    def _dequantized_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
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
            raise RuntimeError("Persona KV cache layer is not initialized")
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
            raise ValueError("Persona model cache layout is not supported")
        layers = cast(
            list[CacheLayerMixin],
            [_QuantizedKVCacheLayer(mode, residual_length) for _ in layer_types],
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
            if not isinstance(layer, _QuantizedKVCacheLayer):
                raise TypeError("unexpected Persona cache layer")
            if not layer.is_initialized:
                yield layer.keys, layer.values, None
                continue
            keys, values = layer._dequantized_cache()
            yield keys, values, None


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
