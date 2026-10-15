# SPDX-License-Identifier: Apache-2.0
"""Quantized cache adapter for VoxCPM's custom static decoder cache."""

from __future__ import annotations

from types import MethodType
from collections.abc import Callable
from typing import Optional, Protocol, Union, cast

import torch

from ...kv_cache import QuantizedKVCacheMode, quantized_kv_cache_mode


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


class _VoxCPMModel(Protocol):
    """Loaded VoxCPM model surface used by the cache adapter."""

    tts_model: _VoxCPMRuntime


class _QuantizedStaticKVLayer:
    """Layer handle that keeps VoxCPM keys and values in compact storage."""

    def __init__(self, cache: QuantizedStaticKVCache, layer_index: int) -> None:
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

    def _store_step(
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

    def _attention_states(
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
    del position_id
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
    cache_layer.cache._store_step(
        cache_layer.layer_index,
        position,
        key_states,
        value_states,
    )
    key_cache, value_cache = cache_layer.cache._attention_states(
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


def install_voxcpm_quantized_cache(model: _VoxCPMModel, enabled: bool) -> bool:
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
