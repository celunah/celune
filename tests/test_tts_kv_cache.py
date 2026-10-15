# SPDX-License-Identifier: Apache-2.0
"""Tests for supported TTS KV-cache quantization adapters."""

from unittest.mock import patch
from typing import Optional, Union, cast

import torch
from torch import nn

from celune.backends.tts.voxcpm_cache import (
    _VoxCPMModel,
    QuantizedStaticKVCache,
    install_voxcpm_quantized_cache,
)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    _cos: torch.Tensor,
    _sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the toy attention test independent of positional encoding."""
    return query_states, key_states


class _ToyAttention(nn.Module):
    """Small grouped-query attention layer with VoxCPM's cache contract."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.head_dim = 2
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Provide the unused module forward surface required by PyTorch."""
        return hidden_states

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        position_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        position_id: Union[int, torch.Tensor],
        kv_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the original full-precision static-cache attention step."""
        del position_emb
        if isinstance(position_id, torch.Tensor):
            position_id = int(position_id.reshape(-1)[0].item())
        batch_size = hidden_states.size(0)
        query = self.q_proj(hidden_states).view(batch_size, 1, 2, 2).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch_size, 1, 1, 2).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch_size, 1, 1, 2).transpose(1, 2)
        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = key
        value_cache[:, :, position_id, :] = value
        key_states = key_cache[..., : position_id + 1, :].repeat_interleave(
            self.num_heads // self.num_key_value_heads,
            dim=1,
        )
        value_states = value_cache[..., : position_id + 1, :].repeat_interleave(
            self.num_heads // self.num_key_value_heads,
            dim=1,
        )
        scores = torch.matmul(query, key_states.transpose(-1, -2))
        weights = torch.softmax(scores * self.head_dim**-0.5, dim=-1)
        output = torch.matmul(weights, value_states)
        return self.o_proj(output.transpose(1, 2).reshape(batch_size, 4))


class _ToyCache:
    """Minimal static cache matching VoxCPM's public adapter boundary."""

    def __init__(self) -> None:
        self.max_length = 6
        self.kv_cache = torch.zeros(2, 1, 1, 1, 6, 2)
        self.current_length = 0

    def get_layer_cache(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the full-precision key and value storage for one layer."""
        return self.kv_cache[0, layer_index], self.kv_cache[1, layer_index]

    def step(self) -> int:
        """Reserve the next token slot in the fake static cache."""
        if self.current_length >= self.max_length:
            raise ValueError("fake cache is full")
        position = self.current_length
        self.current_length += 1
        return position

    def fill_caches(
        self,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Copy prefill key/value states into the fake static storage."""
        self.current_length = caches[0][0].shape[-2]
        self.kv_cache.zero_()
        for layer_index, (keys, values) in enumerate(caches):
            self.kv_cache[0, layer_index, ..., : self.current_length, :] = keys
            self.kv_cache[1, layer_index, ..., : self.current_length, :] = values


class _ToyLanguageModel:
    """One-layer language model with a static cache and attention module."""

    def __init__(self) -> None:
        self.kv_cache = _ToyCache()
        self.layers = [type("Layer", (), {"self_attn": _ToyAttention()})()]


class _ToyRuntime:
    """VoxCPM runtime with both autoregressive language models."""

    def __init__(self) -> None:
        self.base_lm = _ToyLanguageModel()
        self.residual_lm = _ToyLanguageModel()


class _ToyModel:
    """Loaded VoxCPM model fixture."""

    def __init__(self) -> None:
        self.tts_model = _ToyRuntime()


def test_voxcpm_cache_quantizes_old_prefix_and_keeps_recent_tail() -> None:
    """Store older prompt tokens compactly and retain the newest tokens."""
    cache = QuantizedStaticKVCache(
        shape=torch.Size((2, 1, 1, 1, 5, 2)),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_length=5,
        mode="int8",
        residual_length=2,
    )
    keys = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    values = keys + 1

    cache.fill_caches([(keys, values)])

    assert cache.kv_cache.dtype == torch.int8
    assert cache.residual.shape[-2] == 2
    assert cache.current_length == 3
    torch.testing.assert_close(
        cache._attention_states(0, 3)[0],
        keys,
        atol=0.03,
        rtol=0,
    )

    position = cache.step()
    next_key = torch.tensor([[[[7.0, 8.0]]]])
    next_value = next_key + 1
    cache.get_layer_cache(0).cache._store_step(
        0,
        position,
        next_key,
        next_value,
    )
    expected_keys = torch.cat((keys, next_key), dim=-2)
    expected_values = torch.cat((values, next_value), dim=-2)
    actual_keys, actual_values = cache._attention_states(0, 4)
    torch.testing.assert_close(actual_keys, expected_keys, atol=0.04, rtol=0)
    torch.testing.assert_close(actual_values, expected_values, atol=0.04, rtol=0)


def test_voxcpm_adapter_patches_both_cache_backed_language_models() -> None:
    """Use the compact cache for base and residual decoding paths."""
    model = _ToyModel()
    with patch(
        "celune.backends.tts.voxcpm_cache.quantized_kv_cache_mode",
        return_value="int8",
    ):
        assert install_voxcpm_quantized_cache(cast(_VoxCPMModel, model), enabled=True)

    for language_model in (
        model.tts_model.base_lm,
        model.tts_model.residual_lm,
    ):
        cache = cast(QuantizedStaticKVCache, language_model.kv_cache)
        assert isinstance(cache, QuantizedStaticKVCache)
        cache.fill_caches([(torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))])
        position = cache.step()
        output = language_model.layers[0].self_attn.forward_step(
            hidden_states=torch.ones(1, 4),
            position_emb=None,
            position_id=torch.tensor([position]),
            kv_cache=cache.get_layer_cache(0),
        )
        assert output.shape == (1, 4)
        assert torch.isfinite(output).all()


def test_voxcpm_adapter_leaves_native_caches_when_disabled() -> None:
    """Keep both original caches when cache quantization is disabled."""
    model = _ToyModel()
    original_caches = (
        model.tts_model.base_lm.kv_cache,
        model.tts_model.residual_lm.kv_cache,
    )

    with patch(
        "celune.backends.tts.voxcpm_cache.quantized_kv_cache_mode",
        return_value=None,
    ):
        assert not install_voxcpm_quantized_cache(
            cast(_VoxCPMModel, model),
            enabled=False,
        )

    assert model.tts_model.base_lm.kv_cache is original_caches[0]
    assert model.tts_model.residual_lm.kv_cache is original_caches[1]
