# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona's compact attention-cache storage."""

from unittest.mock import patch
from typing import ClassVar, cast

import pytest
import torch
from transformers.configuration_utils import PreTrainedConfig

from celune.modeling import (
    GraphQuantizedKVCacheLayer,
    QuantizedKVCache,
    QuantizedKVCacheLayer,
    _GraphQuantizedKVWorkspace,
    quantized_kv_cache_mode,
)


class _CacheConfig:
    """Minimal decoder configuration accepted by the cache factory."""

    num_hidden_layers = 2
    layer_types: ClassVar[list[str]] = ["full_attention", "full_attention"]
    sliding_window = None
    attention_chunk_size = None

    def get_text_config(self, *, decoder: bool) -> "_CacheConfig":
        """Return this decoder-only fixture configuration."""
        del decoder
        return self


def test_int8_cache_keeps_recent_tail_in_compute_dtype() -> None:
    """Quantize only the older prefix and preserve a recent BF16 tail."""
    cache = QuantizedKVCache(
        config=cast(PreTrainedConfig, _CacheConfig()),
        mode="int8",
        residual_length=2,
    )
    keys = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)
    values = keys + 1

    first_keys, first_values = cache.update(keys[..., :2, :], values[..., :2, :], 0)
    second_keys, second_values = cache.update(keys[..., 2:, :], values[..., 2:, :], 0)

    assert first_keys.shape == (1, 2, 2, 2)
    assert first_values.shape == (1, 2, 2, 2)
    assert second_keys.shape == keys.shape
    assert second_values.shape == values.shape
    assert cache.get_seq_length() == 4
    layer = cast(QuantizedKVCacheLayer, cache.layers[0])
    assert layer._quantized_keys is not None
    assert layer.keys is not None
    assert layer._quantized_keys.dtype == torch.int8
    assert layer.keys.shape[-2] == 2
    assert (
        layer.keys.untyped_storage().nbytes()
        == layer.keys.numel() * layer.keys.element_size()
    )
    assert (
        layer._quantized_keys.untyped_storage().nbytes()
        == layer._quantized_keys.numel()
    )
    torch.testing.assert_close(second_keys[..., -2:, :], keys[..., -2:, :])
    torch.testing.assert_close(cache[0][0], second_keys)
    torch.testing.assert_close(cache.to_legacy_cache()[0][1], second_values)


def test_quantized_cache_crop_and_reset_release_logical_tokens() -> None:
    """Cropping and reset update both compact and residual cache state."""
    cache = QuantizedKVCache(
        config=cast(PreTrainedConfig, _CacheConfig()),
        mode="int8",
        residual_length=2,
    )
    states = torch.ones(1, 2, 5, 2)
    cache.update(states, states, 0)

    cache.crop(3)
    assert cache.get_seq_length() == 3
    layer = cast(QuantizedKVCacheLayer, cache.layers[0])
    assert layer.keys is not None
    cropped_keys, cropped_values = layer.dequantized_cache()
    assert cropped_keys.shape[-2] == 3
    assert cropped_values.shape[-2] == 3

    cache.reset()
    assert cache.get_seq_length() == 0
    assert layer.keys.shape[-2] == 0
    assert layer._quantized_keys is not None
    assert layer._quantized_keys.untyped_storage().nbytes() == 0


def test_quantized_kv_cache_mode_selects_hardware_policy() -> None:
    """Use INT8 on Ampere and FP8 on Ada-class CUDA devices."""
    with (
        patch(
            "celune.modeling.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.modeling.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
    ):
        assert quantized_kv_cache_mode(True) == "int8"

    with (
        patch(
            "celune.modeling.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.modeling.torch.cuda.get_device_capability",
            return_value=(8, 9),
        ),
    ):
        assert quantized_kv_cache_mode(True) == "fp8"

    assert quantized_kv_cache_mode(False) is None


def test_fp8_graph_cache_copies_quantized_prefix_storage() -> None:
    """Write FP8 quantized states through the graph cache's byte view."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("PyTorch does not expose FP8 cache storage")

    layer = GraphQuantizedKVCacheLayer(
        mode="fp8",
        max_cache_len=4,
        residual_length=2,
        workspace=_GraphQuantizedKVWorkspace(max_cache_len=4),
    )
    key_states = [
        torch.tensor([[[[1.0, -0.5]]]]),
        torch.tensor([[[[-0.25, 0.75]]]]),
        torch.tensor([[[[0.5, 0.25]]]]),
    ]
    value_states = [states + 1 for states in key_states]

    for position, (keys, values) in enumerate(zip(key_states, value_states)):
        layer.update(
            keys,
            values,
            {"cache_position": torch.tensor([position])},
        )

    quantized_keys = cast(torch.Tensor, layer._quantized_keys)
    expected_keys, _ = layer._quantize(key_states[0])
    assert quantized_keys.dtype == fp8_dtype
    assert torch.equal(
        quantized_keys[..., :1, :].view(torch.uint8),
        expected_keys.view(torch.uint8),
    )
