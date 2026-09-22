# SPDX-License-Identifier: Apache-2.0
"""Tests for Persona's compact attention-cache storage."""

from typing import ClassVar, cast
from unittest.mock import patch

import torch
from transformers.configuration_utils import PreTrainedConfig

from celune.persona.cache import (
    QuantizedKVCache,
    _QuantizedKVCacheLayer,
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
    layer = cast(_QuantizedKVCacheLayer, cache.layers[0])
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
    layer = cast(_QuantizedKVCacheLayer, cache.layers[0])
    assert layer.keys is not None
    cropped_keys, cropped_values = layer._dequantized_cache()
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
            "celune.persona.cache.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.persona.cache.torch.cuda.get_device_capability",
            return_value=(8, 6),
        ),
    ):
        assert quantized_kv_cache_mode(True) == "int8"

    with (
        patch(
            "celune.persona.cache.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "celune.persona.cache.torch.cuda.get_device_capability",
            return_value=(8, 9),
        ),
    ):
        assert quantized_kv_cache_mode(True) == "fp8"

    assert quantized_kv_cache_mode(False) is None
