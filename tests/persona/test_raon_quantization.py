# SPDX-License-Identifier: Apache-2.0
"""Tests for Raon PLE lookup-table quantization."""

import math

import pytest
import torch
from torch import nn

from celune.persona.raon_quantization import quantize_raon_per_layer_table


class _FakePerLayerInputs(nn.Module):
    """Small test module matching Raon's per-layer input table contract."""

    _hf_hook: object

    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 2
        self.width = 128
        self.table = nn.Parameter(torch.randn(8, 256, dtype=torch.bfloat16))
        self.embedding_projection = nn.Parameter(
            torch.randn(256, 4, dtype=torch.bfloat16)
        )
        self.projection_norm = nn.Identity()

    def forward(
        self, input_ids: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Mirror the upstream PLE operation before Celune quantizes the table."""
        layers = (*embeddings.shape[:2], self.num_layers, self.width)
        projected = torch.matmul(
            embeddings,
            self.embedding_projection.transpose(0, 1),
        ).reshape(layers)
        rows = self.table[input_ids].reshape(layers) * math.sqrt(self.width)
        return (projected + rows) * (1.0 / math.sqrt(2.0))


class _FakeRaonBody(nn.Module):
    """Expose Raon's PLE module inside its causal-LM body."""

    per_layer_inputs: _FakePerLayerInputs

    def __init__(self) -> None:
        super().__init__()
        self.per_layer_inputs = _FakePerLayerInputs()

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the fake PLE module."""
        return self.per_layer_inputs(input_ids, embeddings)


class _FakeRaonModel(nn.Module):
    """Expose the nested PLE module at Raon's native model path."""

    model: _FakeRaonBody
    hf_device_map: dict[str, str]

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeRaonBody()

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the fake causal-LM body."""
        return self.model(input_ids, embeddings)


@pytest.mark.parametrize("bits", (4, 8))
def test_raon_per_layer_table_quantization_preserves_forward_shape(bits: int) -> None:
    """Compress the PLE table and retain finite outputs with its original shape."""
    torch.manual_seed(17)
    model = _FakeRaonModel()
    module = model.model.per_layer_inputs
    input_ids = torch.tensor([[0, 3, 7]])
    embeddings = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    expected = module(input_ids, embeddings)
    original_bytes = module.table.numel() * module.table.element_size()

    bytes_saved = quantize_raon_per_layer_table(model, bits)
    actual = module(input_ids, embeddings)

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    absolute_error = (actual.float() - expected.float()).abs()
    if bits == 4:
        assert absolute_error.mean() < 1.5
    else:
        assert torch.allclose(actual.float(), expected.float(), atol=0.25, rtol=0.05)
    assert bytes_saved > 0
    assert module._celune_ple_bits == bits
    assert module.table.requires_grad is False
    assert module.table.element_size() == 1
    assert module.table.numel() * module.table.element_size() < original_bytes
    assert quantize_raon_per_layer_table(model, bits) == 0


def test_raon_per_layer_table_quantization_rejects_model_sharding() -> None:
    """Do not replace PLE parameters behind Accelerate's dispatch hooks."""
    model = _FakeRaonModel()
    model.hf_device_map = {"model": "cuda:0", "lm_head": "cpu"}
    model.model.per_layer_inputs._hf_hook = object()
    original = model.model.per_layer_inputs.table

    with pytest.raises(RuntimeError, match="Accelerate hook"):
        quantize_raon_per_layer_table(model, 4)

    assert model.model.per_layer_inputs.table is original


def test_raon_per_layer_table_quantization_rejects_invalid_precision() -> None:
    """Restrict this implementation to supported signed int4 and int8 formats."""
    with pytest.raises(ValueError, match="4 or 8 bits"):
        quantize_raon_per_layer_table(_FakeRaonModel(), 6)
