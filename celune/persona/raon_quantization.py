# SPDX-License-Identifier: Apache-2.0
"""Memory-efficient lookup quantization for Raon-1-nano's PLE table."""

from __future__ import annotations

import gc
import math
from types import MethodType
from typing import Protocol, cast
from collections.abc import Callable, Mapping

import torch
from torch import nn

_GROUP_SIZE = 128
_ROWS_PER_CHUNK = 512


class _Normalization(Protocol):
    """The tensor normalization operation used by the PLE module."""

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize one projected PLE tensor."""


class _RaonPerLayerInputs(Protocol):
    """The small structural surface used by Raon's per-layer input module."""

    num_layers: int
    width: int
    table: nn.Parameter
    embedding_projection: torch.Tensor
    projection_norm: _Normalization
    _celune_ple_scales: torch.Tensor
    _celune_ple_bits: int
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _quantized_forward(
    module: nn.Module,
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Run Raon's PLE lookup after unpacking only the requested table rows."""
    state = cast(_RaonPerLayerInputs, module)
    batch_shape = tuple(input_ids.shape)
    packed_rows = state.table[input_ids]
    scale_rows = state._celune_ple_scales[input_ids].float()

    if state._celune_ple_bits == 4:
        nibbles = packed_rows.to(torch.int16)
        low = ((nibbles & 0x0F) ^ 0x08) - 0x08
        high = (((nibbles >> 4) & 0x0F) ^ 0x08) - 0x08
        quantized = torch.stack((low, high), dim=-1).flatten(-2)
    else:
        quantized = packed_rows.to(torch.float32)

    quantized = quantized.reshape(
        *batch_shape,
        state.num_layers * state.width // _GROUP_SIZE,
        _GROUP_SIZE,
    )
    rows = (quantized.float() * scale_rows.unsqueeze(-1)).to(embeddings.dtype)
    layer_shape = (*embeddings.shape[:2], state.num_layers, state.width)
    projected = state.projection_norm(
        torch.matmul(embeddings, state.embedding_projection.transpose(0, 1)).reshape(
            layer_shape
        )
    )
    rows = rows.reshape(layer_shape) * math.sqrt(state.width)
    return (projected + rows) * (1.0 / math.sqrt(2.0))


def _quantize_chunks(
    table: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a CPU table in bounded chunks with per-group scales."""
    rows, columns = table.shape
    groups = columns // _GROUP_SIZE
    packed_columns = columns // 2 if bits == 4 else columns
    packed = torch.empty(
        (rows, packed_columns), dtype=torch.uint8 if bits == 4 else torch.int8
    )
    scales = torch.empty((rows, groups), dtype=torch.float16)
    maximum = 7 if bits == 4 else 127

    for start in range(0, rows, _ROWS_PER_CHUNK):
        end = min(start + _ROWS_PER_CHUNK, rows)
        values = (
            table[start:end]
            .to(torch.float32)
            .reshape(
                end - start,
                groups,
                _GROUP_SIZE,
            )
        )
        scale = values.abs().amax(dim=-1).clamp_min(1e-8) / maximum
        quantized = torch.round(values / scale.unsqueeze(-1)).clamp(
            -maximum,
            maximum,
        )
        scales[start:end] = scale.to(torch.float16)

        if bits == 4:
            nibbles = quantized.to(torch.int16) & 0x0F
            low = nibbles[..., 0::2]
            high = nibbles[..., 1::2] << 4
            packed[start:end] = (
                (low | high)
                .to(torch.uint8)
                .reshape(
                    end - start,
                    packed_columns,
                )
            )
        else:
            packed[start:end] = quantized.to(torch.int8).reshape(
                end - start,
                packed_columns,
            )

    return packed, scales


def _validate_dispatch(model: nn.Module, module: nn.Module) -> None:
    """Reject Accelerate-managed layouts along the PLE module's parent path."""
    module_path = next(
        (name for name, candidate in model.named_modules() if candidate is module),
        None,
    )
    if module_path is None:
        raise RuntimeError("Raon PLE module is not registered in its causal-LM model")

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        parent_devices = [
            (len(name), str(device).casefold())
            for name, device in device_map.items()
            if not name or module_path == name or module_path.startswith(f"{name}.")
        ]
        if (
            parent_devices
            and max(parent_devices, key=lambda item: item[0])[1] == "disk"
        ):
            raise RuntimeError("Raon PLE table is offloaded to disk")

    current = model
    if getattr(current, "_hf_hook", None) is not None:
        raise RuntimeError("Raon PLE path is managed by an Accelerate hook")
    for part in module_path.split("."):
        current = getattr(current, part)
        if getattr(current, "_hf_hook", None) is not None:
            raise RuntimeError("Raon PLE path is managed by an Accelerate hook")


def quantize_raon_per_layer_table(model: nn.Module, bits: int) -> int:
    """Quantize Raon's large per-layer embedding table in place.

    The upstream table is a raw parameter, so generic BitsAndBytes loading does
    not quantize it. This replaces that parameter with signed group-wise int4
    or int8 storage and dequantizes only the token rows used by each forward.
    The PLE module must not be managed by an Accelerate dispatch or offload hook.

    Args:
        model: A loaded ``Raon1NanoForCausalLM`` model.
        bits: Storage precision, either four or eight bits.

    Returns:
        int: Number of bytes saved for the PLE table and its scales.

    Raises:
        TypeError: The loaded model does not expose the expected Raon table.
        RuntimeError: Its device dispatch cannot be safely updated.
        ValueError: The requested precision or table dimensions are unsupported.
    """
    if bits not in {4, 8}:
        raise ValueError("Raon PLE precision must be 4 or 8 bits")

    body = getattr(model, "model", None)
    module = getattr(body, "per_layer_inputs", None)
    table = getattr(module, "table", None)
    if not isinstance(module, nn.Module) or not isinstance(table, nn.Parameter):
        raise TypeError("Loaded Raon model does not expose its expected PLE table")
    existing_bits = getattr(module, "_celune_ple_bits", None)
    if isinstance(existing_bits, int):
        if existing_bits == bits:
            return 0
        raise RuntimeError("Raon PLE table is already quantized at another precision")
    if "_celune_ple_scales" in module._buffers:
        raise RuntimeError("Raon PLE scale buffer already exists")
    _validate_dispatch(model, module)

    state = cast(_RaonPerLayerInputs, module)
    if (
        table.ndim != 2
        or table.shape[1] != state.num_layers * state.width
        or table.shape[1] % _GROUP_SIZE
    ):
        raise ValueError("Raon PLE table dimensions are not group-quantizable")
    if table.device.type == "meta":
        raise RuntimeError("Raon PLE table is not materialized")

    module_parameters = tuple(module.parameters())
    module_devices = {parameter.device for parameter in module_parameters}
    if len(module_devices) != 1 or table.device not in module_devices:
        raise RuntimeError("Raon PLE module parameters span multiple devices")

    device = table.device
    requires_grad = table.requires_grad
    state = cast(_RaonPerLayerInputs, module)
    original_forward = cast(
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        module.forward,
    )
    source = table.detach().to(device="cpu")
    packed_cpu, scales_cpu = _quantize_chunks(source, bits)
    original_bytes = table.numel() * table.element_size()
    quantized_bytes = packed_cpu.numel() * packed_cpu.element_size()
    quantized_bytes += scales_cpu.numel() * scales_cpu.element_size()

    module.register_parameter("table", None)
    del table
    del module_parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        module.register_parameter(
            "table",
            nn.Parameter(packed_cpu.to(device), requires_grad=False),
        )
        module.register_buffer(
            "_celune_ple_scales",
            scales_cpu.to(device),
            persistent=False,
        )
        state._celune_ple_bits = bits
        state.forward = cast(
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            MethodType(_quantized_forward, module),
        )
    except Exception:
        module._buffers.pop("_celune_ple_scales", None)
        if hasattr(module, "_celune_ple_bits"):
            delattr(module, "_celune_ple_bits")
        state.forward = original_forward
        module.register_parameter(
            "table",
            nn.Parameter(source.to(device), requires_grad=requires_grad),
        )
        raise

    del source
    del packed_cpu
    del scales_cpu
    return original_bytes - quantized_bytes
