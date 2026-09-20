# SPDX-License-Identifier: Apache-2.0
"""Contract-driven runtime weight quantization for TTS models."""

import contextlib
import gc
import importlib
from collections.abc import Callable
from typing import Literal, Optional, Protocol, cast

import torch
from torch import nn

from ...compat import torchao_compatibility
from ...exceptions import BackendError
from .contracts import ModelComponentContract, QuantizationRule
from .contracts import validate_model_state

__all__ = [
    "QuantizationMode",
    "quantization_mode",
    "quantize_component",
]


QuantizationMode = Literal["int8", "fp8"]
_QUANTIZED_DTYPES = {
    "torch.float8_e4m3fn",
    "torch.float8_e4m3fnuz",
    "torch.float8_e5m2",
    "torch.float8_e5m2fnuz",
    "torch.int8",
    "torch.uint8",
}


class _QuantizationConfig(Protocol):
    """Marker protocol for a TorchAO quantization configuration."""


def quantization_mode(enabled: bool) -> Optional[QuantizationMode]:
    """Select a supported weight format from the active CUDA capability.

    Args:
        enabled: Whether the user enabled runtime TTS quantization.

    Returns:
        Optional[QuantizationMode]: ``"int8"`` for pre-FP8 Ampere,
        ``"fp8"`` for Ada and newer, or ``None`` when quantization is not
        requested or the device cannot use the supported CUDA path.
    """
    if not enabled or not torch.cuda.is_available():
        return None

    capability = torch.cuda.get_device_capability()
    if capability >= (8, 9):
        return "fp8"
    if capability >= (8, 0):
        return "int8"
    return None


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Return whether TorchAO replaced a parameter with a quantized tensor."""
    if str(tensor.dtype) in _QUANTIZED_DTYPES:
        return True
    return tensor.__class__.__module__.startswith("torchao.")


def _is_safe_linear(
    module: nn.Module,
    fully_qualified_name: str,
    rule: QuantizationRule,
) -> bool:
    """Return whether one module is explicitly approved by its contract."""
    if not isinstance(module, nn.Linear):
        return False
    if fully_qualified_name.rsplit(".", 1)[-1] not in rule.module_suffixes:
        return False
    return not any(
        token in fully_qualified_name.split(".") for token in rule.excluded_module_names
    )


def _import_torchao_quantization():
    """Import TorchAO without repeating PyTorch's native Enum registration."""
    with torchao_compatibility():
        quantization = importlib.import_module("torchao.quantization")
    return quantization


def _torchao_api() -> tuple[
    Callable[..., None],
    Callable[[], _QuantizationConfig],
    Callable[[], _QuantizationConfig],
]:
    """Load the TorchAO APIs lazily inside the selected backend environment."""
    try:
        quantization = _import_torchao_quantization()
        return (
            cast(Callable[..., None], quantization.quantize_),
            cast(
                Callable[[], _QuantizationConfig],
                quantization.Int8WeightOnlyConfig,
            ),
            cast(
                Callable[[], _QuantizationConfig],
                quantization.Float8WeightOnlyConfig,
            ),
        )
    except (ImportError, AttributeError) as exc:
        raise BackendError(
            "torchao is required for TTS quantization",
            error_code="tts_quantization_dependency",
            error_type=type(exc).__name__,
        ) from exc


def _quantization_config(
    mode: QuantizationMode,
    int8_config: Callable[[], _QuantizationConfig],
    fp8_config: Callable[[], _QuantizationConfig],
) -> _QuantizationConfig:
    """Build the TorchAO weight-only config for the selected GPU family."""
    if mode == "int8":
        return int8_config()
    return fp8_config()


def _release_quantization_temporaries() -> None:
    """Release temporary storage left after replacing high-precision weights."""
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def _module_device(module: nn.Module) -> Optional[torch.device]:
    """Return the device of the first parameter or buffer in one component."""
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def quantize_component(
    module: nn.Module,
    component: ModelComponentContract,
    *,
    mode: QuantizationMode,
    backend: str,
) -> int:
    """Quantize approved linear layers and validate the resulting state.

    Args:
        module: Loaded model component whose parameters are already on device.
        component: Contract describing the component and its safe layers.
        mode: Weight-only format selected for the active GPU.
        backend: Celune backend identifier used in diagnostics.

    Returns:
        int: Number of linear modules converted by TorchAO.

    Raises:
        BackendError: TorchAO is unavailable or the conversion failed.
        InvalidCheckpoint: The loaded or converted state violates its contract.
    """
    rule = component.quantization
    if rule is None:
        return 0

    validate_model_state(
        module.state_dict(),
        component.inventory,
        name=component.name,
        backend=backend,
        filename=component.artifact_path.rsplit("/", 1)[-1],
        path=component.artifact_path,
        runtime_dtypes=component.runtime_dtypes,
    )
    quantize, int8_config, fp8_config = _torchao_api()
    config = _quantization_config(mode, int8_config, fp8_config)
    original_device = _module_device(module)

    try:
        if original_device is not None and original_device.type == "cuda":
            module.to(device=torch.device("cpu"))
        try:
            quantize(
                module,
                config,
                filter_fn=lambda candidate, name: _is_safe_linear(
                    candidate, name, rule
                ),
            )
        except Exception as exc:
            raise BackendError(
                f"{backend} {component.name} {mode} quantization failed",
                error_code="tts_quantization_failed",
                error_type=type(exc).__name__,
            ) from exc

        converted = sum(
            1
            for name, candidate in module.named_modules()
            if _is_safe_linear(candidate, name, rule)
            and _is_quantized_tensor(candidate.weight)
        )
        if converted == 0:
            raise BackendError(
                f"{backend} {component.name} did not quantize an approved layer",
                error_code="tts_quantization_empty",
            )

        validate_model_state(
            module.state_dict(),
            component.inventory,
            name=component.name,
            backend=backend,
            filename=component.artifact_path.rsplit("/", 1)[-1],
            path=component.artifact_path,
            runtime_dtypes=component.runtime_dtypes,
            allow_quantized=True,
            quantization=rule,
        )
        return converted
    finally:
        if original_device is not None and original_device.type == "cuda":
            with contextlib.suppress(Exception):
                module.to(device=original_device)
        _release_quantization_temporaries()
