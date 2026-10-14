# SPDX-License-Identifier: Apache-2.0
"""Runtime validation for pinned TTS model artifacts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from collections import Counter
from typing import Union, Optional
from collections.abc import Mapping

import torch

from ...registry.tts import (
    QuantizationRule,
    RuntimeDtypeRule,
    WeightArtifactContract,
    TensorInventoryContract,
)
from ...exceptions import InvalidCheckpoint, ModelContractError

__all__ = ["validate_model_state", "validate_safetensors_artifact"]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local weight artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_inventory(
    tensors: Mapping[str, tuple[tuple[int, ...], str]],
) -> str:
    """Hash sorted tensor names, shapes, and dtypes as a stable inventory."""
    payload = [
        {"name": name, "shape": list(shape), "dtype": dtype}
        for name, (shape, dtype) in sorted(tensors.items())
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dtype_name(dtype: object) -> str:
    """Normalize a tensor or safetensors dtype to the contract spelling."""
    value = str(dtype)
    if value.startswith("torch."):
        return value
    return {
        "BF16": "torch.bfloat16",
        "F16": "torch.float16",
        "F32": "torch.float32",
        "F64": "torch.float64",
        "I64": "torch.int64",
        "I32": "torch.int32",
        "I8": "torch.int8",
        "U8": "torch.uint8",
    }.get(value, value)


def _inventory_from_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, tuple[tuple[int, ...], str]], Counter[str], int]:
    """Build a structural inventory from a loaded PyTorch state dictionary."""
    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    dtype_counts: Counter[str] = Counter()
    parameter_count = 0
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ModelContractError("model state contains an invalid tensor entry")
        shape = tuple(int(value) for value in tensor.shape)
        dtype = _dtype_name(tensor.dtype)
        tensors[name] = (shape, dtype)
        dtype_counts[dtype] += 1
        parameter_count += tensor.numel()
    return tensors, dtype_counts, parameter_count


def _checkpoint_layer_name(tensor_name: str) -> str:
    """Return the owning layer name for a tensor parameter name."""
    for suffix in (".weight", ".bias"):
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _dtype_value(dtype: str) -> Union[str, torch.dtype]:
    """Return a PyTorch dtype for a contract dtype spelling when available."""
    return {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
    }.get(dtype, dtype)


def _is_quantized_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a tensor uses a supported quantized representation."""
    if _dtype_name(tensor.dtype) in {
        "torch.float8_e4m3fn",
        "torch.float8_e4m3fnuz",
        "torch.float8_e5m2",
        "torch.float8_e5m2fnuz",
        "torch.int8",
        "torch.uint8",
    }:
        return True
    return tensor.__class__.__module__.startswith("torchao.")


def _allows_quantized_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    quantization: QuantizationRule,
) -> bool:
    """Return whether a tensor is an approved quantized layer parameter."""
    if not _is_quantized_tensor(tensor) or not tensor_name.endswith(".weight"):
        return False
    layer_name = _checkpoint_layer_name(tensor_name)
    if layer_name.rsplit(".", 1)[-1] not in quantization.module_suffixes:
        return False
    return not any(
        token in layer_name.split(".") for token in quantization.excluded_module_names
    )


def _invalid_state(
    *,
    backend: str,
    filename: str,
    path: str,
    reason: str,
    tensor_name: Optional[str] = None,
    shape: Optional[tuple[int, ...]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    expected: Optional[Union[str, torch.dtype]] = None,
    actual: Optional[Union[str, torch.dtype]] = None,
) -> InvalidCheckpoint:
    """Build a structured checkpoint failure for a loaded model state."""
    return InvalidCheckpoint(
        backend=backend,
        filename=filename,
        path=path,
        tensor_name=tensor_name,
        layer_name=(
            _checkpoint_layer_name(tensor_name) if tensor_name is not None else None
        ),
        shape=shape,
        dtype=dtype,
        expected=expected,
        actual=actual,
        reason=reason,
    )


def validate_model_state(
    state: Mapping[str, torch.Tensor],
    contract: TensorInventoryContract,
    *,
    name: str,
    backend: str = "unknown",
    filename: Optional[str] = None,
    path: Optional[str] = None,
    runtime_dtypes: tuple[RuntimeDtypeRule, ...] = (),
    allow_quantized: bool = False,
    quantization: Optional[QuantizationRule] = None,
) -> None:
    """Validate a loaded model state against one tensor inventory contract.

    Args:
        state: State dictionary from the loaded backend component.
        contract: Expected tensor inventory derived from the pinned artifact.
        name: Diagnostic component name.
        backend: Celune backend identifier.
        filename: Checkpoint filename for diagnostics.
        path: Checkpoint-relative path for diagnostics.
        runtime_dtypes: Optional prefix-specific runtime dtype rules.
        allow_quantized: Permit approved quantized layer weights while checking
            the unquantized structure and non-quantized dtypes.
        quantization: Contract rule that identifies approved quantized layers.

    Raises:
        InvalidCheckpoint: The state is structurally incomplete, inconsistent, or invalid.
    """
    checkpoint_filename = filename or "model.safetensors"
    checkpoint_path = path or name
    try:
        tensors, dtype_counts, parameter_count = _inventory_from_state(state)
    except ModelContractError as exc:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=str(exc),
        ) from exc
    quantized_state = bool(
        allow_quantized
        and quantization is not None
        and any(
            _allows_quantized_tensor(tensor_name, state[tensor_name], quantization)
            for tensor_name in tensors
        )
    )
    if len(tensors) != contract.tensor_count and not quantized_state:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=f"{name} tensor count mismatch: expected {contract.tensor_count}, got {len(tensors)}",
        )
    if parameter_count != contract.parameter_count and not quantized_state:
        raise _invalid_state(
            backend=backend,
            filename=checkpoint_filename,
            path=checkpoint_path,
            reason=f"{name} parameter count mismatch: expected {contract.parameter_count}, got {parameter_count}",
        )
    if not allow_quantized:
        expected_counts = dict(contract.dtype_counts)
        if dict(dtype_counts) != expected_counts:
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} dtype counts mismatch: expected {expected_counts}, got {dict(dtype_counts)}",
            )
        inventory_digest = _canonical_inventory(tensors)
        if inventory_digest != contract.inventory_sha256:
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor inventory mismatch: expected {contract.inventory_sha256}, got {inventory_digest}",
            )

    for tensor_name, (_shape, dtype) in tensors.items():
        matching_rules = tuple(
            rule
            for rule in runtime_dtypes
            if rule.prefix == ""
            or tensor_name == rule.prefix
            or tensor_name.startswith(f"{rule.prefix}.")
        )
        quantized = bool(
            allow_quantized
            and quantization is not None
            and _allows_quantized_tensor(tensor_name, state[tensor_name], quantization)
        )
        if (
            matching_rules
            and not quantized
            and not any(dtype in rule.dtypes for rule in matching_rules)
        ):
            expected = sorted({item for rule in matching_rules for item in rule.dtypes})
            tensor = state[tensor_name]
            expected_value = _dtype_value(expected[0]) if len(expected) == 1 else None
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor has dtype {dtype}; expected one of {expected}",
                tensor_name=tensor_name,
                shape=tensors[tensor_name][0],
                dtype=tensor.dtype,
                expected=expected_value,
                actual=tensor.dtype,
            )
        tensor = state[tensor_name]
        if (
            not quantized
            and tensor.is_floating_point()
            and not bool(torch.isfinite(tensor).all())
        ):
            raise _invalid_state(
                backend=backend,
                filename=checkpoint_filename,
                path=checkpoint_path,
                reason=f"{name} tensor contains non-finite values",
                tensor_name=tensor_name,
                shape=tensors[tensor_name][0],
                dtype=tensor.dtype,
                actual=tensor.dtype,
            )


def validate_safetensors_artifact(
    root: Union[str, Path],
    artifact: WeightArtifactContract,
    *,
    backend: str = "unknown",
) -> None:
    """Validate one local safetensors artifact and its header inventory.

    Args:
        root: Root directory of the local model snapshot.
        artifact: Expected artifact metadata.
        backend: Celune backend identifier.

    Raises:
        InvalidCheckpoint: The artifact is missing, changed, unreadable, or structurally invalid.
    """
    path = Path(root) / artifact.path
    filename = Path(artifact.path).name
    if not path.is_file():
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact is missing",
        )
    try:
        size = path.stat().st_size
        if size != artifact.size:
            raise InvalidCheckpoint(
                backend=backend,
                filename=filename,
                path=artifact.path,
                reason=f"model artifact size mismatch: expected {artifact.size}, got {size}",
            )
        digest = _sha256(path)
    except InvalidCheckpoint:
        raise
    except OSError as exc:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact could not be read",
        ) from exc
    if digest != artifact.sha256:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact checksum mismatch",
        )
    if artifact.inventory is None:
        return

    try:
        tensors = _read_safetensors_inventory(path)
    except Exception as exc:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact could not be parsed",
        ) from exc

    actual = _inventory_from_header(tensors)
    if actual != artifact.inventory:
        raise InvalidCheckpoint(
            backend=backend,
            filename=filename,
            path=artifact.path,
            reason="model artifact tensor inventory mismatch",
        )


def _inventory_from_header(
    tensors: Mapping[str, tuple[tuple[int, ...], str]],
) -> TensorInventoryContract:
    """Convert a safetensors header map into a compact inventory."""
    dtype_counts = Counter(dtype for _shape, dtype in tensors.values())
    parameter_count = sum(
        _shape_parameter_count(shape) for shape, _dtype in tensors.values()
    )
    return TensorInventoryContract(
        tensor_count=len(tensors),
        parameter_count=parameter_count,
        dtype_counts=tuple(sorted(dtype_counts.items())),
        inventory_sha256=_canonical_inventory(tensors),
    )


def _read_safetensors_inventory(
    path: Path,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read the safetensors header without materializing model tensors."""
    try:
        with path.open("rb") as handle:
            header_length_bytes = handle.read(8)
            if len(header_length_bytes) != 8:
                raise ModelContractError("safetensors header length is truncated")
            header_length = int.from_bytes(header_length_bytes, "little")
            header = json.loads(handle.read(header_length).decode("utf-8"))
    except ModelContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, OverflowError) as exc:
        raise ModelContractError("safetensors header is invalid") from exc

    if not isinstance(header, dict):
        raise ModelContractError("safetensors header is not an object")

    tensors: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw_tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_tensor, dict):
            raise ModelContractError("safetensors tensor entry is invalid")
        raw_shape = raw_tensor.get("shape")
        raw_dtype = raw_tensor.get("dtype")
        if not isinstance(raw_shape, list) or not isinstance(raw_dtype, str):
            raise ModelContractError(f"safetensors tensor metadata is invalid: {name}")
        tensors[name] = (
            tuple(int(dimension) for dimension in raw_shape),
            _dtype_name(raw_dtype),
        )
    return tensors


def _shape_parameter_count(shape: tuple[int, ...]) -> int:
    """Return the number of scalar values represented by one tensor shape."""
    count = 1
    for dimension in shape:
        count *= dimension
    return count
