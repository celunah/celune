# SPDX-License-Identifier: Apache-2.0
"""VRAM preset resolution and runtime accounting helpers for Celune."""

import math
from dataclasses import dataclass
from collections.abc import Iterator, Mapping
from typing import Optional, cast

import torch
from torch import nn

from .constants import TIERS, VRAM_REQUIREMENTS
from .typing.common import JSON, JSONSerializable, VramTier

QWEN3_0_6B_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_1_7B_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

TEST_BACKENDS = ("fake", "counting")
PERSONA_HIGH_ALLOWED_BACKENDS = ("mini", "qwen3", "luxtts", *TEST_BACKENDS)
BACKENDS_ALLOWED: Mapping[VramTier, list[str]] = {
    "low": ["mini", "qwen3", "luxtts", *TEST_BACKENDS],
    "medium": ["mini", "qwen3", "luxtts", *TEST_BACKENDS],
    "high": [
        "mini",
        "qwen3",
        "dotstts",
        "voxcpm2",
        "luxtts",
        *TEST_BACKENDS,
    ],
    "xhigh": [
        "mini",
        "qwen3",
        "dotstts",
        "voxcpm2",
        "luxtts",
        *TEST_BACKENDS,
    ],
}


def _persona_requested(
    config: Optional[Mapping[str, JSONSerializable]],
) -> bool:
    """Return whether the configuration leaves Persona enabled."""
    if config is None:
        return False

    raw_persona = config.get("persona", config.get("pyop", {}))
    if isinstance(raw_persona, bool):
        return raw_persona
    if not isinstance(raw_persona, dict):
        return True

    return bool(raw_persona.get("enabled", True))


@dataclass(frozen=True, slots=True)
class VramPreset:
    """Resolved runtime capabilities for one VRAM tier."""

    tier: VramTier
    default_backend: str
    allow_voxcpm2: bool
    qwen3_clone_model_id: str
    persona_enabled: bool
    persona_quantization: str
    normalizer_device: str


def _allowed_backends(
    config: Optional[Mapping[str, JSONSerializable]],
    preset: VramPreset,
) -> tuple[str, ...]:
    """Return backend names allowed by the preset and Persona configuration."""
    if preset.tier == "high" and preset.persona_enabled and _persona_requested(config):
        return PERSONA_HIGH_ALLOWED_BACKENDS
    return tuple(BACKENDS_ALLOWED[preset.tier])


def vram_tier(config: Optional[Mapping[str, JSONSerializable]]) -> VramTier:
    """Return the configured VRAM tier with a safe fallback.

    Args:
        config: Celune's current configuration.

    Returns:
        VramTier: The VRAM tier from current configuration.
    """
    if config is not None:
        raw = config.get("vram")
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"low", "medium", "high", "xhigh"}:
                return cast(VramTier, normalized)
    return "medium"


def validate_vram_preset(
    config: Optional[Mapping[str, JSONSerializable]],
) -> Optional[str]:
    """Validate a VRAM preset and return an appropriate warning message.

    Args:
        config: Celune's current configuration.

    Returns:
        Optional[str]: The warning message, if applicable.
    """
    configured_tier = vram_tier(config)

    if torch.cuda.is_available():
        _, total_bytes = torch.cuda.mem_get_info(0)
        total_gb = math.ceil(total_bytes / 1024**3)

        tier = configured_tier

        while VRAM_REQUIREMENTS[tier] > total_gb and tier != "low":
            tier = TIERS[TIERS.index(tier) - 1]

        if tier != configured_tier:
            return (
                f"You don't have enough VRAM ({total_gb} GB) for the "
                f"'{configured_tier}' preset. "
                f"Setting '{tier}' instead."
            )

    return None


def resolve_vram_preset(
    config: Optional[Mapping[str, JSONSerializable]],
) -> VramPreset:
    """Resolve Celune runtime settings from the documented VRAM presets.

    Args:
        config: Celune's current configuration.

    Returns:
        VramPreset: The resolved VRAM preset from configuration.
    """
    configured_tier = vram_tier(config)
    tier = configured_tier

    # downgrade VRAM preset if user doesn't have enough VRAM for the currently selected preset
    if torch.cuda.is_available():
        _, total_bytes = torch.cuda.mem_get_info(0)
        total_gb = math.ceil(total_bytes / 1024**3)

        while VRAM_REQUIREMENTS[tier] > total_gb and tier != "low":
            tier = TIERS[TIERS.index(tier) - 1]

    if tier == "low":
        return VramPreset(
            tier="low",
            default_backend="mini",
            allow_voxcpm2="voxcpm2" in BACKENDS_ALLOWED["low"],
            qwen3_clone_model_id=QWEN3_0_6B_MODEL,
            persona_enabled=False,
            persona_quantization="4bit",
            normalizer_device="cpu",
        )

    if tier == "medium":
        return VramPreset(
            tier="medium",
            default_backend="qwen3",
            allow_voxcpm2="voxcpm2" in BACKENDS_ALLOWED["medium"],
            qwen3_clone_model_id=QWEN3_1_7B_MODEL,
            persona_enabled=False,
            persona_quantization="4bit",
            normalizer_device="cpu",
        )

    if tier == "high":
        return VramPreset(
            tier="high",
            default_backend="qwen3",
            allow_voxcpm2=(
                "voxcpm2" in BACKENDS_ALLOWED["high"] and not _persona_requested(config)
            ),
            qwen3_clone_model_id=QWEN3_1_7B_MODEL,
            persona_enabled=True,
            persona_quantization="4bit",
            normalizer_device="cpu",
        )

    return VramPreset(
        tier="xhigh",
        default_backend="qwen3",
        allow_voxcpm2="voxcpm2" in BACKENDS_ALLOWED["xhigh"],
        qwen3_clone_model_id=QWEN3_1_7B_MODEL,
        persona_enabled=True,
        persona_quantization="8bit",
        normalizer_device="cuda",
    )


def resolve_backend_name(
    config: Optional[Mapping[str, JSONSerializable]],
    requested_backend: Optional[str],
) -> str:
    """Return the backend permitted by the configured VRAM tier.

    Args:
        config: Celune's current configuration.
        requested_backend: A backend name requested by the caller.

    Returns:
        str: The resolved permitted TTS backend by the currently configured VRAM tier.
    """
    preset = resolve_vram_preset(config)
    if requested_backend is None:
        return preset.default_backend

    normalized = requested_backend.strip().lower()
    if normalized in _allowed_backends(config, preset):
        return normalized
    return preset.default_backend


def backend_allowed(
    config: Optional[Mapping[str, JSONSerializable]],
    backend_name: str,
) -> bool:
    """Return whether the named backend is permitted by the VRAM tier.

    Args:
        config: Celune's current configuration.
        backend_name: A backend name requested by the caller.

    Returns:
        bool: Whether this backend is allowed by this VRAM tier, or ``False`` if the name is not a known Celune backend
        type name.
    """
    normalized = backend_name.strip().lower()
    preset = resolve_vram_preset(config)
    return normalized in _allowed_backends(config, preset)


def agent_vram_compatible(
    config: Optional[Mapping[str, JSONSerializable]],
) -> bool:
    """Return whether the resolved VRAM preset supports agent mode."""
    return resolve_vram_preset(config).tier == "xhigh"


def _iter_cuda_tensors(
    value: object,
    seen: set[int],
    *,
    depth: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield CUDA tensors retained by one runtime object."""
    if depth > 8:
        return

    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            yield value
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, nn.Module):
        for tensor in value.parameters(recurse=True):
            if tensor.device.type == "cuda":
                yield tensor
        for tensor in value.buffers(recurse=True):
            if tensor.device.type == "cuda":
                yield tensor
        attributes = vars(value)
        for name, attribute in attributes.items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            yield from _iter_cuda_tensors(attribute, seen, depth=depth + 1)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_cuda_tensors(item, seen, depth=depth + 1)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_cuda_tensors(item, seen, depth=depth + 1)
        return

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for attribute in attributes.values():
            yield from _iter_cuda_tensors(attribute, seen, depth=depth + 1)


def _tensor_storage_size(tensor: torch.Tensor) -> tuple[int, int]:
    """Return one tensor's storage pointer and allocated byte count."""
    try:
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.nbytes()
    except (AttributeError, RuntimeError, TypeError):
        return tensor.data_ptr(), tensor.numel() * tensor.element_size()


def vram_report_int(
    report: Mapping[str, JSONSerializable],
    key: str,
) -> int:
    """Return one integer field from a JSON-shaped VRAM report."""
    value = report.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def cuda_component_usage(name: str, value: object) -> JSON:
    """Return the resident CUDA tensor footprint of one runtime component.

    Args:
        name: Display name of the component being measured.
        value: Runtime object that owns the component's tensors.

    Returns:
        JSON: Component name, load state, tensor count, and deduplicated storage bytes.
    """
    storage_keys: set[tuple[str, Optional[int], int, int]] = set()
    tensor_bytes = 0
    tensor_count = 0
    for tensor in _iter_cuda_tensors(value, set()):
        pointer, size = _tensor_storage_size(tensor)
        key = (tensor.device.type, tensor.device.index, pointer, size)
        if key in storage_keys:
            continue
        storage_keys.add(key)
        tensor_bytes += size
        tensor_count += 1

    return {
        "name": name,
        "loaded": value is not None,
        "available": True,
        "tensor_bytes": tensor_bytes,
        "tensor_count": tensor_count,
    }


def cuda_process_memory() -> JSON:
    """Return allocated, reserved, and peak CUDA memory for this process."""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "device_count": 0,
        }

    allocated = 0
    reserved = 0
    peak_allocated = 0
    device_count = torch.cuda.device_count()
    for device_index in range(device_count):
        allocated += torch.cuda.memory_allocated(device_index)
        reserved += torch.cuda.memory_reserved(device_index)
        peak_allocated += torch.cuda.max_memory_allocated(device_index)

    return {
        "available": True,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "device_count": device_count,
    }


def backend_vram_report(name: str, model: object) -> JSON:
    """Return one backend component and its owning process memory report."""
    process = cuda_process_memory()
    return {
        "component": cuda_component_usage(name, model),
        "process_available": process["available"],
        "process_allocated_bytes": process["allocated_bytes"],
        "process_reserved_bytes": process["reserved_bytes"],
        "process_peak_allocated_bytes": process["peak_allocated_bytes"],
        "process_scope": "main",
    }


def _backend_report(backend: object, name: str) -> JSON:
    """Return a backend-owned report, including reports from isolated workers."""
    report_method = getattr(backend, "vram_report", None)
    if callable(report_method):
        try:
            report = report_method()
        except Exception:
            return {
                "component": {
                    "name": name,
                    "loaded": getattr(backend, "model", None) is not None,
                    "available": False,
                    "tensor_bytes": 0,
                    "tensor_count": 0,
                }
            }
        if isinstance(report, dict):
            component = report.get("component")
            if isinstance(component, dict):
                component["name"] = name
            return cast(JSON, report)

    return backend_vram_report(name, getattr(backend, "model", None))


def runtime_vram_report(runtime: object) -> JSON:
    """Return process and per-component CUDA usage for an active Celune runtime.

    Args:
        runtime: Celune runtime object whose active components should be measured.

    Returns:
        JSON: Aggregate CUDA memory and component-level resident tensor usage.
    """
    process = cuda_process_memory()
    allocated = cast(int, process["allocated_bytes"])
    reserved = cast(int, process["reserved_bytes"])
    peak_allocated = cast(int, process["peak_allocated_bytes"])
    available = cast(bool, process["available"])
    components: list[JSONSerializable] = []
    seen_backends: set[int] = set()

    for category, backend in (
        ("tts", getattr(runtime, "backend", None)),
        ("vc", getattr(runtime, "vc_backend", None)),
    ):
        if backend is None or id(backend) in seen_backends:
            continue
        seen_backends.add(id(backend))
        backend_name = str(getattr(backend, "name", category))
        report = _backend_report(backend, f"{category}/{backend_name}")
        component = report.get("component")
        if isinstance(component, dict):
            components.append(cast(JSONSerializable, component))
        if report.get("process_scope") == "worker":
            available = available or report.get("process_available") is True
            allocated += vram_report_int(report, "process_allocated_bytes")
            reserved += vram_report_int(report, "process_reserved_bytes")
            peak_allocated += vram_report_int(report, "process_peak_allocated_bytes")

    vision = getattr(runtime, "vision", None)
    persona_runtime = getattr(vision, "runtime", None)
    persona_backend = getattr(persona_runtime, "backend", None)
    persona_model = getattr(persona_backend, "model", None)
    if persona_model is not None:
        components.append(cuda_component_usage("persona", persona_model))

    normalizer = getattr(runtime, "llm", None)
    if normalizer is not None:
        components.append(cuda_component_usage("normalizer", normalizer))

    selector = getattr(runtime, "_agent_needle_selector", None)
    agent_handler = getattr(selector, "handler", None)
    agent_model = getattr(agent_handler, "model", None)
    if agent_model is not None:
        components.append(cuda_component_usage("agent", agent_model))

    return {
        "available": available,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "peak_allocated_bytes": peak_allocated,
        "device_count": cast(int, process["device_count"]),
        "components": components,
    }


def format_vram_bytes(value: int) -> str:
    """Format a byte count using binary units for compact UI diagnostics."""
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} TiB"
