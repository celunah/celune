# SPDX-License-Identifier: Apache-2.0
"""VRAM preset resolution and runtime accounting helpers for Celune."""

import gc
import os
import math
import contextlib
from typing import Optional, cast
from dataclasses import dataclass
from collections.abc import Mapping, Iterator

import torch
from torch import nn

from .i18n import string
from .constants import (
    TIERS,
    VRAM_BUDGETS,
    VRAM_REQUIREMENTS,
    AGENT_CONTEXT_SPACE,
    PERSONA_CONTEXT_SPACE,
    VRAM_SYSTEM_RESERVE_GIB,
    PERSONA_DEFAULT_MODEL_ID,
    persona_model_tier,
    remote_code_model_revision,
)
from .typing.common import JSON, VramTier, JSONSerializable

QWEN3_0_6B_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_1_7B_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

TEST_BACKENDS = ("fake", "counting")
KNOWN_TTS_BACKENDS = (
    "mini",
    "qwen3",
    "dotstts",
    "voxcpm2",
    "luxtts",
    "fireredtts3",
    *TEST_BACKENDS,
)
BACKENDS_ALLOWED: Mapping[VramTier, list[str]] = {
    "low": ["mini", "qwen3", "luxtts", *TEST_BACKENDS],
    "medium": list(KNOWN_TTS_BACKENDS),
    "high": list(KNOWN_TTS_BACKENDS),
    "xhigh": list(KNOWN_TTS_BACKENDS),
}


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


@dataclass(frozen=True, slots=True)
class VramProfile:
    """Measured peak VRAM for one exact Celune hardware/configuration key."""

    gpu_name: str
    configuration: tuple[str, ...]
    model_revision: str
    tts_revision: str
    dtype: str
    quantization: str
    components: tuple[str, ...]
    context_size: int
    load_peak_bytes: int
    warmup_peak_bytes: int
    inference_peak_bytes: int

    @property
    def peak_bytes(self) -> int:
        """Return the greatest recorded phase peak."""
        return max(
            self.load_peak_bytes,
            self.warmup_peak_bytes,
            self.inference_peak_bytes,
        )


VRAM_PROFILES: dict[tuple[str, ...], VramProfile] = {}


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Return whether an exception reports a CUDA allocation failure."""
    out_of_memory_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(out_of_memory_type, type) and isinstance(error, out_of_memory_type):
        return True
    message = str(error).casefold()
    return "cuda" in message and (
        "out of memory" in message or "alloc_failed" in message
    )


def release_cuda_after_oom() -> None:
    """Collect released model references and return unused CUDA blocks."""
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(RuntimeError, AssertionError):
            torch.cuda.empty_cache()


def vram_profile_key(
    config: Optional[Mapping[str, JSONSerializable]],
) -> tuple[str, ...]:
    """Return the exact hardware and model configuration key for profile lookup."""
    from .backends.tts.contracts import MODEL_CONTRACTS

    values = config or {}
    preset = resolve_vram_preset(config)
    raw_persona = values.get("persona")
    persona = raw_persona if isinstance(raw_persona, dict) else {}
    raw_agent = values.get("agent")
    agent = raw_agent if isinstance(raw_agent, dict) else {}
    backend_value = values.get("backend")
    backend = (
        backend_value.strip().lower()
        if isinstance(backend_value, str) and backend_value.strip()
        else preset.default_backend
    )
    model_id = persona.get("model_id")
    model_id = model_id.strip() if isinstance(model_id, str) else "default"
    resolved_model_id = model_id
    if resolved_model_id == "default":
        resolved_model_id = PERSONA_DEFAULT_MODEL_ID
    model_revision = remote_code_model_revision(resolved_model_id) or "unknown"
    tts_contracts = tuple(
        contract
        for contract in MODEL_CONTRACTS
        if contract.backend_id == backend
        and (backend != "qwen3" or contract.model_id == preset.qwen3_clone_model_id)
    )
    tts_model_ids = ",".join(sorted({contract.model_id for contract in tts_contracts}))
    tts_revisions = ",".join(
        sorted(
            {
                f"{contract.revision}:{contract.variant or 'default'}"
                for contract in tts_contracts
            }
        )
    )
    tts_components = ",".join(
        sorted(
            {
                component.name
                for contract in tts_contracts
                for component in contract.components
            }
        )
    )
    tts_dtypes = ",".join(
        sorted(
            {
                f"{component.name}:{rule.prefix}:{'|'.join(rule.dtypes)}"
                for contract in tts_contracts
                for component in contract.components
                for rule in component.runtime_dtypes
            }
        )
    )
    gpu_name = "unavailable"
    capability: tuple[int, int] = (0, 0)
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
        except (RuntimeError, AssertionError):
            pass
    mode = values.get("mode")
    mode_name = mode.strip().lower() if isinstance(mode, str) else "converse"
    persona_requested = bool(persona.get("enabled", True))
    known_persona_tier = persona_model_tier(resolved_model_id)
    persona_active = (
        preset.persona_enabled
        and persona_requested
        and not (preset.tier == "high" and known_persona_tier == "smart")
        and mode_name in {"converse", "agent"}
    )
    agent_active = (
        mode_name == "agent" and persona_active and agent_vram_compatible(config)
    )
    persona_context = persona.get("context_size", PERSONA_CONTEXT_SPACE)
    agent_context = agent.get("context_size", AGENT_CONTEXT_SPACE)
    quantize_setting = os.getenv("CELUNE_TTS_QUANTIZE")
    quantize_requested = (
        quantize_setting.strip().lower() in {"1", "true", "on", "yes", "enabled"}
        if quantize_setting is not None
        else values.get("quantize", True) is True
    )
    quantization = "native-lux"
    if backend != "luxtts":
        quantization = "bf16"
        if quantize_requested:
            if capability >= (8, 9):
                quantization = "fp8"
            elif capability >= (8, 0):
                quantization = "int8"
    return (
        gpu_name,
        preset.tier,
        mode_name,
        backend,
        tts_model_ids or "unknown",
        tts_revisions or "unknown",
        tts_dtypes or "unknown",
        tts_components or "unknown",
        model_id,
        model_revision,
        preset.persona_quantization,
        str(min(persona_context, PERSONA_CONTEXT_SPACE))
        if isinstance(persona_context, int) and not isinstance(persona_context, bool)
        else str(PERSONA_CONTEXT_SPACE),
        str(min(agent_context, AGENT_CONTEXT_SPACE))
        if isinstance(agent_context, int) and not isinstance(agent_context, bool)
        else str(AGENT_CONTEXT_SPACE),
        quantization,
        f"persona-kv:{bool(persona.get('quantize_kv_cache', values.get('quantize_kv_cache', True)))}",
        "persona" if persona_active else "no-persona",
        "agent" if agent_active else "no-agent",
        "normalizer" if values.get("use_normalizer", False) else "no-normalizer",
        "whisper-on-demand" if persona_active else "no-whisper",
    )


def vram_profile_fits(
    config: Optional[Mapping[str, JSONSerializable]],
) -> Optional[bool]:
    """Return whether a confirmed profile fits both preset and current headroom.

    ``None`` means no exact hardware/configuration profile is available and the
    caller should allow startup while warning the user.
    """
    profile = VRAM_PROFILES.get(vram_profile_key(config))
    if profile is None:
        return None

    tier = resolve_vram_preset(config).tier
    budget_bytes = VRAM_BUDGETS[tier] * 1024**3
    if profile.peak_bytes > budget_bytes:
        return False

    if torch.cuda.is_available():
        try:
            available_bytes, _ = torch.cuda.mem_get_info(0)
            current_allocated = torch.cuda.memory_allocated(0)
        except (RuntimeError, AssertionError):
            return None
        system_reserve_bytes = VRAM_SYSTEM_RESERVE_GIB * 1024**3
        additional_bytes = max(0, profile.peak_bytes - current_allocated)
        if additional_bytes > max(0, available_bytes - system_reserve_bytes):
            return False
    return True


def _allowed_backends(
    preset: VramPreset,
) -> tuple[str, ...]:
    """Return backend names whose known weight floor fits the preset."""
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

    warnings: list[str] = []
    if torch.cuda.is_available():
        _, total_bytes = torch.cuda.mem_get_info(0)
        total_gb = math.ceil(total_bytes / 1024**3)

        tier = configured_tier

        while VRAM_REQUIREMENTS[tier] > total_gb and tier != "low":
            tier = TIERS[TIERS.index(tier) - 1]

        if tier != configured_tier:
            warnings.append(
                string(
                    "vram.preset_reduced",
                    total=total_gb,
                    requested=configured_tier,
                    selected=tier,
                )
            )

    if configured_tier == "high" and config is not None:
        raw_persona = config.get("persona")
        persona = raw_persona if isinstance(raw_persona, dict) else {}
        model_id = persona.get("model_id")
        if isinstance(model_id, str) and persona_model_tier(model_id) == "smart":
            warnings.append(string("vram.smart_model_requires_xhigh"))

    profile_status = vram_profile_fits(config)
    if profile_status is None:
        warnings.append(string("vram.configuration_unprofiled"))
    elif not profile_status:
        warnings.append(string("vram.profile_exceeds_budget"))

    return "\n".join(warnings) if warnings else None


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
            allow_voxcpm2="voxcpm2" in BACKENDS_ALLOWED["high"],
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
    if normalized in _allowed_backends(preset):
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
    return normalized in _allowed_backends(preset)


def agent_vram_compatible(
    config: Optional[Mapping[str, JSONSerializable]],
) -> bool:
    """Return whether the resolved VRAM preset supports agent mode."""
    tier = resolve_vram_preset(config).tier
    if tier not in {"high", "xhigh"}:
        return False
    if tier == "high" and config is not None:
        raw_persona = config.get("persona")
        persona = raw_persona if isinstance(raw_persona, dict) else {}
        model_id = persona.get("model_id")
        if isinstance(model_id, str) and persona_model_tier(model_id) == "smart":
            return False
    return True


def _iter_component_tensors(
    value: object,
    seen: set[int],
    *,
    depth: int = 0,
) -> Iterator[torch.Tensor]:
    """Yield tensors retained by one runtime object."""
    if depth > 8:
        return

    if isinstance(value, torch.Tensor):
        yield value
        return

    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, nn.Module):
        yield from value.parameters(recurse=True)
        yield from value.buffers(recurse=True)
        attributes = vars(value)
        for name, attribute in attributes.items():
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            yield from _iter_component_tensors(attribute, seen, depth=depth + 1)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_component_tensors(item, seen, depth=depth + 1)
        return

    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_component_tensors(item, seen, depth=depth + 1)
        return

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for attribute in attributes.values():
            yield from _iter_component_tensors(attribute, seen, depth=depth + 1)


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


def component_memory_usage(
    name: str,
    value: object,
    process: Optional[Mapping[str, JSONSerializable]] = None,
) -> JSON:
    """Return memory statistics for one runtime component.

    Args:
        name: Display name of the component being measured.
        value: Runtime object that owns the component's tensors.
        process: Owning process CUDA allocator statistics, when available.

    Returns:
        JSON: Component name, load state, device, and memory statistics.
    """
    storage_keys: set[tuple[str, Optional[int], int, int]] = set()
    devices: set[str] = set()
    tensor_bytes = 0
    tensor_count = 0
    for tensor in _iter_component_tensors(value, set()):
        pointer, size = _tensor_storage_size(tensor)
        key = (tensor.device.type, tensor.device.index, pointer, size)
        if key in storage_keys:
            continue
        storage_keys.add(key)
        devices.add(str(tensor.device))
        tensor_bytes += size
        tensor_count += 1

    if not devices:
        device = getattr(value, "device", None)
        device_name = str(device) if isinstance(device, torch.device) else "unknown"
        devices.add(device_name)
    device_name = ",".join(sorted(devices))

    reserved_bytes = tensor_bytes
    peak_allocated_bytes = tensor_bytes
    if any(device.startswith("cuda") for device in devices) and process is not None:
        reserved_bytes = vram_report_int(process, "reserved_bytes")
        peak_allocated_bytes = vram_report_int(process, "peak_allocated_bytes")

    return {
        "name": name,
        "loaded": value is not None,
        "available": True,
        "device": device_name,
        "allocated_bytes": tensor_bytes,
        "reserved_bytes": reserved_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
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
        "component": component_memory_usage(name, model, process),
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
                    "device": "unknown",
                    "allocated_bytes": 0,
                    "reserved_bytes": 0,
                    "peak_allocated_bytes": 0,
                    "tensor_count": 0,
                }
            }
        if isinstance(report, dict):
            component = report.get("component")
            if isinstance(component, dict):
                component["name"] = name
            return cast(JSON, report)

    return backend_vram_report(name, getattr(backend, "model", None))


def runtime_vram_report(
    runtime: object,
    extra_components: Optional[Mapping[str, object]] = None,
) -> JSON:
    """Return process and per-component memory usage for an active runtime.

    Args:
        runtime: Celune runtime object whose active components should be measured.
        extra_components: Additional main-process component names and tensor-owning
            runtime objects to include in the report.

    Returns:
        JSON: Aggregate allocator memory and component-level memory usage.
    """
    process = cuda_process_memory()
    allocated = cast(int, process["allocated_bytes"])
    reserved = cast(int, process["reserved_bytes"])
    peak_allocated = cast(int, process["peak_allocated_bytes"])
    cuda_available = cast(bool, process["available"])
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
            cuda_available = cuda_available or report.get("process_available") is True
            allocated += vram_report_int(report, "process_allocated_bytes")
            reserved += vram_report_int(report, "process_reserved_bytes")
            peak_allocated += vram_report_int(report, "process_peak_allocated_bytes")

    vision = getattr(runtime, "vision", None)
    persona_runtime = getattr(vision, "runtime", None)
    persona_backend = getattr(persona_runtime, "backend", None)
    persona_model = getattr(persona_backend, "model", None)
    if persona_model is not None:
        components.append(component_memory_usage("persona", persona_model, process))

    normalizer = getattr(runtime, "llm", None)
    if normalizer is not None:
        components.append(
            component_memory_usage(
                "normalizer",
                (normalizer, getattr(runtime, "tokenizer", None)),
                process,
            )
        )

    selector = getattr(runtime, "_agent_needle_selector", None)
    agent_handler = getattr(selector, "handler", None)
    agent_model = getattr(agent_handler, "model", None)
    if agent_model is not None:
        components.append(component_memory_usage("agent", agent_model, process))

    if extra_components is not None:
        for name, value in extra_components.items():
            if value is not None:
                components.append(component_memory_usage(name, value, process))

    return {
        "available": True,
        "cuda_available": cuda_available,
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
