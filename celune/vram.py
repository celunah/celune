# SPDX-License-Identifier: MIT
"""VRAM preset resolution helpers for Celune."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal, Optional, cast

import torch

from .constants import JSONSerializable, VRAM_REQUIREMENTS, TIERS

type VramTier = Literal["low", "medium", "high", "xhigh"]

QWEN3_0_6B_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_1_7B_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


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
    qwen3_native_supported: bool


def vram_tier(config: Optional[Mapping[str, JSONSerializable]]) -> VramTier:
    """Return the configured VRAM tier with a safe fallback."""
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
    """Validate a VRAM preset and return an appropriate warning message."""
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
    """Resolve Celune runtime settings from the documented VRAM presets."""
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
            default_backend="qwen3",
            allow_voxcpm2=False,
            qwen3_clone_model_id=QWEN3_0_6B_MODEL,
            persona_enabled=False,
            persona_quantization="4bit",
            normalizer_device="cpu",
            qwen3_native_supported=False,
        )

    if tier == "medium":
        return VramPreset(
            tier="medium",
            default_backend="qwen3",
            allow_voxcpm2=False,
            qwen3_clone_model_id=QWEN3_1_7B_MODEL,
            persona_enabled=False,
            persona_quantization="4bit",
            normalizer_device="cpu",
            qwen3_native_supported=True,
        )

    if tier == "high":
        return VramPreset(
            tier="high",
            default_backend="qwen3",
            allow_voxcpm2=True,
            qwen3_clone_model_id=QWEN3_1_7B_MODEL,
            persona_enabled=True,
            persona_quantization="4bit",
            normalizer_device="cpu",
            qwen3_native_supported=True,
        )

    return VramPreset(
        tier="xhigh",
        default_backend="qwen3",
        allow_voxcpm2=True,
        qwen3_clone_model_id=QWEN3_1_7B_MODEL,
        persona_enabled=True,
        persona_quantization="8bit",
        normalizer_device="cuda",
        qwen3_native_supported=True,
    )


def resolve_backend_name(
    config: Optional[Mapping[str, JSONSerializable]],
    requested_backend: Optional[str],
) -> str:
    """Return the backend permitted by the configured VRAM tier."""
    preset = resolve_vram_preset(config)
    if requested_backend is None:
        return preset.default_backend

    normalized = requested_backend.strip().lower()
    if normalized == "voxcpm2" and not preset.allow_voxcpm2:
        return preset.default_backend
    if normalized in {"qwen3", "voxcpm2"}:
        return normalized
    return preset.default_backend


def backend_allowed(
    config: Optional[Mapping[str, JSONSerializable]],
    backend_name: str,
) -> bool:
    """Return whether the named backend is permitted by the VRAM tier."""
    normalized = backend_name.strip().lower()
    preset = resolve_vram_preset(config)
    if normalized == "qwen3":
        return True
    if normalized == "voxcpm2":
        return preset.allow_voxcpm2
    return True
