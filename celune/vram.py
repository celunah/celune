# SPDX-License-Identifier: MIT
"""VRAM preset resolution helpers for Celune."""

import math
from dataclasses import dataclass
from typing import Optional, cast
from collections.abc import Mapping

import torch

from .typing.common import VramTier
from .constants import JSONSerializable, VRAM_REQUIREMENTS, TIERS

QWEN3_0_6B_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN3_1_7B_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

TEST_BACKENDS = ("fake", "counting")
BACKENDS_ALLOWED: Mapping[VramTier, list[str]] = {
    "low": ["mini", "qwen3", *TEST_BACKENDS],
    "medium": ["mini", "qwen3", *TEST_BACKENDS],
    "high": ["mini", "qwen3", "dotstts", "voxcpm2", *TEST_BACKENDS],
    "xhigh": ["mini", "qwen3", "dotstts", "voxcpm2", *TEST_BACKENDS],
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
    if normalized in BACKENDS_ALLOWED[preset.tier]:
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
    return normalized in BACKENDS_ALLOWED[preset.tier]
