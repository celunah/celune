# SPDX-License-Identifier: Apache-2.0
"""VRAM preset resolution and runtime accounting helpers for Celune."""

import gc
import os
import sys
import math
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, cast
from collections.abc import Mapping, Iterator

import torch
import psutil
from torch import nn

from .i18n import string
from .constants import (
    TIERS,
    APP_NAME,
    VRAM_BUDGETS,
    VRAM_REQUIREMENTS,
    AGENT_CONTEXT_SPACE,
    PERSONA_CONTEXT_SPACE,
    NORMALIZER_MODEL_ID,
    VRAM_SYSTEM_RESERVE_GIB,
    PERSONA_DEFAULT_MODEL_ID,
    DEFAULT_PERSONA_SPEECH_MODEL_ID,
    DEFAULT_PERSONA_SPEECH_MODEL_REVISION,
    persona_model_tier,
    remote_code_model_revision,
)
from .paths import huggingface_hub_cache_dir
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


def _windows_shared_memory_capacity_bytes() -> int:
    """Return the maximum WDDM shared-memory pool from installed system RAM."""
    if sys.platform != "win32":
        return 0

    total_bytes = psutil.virtual_memory().total
    return min(
        total_bytes * 4 // 5,
        max(total_bytes - 16 * 1024**3, total_bytes // 2),
    )


def _windows_shared_memory_headroom_bytes() -> int:
    """Return the shared-memory capacity currently free above the system reserve."""
    if sys.platform != "win32":
        return 0

    available_bytes = psutil.virtual_memory().available
    reserve_bytes = VRAM_SYSTEM_RESERVE_GIB * 1024**3
    return min(
        _windows_shared_memory_capacity_bytes(),
        max(0, available_bytes - reserve_bytes),
    )


def _cuda_total_capacity_bytes() -> int:
    """Return dedicated CUDA memory plus maximum Windows WDDM shared memory."""
    _, dedicated_bytes = torch.cuda.mem_get_info(0)
    return dedicated_bytes + _windows_shared_memory_capacity_bytes()


def _cached_hub_revision(model_id: str) -> str:
    """Return the cached commit selected by the model's ``main`` reference."""
    if "/" not in model_id:
        return "unknown"

    configured_cache = os.environ.get("HF_HUB_CACHE")
    configured_home = os.environ.get("HF_HOME")
    cache_root = (
        Path(configured_cache)
        if configured_cache
        else Path(configured_home) / "hub"
        if configured_home
        else huggingface_hub_cache_dir()
    )
    revision_path = (
        cache_root / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    )
    with contextlib.suppress(OSError, UnicodeError):
        revision = revision_path.read_text(encoding="utf-8").strip()
        if revision:
            return revision
    return "unknown"


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
    mode = values.get("mode")
    mode_name = mode.strip().lower() if isinstance(mode, str) else "converse"
    model_id = persona.get("model_id")
    model_id = model_id.strip() if isinstance(model_id, str) else "default"
    resolved_model_id = model_id
    if resolved_model_id == "default":
        resolved_model_id = PERSONA_DEFAULT_MODEL_ID
    persona_requested = bool(persona.get("enabled", True))
    known_persona_tier = persona_model_tier(resolved_model_id)
    persona_active = (
        preset.persona_enabled
        and persona_requested
        and not (preset.tier == "high" and known_persona_tier == "smart")
        and mode_name in {"converse", "agent"}
    )
    model_revision = (
        remote_code_model_revision(resolved_model_id) or "unknown"
        if persona_active
        else "no-persona-revision"
    )
    speech_model_value = persona.get("speech_model_id")
    speech_model_id = (
        speech_model_value.strip()
        if isinstance(speech_model_value, str) and speech_model_value.strip()
        else DEFAULT_PERSONA_SPEECH_MODEL_ID
    )
    speech_revision = "no-whisper-revision"
    if persona_active:
        speech_revision = (
            DEFAULT_PERSONA_SPEECH_MODEL_REVISION
            if speech_model_id == DEFAULT_PERSONA_SPEECH_MODEL_ID
            else "unknown"
        )
    normalizer_enabled = bool(values.get("use_normalizer", False))
    normalizer_revision = (
        _cached_hub_revision(NORMALIZER_MODEL_ID)
        if normalizer_enabled
        else "no-normalizer-revision"
    )
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
    agent_active = (
        mode_name == "agent" and persona_active and agent_vram_compatible(config)
    )
    needle_revision = "no-needle"
    if agent_active:
        from .agent.needle.checkpoints import NEEDLE_MODEL_REVISION

        needle_revision = NEEDLE_MODEL_REVISION
    agent_context = agent.get("context_size", AGENT_CONTEXT_SPACE)
    persona_context = (
        agent_context
        if agent_active
        else persona.get("context_size", PERSONA_CONTEXT_SPACE)
    )
    context_limit = AGENT_CONTEXT_SPACE if agent_active else PERSONA_CONTEXT_SPACE
    if not persona_active:
        context_key = "no-persona-context"
    elif isinstance(persona_context, int) and not isinstance(persona_context, bool):
        context_key = str(min(persona_context, context_limit))
    else:
        context_key = str(context_limit)
    agent_context_key = (
        str(min(agent_context, AGENT_CONTEXT_SPACE))
        if agent_active
        and isinstance(agent_context, int)
        and not isinstance(agent_context, bool)
        else str(AGENT_CONTEXT_SPACE)
        if agent_active
        else "no-agent-context"
    )
    quantize_setting = os.getenv("CELUNE_TTS_QUANTIZE")
    quantize_requested = (
        quantize_setting.strip().lower() in {"1", "true", "on", "yes", "enabled"}
        if quantize_setting is not None
        else values.get("quantize", True) is True
    )
    quantization = "native-lux" if backend == "luxtts" else "bf16"
    if backend not in {"luxtts", "mini"}:
        if quantize_requested:
            if capability >= (8, 9):
                quantization = "fp8"
            elif capability >= (8, 0):
                quantization = "int8"
    elif backend == "mini":
        quantization = "cpu"
    cache_quantization = bool(
        persona.get("quantize_kv_cache", values.get("quantize_kv_cache", True))
    )
    kv_cache_signature = (
        f"persona-kv:{cache_quantization}" if persona_active else "no-persona-kv"
    )
    tts_kv_cache_signature = (
        (f"tts-kv:{bool(values.get('quantize_kv_cache', True))}",)
        if backend in {"qwen3", "fireredtts3", "voxcpm2"}
        else ()
    )
    return (
        gpu_name,
        sys.platform,
        preset.tier,
        mode_name,
        backend,
        tts_model_ids or "unknown",
        tts_revisions or "unknown",
        tts_dtypes or "unknown",
        tts_components or "unknown",
        resolved_model_id if persona_active else "no-persona-model",
        model_revision,
        preset.persona_quantization if persona_active else "no-persona-quantization",
        context_key,
        agent_context_key,
        quantization,
        kv_cache_signature,
        *tts_kv_cache_signature,
        "persona" if persona_active else "no-persona",
        "agent" if agent_active else "no-agent",
        "normalizer" if normalizer_enabled else "no-normalizer",
        f"normalizer-id:{NORMALIZER_MODEL_ID}"
        if normalizer_enabled
        else "no-normalizer-id",
        f"normalizer-revision:{normalizer_revision}",
        "whisper-on-demand" if persona_active else "no-whisper",
        f"whisper-id:{speech_model_id}" if persona_active else "no-whisper-id",
        f"whisper-revision:{speech_revision}"
        if persona_active
        else "no-whisper-revision",
        "whisper-quantization:int8" if persona_active else "no-whisper-quantization",
        f"needle-revision:{needle_revision}",
    )


_MEASURED_GPU_NAME = "NVIDIA GeForce RTX 5070"
_MEASURED_PLATFORM = "win32"


def _tts_profile_key(
    backend: str,
    model_id: str,
    revision: str,
    dtype: str,
    components: str,
    quantization: str,
) -> tuple[str, ...]:
    """Build a fixed key for a measured RTX 5070 TTS-only configuration."""
    tts_kv_cache_signature = (
        ("tts-kv:False",) if backend in {"qwen3", "fireredtts3", "voxcpm2"} else ()
    )
    return (
        _MEASURED_GPU_NAME,
        _MEASURED_PLATFORM,
        "xhigh",
        "speak",
        backend,
        model_id,
        revision,
        dtype,
        components,
        "no-persona-model",
        "no-persona-revision",
        "no-persona-quantization",
        "no-persona-context",
        "no-agent-context",
        quantization,
        "no-persona-kv",
        *tts_kv_cache_signature,
        "no-persona",
        "no-agent",
        "no-normalizer",
        "no-normalizer-id",
        "normalizer-revision:no-normalizer-revision",
        "no-whisper",
        "no-whisper-id",
        "no-whisper-revision",
        "no-whisper-quantization",
        "needle-revision:no-needle",
    )


def _reported_peak_bytes(reported_gib: float) -> int:
    """Store a two-decimal GiB sample with a 0.01 GiB upward allowance."""
    return math.ceil((reported_gib + 0.01) * 1024**3)


def _measured_vram_profile(
    key: tuple[str, ...],
    phase_peaks_gib: tuple[float, float, float],
    *,
    model_revision: str,
    quantization: str,
    components: tuple[str, ...],
    context_size: int = 0,
    dtype: Optional[str] = None,
) -> VramProfile:
    """Create a measured profile from conservatively rounded phase peaks."""
    return VramProfile(
        gpu_name=key[0],
        configuration=key,
        model_revision=model_revision,
        tts_revision=key[6],
        dtype=dtype or key[7],
        quantization=quantization,
        components=components,
        context_size=context_size,
        load_peak_bytes=_reported_peak_bytes(phase_peaks_gib[0]),
        warmup_peak_bytes=_reported_peak_bytes(phase_peaks_gib[1]),
        inference_peak_bytes=_reported_peak_bytes(phase_peaks_gib[2]),
    )


_QWEN3_XHIGH_KEY = _tts_profile_key(
    "qwen3",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "fd4b254389122332181a7c3db7f27e918eec64e3:default",
    "speech_tokenizer::torch.float32,talker::torch.bfloat16",
    "speech_tokenizer,talker",
    "fp8",
)
_DOTSTTS_XHIGH_KEY = _tts_profile_key(
    "dotstts",
    "rednote-hilab/dots.tts-mf",
    "c28105adc8228143392b4e346994ff613ee48a06:default",
    "core::torch.bfloat16,speaker_encoder::torch.float32|torch.int64,vocoder::torch.float32",
    "core,speaker_encoder,vocoder",
    "fp8",
)
_FIREREDTTS3_XHIGH_KEY = _tts_profile_key(
    "fireredtts3",
    "FireRedTeam/FireRedTTS3",
    "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610:default",
    (
        "redae:decoder:torch.float32,redae:encoder:torch.bfloat16,"
        "tts_core:backbone_llm:torch.bfloat16,tts_core:dit:torch.float32,"
        "tts_core:dit_head:torch.float32,tts_core:patch_encoder:torch.float32,"
        "tts_core:spk_proj_dit:torch.float32,tts_core:spk_proj_llm:torch.float32,"
        "tts_core:stop_head:torch.bfloat16"
    ),
    "redae,tts_core",
    "fp8",
)
_LUXTTS_XHIGH_KEY = _tts_profile_key(
    "luxtts",
    "YatharthS/LuxTTS",
    "527f245a276a0eb42ea103a7a512bcfd771eb9b6:default",
    "unknown",
    "unknown",
    "native-lux",
)
_MINI_XHIGH_KEY = _tts_profile_key(
    "mini",
    "lunahr/pocket-tts-ungated",
    (
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:english,"
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:french_24l,"
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:german,"
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:italian,"
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:portuguese,"
        "d03cd73415a8d46d8eb115c7b524aebb0a729f4a:spanish"
    ),
    "flow_lm::torch.bfloat16",
    "flow_lm",
    "cpu",
)
_QWEN3_PERSONA_WHISPER_HIGH_KEY = (
    _MEASURED_GPU_NAME,
    _MEASURED_PLATFORM,
    "high",
    "converse",
    "qwen3",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "fd4b254389122332181a7c3db7f27e918eec64e3:default",
    "speech_tokenizer::torch.float32,talker::torch.bfloat16",
    "speech_tokenizer,talker",
    "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
    "ce72a7c22aacb493fb94478de3bfbe834c61844a",
    "4bit",
    "2048",
    "no-agent-context",
    "fp8",
    "persona-kv:True",
    "tts-kv:False",
    "persona",
    "no-agent",
    "no-normalizer",
    "no-normalizer-id",
    "normalizer-revision:no-normalizer-revision",
    "whisper-on-demand",
    "whisper-id:openai/whisper-large-v3-turbo",
    "whisper-revision:41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "whisper-quantization:int8",
    "needle-revision:no-needle",
)

VRAM_PROFILES.update(
    {
        _QWEN3_XHIGH_KEY: _measured_vram_profile(
            _QWEN3_XHIGH_KEY,
            (4.53, 4.91, 4.91),
            model_revision="no-persona",
            quantization="fp8 requested; backend inactive",
            components=("tts:qwen3", "speech_tokenizer", "talker"),
        ),
        _DOTSTTS_XHIGH_KEY: _measured_vram_profile(
            _DOTSTTS_XHIGH_KEY,
            (5.43, 6.26, 6.26),
            model_revision="no-persona",
            quantization="fp8 requested; backend inactive",
            components=("tts:dotstts", "core", "speaker_encoder", "vocoder"),
        ),
        _FIREREDTTS3_XHIGH_KEY: _measured_vram_profile(
            _FIREREDTTS3_XHIGH_KEY,
            (6.39, 6.39, 6.39),
            model_revision="no-persona",
            quantization="fp8 requested; backend inactive",
            components=("tts:fireredtts3", "redae", "tts_core"),
        ),
        _LUXTTS_XHIGH_KEY: _measured_vram_profile(
            _LUXTTS_XHIGH_KEY,
            (1.62, 1.62, 1.64),
            model_revision="no-persona",
            quantization="ONNX; not applicable",
            components=("tts:luxtts",),
        ),
        _MINI_XHIGH_KEY: _measured_vram_profile(
            _MINI_XHIGH_KEY,
            (0.46, 0.46, 0.46),
            model_revision="no-persona",
            quantization="CPU model; not applicable",
            components=("tts:mini", "flow_lm"),
        ),
        _QWEN3_PERSONA_WHISPER_HIGH_KEY: _measured_vram_profile(
            _QWEN3_PERSONA_WHISPER_HIGH_KEY,
            (8.67, 9.00, 9.98),
            model_revision=(
                "ce72a7c22aacb493fb94478de3bfbe834c61844a; "
                "Whisper 41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
            ),
            dtype=("TTS mixed; Persona bfloat16 compute; Whisper bfloat16 compute"),
            quantization=("Persona 4bit; Whisper int8; TTS fp8 requested but inactive"),
            components=("tts:qwen3", "persona:qwen3-vl-4b", "whisper"),
            context_size=2048,
        ),
    }
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
        if sys.platform == "win32":
            available_bytes += _windows_shared_memory_headroom_bytes()
            system_reserve_bytes = 0
        else:
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
        total_bytes = _cuda_total_capacity_bytes()
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
        warnings.append(string("vram.profile_exceeds_budget", app_name=APP_NAME))

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
        total_bytes = _cuda_total_capacity_bytes()
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
