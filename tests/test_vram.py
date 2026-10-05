# SPDX-License-Identifier: Apache-2.0
"""Tests for VRAM preset and backend compatibility rules."""

from unittest import mock

import torch
from torch import nn

from celune.vram import (
    VRAM_PROFILES,
    VramProfile,
    backend_allowed,
    vram_profile_key,
    format_vram_bytes,
    vram_profile_fits,
    resolve_vram_preset,
    resolve_backend_name,
    validate_vram_preset,
    is_cuda_out_of_memory,
    component_memory_usage,
)
from celune.celune import _tts_quantization_enabled
from celune.constants import VRAM_BUDGETS
from celune.typing.common import Config

from .support import CeluneTestCase


class TestVram(CeluneTestCase):
    """Test VRAM-aware backend selection."""

    def test_agent_mode_is_enabled_at_high_with_standard_persona(self) -> None:
        """Verify standard Persona agent mode fits the high preset policy."""
        config: Config = {
            "mode": "agent",
            "vram": "high",
            "persona": {"enabled": True},
        }
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            preset = resolve_vram_preset(config)
            assert preset.tier == "high"
            assert preset.persona_quantization == "4bit"
            assert resolve_backend_name(config, "dotstts") == "dotstts"

    def test_unprofiled_high_persona_backend_is_allowed(self) -> None:
        """Verify an unprofiled TTS combination proceeds at high VRAM."""
        config: Config = {"vram": "high", "persona": {"enabled": True}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert resolve_backend_name(config, "mini") == "mini"
            assert resolve_backend_name(config, "qwen3") == "qwen3"
            assert resolve_backend_name(config, "dotstts") == "dotstts"
            assert resolve_backend_name(config, "voxcpm2") == "voxcpm2"
            assert backend_allowed(config, "dotstts")
            assert backend_allowed(config, "voxcpm2")
            assert resolve_vram_preset(config).allow_voxcpm2

    def test_high_tier_allows_heavy_unprofiled_tts_without_persona(self) -> None:
        """Verify high-tier heavy TTS backends remain available without Persona."""
        config: Config = {"vram": "high", "persona": {"enabled": False}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert resolve_backend_name(config, "dotstts") == "dotstts"
            assert resolve_backend_name(config, "voxcpm2") == "voxcpm2"
            assert backend_allowed(config, "dotstts")
            assert backend_allowed(config, "voxcpm2")
            assert resolve_vram_preset(config).allow_voxcpm2

    def test_xhigh_persona_allows_heavy_tts_backends(self) -> None:
        """Verify xhigh-tier Persona sessions allow the other TTS backends."""
        config: Config = {"vram": "xhigh", "persona": {"enabled": True}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert resolve_backend_name(config, "dotstts") == "dotstts"
            assert resolve_backend_name(config, "voxcpm2") == "voxcpm2"
            assert backend_allowed(config, "dotstts")
            assert backend_allowed(config, "voxcpm2")

    def test_component_usage_reports_cpu_storage_and_device(self) -> None:
        """Verify component accounting reports CPU storage and its device."""
        report = component_memory_usage("test", nn.Linear(4, 4, dtype=torch.float32))

        assert report["loaded"] is True
        assert report["device"] == "cpu"
        assert report["allocated_bytes"] == 80
        assert report["reserved_bytes"] == 80
        assert report["peak_allocated_bytes"] == 80
        assert report["tensor_count"] == 2

    def test_format_vram_bytes_uses_binary_units(self) -> None:
        """Verify the diagnostic formatter keeps byte units readable."""
        assert format_vram_bytes(0) == "0 B"
        assert format_vram_bytes(1024**3) == "1.00 GiB"

    def test_tts_quantization_defaults_on_except_for_luxtts(self) -> None:
        """Verify default quantization covers TTS models except LuxTTS."""
        with mock.patch(
            "celune.config.env_bool",
            side_effect=lambda _name, fallback=False: fallback,
        ):
            assert _tts_quantization_enabled(None, "qwen3")
            assert not _tts_quantization_enabled(None, "luxtts")
            assert not _tts_quantization_enabled({"quantize": False}, "qwen3")

    def test_cuda_oom_classifier_accepts_worker_wrapped_errors(self) -> None:
        """Verify local and CEDTS-wrapped CUDA OOM errors are recognized."""
        assert is_cuda_out_of_memory(torch.cuda.OutOfMemoryError("CUDA OOM"))
        assert is_cuda_out_of_memory(
            RuntimeError("backend_worker_error: CUDA error: out of memory")
        )
        assert not is_cuda_out_of_memory(RuntimeError("invalid CUDA device index"))

    def test_presets_reserve_two_gib_and_unprofiled_is_warning_only(self) -> None:
        """Verify each preset preserves system headroom without rejecting unknown profiles."""
        config: Config = {"vram": "high", "backend": "voxcpm2"}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert validate_vram_preset(config) == (
                "This configuration is unprofiled and may not work on your hardware configuration."
            )
            assert VRAM_BUDGETS == {
                "low": 4,
                "medium": 6,
                "high": 10,
                "xhigh": 14,
            }

    def test_confirmed_profile_fits_only_when_reserve_remains_free(self) -> None:
        """Verify a confirmed combination must fit preset and live GPU budgets."""
        config: Config = {"vram": "high", "backend": "qwen3"}
        with (
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch(
                "celune.vram.torch.cuda.get_device_name", return_value="test GPU"
            ),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
        ):
            key = vram_profile_key(config)
        profile = VramProfile(
            gpu_name=key[0],
            configuration=key,
            model_revision="persona-revision",
            tts_revision="tts-revision",
            dtype="bfloat16",
            quantization="int8",
            components=("tts",),
            context_size=2048,
            load_peak_bytes=7 * 1024**3,
            warmup_peak_bytes=8 * 1024**3,
            inference_peak_bytes=9 * 1024**3,
        )
        with (
            mock.patch.dict(VRAM_PROFILES, {key: profile}),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch(
                "celune.vram.torch.cuda.get_device_name", return_value="test GPU"
            ),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
            mock.patch(
                "celune.vram.torch.cuda.mem_get_info",
                return_value=(12 * 1024**3, 12 * 1024**3),
            ),
            mock.patch("celune.vram.torch.cuda.memory_allocated", return_value=0),
        ):
            assert vram_profile_fits(config) is True
            assert validate_vram_preset(config) is None

        with (
            mock.patch.dict(VRAM_PROFILES, {key: profile}),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch("celune.vram.torch.cuda.get_device_name", return_value=key[0]),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
            mock.patch(
                "celune.vram.torch.cuda.mem_get_info",
                return_value=(10 * 1024**3, 12 * 1024**3),
            ),
            mock.patch("celune.vram.torch.cuda.memory_allocated", return_value=0),
        ):
            assert vram_profile_fits(config) is False

    def test_confirmed_profile_over_budget_is_rejected(self) -> None:
        """Verify a measured peak above the preset budget is known-incompatible."""
        config: Config = {"vram": "high", "backend": "qwen3"}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            key = vram_profile_key(config)
        profile = VramProfile(
            gpu_name=key[0],
            configuration=key,
            model_revision="persona-revision",
            tts_revision="tts-revision",
            dtype="bfloat16",
            quantization="int8",
            components=("tts",),
            context_size=2048,
            load_peak_bytes=11 * 1024**3,
            warmup_peak_bytes=10 * 1024**3,
            inference_peak_bytes=9 * 1024**3,
        )
        with (
            mock.patch.dict(VRAM_PROFILES, {key: profile}),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=False),
        ):
            assert vram_profile_fits(config) is False
            assert validate_vram_preset(config) is not None
