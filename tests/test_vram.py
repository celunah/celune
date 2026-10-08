# SPDX-License-Identifier: Apache-2.0
"""Tests for VRAM preset and backend compatibility rules."""

from unittest import mock

import torch
from torch import nn

from celune.vram import (
    VRAM_PROFILES,
    VramProfile,
    backend_allowed,
    _cuda_total_capacity_bytes,
    vram_profile_key,
    format_vram_bytes,
    vram_profile_fits,
    resolve_vram_preset,
    resolve_backend_name,
    validate_vram_preset,
    is_cuda_out_of_memory,
    component_memory_usage,
    _windows_shared_memory_capacity_bytes,
    _windows_shared_memory_headroom_bytes,
)

from celune.constants import VRAM_BUDGETS
from celune.typing.common import Config
from celune.celune import _tts_quantization_enabled

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

    def test_windows_shared_capacity_is_stable_and_headroom_keeps_reserve(self) -> None:
        """Separate Windows shared capacity from live system-memory headroom."""
        system_total = 32 * 1024**3
        with (
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch(
                "celune.vram.psutil.virtual_memory",
                return_value=mock.Mock(total=system_total, available=4 * 1024**3),
            ),
            mock.patch(
                "celune.vram.torch.cuda.mem_get_info",
                return_value=(1024**3, 12 * 1024**3),
            ),
        ):
            assert _windows_shared_memory_capacity_bytes() == 16 * 1024**3
            assert _windows_shared_memory_headroom_bytes() == 2 * 1024**3
            assert _cuda_total_capacity_bytes() == 28 * 1024**3

        with (
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch(
                "celune.vram.psutil.virtual_memory",
                return_value=mock.Mock(total=64 * 1024**3),
            ),
        ):
            assert _windows_shared_memory_capacity_bytes() == 48 * 1024**3

        with (
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch(
                "celune.vram.psutil.virtual_memory",
                return_value=mock.Mock(total=system_total, available=1024**3),
            ),
        ):
            assert _windows_shared_memory_headroom_bytes() == 0

        with mock.patch("celune.vram.sys.platform", "linux"):
            assert _windows_shared_memory_capacity_bytes() == 0
            assert _windows_shared_memory_headroom_bytes() == 0

    def test_windows_profile_headroom_combines_dedicated_and_shared_free_memory(
        self,
    ) -> None:
        """Use current shared headroom once without subtracting the reserve twice."""
        config: Config = {"vram": "high", "backend": "qwen3"}
        with (
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch(
                "celune.vram.torch.cuda.get_device_name", return_value="test GPU"
            ),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
            mock.patch(
                "celune.vram._cuda_total_capacity_bytes", return_value=12 * 1024**3
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
            load_peak_bytes=10 * 1024**3,
            warmup_peak_bytes=10 * 1024**3,
            inference_peak_bytes=10 * 1024**3,
        )
        with (
            mock.patch.dict(VRAM_PROFILES, {key: profile}),
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch(
                "celune.vram.torch.cuda.get_device_name", return_value="test GPU"
            ),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
            mock.patch(
                "celune.vram._cuda_total_capacity_bytes", return_value=12 * 1024**3
            ),
            mock.patch(
                "celune.vram.torch.cuda.mem_get_info",
                return_value=(1024**3, 12 * 1024**3),
            ),
            mock.patch("celune.vram.torch.cuda.memory_allocated", return_value=0),
            mock.patch(
                "celune.vram._windows_shared_memory_headroom_bytes",
                return_value=10 * 1024**3,
            ),
        ):
            assert vram_profile_fits(config) is True

    def test_measured_profiles_match_model_identity_and_leave_agent_unprofiled(
        self,
    ) -> None:
        """Match measured revisions exactly while keeping agent mode unprofiled."""
        whisper_revision = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
        persona_config: Config = {
            "enabled": True,
            "model_id": "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
            "context_size": 2048,
            "speech_model_id": "openai/whisper-large-v3-turbo",
        }
        config: Config = {
            "mode": "converse",
            "vram": "high",
            "backend": "qwen3",
            "quantize": True,
            "use_normalizer": False,
            "persona": persona_config,
        }
        with (
            mock.patch("celune.vram.sys.platform", "win32"),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=True),
            mock.patch(
                "celune.vram.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 5070",
            ),
            mock.patch(
                "celune.vram.torch.cuda.get_device_capability", return_value=(8, 9)
            ),
            mock.patch(
                "celune.vram._cuda_total_capacity_bytes", return_value=28 * 1024**3
            ),
            mock.patch(
                "celune.vram._cached_hub_revision", return_value=whisper_revision
            ),
            mock.patch.dict("celune.vram.os.environ", {"CELUNE_TTS_QUANTIZE": "true"}),
        ):
            key = vram_profile_key(config)
            assert "tts-kv:True" in key
            assert key not in VRAM_PROFILES

            persona_config["quantize_kv_cache"] = True
            config["quantize_kv_cache"] = False
            native_cache_key = vram_profile_key(config)
            assert native_cache_key in VRAM_PROFILES
            assert native_cache_key[23] == "whisper-id:openai/whisper-large-v3-turbo"
            assert native_cache_key[24] == f"whisper-revision:{whisper_revision}"

            persona_config["speech_model_id"] = "openai/whisper-small"
            custom_whisper_key = vram_profile_key(config)
            assert custom_whisper_key not in VRAM_PROFILES

            agent_config = dict(config)
            agent_config["mode"] = "agent"
            with mock.patch("celune.vram.agent_vram_compatible", return_value=True):
                agent_key = vram_profile_key(agent_config)
            assert agent_key[-1].startswith("needle-revision:")
            assert agent_key not in VRAM_PROFILES

    def test_confirmed_profile_fits_only_when_reserve_remains_free(self) -> None:
        """Verify a confirmed combination must fit preset and live GPU budgets."""
        config: Config = {"vram": "high", "backend": "qwen3"}
        with (
            mock.patch("celune.vram.sys.platform", "linux"),
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
            mock.patch("celune.vram.sys.platform", "linux"),
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
            mock.patch("celune.vram.sys.platform", "linux"),
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
