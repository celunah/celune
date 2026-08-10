# SPDX-License-Identifier: Apache-2.0
"""Tests for VRAM preset and backend compatibility rules."""

from unittest import mock

from celune.typing.common import Config
from celune.vram import backend_allowed, resolve_backend_name, resolve_vram_preset

from .support import CeluneTestCase


class TestVram(CeluneTestCase):
    """Test VRAM-aware backend selection."""

    def test_high_persona_allows_only_light_tts_backends(self) -> None:
        """Verify high-tier Persona sessions select Mini or Qwen3 only."""
        config: Config = {"vram": "high", "persona": {"enabled": True}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            assert resolve_backend_name(config, "mini") == "mini"
            assert resolve_backend_name(config, "qwen3") == "qwen3"
            assert resolve_backend_name(config, "dotstts") == "qwen3"
            assert resolve_backend_name(config, "voxcpm2") == "qwen3"
            assert not backend_allowed(config, "dotstts")
            assert not backend_allowed(config, "voxcpm2")
            assert not resolve_vram_preset(config).allow_voxcpm2

    def test_high_persona_restriction_can_be_disabled(self) -> None:
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
