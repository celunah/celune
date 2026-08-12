# SPDX-License-Identifier: Apache-2.0
"""Tests for VRAM preset and backend compatibility rules."""

from unittest import TestCase, mock

from celune.typing.common import Config
from celune.vram import backend_allowed, resolve_backend_name, resolve_vram_preset


class VramTests(TestCase):
    """Test VRAM-aware backend selection."""

    def test_high_persona_allows_only_light_tts_backends(self) -> None:
        """Verify high-tier Persona sessions select Mini or Qwen3 only."""
        config: Config = {"vram": "high", "persona": {"enabled": True}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_backend_name(config, "mini"), "mini")
            self.assertEqual(resolve_backend_name(config, "qwen3"), "qwen3")
            self.assertEqual(resolve_backend_name(config, "dotstts"), "qwen3")
            self.assertEqual(resolve_backend_name(config, "voxcpm2"), "qwen3")
            self.assertFalse(backend_allowed(config, "dotstts"))
            self.assertFalse(backend_allowed(config, "voxcpm2"))
            self.assertFalse(resolve_vram_preset(config).allow_voxcpm2)

    def test_high_persona_restriction_can_be_disabled(self) -> None:
        """Verify high-tier heavy TTS backends remain available without Persona."""
        config: Config = {"vram": "high", "persona": {"enabled": False}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_backend_name(config, "dotstts"), "dotstts")
            self.assertEqual(resolve_backend_name(config, "voxcpm2"), "voxcpm2")
            self.assertTrue(backend_allowed(config, "dotstts"))
            self.assertTrue(backend_allowed(config, "voxcpm2"))
            self.assertTrue(resolve_vram_preset(config).allow_voxcpm2)

    def test_xhigh_persona_allows_heavy_tts_backends(self) -> None:
        """Verify xhigh-tier Persona sessions allow the other TTS backends."""
        config: Config = {"vram": "xhigh", "persona": {"enabled": True}}
        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_backend_name(config, "dotstts"), "dotstts")
            self.assertEqual(resolve_backend_name(config, "voxcpm2"), "voxcpm2")
            self.assertTrue(backend_allowed(config, "dotstts"))
            self.assertTrue(backend_allowed(config, "voxcpm2"))
