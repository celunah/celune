# SPDX-License-Identifier: Apache-2.0
"""Tests for detected multichannel speaker layouts."""

from celune.audio.speakers import _windows_channel_positions

from .support import CeluneTestCase


class TestSpeakerChannelLayout(CeluneTestCase):
    """Verify Windows channel masks preserve speaker positions and order."""

    def test_surround_layouts_preserve_back_and_side_channels(self) -> None:
        """Map 5.1 and 7.1 masks to their exact output channel positions."""
        assert _windows_channel_positions(0x3F, 6) == (
            "front_left",
            "front_right",
            "front_center",
            "lfe",
            "back_left",
            "back_right",
        )
        assert _windows_channel_positions(0x60F, 6) == (
            "front_left",
            "front_right",
            "front_center",
            "lfe",
            "side_left",
            "side_right",
        )
        assert _windows_channel_positions(0x63F, 8) == (
            "front_left",
            "front_right",
            "front_center",
            "lfe",
            "back_left",
            "back_right",
            "side_left",
            "side_right",
        )
