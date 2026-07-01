# SPDX-License-Identifier: MIT
"""Compatibility wrappers for VC live audio helpers."""

from .vc_runtime import LiveVoiceActivityDetector, create_live_voice_activity_detector

__all__ = ["LiveVoiceActivityDetector", "create_live_voice_activity_detector"]
