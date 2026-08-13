# SPDX-License-Identifier: Apache-2.0
"""Passthrough voice-conversion backend for plumbing and tests."""

import numpy as np

from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from .base import CeluneVCBackend

__all__ = ["CelunePassthroughVCBackend"]


class CelunePassthroughVCBackend(CeluneVCBackend):
    """Dummy voice-conversion backend that returns the input audio unchanged."""

    name = "passthrough"

    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Return the source audio as playable output without modification.

        Args:
            request: The voice-conversion request to convert.

        Returns:
            AudioOutput: The unmodified source audio and sample rate.
        """
        return AudioOutput(
            audio=np.asarray(request.source_audio, dtype=np.float32).copy(),
            sample_rate=request.sample_rate,
            label=request.label,
        )
