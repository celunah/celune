# SPDX-License-Identifier: MIT
"""Dependency-light audio resampling shared by Celune runtimes."""

import math

import numpy as np
from scipy.signal import resample_poly


def resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Resample audio with Celune's shared polyphase implementation."""
    if source_sample_rate == target_sample_rate:
        return np.ascontiguousarray(audio, dtype=np.float32)

    factor = math.gcd(source_sample_rate, target_sample_rate)
    return np.ascontiguousarray(
        resample_poly(
            np.asarray(audio, dtype=np.float32),
            up=target_sample_rate // factor,
            down=source_sample_rate // factor,
            axis=axis,
        ),
        dtype=np.float32,
    )
