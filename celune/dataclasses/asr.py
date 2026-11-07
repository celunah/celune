# SPDX-License-Identifier: Apache-2.0
"""Timestamped speech-recognition result records."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WhisperWord:
    """One Whisper-aligned word with audio-relative timestamps."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class WhisperSegment:
    """One timestamped segment returned by Whisper."""

    text: str
    start: float
    end: float
    words: tuple[WhisperWord, ...] = ()
