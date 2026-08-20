# SPDX-License-Identifier: MIT
"""API-specific type aliases."""

from typing import Union, Literal, Optional

from .common import JSONSerializable
from .aliases import AudioChunk, AudioChunkNonNormalized

type WebUiUpdate = dict[str, JSONSerializable]
type WebUiAudioValue = Optional[tuple[int, AudioChunk]]
type WebUiInputArray = Union[AudioChunk, AudioChunkNonNormalized]
type WebUiInputAudioValue = Optional[tuple[int, WebUiInputArray]]
type TaskEventName = Literal[
    "started",
    "progress",
    "log",
    "completed",
    "failed",
    "cancelled",
]
type TaskStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
type TaskCommandName = Literal["cancel"]
