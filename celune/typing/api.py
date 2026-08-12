# SPDX-License-Identifier: Apache-2.0
"""API-specific type aliases."""

from typing import Literal, Optional, Union

from .aliases import AudioChunk, AudioChunkNonNormalized
from .common import JSONSerializable

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
