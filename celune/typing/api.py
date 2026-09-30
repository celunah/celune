# SPDX-License-Identifier: Apache-2.0
"""API-specific type aliases."""

from typing import Union, Literal, Final, Optional

from .common import JSONSerializable, Sentinel
from .aliases import AudioChunk, AudioChunkNonNormalized


class WebUiUnset(Sentinel):
    """Sentinel type for an omitted WebUI input update."""


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

WEBUI_UNSET: Final[WebUiUnset] = WebUiUnset()
