"""API-specific type aliases."""

from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from .common import JSONSerializable

type WebUiUpdate = dict[str, JSONSerializable]
type WebUiAudioValue = Optional[tuple[int, npt.NDArray[np.float32]]]
type WebUiInputArray = Union[npt.NDArray[np.float32], npt.NDArray[np.int16]]
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
