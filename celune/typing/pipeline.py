"""Speech pipeline aliases."""

import queue
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ..constants import PipelineStates
from ..dataclasses.pipeline import PlaybackChunk, PlaybackSourceDone, SpeechRequest

SpeechStreamItem = Optional[Union[npt.NDArray[np.float32], Exception]]
SpeechStreamQueue = queue.Queue[SpeechStreamItem]
TextQueueItem = Union[SpeechRequest, PipelineStates]
AudioChunk = PlaybackChunk
AudioQueueItem = Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]
