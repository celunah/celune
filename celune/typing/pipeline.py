"""Speech pipeline aliases."""

import queue
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ..constants import PipelineStates
from ..dataclasses.pipeline import (
    AudioOutput,
    AudioInputRequest,
    PlaybackChunk,
    PlaybackSourceDone,
    SpeechRequest,
    VoiceConversionRequest,
)

SpeechStreamItem = Optional[Union[npt.NDArray[np.float32], Exception]]
SpeechStreamQueue = queue.Queue[SpeechStreamItem]
TextQueueItem = Union[SpeechRequest, PipelineStates]
AudioInputItem = AudioInputRequest
AudioOutputItem = AudioOutput
VoiceConversionInputItem = VoiceConversionRequest
AudioChunk = PlaybackChunk
AudioQueueItem = Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]
