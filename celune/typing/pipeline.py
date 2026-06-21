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

type SpeechStreamItem = Optional[Union[npt.NDArray[np.float32], Exception]]
type SpeechStreamQueue = queue.Queue[SpeechStreamItem]
type TextQueueItem = Union[SpeechRequest, PipelineStates]
type AudioInputItem = AudioInputRequest
type AudioOutputItem = AudioOutput
type VoiceConversionInputItem = VoiceConversionRequest
type AudioChunk = PlaybackChunk
type AudioQueueItem = Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]
