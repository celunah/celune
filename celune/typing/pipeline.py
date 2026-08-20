# SPDX-License-Identifier: MIT
"""Speech pipeline aliases."""

import queue
from typing import Union, Optional

from .aliases import AudioChunk
from ..constants import PipelineStates
from ..dataclasses.pipeline import (
    AudioOutput,
    PlaybackChunk,
    SpeechRequest,
    AudioInputRequest,
    PlaybackSourceDone,
    VoiceConversionRequest,
)

type SpeechStreamItem = Optional[Union[AudioChunk, Exception]]
type SpeechStreamQueue = queue.Queue[SpeechStreamItem]
type TextQueueItem = Union[SpeechRequest, PipelineStates]
type AudioInputItem = AudioInputRequest
type AudioOutputItem = AudioOutput
type VoiceConversionInputItem = VoiceConversionRequest
type AudioQueueItem = Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]
