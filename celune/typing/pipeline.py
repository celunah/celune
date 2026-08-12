# SPDX-License-Identifier: Apache-2.0
"""Speech pipeline aliases."""

import queue
from typing import Optional, Union

from ..constants import PipelineStates
from ..dataclasses.pipeline import (
    AudioInputRequest,
    AudioOutput,
    PlaybackChunk,
    PlaybackSourceDone,
    SpeechRequest,
    VoiceConversionRequest,
)
from .aliases import AudioChunk

type SpeechStreamItem = Optional[Union[AudioChunk, Exception]]
type SpeechStreamQueue = queue.Queue[SpeechStreamItem]  # noqa
type TextQueueItem = Union[SpeechRequest, PipelineStates]
type AudioInputItem = AudioInputRequest
type AudioOutputItem = AudioOutput
type VoiceConversionInputItem = VoiceConversionRequest
type AudioQueueItem = Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]
