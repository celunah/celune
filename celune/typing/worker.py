# SPDX-License-Identifier: Apache-2.0
"""Types for Celune's isolated backend worker protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, Union

from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from .aliases import AudioChunk
from .backends import BackendDescription
from .common import JSONSerializable

type WorkerValue = Union[
    JSONSerializable,
    bytes,
    AudioChunk,
    AudioOutput,
    BackendDescription,
    VoiceConversionRequest,
    list["WorkerValue"],
    tuple["WorkerValue", ...],
    dict[str, "WorkerValue"],
]
type WorkerArguments = dict[str, WorkerValue]


class WorkerRequest(TypedDict, total=False):
    """Request frame sent to an isolated backend worker."""

    operation: str
    arguments: WorkerArguments


class WorkerResponse(TypedDict, total=False):
    """Response or streamed-value frame sent by an isolated backend worker."""

    ok: bool
    value: WorkerValue
    error: str
    stream: bool
    done: bool
    fatal: bool


type WorkerMessageValue = WorkerValue
type WorkerMessage = Mapping[str, WorkerMessageValue]
