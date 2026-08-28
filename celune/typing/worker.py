# SPDX-License-Identifier: Apache-2.0
"""Types for Celune's isolated backend worker protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union, Optional, TypedDict

from .aliases import AudioChunk
from .common import JSONSerializable
from .backends import BackendDescription
from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest

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


class WorkerPayloadDescriptor(TypedDict, total=False):
    """Metadata declared for one binary CEDTS payload."""

    id: str
    media_type: str
    byte_length: int
    dtype: str
    shape: list[int]
    sample_rate: int
    channels: int


type WorkerControlMessage = dict[str, JSONSerializable]


class WorkerPacket(TypedDict, total=False):
    """Common CEDTS packet envelope shared by core and worker."""

    cedts_version: int
    kind: str
    message_id: str
    reply_to: Optional[str]
    operation: str
    data: dict[str, JSONSerializable]
    payloads: list[WorkerPayloadDescriptor]


class WorkerRequest(TypedDict, total=False):
    """Request frame sent to an isolated backend worker."""

    operation: str
    arguments: WorkerArguments


class WorkerResponse(TypedDict, total=False):
    """Response or streamed-value frame sent by an isolated backend worker."""

    ok: bool
    value: WorkerValue
    error: str
    error_type: str
    stream: bool
    done: bool
    cancelled: bool
    fatal: bool
    payloads: list[WorkerPayloadDescriptor]


type WorkerMessageValue = WorkerValue
type WorkerMessage = Mapping[str, WorkerMessageValue]
