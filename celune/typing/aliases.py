# SPDX-License-Identifier: Apache-2.0
"""Shared type aliases moved out of runtime implementation modules."""

from __future__ import annotations

import unittest.mock
from enum import Enum
from collections.abc import Callable, Hashable, Generator
from typing import TYPE_CHECKING, Union, Literal, Optional, Protocol

from .common import JSONSerializable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import sounddevice as sd
    from torch import Tensor

    # noinspection PyPep8Naming
    from torch import dtype as DType

    # noinspection PyPep8Naming
    from torch import device as Device
    from transformers import PreTrainedModel, TokenizersBackend, SentencePieceBackend

    # noinspection PyUnresolvedReferences
    from .events import EventPayload

type RuntimeValue = Union[
    str,
    bytes,
    bytearray,
    int,
    float,
    bool,
    None,
    dict[Hashable, "RuntimeValue"],
    list["RuntimeValue"],
    set["RuntimeValue"],
    tuple["RuntimeValue", ...],
    "SupportsCloseHook",
    "SupportsUnloadHook",
    "SupportsRuntimeAttributes",
    unittest.mock.NonCallableMock,
]

type AudioChunk = npt.NDArray[np.float32]
type AudioChunkNonNormalized = npt.NDArray[np.int16]
type AudioChunkBroad = npt.NDArray[np.floating]
type AudioChunks = list[AudioChunk]

type SeedVCArgument = Union[str, int, float, bool]
type SeedVCGenerator = Generator[Optional[AudioChunk], None, AudioChunk]

type LogLevel = Literal["info", "verbose", "debug"]


class LogCallback(Protocol):
    """Callback receiving a message and its display severity."""

    def __call__(
        self,
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Receive one log message."""


type DevLogCallback = LogCallback
type _DispatcherCallback = Callable[["EventPayload"], None]
type DispatcherCallback = _DispatcherCallback
type EmbeddingVector = npt.NDArray[np.float32]
type TokenizerBackend = Union[SentencePieceBackend, TokenizersBackend]
type _EmbeddingBackend = tuple[TokenizerBackend, PreTrainedModel]
type EmbeddingBackend = _EmbeddingBackend

type _AudioDeviceScalar = Union[bool, int, float, str]
type AudioDeviceScalar = _AudioDeviceScalar
type _VCAudioCallback = Callable[
    [
        npt.NDArray[np.float32],
        int,
        Optional[tuple[float, float, float]],
        Optional[sd.CallbackFlags],
    ],
    None,
]
type VCAudioCallback = _VCAudioCallback
type ConstantPropertyValue = Union[JSONSerializable, Enum]

type _RecordedKwargValue = Optional[
    Union[
        str,
        bool,
        bytes,
        list[bytes],
        Tensor,
        DType,
        Device,
    ]
]
type RecordedKwargValue = _RecordedKwargValue


class SupportsCloseHook(Protocol):
    """Protocol for runtime objects exposing a close hook."""

    def close(self) -> None:
        """Release runtime resources."""


class SupportsUnloadHook(Protocol):
    """Protocol for runtime objects exposing an unload hook."""

    def unload(self) -> None:
        """Unload runtime state."""


class SupportsRuntimeAttributes(Protocol):
    """Protocol for runtime objects that keep nested state in ``__dict__``."""

    __dict__: dict[str, RuntimeValue]
