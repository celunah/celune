"""Shared type aliases moved out of runtime implementation modules."""

import unittest.mock
from collections.abc import Generator, Hashable
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Union

import numpy as np
import numpy.typing as npt
import sounddevice as sd

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel  # noqa
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # noqa

    from .events import EventPayload  # noqa

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
    tuple["RuntimeValue", ...],  # noqa
    "SupportsCloseHook",
    "SupportsUnloadHook",
    "SupportsRuntimeAttributes",
    unittest.mock.NonCallableMock,
]

type AudioChunk = npt.NDArray[np.float32]  # noqa
type AudioChunkNonNormalized = npt.NDArray[np.int16]  # noqa
type AudioChunkBroad = npt.NDArray[np.floating]  # noqa
type AudioChunks = list[AudioChunk]  # noqa

type SeedVCArgument = Union[str, int, float, bool]
type SeedVCGenerator = Generator[
    Optional[AudioChunk], None, AudioChunk  # noqa
]

type DevLogCallback = Callable[[str, str], None]
type _DispatcherCallback = Callable[["EventPayload"], None]
type DispatcherCallback = _DispatcherCallback
type EmbeddingVector = npt.NDArray[np.float32]  # noqa
type _EmbeddingBackend = tuple["PreTrainedTokenizerBase", "PreTrainedModel"]
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
