"""Typed event names, payload protocols, and callback aliases for extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, Union

from ..dataclasses.events import (
    AudioEndEvent,
    AudioStartEvent,
    CharacterChangedEvent,
    CharacterLoadedEvent,
    CharacterUnloadedEvent,
    ErrorEvent,
    FatalEvent,
    GenerationEndEvent,
    GenerationErrorEvent,
    GenerationStartEvent,
    ReadyEvent,
    ShutdownEvent,
    StateChangedEvent,
    VoiceChangedEvent,
)


EventName = Literal[
    "ready",
    "shutdown",
    "fatal",
    "error",
    "voice_changed",
    "state_changed",
    "generation_start",
    "generation_end",
    "generation_error",
    "audio_start",
    "audio_end",
    "character_changed",
    "character_loaded",
    "character_unloaded",
]


class CeluneEventProtocol(Protocol):
    """Protocol shared by all Celune event payloads."""

    celune: Celune


class ErrorEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol shared by error-like event payloads."""

    error: Exception
    source: str


class StateChangedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for runtime-state transition payloads."""

    old_state: str
    new_state: str


class VoiceChangedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for voice transition payloads."""

    old_voice: str
    new_voice: str


class CharacterLoadedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for CEVOICE bundle activation payloads."""

    character_name: str
    bundle_path: Optional[str]
    is_default: bool


class CharacterUnloadedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for CEVOICE bundle unload payloads."""

    character_name: str
    bundle_path: Optional[str]


class CharacterChangedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for CEVOICE bundle replacement payloads."""

    old_character: str
    new_character: str
    old_bundle_path: Optional[str]
    new_bundle_path: Optional[str]
    new_is_default: bool


class GenerationEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol shared by speech-generation payloads."""

    text: str
    display_text: str
    save: bool


class AudioEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol shared by audio playback payloads."""

    source_id: int
    label: str
    kind: str
    saved_path: Optional[str]


if TYPE_CHECKING:
    from ..celune import Celune


ReadyEventCallback = Callable[[ReadyEvent], None]
ShutdownEventCallback = Callable[[ShutdownEvent], None]
FatalEventCallback = Callable[[FatalEvent], None]
ErrorEventCallback = Callable[[ErrorEvent], None]
VoiceChangedEventCallback = Callable[[VoiceChangedEvent], None]
StateChangedEventCallback = Callable[[StateChangedEvent], None]
GenerationStartEventCallback = Callable[[GenerationStartEvent], None]
GenerationEndEventCallback = Callable[[GenerationEndEvent], None]
GenerationErrorEventCallback = Callable[[GenerationErrorEvent], None]
AudioStartEventCallback = Callable[[AudioStartEvent], None]
AudioEndEventCallback = Callable[[AudioEndEvent], None]
CharacterChangedEventCallback = Callable[[CharacterChangedEvent], None]
CharacterLoadedEventCallback = Callable[[CharacterLoadedEvent], None]
CharacterUnloadedEventCallback = Callable[[CharacterUnloadedEvent], None]

EventPayload = Union[
    ReadyEvent,
    ShutdownEvent,
    FatalEvent,
    ErrorEvent,
    VoiceChangedEvent,
    StateChangedEvent,
    GenerationStartEvent,
    GenerationEndEvent,
    GenerationErrorEvent,
    AudioStartEvent,
    AudioEndEvent,
    CharacterChangedEvent,
    CharacterLoadedEvent,
    CharacterUnloadedEvent,
]

EventCallback = Union[
    ReadyEventCallback,
    ShutdownEventCallback,
    FatalEventCallback,
    ErrorEventCallback,
    VoiceChangedEventCallback,
    StateChangedEventCallback,
    GenerationStartEventCallback,
    GenerationEndEventCallback,
    GenerationErrorEventCallback,
    AudioStartEventCallback,
    AudioEndEventCallback,
    CharacterChangedEventCallback,
    CharacterLoadedEventCallback,
    CharacterUnloadedEventCallback,
]
