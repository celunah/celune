# SPDX-License-Identifier: Apache-2.0
"""Typed event names, payload protocols, and callback aliases for extensions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Union, Literal, Optional, Protocol

from .common import JSON
from .agent import (
    AgentTaskState,
    AgentAbortReason,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentApprovalRequest,
    AgentCancellationReason,
)
from ..dataclasses.events import (
    ErrorEvent,
    FatalEvent,
    ReadyEvent,
    AudioEndEvent,
    ShutdownEvent,
    AudioStartEvent,
    StateChangedEvent,
    VoiceChangedEvent,
    GenerationEndEvent,
    CharacterLoadedEvent,
    GenerationErrorEvent,
    GenerationStartEvent,
    CharacterChangedEvent,
    AgentTaskFinishedEvent,
    CharacterUnloadedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)

type EventName = Literal[
    "agent_task_state_changed",
    "agent_approval_requested",
    "agent_choice_requested",
    "agent_task_finished",
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


class AgentTaskStateChangedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for agent task lifecycle transitions."""

    task_id: str
    session_id: str
    old_state: AgentTaskState
    new_state: AgentTaskState


class AgentApprovalRequestedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for agent tool approval requests."""

    task_id: str
    session_id: str
    request: AgentApprovalRequest


class AgentChoiceRequestedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for agent user-choice requests."""

    task_id: str
    session_id: str
    request: AgentChoiceRequest


class AgentTaskFinishedEventProtocol(CeluneEventProtocol, Protocol):
    """Protocol for terminal agent task outcomes."""

    task_id: str
    session_id: str
    state: AgentTaskState
    abort_reason: Optional[AgentAbortReason]
    failure_reason: Optional[AgentFailureReason]
    cancellation_reason: Optional[AgentCancellationReason]
    completion_metadata: Optional[JSON]


if TYPE_CHECKING:
    from ..celune import Celune


type AgentTaskStateChangedEventCallback = Callable[[AgentTaskStateChangedEvent], None]
type AgentApprovalRequestedEventCallback = Callable[[AgentApprovalRequestedEvent], None]
type AgentChoiceRequestedEventCallback = Callable[[AgentChoiceRequestedEvent], None]
type AgentTaskFinishedEventCallback = Callable[[AgentTaskFinishedEvent], None]
type ReadyEventCallback = Callable[[ReadyEvent], None]
type ShutdownEventCallback = Callable[[ShutdownEvent], None]
type FatalEventCallback = Callable[[FatalEvent], None]
type ErrorEventCallback = Callable[[ErrorEvent], None]
type VoiceChangedEventCallback = Callable[[VoiceChangedEvent], None]
type StateChangedEventCallback = Callable[[StateChangedEvent], None]
type GenerationStartEventCallback = Callable[[GenerationStartEvent], None]
type GenerationEndEventCallback = Callable[[GenerationEndEvent], None]
type GenerationErrorEventCallback = Callable[[GenerationErrorEvent], None]
type AudioStartEventCallback = Callable[[AudioStartEvent], None]
type AudioEndEventCallback = Callable[[AudioEndEvent], None]
type CharacterChangedEventCallback = Callable[[CharacterChangedEvent], None]
type CharacterLoadedEventCallback = Callable[[CharacterLoadedEvent], None]
type CharacterUnloadedEventCallback = Callable[[CharacterUnloadedEvent], None]

type EventPayload = Union[
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskFinishedEvent,
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

type EventCallback = Union[
    AgentTaskStateChangedEventCallback,
    AgentApprovalRequestedEventCallback,
    AgentChoiceRequestedEventCallback,
    AgentTaskFinishedEventCallback,
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
