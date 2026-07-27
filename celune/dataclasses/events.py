# SPDX-License-Identifier: MIT
"""Typed Celune lifecycle and extension event payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..celune import Celune


@dataclass(slots=True)
class ReadyEvent:
    """Payload emitted once Celune is fully initialized."""

    celune: Celune


@dataclass(slots=True)
class ShutdownEvent:
    """Payload emitted when Celune begins shutting down."""

    celune: Celune


@dataclass(slots=True)
class FatalEvent:
    """Payload emitted when Celune enters a fatal runtime error state."""

    celune: Celune
    error: Exception
    source: str


@dataclass(slots=True)
class ErrorEvent:
    """Payload emitted for non-fatal engine-level errors."""

    celune: Celune
    error: Exception
    source: str


@dataclass(slots=True)
class VoiceChangedEvent:
    """Payload emitted after Celune switches to a different voice."""

    celune: Celune
    old_voice: str
    new_voice: str


@dataclass(slots=True)
class StateChangedEvent:
    """Payload emitted whenever Celune changes runtime state."""

    celune: Celune
    old_state: str
    new_state: str


@dataclass(slots=True)
class GenerationStartEvent:
    """Payload emitted when a speech request starts generating."""

    celune: Celune
    text: str
    display_text: str
    save: bool
    language: str


@dataclass(slots=True)
class GenerationEndEvent:
    """Payload emitted when a speech request finishes generating."""

    celune: Celune
    text: str
    display_text: str
    save: bool
    language: str
    saved_path: Optional[str] = None


@dataclass(slots=True)
class GenerationErrorEvent:
    """Payload emitted when generation fails for one speech request."""

    celune: Celune
    text: str
    display_text: str
    save: bool
    language: str
    error: Exception


@dataclass(slots=True)
class AudioStartEvent:
    """Payload emitted when an audio source begins playback."""

    celune: Celune
    source_id: int
    label: str
    kind: str
    saved_path: Optional[str] = None


@dataclass(slots=True)
class AudioEndEvent:
    """Payload emitted when an audio source finishes playback."""

    celune: Celune
    source_id: int
    label: str
    kind: str
    saved_path: Optional[str] = None


@dataclass(slots=True)
class CharacterLoadedEvent:
    """Payload emitted when one CEVOICE character bundle becomes active."""

    celune: Celune
    character_name: str
    bundle_path: Optional[str]
    is_default: bool


@dataclass(slots=True)
class CharacterUnloadedEvent:
    """Payload emitted when one CEVOICE character bundle is unloaded."""

    celune: Celune
    character_name: str
    bundle_path: Optional[str]


@dataclass(slots=True)
class CharacterChangedEvent:
    """Payload emitted when Celune switches between CEVOICE characters."""

    celune: Celune
    old_character: str
    new_character: str
    old_bundle_path: Optional[str]
    new_bundle_path: Optional[str]
    new_is_default: bool
