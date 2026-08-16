# SPDX-License-Identifier: MIT
"""Typed ownership contracts for Celune component operations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .common import JSON


class ComponentLockName(str, Enum):
    """Exclusive Celune resources that can participate in one operation."""

    VLM = "vlm"
    TTS = "tts"
    SPEECH_QUEUE = "speech_queue"
    AUDIO_PLAYBACK = "audio_playback"
    ASR = "asr"
    MICROPHONE = "microphone"
    AGENT = "agent"
    MODEL_LOADING = "model_loading"


@dataclass(frozen=True, slots=True)
class ComponentLockRequirement:
    """One component required before an operation may begin."""

    component: ComponentLockName

    def __post_init__(self) -> None:
        """Validate the component requirement."""
        if not isinstance(self.component, ComponentLockName):
            raise TypeError("component lock requirements need a known component")


@dataclass(frozen=True, slots=True)
class ComponentLockOwner:
    """Identity attached to every component ownership claim."""

    operation_id: str
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    generation_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate operation and optional task-generation identity."""
        if not isinstance(self.operation_id, str):
            raise TypeError("component lock operation_id must be a string")
        if not self.operation_id.strip():
            raise ValueError("component lock operation_id must not be empty")
        for name, value in (
            ("task_id", self.task_id),
            ("session_id", self.session_id),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"component lock {name} must be a string")
            if value is not None and not value.strip():
                raise ValueError(f"component lock {name} must not be empty")
        if self.generation_id is not None:
            if isinstance(self.generation_id, bool) or not isinstance(
                self.generation_id, int
            ):
                raise TypeError("component lock generation_id must be an integer")
            if self.generation_id < 0:
                raise ValueError("component lock generation_id must be non-negative")

    def to_json(self) -> JSON:
        """Serialize ownership metadata for diagnostics and lifecycle records."""
        return {
            "operation_id": self.operation_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentBusyResult:
    """Typed explanation for an operation that could not acquire its resources."""

    components: tuple[ComponentLockName, ...]
    owners: tuple[tuple[ComponentLockName, Optional[ComponentLockOwner]], ...]

    def __post_init__(self) -> None:
        """Validate that a busy result identifies at least one unavailable resource."""
        if not self.components:
            raise ValueError("component busy results need at least one component")
        if any(
            not isinstance(component, ComponentLockName)
            for component in self.components
        ):
            raise TypeError("component busy results need known components")
        if len(set(self.components)) != len(self.components):
            raise ValueError("component busy results cannot repeat components")
        component_set = set(self.components)
        owner_components = {component for component, _ in self.owners}
        if component_set != owner_components:
            raise ValueError("component busy owners must match unavailable components")

    def to_json(self) -> JSON:
        """Serialize the unavailable components and their current owners."""
        return {
            "components": [component.value for component in self.components],
            "owners": [
                {
                    "component": component.value,
                    "owner": owner.to_json() if owner is not None else None,
                }
                for component, owner in self.owners
            ],
        }


@dataclass(frozen=True, slots=True)
class ComponentLockAcquisition:
    """Atomic result of attempting to acquire an operation's requirements."""

    owner: ComponentLockOwner
    components: tuple[ComponentLockName, ...]
    busy: Optional[ComponentBusyResult] = None

    @property
    def acquired(self) -> bool:
        """Return whether every requested component was acquired."""
        return self.busy is None

    def to_json(self) -> JSON:
        """Serialize the acquisition result without exposing synchronization state."""
        return {
            "acquired": self.acquired,
            "owner": self.owner.to_json(),
            "components": [component.value for component in self.components],
            "busy": self.busy.to_json() if self.busy is not None else None,
        }


ComponentName = ComponentLockName
