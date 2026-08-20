# SPDX-License-Identifier: MIT
"""Celune's internal extension event dispatcher and decorators."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar, Optional, cast, overload

from ..utils import format_error
from ..typing.aliases import LogLevel, DispatcherCallback
from ..typing.events import (
    EventName,
    EventPayload,
    EventCallback,
    ErrorEventCallback,
    FatalEventCallback,
    ReadyEventCallback,
    AudioEndEventCallback,
    ShutdownEventCallback,
    AudioStartEventCallback,
    StateChangedEventCallback,
    VoiceChangedEventCallback,
    GenerationEndEventCallback,
    CharacterLoadedEventCallback,
    GenerationErrorEventCallback,
    GenerationStartEventCallback,
    CharacterChangedEventCallback,
    AgentTaskFinishedEventCallback,
    CharacterUnloadedEventCallback,
    AgentChoiceRequestedEventCallback,
    AgentTaskStateChangedEventCallback,
    AgentApprovalRequestedEventCallback,
)

if TYPE_CHECKING:
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


EVENT_NAMES: tuple[EventName, ...] = (
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
)
_EVENT_NAME_SET = frozenset(EVENT_NAMES)
EVENT_HANDLER_METADATA_ATTR = "__celune_event_subscriptions__"
_DecoratedCallback = TypeVar("_DecoratedCallback", bound=Callable[..., None])


@dataclass(frozen=True)
class EventSubscription:
    """Declared event subscription metadata stored on callbacks."""

    event_name: EventName
    enabled: bool


@dataclass(frozen=True)
class RegisteredEventHandler:
    """Stored event-subscription metadata used for later cleanup."""

    event_name: EventName
    callback: EventCallback
    owner_name: str


class EventDispatcher:
    """Internal event dispatcher for Celune extension callbacks."""

    def __init__(
        self,
        *,
        log_warning: Callable[[str, str], None],
        log_level: LogLevel = "info",
        log_debug: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._log_warning = log_warning
        self._log_level = log_level
        self._log_debug = log_debug
        self._lock = threading.RLock()
        self._callbacks: dict[EventName, list[DispatcherCallback]] = defaultdict(list)
        self._owners: dict[tuple[EventName, DispatcherCallback], str] = {}

    @overload
    def subscribe(
        self,
        event_name: Literal["agent_task_state_changed"],
        callback: AgentTaskStateChangedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["agent_approval_requested"],
        callback: AgentApprovalRequestedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["agent_choice_requested"],
        callback: AgentChoiceRequestedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["agent_task_finished"],
        callback: AgentTaskFinishedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["ready"],
        callback: ReadyEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["shutdown"],
        callback: ShutdownEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["fatal"],
        callback: FatalEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["error"],
        callback: ErrorEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["voice_changed"],
        callback: VoiceChangedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["state_changed"],
        callback: StateChangedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["generation_start"],
        callback: GenerationStartEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["generation_end"],
        callback: GenerationEndEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["generation_error"],
        callback: GenerationErrorEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["audio_start"],
        callback: AudioStartEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["audio_end"],
        callback: AudioEndEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["character_changed"],
        callback: CharacterChangedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["character_loaded"],
        callback: CharacterLoadedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: Literal["character_unloaded"],
        callback: CharacterUnloadedEventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    @overload
    def subscribe(
        self,
        event_name: EventName,
        callback: EventCallback,
        owner_name: Optional[str] = None,
    ) -> None: ...

    def subscribe(
        self,
        event_name: EventName,
        callback: Callable[..., None],
        owner_name: Optional[str] = None,
    ) -> None:
        """Register one callback for an event.

        Args:
            event_name: Event name to subscribe to.
            callback: Callback invoked for matching event payloads.
            owner_name: Optional display name used in failure logs.
        """
        self.validate_event_name(event_name)
        with self._lock:
            callbacks = self._callbacks[event_name]
            typed_callback = cast(DispatcherCallback, callback)
            resolved_owner = owner_name or self._describe_callback(typed_callback)
            if typed_callback not in callbacks:
                callbacks.append(typed_callback)
                self._owners[(event_name, typed_callback)] = resolved_owner
            self._debug(f"[EVENT] subscribed name={event_name} owner={resolved_owner}")

    @overload
    def unsubscribe(
        self,
        event_name: Literal["agent_task_state_changed"],
        callback: AgentTaskStateChangedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["agent_approval_requested"],
        callback: AgentApprovalRequestedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["agent_choice_requested"],
        callback: AgentChoiceRequestedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["agent_task_finished"],
        callback: AgentTaskFinishedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["ready"],
        callback: ReadyEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["shutdown"],
        callback: ShutdownEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["fatal"],
        callback: FatalEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["error"],
        callback: ErrorEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["voice_changed"],
        callback: VoiceChangedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["state_changed"],
        callback: StateChangedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["generation_start"],
        callback: GenerationStartEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["generation_end"],
        callback: GenerationEndEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["generation_error"],
        callback: GenerationErrorEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["audio_start"],
        callback: AudioStartEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["audio_end"],
        callback: AudioEndEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["character_changed"],
        callback: CharacterChangedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["character_loaded"],
        callback: CharacterLoadedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: Literal["character_unloaded"],
        callback: CharacterUnloadedEventCallback,
    ) -> None: ...

    @overload
    def unsubscribe(
        self,
        event_name: EventName,
        callback: EventCallback,
    ) -> None: ...

    def unsubscribe(
        self,
        event_name: EventName,
        callback: Callable[..., None],
    ) -> None:
        """Unregister one callback for an event.

        Args:
            event_name: Event name to remove the callback from.
            callback: Previously registered callback to remove.
        """
        self.validate_event_name(event_name)
        with self._lock:
            callbacks = self._callbacks.get(event_name)
            if callbacks is None:
                return
            typed_callback = cast(DispatcherCallback, callback)
            try:
                callbacks.remove(typed_callback)
            except ValueError:
                return
            self._owners.pop((event_name, typed_callback), None)
            if not callbacks:
                self._callbacks.pop(event_name, None)
        self._debug(
            f"[EVENT] unsubscribed name={event_name} callback={self._describe_callback(typed_callback)}"
        )

    @overload
    def emit(
        self,
        event_name: Literal["agent_task_state_changed"],
        event: AgentTaskStateChangedEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["agent_approval_requested"],
        event: AgentApprovalRequestedEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["agent_choice_requested"],
        event: AgentChoiceRequestedEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["agent_task_finished"],
        event: AgentTaskFinishedEvent,
    ) -> None: ...

    @overload
    def emit(self, event_name: Literal["ready"], event: ReadyEvent) -> None: ...

    @overload
    def emit(self, event_name: Literal["shutdown"], event: ShutdownEvent) -> None: ...

    @overload
    def emit(self, event_name: Literal["fatal"], event: FatalEvent) -> None: ...

    @overload
    def emit(self, event_name: Literal["error"], event: ErrorEvent) -> None: ...

    @overload
    def emit(
        self, event_name: Literal["voice_changed"], event: VoiceChangedEvent
    ) -> None: ...

    @overload
    def emit(
        self, event_name: Literal["state_changed"], event: StateChangedEvent
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["generation_start"],
        event: GenerationStartEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["generation_end"],
        event: GenerationEndEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["generation_error"],
        event: GenerationErrorEvent,
    ) -> None: ...

    @overload
    def emit(
        self, event_name: Literal["audio_start"], event: AudioStartEvent
    ) -> None: ...

    @overload
    def emit(self, event_name: Literal["audio_end"], event: AudioEndEvent) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["character_changed"],
        event: CharacterChangedEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["character_loaded"],
        event: CharacterLoadedEvent,
    ) -> None: ...

    @overload
    def emit(
        self,
        event_name: Literal["character_unloaded"],
        event: CharacterUnloadedEvent,
    ) -> None: ...

    @overload
    def emit(self, event_name: EventName, event: EventPayload) -> None: ...

    def emit(self, event_name: EventName, event: EventPayload) -> None:
        """Dispatch an event to all current subscribers.

        Args:
            event_name: Event name being dispatched.
            event: Event payload delivered to subscribers.
        """
        self.validate_event_name(event_name)
        with self._lock:
            callbacks = list(self._callbacks.get(event_name, ()))
            owners = {
                callback: self._owners.get(
                    (event_name, callback), self._describe_callback(callback)
                )
                for callback in callbacks
            }

        self._debug(
            f"[EVENT] dispatch name={event_name} callbacks={len(callbacks)} "
            f"payload={type(event).__name__}"
        )

        for callback in callbacks:
            owner_name = owners.get(callback, self._describe_callback(callback))
            self._debug(f"[EVENT] invoke name={event_name} owner={owner_name}")
            try:
                callback(event)
            except Exception as exc:
                self._log_warning(
                    (
                        f"[Core] Event callback failed for '{event_name}' in "
                        f"'{owner_name}': {format_error(exc, self._log_level)}"
                    ),
                    "warning",
                )
                self._debug(
                    f"[EVENT] callback_error name={event_name} owner={owner_name} "
                    f"error={type(exc).__name__}"
                )
            else:
                self._debug(
                    f"[EVENT] callback_complete name={event_name} owner={owner_name}"
                )

        self._debug(f"[EVENT] dispatch_complete name={event_name}")

    def _debug(self, message: str) -> None:
        """Emit a low-level dispatcher trace when one was configured."""
        if self._log_debug is not None:
            self._log_debug(message)

    @staticmethod
    def validate_event_name(event_name: EventName) -> None:
        """Reject unknown event names.

        Args:
            event_name: Value for `event_name`.

        Raises:
            ValueError: If `ValueError` needs to be raised.
        """
        if event_name not in _EVENT_NAME_SET:
            raise ValueError(f"unknown event name: {event_name}")

    @staticmethod
    def _describe_callback(callback: DispatcherCallback) -> str:
        """Return a useful callback label for logs."""
        qualname = getattr(callback, "__qualname__", None)
        if isinstance(qualname, str) and qualname:
            return qualname
        name = getattr(callback, "__name__", None)
        if isinstance(name, str) and name:
            return name
        return callback.__class__.__name__


def _store_subscription_metadata(
    callback: Callable[..., None],
    event_name: EventName,
    enabled: bool,
) -> Callable[..., None]:
    """Attach one declared event subscription to a callback."""
    EventDispatcher.validate_event_name(event_name)
    subscription = EventSubscription(event_name=event_name, enabled=enabled)
    existing = getattr(callback, EVENT_HANDLER_METADATA_ATTR, None)
    if isinstance(existing, tuple):
        filtered = tuple(
            item
            for item in existing
            if not isinstance(item, EventSubscription) or item.event_name != event_name
        )
        setattr(callback, EVENT_HANDLER_METADATA_ATTR, filtered + (subscription,))
        return callback
    setattr(callback, EVENT_HANDLER_METADATA_ATTR, (subscription,))
    return callback


def iter_subscriptions(
    callback: Callable[..., None],
) -> tuple[EventSubscription, ...]:
    """Return the declared event subscriptions stored on a callback.

    Args:
        callback: Function or method that may carry subscription metadata.

    Returns:
        tuple[EventSubscription, ...]: Declared event subscriptions attached by ``@subscribe``.
    """
    subscriptions = getattr(callback, EVENT_HANDLER_METADATA_ATTR, ())
    if not isinstance(subscriptions, tuple):
        return ()
    normalized: list[EventSubscription] = []
    for item in subscriptions:
        if isinstance(item, EventSubscription):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(
                EventSubscription(event_name=cast(EventName, item), enabled=True)
            )
    return tuple(normalized)


@overload
def subscribe(
    event_name: Literal["agent_task_state_changed"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["agent_approval_requested"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["agent_choice_requested"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["agent_task_finished"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["ready"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["shutdown"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["fatal"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["error"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["voice_changed"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["state_changed"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["generation_start"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["generation_end"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["generation_error"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["audio_start"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["audio_end"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["character_changed"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["character_loaded"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


@overload
def subscribe(
    event_name: Literal["character_unloaded"],
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]: ...


def subscribe(
    event_name: EventName,
    *,
    enabled: bool = True,
) -> Callable[[_DecoratedCallback], _DecoratedCallback]:
    """Declare an event subscription on a function or bound-method definition.

    Args:
        event_name: Event name that should be attached to the callback metadata.
        enabled: Whether the handler should be auto-registered when discovered.

    Returns:
        Callable[..., Callable[..., None]]: Decorator that records the event name on the callback.
    """

    def decorator(callback: _DecoratedCallback) -> _DecoratedCallback:
        return cast(
            _DecoratedCallback,
            _store_subscription_metadata(
                cast(DispatcherCallback, callback),
                event_name,
                enabled,
            ),
        )

    return decorator
