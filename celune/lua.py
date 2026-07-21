# SPDX-License-Identifier: MIT
"""Lua runtime hosting and Celune bindings."""

from __future__ import annotations

import threading
from dataclasses import fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, cast

from .extensions.base import CeluneExtension
from .extensions.events import RegisteredEventHandler
from .dataclasses.lua import _LuaRuntime, _LuaTable
from .typing.common import JSONSerializable
from .typing.events import EventName, EventPayload
from .typing.lua import LuaValue, _LuaScalar
from .utils import format_error

if TYPE_CHECKING:
    from .celune import Celune
    from .extensions.base import CeluneContext
    from .extensions.events import EventDispatcher


_LUA_EVENT_NAMES: tuple[EventName, ...] = (
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


def _lua_event_value(value: _LuaScalar) -> JSONSerializable:
    """Convert one event value into a value safe to pass into Lua."""
    if isinstance(value, Exception):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported Lua event value: {type(value).__name__}")


def _event_table_data(
    event_name: EventName, event: EventPayload
) -> dict[str, LuaValue]:
    """Build the serializable event table exposed to Lua callbacks."""
    if not is_dataclass(event):
        return {"name": event_name}

    data: dict[str, LuaValue] = {"name": event_name}
    for event_field in fields(event):
        if event_field.name == "celune":
            continue
        data[event_field.name] = cast(
            LuaValue,
            _lua_event_value(cast(_LuaScalar, getattr(event, event_field.name))),
        )
    return data


class LuaRuntimeManager:
    """Own one isolated Lua runtime and its Celune event registrations."""

    def __init__(
        self,
        core: "Celune",
        dispatcher: "EventDispatcher",
        *,
        max_memory: Optional[int] = None,
    ) -> None:
        try:
            lupa_module = import_module("lupa")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Lupa is required for Lua extensions") from exc

        lua_runtime_factory = cast(
            Callable[..., _LuaRuntime],
            getattr(lupa_module, "LuaRuntime"),
        )
        self._core = core
        self._dispatcher = dispatcher
        self._lock = threading.RLock()
        self._registration_lock = threading.Lock()
        self._closed = False
        self._registrations: list["RegisteredEventHandler"] = []
        self._runtime = lua_runtime_factory(
            encoding="utf-8",
            register_eval=False,
            register_builtins=False,
            max_memory=max_memory,
        )
        self._globals = self._runtime.globals()
        self._globals["celune"] = core
        self._globals["subscribe"] = self.subscribe
        self._globals["CELUNE_VERSION"] = getattr(core, "version", "")

    def load(self, path: Path) -> str:
        """Execute one Lua extension file and return its registered name.

        Args:
            path: Lua source file to execute.

        Returns:
            str: The configured extension name, or the file stem when omitted.
        """
        source = path.read_text(encoding="utf-8")
        with self._lock:
            self._ensure_open()
            self._runtime.execute(source, name=str(path))
            extension_name = self._globals["EXTENSION_NAME"]
            if not isinstance(extension_name, str) or not extension_name.strip():
                extension_name = path.stem
            return extension_name.strip()

    def invoke(self, *args: LuaValue, **kwargs: LuaValue) -> None:
        """Invoke the optional global Lua ``invoke`` function.

        Args:
            args: Positional values forwarded to the Lua function.
            kwargs: Keyword values forwarded to the Lua function.
        """
        with self._lock:
            self._ensure_open()
            invoke = self._globals["invoke"]
            if not callable(invoke):
                return
            cast(Callable[..., None], invoke)(*args, **kwargs)

    def subscribe(
        self,
        event_name: str,
        callback: Callable[[LuaValue], None],
        enabled: bool = True,
    ) -> None:
        """Register one Lua callback with Celune's event dispatcher.

        Args:
            event_name: Event name that should trigger the callback.
            callback: Lua function invoked with a serialized event table.
            enabled: Whether the callback should be registered with Celune.

        Notes:
            Lua callbacks run on daemon workers so blocking code and nested
            Celune events cannot delay Celune's state transitions.

        Raises:
            ValueError: If the event name is not supported.
            TypeError: If the callback is not callable.
        """
        if event_name not in _LUA_EVENT_NAMES:
            raise ValueError(f"unknown event name: {event_name}")
        if not callable(callback):
            raise TypeError("Lua event callback is not callable")
        if not enabled:
            return
        lua_callback = cast(Callable[[_LuaTable], None], callback)

        typed_event_name = cast(EventName, event_name)

        def invoke_callback(event: EventPayload) -> None:
            try:
                with self._lock:
                    if self._closed:
                        return
                    event_data = _event_table_data(typed_event_name, event)
                    lua_event = self._runtime.table_from(event_data, recursive=True)
                    lua_callback(lua_event)
            except Exception as exc:
                self._core.log(
                    (
                        f"[Core] Lua event callback failed for '{typed_event_name}': "
                        f"{format_error(exc, self._core.dev)}"
                    ),
                    "warning",
                )

        def handler(event: EventPayload) -> None:
            if self._closed:
                return
            threading.Thread(
                target=invoke_callback,
                args=(event,),
                daemon=True,
                name=f"celune-lua-{typed_event_name}-{self.extension_name}",
            ).start()

        self._dispatcher.subscribe(
            typed_event_name,
            handler,
            owner_name=self.extension_name,
        )
        registration = RegisteredEventHandler(
            event_name=typed_event_name,
            callback=handler,
            owner_name=self.extension_name,
        )
        with self._registration_lock:
            if self._closed:
                self._dispatcher.unsubscribe(
                    registration.event_name,
                    registration.callback,
                )
                return
            self._registrations.append(registration)

    @property
    def extension_name(self) -> str:
        """Return the current Lua extension name for diagnostics.

        Returns:
            str: The configured extension name, or a fallback label.
        """
        extension_name = self._globals["EXTENSION_NAME"]
        if isinstance(extension_name, str) and extension_name.strip():
            return extension_name.strip()
        return "Lua extension"

    def close(self) -> None:
        """Unregister callbacks and release the Lua runtime."""
        self._closed = True
        with self._registration_lock:
            registrations = list(self._registrations)
            self._registrations.clear()
        for registration in registrations:
            self._dispatcher.unsubscribe(
                registration.event_name,
                registration.callback,
            )

    def _ensure_open(self) -> None:
        """Reject operations after the runtime has been closed."""
        if self._closed:
            raise RuntimeError("Lua extension runtime is closed")


class LuaExtension(CeluneExtension):
    """Adapter representing one loaded Lua file to the extension manager."""

    def __init__(
        self,
        context: "CeluneContext",
        name: str,
        runtime: LuaRuntimeManager,
    ) -> None:
        super().__init__(context)
        self.EXTENSION_NAME = name
        self.runtime = runtime

    @property
    def name(self) -> str:
        """Return the Lua extension display name.

        Returns:
            str: The extension name exposed to Celune.
        """
        return self.EXTENSION_NAME

    def invoke(self, *args: LuaValue, **kwargs: LuaValue) -> None:
        """Invoke the Lua extension's optional global function.

        Args:
            args: Positional values forwarded to the Lua function.
            kwargs: Keyword values forwarded to the Lua function.
        """
        self.runtime.invoke(*args, **kwargs)

    def close(self) -> None:
        """Close the Lua runtime owned by this extension."""
        self.runtime.close()
