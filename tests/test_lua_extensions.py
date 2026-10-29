# SPDX-License-Identifier: MIT
"""Tests for Lua extension loading and Celune bindings."""

import time
import tempfile
import textwrap
import threading
from pathlib import Path
from unittest import TestCase
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from celune.lua import LuaExtension
from celune.extensions.events import EventDispatcher
from celune.dataclasses.extensions import CeluneContext
from celune.extensions.manager import CeluneExtensionManager
from celune.dataclasses.events import ReadyEvent, VoiceChangedEvent

if TYPE_CHECKING:
    from celune.celune import Celune


class LuaExtensionTests(TestCase):
    """Verify Lua extensions use the existing extension manager lifecycle."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []

        def capture_log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: str = "info",
        ) -> None:
            del loglevel
            self.logs.append((msg, severity))

        self.dispatcher = EventDispatcher(
            log_warning=lambda message, severity="info": self.logs.append(
                (message, severity)
            )
        )
        self.context = CeluneContext(
            log=capture_log,
            say=lambda text, save=True, display_text=None: True,
            think=lambda text: True,
            play=lambda sound_path, keep=False, volume=1.0: True,
            status=lambda msg, severity="info": None,
            set_voice=lambda name: True,
            get_state=lambda: "idle",
            wait_until_ready=lambda timeout=30.0: True,
        )

    def test_manager_injects_core_and_invokes_lua_extension(self) -> None:
        """Verify Lua can call the injected Celune singleton through ``invoke``."""
        invoked = threading.Event()
        core = SimpleNamespace(
            say=lambda text: (self.assertEqual(text, "hello"), invoked.set()),
            version="test",
        )
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "fixture.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    EXTENSION_NAME = "Lua fixture"

                    function invoke(text)
                        celune.say(text)
                    end
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        self.assertEqual(manager.list_extensions(), ["Lua fixture"])
        manager.invoke("Lua fixture", "hello")
        self.assertTrue(invoked.wait(timeout=5))

    def test_lua_event_callbacks_receive_serialized_payloads(self) -> None:
        """Verify Lua callbacks receive event fields without the raw core object."""
        callback_complete = threading.Event()
        core = SimpleNamespace(
            log=lambda message, severity="info": callback_complete.set(),
            version="test",
        )
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "events.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    subscribe("voice_changed", function(event)
                        observed = event.old_voice .. "->" .. event.new_voice
                        has_core = event.celune ~= nil
                        celune.log(observed)
                    end)
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        extension = manager.extensions["events"]
        self.dispatcher.emit(
            "voice_changed",
            VoiceChangedEvent(
                celune=cast("Celune", core),
                old_voice="balanced",
                new_voice="bold",
            ),
        )
        self.assertTrue(callback_complete.wait(timeout=5))

        lua_extension = cast(LuaExtension, extension)
        self.assertEqual(lua_extension.runtime._globals["observed"], "balanced->bold")
        self.assertFalse(lua_extension.runtime._globals["has_core"])

        manager.unregister("events")
        self.assertEqual(manager.list_extensions(), [])

    def test_manager_without_core_rejects_lua_registration(self) -> None:
        """Verify standalone extension managers cannot load Lua core bindings."""
        manager = CeluneExtensionManager(self.context, self.dispatcher)
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            self.assertRaises(RuntimeError),
        ):
            manager.register_lua(Path(temp_dir) / "missing.lua")

    def test_lua_subscriptions_can_be_disabled(self) -> None:
        """Verify Lua supports disabled subscriptions through a third argument."""
        core = SimpleNamespace(version="test")
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "disabled.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    subscribe("ready", function()
                        called = true
                    end, false)
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        self.dispatcher.emit(
            "ready",
            ReadyEvent(celune=cast("Celune", core)),
        )
        lua_extension = cast(LuaExtension, manager.extensions["disabled"])
        self.assertIsNone(lua_extension.runtime._globals["called"])

    def test_closing_lua_extension_stops_new_callbacks(self) -> None:
        """Verify unloading a Lua extension removes callbacks before later events run."""
        core = SimpleNamespace(version="test")
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "closing.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    subscribe("ready", function()
                        called = true
                    end)
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        extension = cast(LuaExtension, manager.extensions["closing"])
        manager.unregister("closing")
        self.dispatcher.emit(
            "ready",
            ReadyEvent(celune=cast("Celune", core)),
        )
        time.sleep(0.05)

        self.assertIsNone(extension.runtime._globals["called"])

    def test_lua_ready_callbacks_do_not_block_core_loading(self) -> None:
        """Verify blocking Lua startup work runs outside the ready dispatcher."""
        callback_started = threading.Event()
        release_callback = threading.Event()
        completed = threading.Event()

        def block_callback(_duration: float) -> None:
            callback_started.set()
            release_callback.wait(timeout=5.0)

        core = SimpleNamespace(
            log=lambda message, severity="info": completed.set(),
            sleep=block_callback,
            version="test",
        )
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "ready.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    subscribe("ready", function()
                        celune.sleep(0.2)
                        celune.log("ready callback complete")
                    end)
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        dispatch_thread = threading.Thread(
            target=self.dispatcher.emit,
            args=("ready", ReadyEvent(celune=cast("Celune", core))),
            daemon=True,
        )
        dispatch_thread.start()
        try:
            self.assertTrue(callback_started.wait(timeout=5.0))
            dispatch_thread.join(timeout=5.0)
            self.assertFalse(dispatch_thread.is_alive())
        finally:
            release_callback.set()
            dispatch_thread.join(timeout=5.0)

        self.assertTrue(completed.wait(timeout=5))

    def test_nested_lua_events_do_not_deadlock_a_reload(self) -> None:
        """Verify a reload-triggered Lua event cannot block its completion signal."""
        completed = threading.Event()
        core = SimpleNamespace(
            log=lambda message, severity="info": completed.set(),
            version="test",
        )

        def set_voice(name: str) -> bool:
            self.dispatcher.emit(
                "voice_changed",
                VoiceChangedEvent(
                    celune=cast("Celune", core),
                    old_voice="balanced",
                    new_voice=name,
                ),
            )
            return True

        core.set_voice = set_voice
        manager = CeluneExtensionManager(
            self.context,
            self.dispatcher,
            core=cast("Celune", core),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_path = Path(temp_dir) / "nested.lua"
            extension_path.write_text(
                textwrap.dedent(
                    """
                    subscribe("voice_changed", function()
                        celune.log("voice callback complete")
                    end)

                    subscribe("ready", function()
                        celune.set_voice("calm")
                        celune.log("ready callback complete")
                    end)
                    """
                ),
                encoding="utf-8",
            )
            manager.autoload(temp_dir)

        self.dispatcher.emit(
            "ready",
            ReadyEvent(celune=cast("Celune", core)),
        )
        self.assertTrue(completed.wait(timeout=5))
