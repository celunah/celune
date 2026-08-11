# SPDX-License-Identifier: MIT
"""Tests for Celune's extension event subsystem."""

import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, mock

from celune import subscribe
from celune.celune import Celune
from celune.dataclasses.events import (
    AudioEndEvent,
    AudioStartEvent,
    CharacterChangedEvent,
    CharacterLoadedEvent,
    CharacterUnloadedEvent,
    ReadyEvent,
    ShutdownEvent,
    StateChangedEvent,
    VoiceChangedEvent,
)
from celune.dataclasses.extensions import CeluneContext
from celune.extensions.base import CeluneExtension
from celune.extensions.events import EventDispatcher, iter_subscriptions
from celune.extensions.manager import CeluneExtensionManager
from celune.typing.events import ReadyEventCallback

from .support import FakeBackend, FakeGlow


class DispatcherTests(TestCase):
    """Tests for low-level event dispatch behavior."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.dispatcher = EventDispatcher(
            log_warning=lambda msg, severity="info": self.logs.append((msg, severity))
        )

    def test_dispatcher_registers_dispatches_and_unregisters_handlers(self) -> None:
        """Verify handlers receive events until they are unsubscribed."""
        calls: list[str] = []
        ready_event = ReadyEvent(celune=cast(Celune, SimpleNamespace()))

        def first(_event: ReadyEvent) -> None:
            calls.append("first")

        def second(_event: ReadyEvent) -> None:
            calls.append("second")

        self.dispatcher.subscribe(
            "ready",
            cast(ReadyEventCallback, first),
            owner_name="first",
        )
        self.dispatcher.subscribe(
            "ready",
            cast(ReadyEventCallback, second),
            owner_name="second",
        )
        self.dispatcher.emit("ready", ready_event)
        self.assertEqual(calls, ["first", "second"])

        self.dispatcher.unsubscribe("ready", first)
        self.dispatcher.emit("ready", ready_event)
        self.assertEqual(calls, ["first", "second", "second"])

    def test_dispatcher_logs_handler_failures_and_continues(self) -> None:
        """Verify one failing handler does not block later handlers."""
        calls: list[str] = []
        ready_event = ReadyEvent(celune=cast(Celune, SimpleNamespace()))

        def broken(_event: ReadyEvent) -> None:
            calls.append("broken")
            raise RuntimeError("boom")

        def healthy(_event: ReadyEvent) -> None:
            calls.append("healthy")

        self.dispatcher.subscribe(
            "ready",
            cast(ReadyEventCallback, broken),
            owner_name="broken",
        )
        self.dispatcher.subscribe(
            "ready",
            cast(ReadyEventCallback, healthy),
            owner_name="healthy",
        )

        self.dispatcher.emit("ready", ready_event)

        self.assertEqual(calls, ["broken", "healthy"])
        self.assertEqual(self.logs[-1][1], "warning")
        self.assertIn("Event callback failed", self.logs[-1][0])


class ManagerEventTests(TestCase):
    """Tests for extension-manager event integration."""

    def setUp(self) -> None:
        self.logs: list[tuple[str, str]] = []
        self.context = CeluneContext(
            log=lambda msg, severity="info", **kwargs: self.logs.append(
                (msg, severity)
            ),
            log_level="verbose",
            say=lambda text, save=True, display_text=None: True,
            think=lambda text: True,
            play=lambda sound_path, keep=False, volume=1.0: True,
            status=lambda msg, severity="info": None,
            set_voice=lambda name: True,
            get_state=lambda: "idle",
            wait_until_ready=lambda timeout=30.0: True,
        )
        self.dispatcher = EventDispatcher(
            log_warning=lambda msg, severity="info": self.logs.append((msg, severity))
        )

    def test_decorator_stores_subscription_metadata(self) -> None:
        """Verify the decorator only records metadata until registration time."""

        @subscribe("ready")
        def on_ready(_event: ReadyEvent) -> None:
            return None

        subscriptions = iter_subscriptions(on_ready)
        self.assertEqual(len(subscriptions), 1)
        self.assertEqual(subscriptions[0].event_name, "ready")
        self.assertEqual(subscriptions[0].enabled, True)

    def test_manager_discovers_extension_handlers_and_unregisters_them(self) -> None:
        """Verify decorated extension methods auto-register and clean up on unload."""
        received: list[str] = []

        class EventExtension(CeluneExtension):
            """Fixture extension used by one event-manager test."""

            EXTENSION_NAME = "Events"

            @subscribe("ready")
            def on_ready(self, _event: ReadyEvent) -> None:
                """React to a mock ready event.

                Args:
                    _event: The ready event payload.
                """
                received.append("ready")

            def invoke(self, *args, **kwargs) -> None:
                return None

        manager = CeluneExtensionManager(self.context, self.dispatcher)
        manager.register(EventExtension)
        self.dispatcher.emit("ready", ReadyEvent(celune=mock.Mock(spec=Celune)))
        self.assertEqual(received, ["ready"])

        manager.unregister("Events")
        self.dispatcher.emit("ready", ReadyEvent(celune=mock.Mock(spec=Celune)))
        self.assertEqual(received, ["ready"])

    def test_manager_can_disable_subscriptions_per_handler(self) -> None:
        """Verify extensions can opt out of individual handlers."""
        received: list[str] = []

        class EventExtension(CeluneExtension):
            """Fixture extension used by one ready-disable test."""

            EXTENSION_NAME = "Events"

            @subscribe("ready", enabled=False)
            def on_ready(self, _event: ReadyEvent) -> None:
                """React to a mock ready event.

                Args:
                    _event: The ready event payload.
                """
                received.append("ready")

            @subscribe("voice_changed", enabled=True)
            def on_voice_changed(self, _event: VoiceChangedEvent) -> None:
                """React to a mock voice change event.

                Args:
                    _event: The voice-change event payload.
                """
                received.append("voice")

            def invoke(self, *args, **kwargs) -> None:
                return None

        manager = CeluneExtensionManager(self.context, self.dispatcher)
        manager.register(EventExtension)
        self.dispatcher.emit("ready", ReadyEvent(celune=mock.Mock(spec=Celune)))
        self.dispatcher.emit(
            "voice_changed",
            VoiceChangedEvent(
                celune=mock.Mock(spec=Celune),
                old_voice="balanced",
                new_voice="bold",
            ),
        )
        self.assertEqual(received, ["voice"])

    def test_manager_autoloads_module_level_handlers(self) -> None:
        """Verify event-only extension modules can subscribe through ``@celune.subscribe``."""
        manager = CeluneExtensionManager(self.context, self.dispatcher)

        with tempfile.TemporaryDirectory() as temp_dir:
            extension_file = Path(temp_dir) / "fixture.py"
            extension_file.write_text(
                textwrap.dedent(
                    """
                    import celune

                    EVENTS = []

                    @celune.subscribe("ready")
                    def on_ready(event):
                        EVENTS.append(event.__class__.__name__)
                    """
                ),
                encoding="utf-8",
            )

            manager.autoload(temp_dir)
            module = sys.modules["user_extension_fixture"]
            self.dispatcher.emit("ready", ReadyEvent(celune=mock.Mock(spec=Celune)))

        self.assertEqual(module.EVENTS, ["ReadyEvent"])


class EngineEventIntegrationTests(TestCase):
    """Tests for Celune runtime event emission."""

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    def _make_celune(self) -> Celune:
        """Create a lightweight Celune instance for event tests."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            celune = Celune(config={}, tts_backend=FakeBackend)
            self.addCleanup(self._close_celune, celune)
            return celune

    def test_state_and_shutdown_events_are_emitted(self) -> None:
        """Verify runtime state transitions and shutdown emit typed payloads."""
        celune = self._make_celune()
        state_events: list[StateChangedEvent] = []
        shutdown_events: list[ShutdownEvent] = []
        celune._event_dispatcher.subscribe("state_changed", state_events.append)
        celune._event_dispatcher.subscribe("shutdown", shutdown_events.append)

        celune.cur_state = "speaking"
        celune.close()
        celune.close()

        self.assertEqual(state_events[-1].old_state, "init")
        self.assertEqual(state_events[-1].new_state, "speaking")
        self.assertEqual(len(shutdown_events), 1)

    def test_voice_change_emits_typed_event(self) -> None:
        """Verify successful voice changes emit a ``VoiceChangedEvent`` payload."""
        celune = self._make_celune()
        celune.current_voice = "balanced"
        celune.voices = ("balanced", "bold")
        celune.model_name = "shared-model"
        celune.loaded = True
        celune.cur_state = "idle"
        celune.backend.model_id_for_voice = mock.Mock(return_value="shared-model")
        events: list[VoiceChangedEvent] = []
        celune._event_dispatcher.subscribe("voice_changed", events.append)

        with mock.patch("celune.celune.play_signal", return_value=False):
            celune.change_voice("bold")

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].old_voice, "balanced")
        self.assertEqual(events[0].new_voice, "bold")

    def test_character_bundle_events_emit_typed_payloads(self) -> None:
        """Verify CEVOICE lifecycle transitions emit load, unload, and change events."""
        celune = self._make_celune()
        celune.backend.uses_voice_bundles = True
        loaded_events: list[CharacterLoadedEvent] = []
        unloaded_events: list[CharacterUnloadedEvent] = []
        changed_events: list[CharacterChangedEvent] = []
        celune._event_dispatcher.subscribe("character_loaded", loaded_events.append)
        celune._event_dispatcher.subscribe("character_unloaded", unloaded_events.append)
        celune._event_dispatcher.subscribe("character_changed", changed_events.append)

        first_bundle = mock.Mock()
        first_bundle.path = Path("celune.cevoice")
        first_bundle.voice_order = ("balanced", "bold")
        first_bundle.metadata = {"name": "Celune", "default_voice": "balanced"}
        first_loader = mock.Mock(bundle=first_bundle)

        second_bundle = mock.Mock()
        second_bundle.path = Path("nova.cevoice")
        second_bundle.voice_order = ("bold", "balanced")
        second_bundle.metadata = {"name": "Nova", "default_voice": "bold"}
        second_loader = mock.Mock(bundle=second_bundle)

        with (
            mock.patch("celune.celune.select_voice_bundle"),
            mock.patch(
                "celune.celune.default_loader",
                side_effect=[
                    None,
                    first_loader,
                    first_loader,
                    second_loader,
                    second_loader,
                    None,
                ],
            ),
        ):
            self.assertEqual(celune.load_voice_bundle(Path("celune.cevoice")), True)
            self.assertEqual(celune.load_voice_bundle(Path("nova.cevoice")), True)
            self.assertEqual(celune.load_voice_bundle(None), True)

        self.assertEqual(len(loaded_events), 1)
        self.assertEqual(loaded_events[0].character_name, "Celune")
        self.assertEqual(loaded_events[0].bundle_path, str(Path("celune.cevoice")))
        self.assertEqual(len(changed_events), 1)
        self.assertEqual(changed_events[0].old_character, "Celune")
        self.assertEqual(changed_events[0].new_character, "Nova")
        self.assertEqual(len(unloaded_events), 1)
        self.assertEqual(unloaded_events[0].character_name, "Nova")

    def test_typed_event_payloads_expose_expected_attributes(self) -> None:
        """Verify runtime event payload dataclasses carry the documented fields."""
        celune = self._make_celune()
        audio_start = AudioStartEvent(
            celune=celune,
            source_id=1,
            label="fixture",
            kind="sfx",
        )
        audio_end = AudioEndEvent(
            celune=celune,
            source_id=1,
            label="fixture",
            kind="sfx",
            saved_path="outputs/test.flac",
        )

        self.assertEqual(audio_start.label, "fixture")
        self.assertEqual(audio_end.saved_path, "outputs/test.flac")
        character_loaded = CharacterLoadedEvent(
            celune=celune,
            character_name="Celune",
            bundle_path="celune.cevoice",
            is_default=True,
        )
        self.assertEqual(character_loaded.character_name, "Celune")
