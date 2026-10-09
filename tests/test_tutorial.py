# SPDX-License-Identifier: Apache-2.0
"""Tests for the spoken tutorial flow."""

from typing import cast
from unittest import mock
from types import SimpleNamespace
from collections.abc import Callable

from celune.i18n import string
from celune.ui import commands
from celune.celune import Celune
from celune.ui.app import CeluneUI
from celune.speech import _say_tutorial


def _tutorial_ui(events: list[str]) -> SimpleNamespace:
    """Build a minimal UI whose thread handoff executes immediately."""
    ui = SimpleNamespace()
    ui.tutorial_token = 0
    ui.tutorial_active = False
    ui.logs = []
    ui.safe_log = lambda message, severity="info": ui.logs.append(
        f"{severity}: {message}"
    )
    ui.call_from_thread = lambda callback, *args: callback(*args)
    ui.begin_tutorial = lambda: (
        setattr(ui, "tutorial_token", ui.tutorial_token + 1),
        setattr(ui, "tutorial_active", True),
    )
    ui.finish_tutorial = lambda: (
        events.append("finish"),
        setattr(ui, "tutorial_active", False),
    )
    ui.cancel_tutorial = lambda stop_audio=True: (
        events.append(f"cancel:{stop_audio}"),
        setattr(ui, "tutorial_active", False),
    )
    ui.pulse_border = lambda selector: events.append(f"pulse:{selector}")
    ui.type_and_send = lambda text, process_commands=False: events.append(
        f"type:{text}:{process_commands}"
    )
    ui.celune = SimpleNamespace(
        cur_state="idle",
        wait_until_idle=lambda **_kwargs: events.append("wait") or True,
        log_level="info",
    )
    return ui


def test_tutorial_keeps_each_utterance_separate_and_syncs_actions(
    monkeypatch,
) -> None:
    """Keep tutorial lines separate and start each UI action in its matching line."""
    events: list[str] = []
    ui = _tutorial_ui(events)

    class ImmediateThread:
        """Run the tutorial worker inline for deterministic coverage."""

        def __init__(
            self,
            target: Callable[..., None],
            args: tuple[object, ...],
            daemon: bool,
        ) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            """Run the stored target synchronously."""
            self.target(*self.args)

    def speak(_engine, text: str) -> bool:
        events.append(f"speech:{text}")
        return True

    monkeypatch.setattr(commands, "threading", SimpleNamespace(Thread=ImmediateThread))
    monkeypatch.setattr(commands, "_say_tutorial", speak)

    commands.tutorial(cast(CeluneUI, ui))

    assert [
        event.removeprefix("speech:") for event in events if event.startswith("speech:")
    ] == [
        string("commands.tutorial_intro"),
        string("commands.tutorial_input"),
        string("commands.tutorial_voice"),
        string("commands.tutorial_help"),
        string("commands.tutorial_help_simple"),
        string("commands.tutorial_help_vibe"),
        string("commands.tutorial_extensions"),
        string("commands.tutorial_extension_example"),
        string("commands.tutorial_extension_code"),
        string("commands.tutorial_local_api"),
        string("commands.tutorial_local_api_usage"),
        string("commands.tutorial_voice_self", app_name=commands.APP_NAME),
        string("commands.tutorial_voice_default", app_name=commands.APP_NAME),
        string("commands.tutorial_voice_pack"),
        string("commands.tutorial_voice_pack_continued"),
        string("commands.tutorial_persona"),
        string("commands.tutorial_persona_chat"),
        string("commands.tutorial_persona_speech"),
        string("commands.tutorial_persona_invitation"),
        string("commands.tutorial_agent"),
        string("commands.tutorial_agent_abilities"),
        string("commands.tutorial_agent_actions"),
        string("commands.tutorial_can_do_more"),
        string("commands.tutorial_variety", app_name=commands.APP_NAME),
        string("commands.tutorial_variety_many"),
        string("commands.tutorial_supported"),
        string("commands.tutorial_wrap_up"),
        string("commands.tutorial_wait"),
    ]
    assert [event for event in events if event.startswith("pulse:")] == [
        "pulse:#input",
        "pulse:#style",
    ]
    assert "type:/help:True" in events
    assert events.count("wait") == 28
    assert events[-1] == "finish"


def test_english_tutorial_uses_the_recorded_wording() -> None:
    """Keep the English tutorial faithful to its original WAV narration."""
    assert [
        string(f"commands.tutorial_{name}", locale="en", **arguments)
        for name, arguments in (
            ("intro", {}),
            ("input", {}),
            ("voice", {}),
            ("help", {}),
            ("help_simple", {}),
            ("help_vibe", {}),
            ("extensions", {}),
            ("extension_example", {}),
            ("extension_code", {}),
            ("local_api", {}),
            ("local_api_usage", {}),
            ("voice_self", {}),
            ("voice_default", {"app_name": commands.APP_NAME}),
            ("voice_pack", {}),
            ("voice_pack_continued", {}),
            ("persona", {}),
            ("persona_chat", {}),
            ("persona_speech", {}),
            ("persona_invitation", {}),
            ("agent", {}),
            ("agent_abilities", {}),
            ("agent_actions", {}),
            ("can_do_more", {}),
            ("variety", {"app_name": commands.APP_NAME}),
            ("variety_many", {}),
            ("supported", {}),
            ("wrap_up", {}),
            ("wait", {}),
        )
    ] == [
        "This is my main control panel.",
        "Type here. It's the loop.",
        "That button on the right, it's the way you reach out for the calm and any other tones.",
        "The slash key will tell you about any additional features.",
        "It's meant to be simple and never overwhelming.",
        "Vibe to the sound of my voice, the easy way.",
        "You can also write custom extensions to programmatically use my abilities.",
        "There's an example extension already present.",
        "Look at the code to see what can I do.",
        "For those that need external access to my voice, I can start an API for you.",
        "You can post stuff to 127.0.0.1, port 2060, and I'll say that for you.",
        "I can also speak in your own voice.",
        f"The voice you are hearing right now is the default in {commands.APP_NAME}.",
        "You can however load your own to provide your own CE voice pack into my voices directory,",
        "and I'll be able to speak as your character and not just myself.",
        "As a version 4.0, a new Persona system has been added and fixed in version 4.3,",
        "allowing you to properly talk to me or anyone else running in the software.",
        "It also includes speech recognition capabilities.",
        "Let your voice be heard and your character respond back.",
        "By the way, I've just gained new abilities.",
        "I now possess something called an agent.",
        "This agent lets me perform actions directly on your machine, with more features coming over time.",
        "I am no longer limited to just talking and speaking.",
        f"There really isn't just one way to use {commands.APP_NAME}.",
        "There's many of them.",
        "No matter how you typically interact with software, I likely already support it.",
        "Go ahead, type something.",
        "I'll stay here until you do.",
    ]


def test_tutorial_stops_after_cancellation(monkeypatch) -> None:
    """Do not queue later tutorial sections after a user cancels playback."""
    events: list[str] = []
    ui = _tutorial_ui(events)
    ui.tutorial_active = True

    def wait_until_idle(**_kwargs) -> bool:
        ui.tutorial_active = False
        ui.tutorial_token += 1
        return True

    ui.celune.wait_until_idle = wait_until_idle
    monkeypatch.setattr(
        commands,
        "_say_tutorial",
        lambda _engine, text: events.append(f"speech:{text}") or True,
    )

    commands._run_tutorial_sequence(
        cast(CeluneUI, ui),
        0,
        (("first section", None), ("second section", None)),
    )

    assert events == ["speech:first section"]


def test_tutorial_cancels_when_speech_cannot_be_queued(monkeypatch) -> None:
    """Report a failed speech request and release tutorial input controls."""
    events: list[str] = []
    ui = _tutorial_ui(events)
    ui.tutorial_active = True
    monkeypatch.setattr(commands, "_say_tutorial", lambda _engine, _text: False)

    commands._run_tutorial_sequence(
        cast(CeluneUI, ui),
        0,
        (("first section", None),),
    )

    assert events == ["cancel:True"]
    assert len(ui.logs) == 1
    assert ui.logs[0][0:8] == "warning:"


def test_tutorial_speech_bypasses_only_the_tutorial_input_guard() -> None:
    """Route tutorial speech through the pipeline without enabling ordinary input."""
    engine = SimpleNamespace(
        is_in_tutorial=True,
        test_finished=False,
        input_mode="text_to_speech",
        cur_state="idle",
        log=mock.Mock(),
    )
    with mock.patch(
        "celune.speech._queue_speech_request",
        return_value=True,
    ) as queue_speech:
        assert _say_tutorial(cast(Celune, engine), "Tutorial line.")

    queue_speech.assert_called_once_with(
        engine,
        "Tutorial line.",
        save=False,
        stream_queue=None,
        display_text="Tutorial line.",
        allow_tutorial=True,
    )
