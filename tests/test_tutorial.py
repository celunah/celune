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
        string("commands.tutorial_voice_pack"),
        string("commands.tutorial_extensions"),
        string("commands.tutorial_persona"),
        string("commands.tutorial_local_api"),
        string("commands.tutorial_agent"),
        string("commands.tutorial_wrap_up"),
    ]
    assert [event for event in events if event.startswith("pulse:")] == [
        "pulse:#input",
        "pulse:#style",
    ]
    assert "type:/help:True" in events
    assert events.count("wait") == 10
    assert events[-1] == "finish"


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
