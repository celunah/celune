# SPDX-License-Identifier: Apache-2.0
"""Tests for the spoken tutorial flow."""

import queue
import threading
from types import SimpleNamespace
from typing import Union, cast
from unittest import mock
from collections.abc import Callable

import numpy as np

from celune import pipeline
from celune.ui import commands
from celune.i18n import string
from celune.celune import Celune
from celune.speech import _play_tutorial_sections, _prepare_tutorial_sections
from celune.ui.app import CeluneUI
from celune.prepared import play_prepared_speech
from celune.dataclasses.pipeline import (
    SpeechRequest,
    PreparedSpeechAudio,
    CaptionAlignmentSection,
)

from .support import make_pipeline_engine


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

    def prepare(_engine, sections, is_active, _timeout):
        events.append("prepare")
        assert is_active()
        return tuple(f"audio:{index}" for index in range(len(sections)))

    def play(_engine, sections, audio, is_active, on_start, _timeout):
        events.append("play")
        assert is_active()
        assert len(audio) == len(sections)
        for index in range(len(sections)):
            on_start(index)
        return True

    monkeypatch.setattr(commands, "threading", SimpleNamespace(Thread=ImmediateThread))
    monkeypatch.setattr(commands, "_prepare_tutorial_sections", prepare)
    monkeypatch.setattr(commands, "_play_tutorial_sections", play)

    commands.tutorial(cast(CeluneUI, ui))

    assert events[:2] == ["prepare", "play"]
    assert [event for event in events if event.startswith("pulse:")] == [
        "pulse:#input",
        "pulse:#style",
    ]
    assert "type:/help:True" in events
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
        "Type. Hear. It's the loop.",
        'That button on the right, it\'s the way you reach out for the "calm", and any other tones.',
        "The slash key will tell you about any additional features.",
        "It's meant to be simple, and never overwhelming.",
        "Vibe to the sound of my voice, the easy way.",
        "You can also write custom extensions to programmatically use my abilities.",
        "There is an example extension you can try out. Type /invoke Test to check it out.",
        "Look at the code to see what I am capable of.",
        "For those that need external access to my voice, I have an API for you.",
        "You can post stuff to 127.0.0.1, port 2060, and I'll say that for you.",
        "I can also speak in your own voice.",
        f"The voice you are hearing right now is the default in {commands.APP_NAME}.",
        'You can however load your own to provide your own "CE voice" pack into my voices directory,',
        "and I'll be able to speak as your character, and not just myself.",
        "As of version 4.0, a new Persona system has been added, and then fixed,",
        "allowing you to properly talk to me, or anyone else running in this software.",
        "It also includes speech recognition capabilities.",
        "Let your voice be heard, and your character respond back.",
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
    """Do not queue playback when preparation is canceled."""
    events: list[str] = []
    ui = _tutorial_ui(events)
    ui.tutorial_active = True

    def prepare(_engine, _sections, _is_active, _timeout):
        ui.tutorial_active = False
        ui.tutorial_token += 1

    monkeypatch.setattr(
        commands,
        "_prepare_tutorial_sections",
        prepare,
    )
    monkeypatch.setattr(
        commands,
        "_play_tutorial_sections",
        lambda *_args: events.append("play") or True,
    )

    commands._run_tutorial_sequence(
        cast(CeluneUI, ui),
        0,
        (("first section", None), ("second section", None)),
    )

    assert not events


def test_tutorial_cancels_when_speech_cannot_be_queued(monkeypatch) -> None:
    """Report a failed speech request and release tutorial input controls."""
    events: list[str] = []
    ui = _tutorial_ui(events)
    ui.tutorial_active = True
    monkeypatch.setattr(
        commands,
        "_prepare_tutorial_sections",
        lambda *_args: None,
    )

    commands._run_tutorial_sequence(
        cast(CeluneUI, ui),
        0,
        (("first section", None),),
    )

    assert events == ["cancel:True"]
    assert len(ui.logs) == 1
    assert ui.logs[0][0:8] == "warning:"


def test_tutorial_generation_is_batched_without_normalization(monkeypatch) -> None:
    """Generate each tutorial section independently and skip CeluneNorm."""
    sections = ("First section.", "Second section.")
    audio = (np.zeros((4, 2), dtype=np.float32), np.ones((5, 2), dtype=np.float32))

    def queue_speech(_engine, _transcript, **kwargs):
        assert kwargs["normalize"] is False
        assert kwargs["synthesis_sections"] == sections
        kwargs["audio_capture_queue"].put(PreparedSpeechAudio(audio))
        return True

    monkeypatch.setattr("celune.speech._queue_speech_request", queue_speech)

    prepared_audio = _prepare_tutorial_sections(
        cast(Celune, SimpleNamespace()), sections, lambda: True, 1.0
    )
    assert prepared_audio is not None
    assert len(prepared_audio) == len(audio)
    for prepared, expected in zip(prepared_audio, audio):
        np.testing.assert_array_equal(prepared, expected)


def test_pipeline_captures_tutorial_sections_without_playback() -> None:
    """Generate each tutorial section without normalization or playback."""
    engine = make_pipeline_engine()
    generated_text: list[str] = []

    def generate_stream(_model, *, text, **_kwargs):
        generated_text.append(text)
        audio = np.full((4, 2), len(generated_text) / 10, dtype=np.float32)
        yield audio, 48000, None

    engine.backend = SimpleNamespace(
        is_fake=True,
        name="fixture",
        supported_languages=("en",),
        generate_stream=generate_stream,
    )
    engine.model = object()
    engine.model_name = "fixture"
    engine.model_lock = threading.Lock()
    engine.speed = 1.0
    engine.can_use_rubberband = False
    engine.chunk_size = 128
    engine._speech_generation = 1
    engine.reverb = SimpleNamespace(
        strength=0.0,
        reset=mock.Mock(),
        flush=lambda: np.empty(0, dtype=np.float32),
    )
    audio_result: queue.Queue[Union[PreparedSpeechAudio, Exception]] = queue.Queue()
    request = SpeechRequest(
        text="First section. Second section.",
        display_text="First section. Second section.",
        save=False,
        normalize=False,
        generation=1,
        synthesis_sections=("First section.", "Second section."),
        audio_capture_queue=audio_result,
    )

    with (
        mock.patch("celune.pipeline._effective_voice_prompt", return_value=None),
        mock.patch("celune.pipeline._smart_buffer_target_seconds", return_value=0.0),
        mock.patch("celune.prepared.is_silent_utterance", return_value=(False, 0)),
    ):
        pipeline._process_generation_request(
            cast(Celune, engine),
            request,
            None,
        )

    result = audio_result.get_nowait()
    assert generated_text == ["First section.", "Second section."]
    assert isinstance(result, PreparedSpeechAudio)
    assert len(result.sections) == 2
    assert engine.normalize.call_count == 0
    assert engine.audio_queue.empty()
    assert engine.progress == [(0, 2), (1, 2), (2, 2)]


def test_tutorial_playback_stitches_audio_with_caption_transcript(monkeypatch) -> None:
    """Queue one stitched source with the combined transcript and timed actions."""
    sections = ("First section.", "Second section.", "Third section.")
    audio = (
        np.zeros((3, 2), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
        np.full((2, 2), 0.5, dtype=np.float32),
    )
    source_id = 7
    engine = SimpleNamespace(
        playback_done=mock.Mock(),
        _playback_source_meta={source_id: {"played_frames": 48_009}},
    )
    engine.playback_done.wait.return_value = True
    captured_audio: list[np.ndarray] = []
    captured_transcript: list[str] = []
    captured_sections: list[tuple[CaptionAlignmentSection, ...]] = []

    def queue_speech(_engine, transcript, **kwargs):
        captured_transcript.append(transcript)
        prepared = kwargs["prepared_audio"]
        assert isinstance(prepared, np.ndarray)
        captured_audio.append(prepared)
        captured_sections.append(kwargs["caption_alignment_sections"])
        kwargs["playback_source_queue"].put(source_id)
        return True

    monkeypatch.setattr("celune.speech._queue_speech_request", queue_speech)
    starts: list[int] = []

    assert _play_tutorial_sections(
        cast(Celune, engine),
        sections,
        audio,
        lambda: True,
        starts.append,
        1.0,
    )

    combined_audio = captured_audio[0]
    assert captured_transcript == ["First section. Second section. Third section."]
    assert combined_audio.shape == (48_009, 2)
    np.testing.assert_array_equal(combined_audio[:3], audio[0])
    assert np.count_nonzero(combined_audio[3:24_003]) == 0
    np.testing.assert_array_equal(combined_audio[24_003:24_007], audio[1])
    assert captured_sections == [
        (
            CaptionAlignmentSection(0, 3, 0, 2, 0, 2),
            CaptionAlignmentSection(24_003, 24_007, 2, 4, 2, 4),
            CaptionAlignmentSection(48_007, 48_009, 4, 6, 4, 6),
        )
    ]
    assert starts == [0, 1, 2]


def test_prepared_tutorial_playback_aligns_combined_transcript(monkeypatch) -> None:
    """Pass the combined tutorial transcript to the caption alignment worker."""
    transcript = "First section. Second section."
    audio = np.ones((8, 2), dtype=np.float32)
    source_queue: queue.Queue[Union[int, Exception]] = queue.Queue()
    alignment_sections = (CaptionAlignmentSection(0, 8, 0, 4, 0, 4),)
    request = SpeechRequest(
        text=transcript,
        display_text=transcript,
        language="English",
        save=False,
        prepared_audio=audio,
        playback_source_queue=source_queue,
        caption_alignment_sections=alignment_sections,
    )
    engine = SimpleNamespace(
        config={"captions": True},
        backend=SimpleNamespace(resolve_generation_language=lambda language: language),
        log=mock.Mock(),
        log_level="info",
        cur_state="idle",
        progress_callback=mock.Mock(),
        error_callback=mock.Mock(),
    )
    caption_worker = mock.Mock()
    monkeypatch.setattr("celune.prepared._next_playback_source_id", lambda _engine: 5)
    monkeypatch.setattr("celune.prepared._register_playback_source", mock.Mock())
    monkeypatch.setattr("celune.prepared.release_pipeline", mock.Mock())
    monkeypatch.setattr("celune.prepared.string", lambda key, **_kwargs: key)

    play_prepared_speech(cast(Celune, engine), request, caption_worker)

    assert caption_worker.submit_chunk.call_args.args[6] == transcript
    assert caption_worker.submit_chunk.call_args.args[15] == alignment_sections
    assert source_queue.get_nowait() == 5
