# SPDX-License-Identifier: Apache-2.0
"""Engine-level speech entry points coordinated with the pipeline."""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..i18n import string
from ..constants import BASE_SR
from ..audio.dsp import _generate_test_tone
from ..pipeline import buffering, submission
from ..pipeline.playback import close_stream
from ..typing.pipeline import SpeechStreamQueue
from ..audio.speakers import detect_speaker_channels

if TYPE_CHECKING:
    from ..celune import Celune

_AUDIO_TEST_TIMEOUT = 90.0
_SUBWOOFER_TEST_FREQUENCY = 60.0


def say(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
    maintain_status: bool = False,
) -> bool:
    """Submit text for ordinary buffered speech generation."""
    if maintain_status:
        return submission.say(engine, text, save, display_text, maintain_status=True)
    return submission.say(engine, text, save, display_text)


async def say_async(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Submit text without blocking an async caller."""
    return await submission.say_async(engine, text, save, display_text)


def say_stream(
    engine: Celune,
    text: str,
    save: bool = True,
) -> Optional[SpeechStreamQueue]:
    """Submit text and return its generated-chunk stream."""
    return submission.say_stream(engine, text, save)


async def say_stream_async(
    engine: Celune,
    text: str,
    save: bool = True,
) -> Optional[SpeechStreamQueue]:
    """Submit streaming text without blocking an async caller."""
    return await submission.say_stream_async(engine, text, save)


def submit_audio(
    engine: Celune,
    audio: np.ndarray,
    sample_rate: int,
    label: str = "audio input",
    f0_condition: Optional[bool] = None,
    log_playback: bool = True,
    reset_ready_announcement: bool = True,
) -> bool:
    """Submit captured or uploaded audio to the active input mode."""
    return submission.submit_audio(
        engine,
        audio,
        sample_rate,
        label,
        f0_condition,
        log_playback,
        reset_ready_announcement,
    )


def release_pipeline(engine: Celune, playback_idle: bool = True) -> None:
    """Release the shared audio pipeline after a request finishes."""
    return buffering.release_pipeline(engine, playback_idle)


def stop_live_audio(engine: Celune) -> None:
    """Reset state held by the active live voice-conversion backend."""
    return submission.stop_live_audio_input(engine)


def play(
    engine: Celune,
    sound_path: str,
    keep: bool = False,
    volume: float = 1.0,
    on_started: Optional[Callable[[], None]] = None,
) -> bool:
    """Queue a sound file through Celune's speech playback pipeline."""
    if (
        engine.test_finished
        or engine.backend_mode == "agent_test"
        or engine.exit_requested
    ):
        return False
    return submission.play(
        engine,
        sound_path,
        keep=keep,
        volume=volume,
        on_started=on_started,
    )


def play_audio(
    engine: Celune,
    audio: np.ndarray,
    sample_rate: int,
    label: str = "uploaded SFX",
    keep: bool = False,
) -> bool:
    """Queue decoded sound through Celune's speech playback pipeline."""
    if (
        engine.test_finished
        or engine.backend_mode == "agent_test"
        or engine.exit_requested
    ):
        return False
    return submission.queue_sfx_audio(
        engine,
        audio,
        sample_rate,
        label,
        keep=keep,
    )


def test_audio(engine: Celune) -> bool:
    """Speak or play a test tone through each configured speaker channel.

    Returns:
        bool: Whether every detected speaker channel completed its test.

    Raises:
        ValueError: A configured output device name matches multiple devices.
    """
    if engine.input_mode != "text_to_speech":
        return False
    if not engine.wait_until_idle(timeout=_AUDIO_TEST_TIMEOUT, wait_for_speech=True):
        return False

    speakers = detect_speaker_channels(engine.config)
    if not speakers:
        return False

    close_stream(engine, abort=True)
    try:
        for channel_index, speaker in enumerate(speakers):
            engine._audio_test_route = (len(speakers), channel_index)
            if speaker.position == "lfe":
                audio = _generate_test_tone(
                    _SUBWOOFER_TEST_FREQUENCY,
                    duration=1.5,
                    sample_rate=BASE_SR,
                )
                queued = engine.play_audio(
                    audio,
                    BASE_SR,
                    label=string("commands.audio_test_subwoofer"),
                )
            else:
                queued = engine.say(speaker.name, save=False)
            if not queued or not engine.wait_until_idle(
                timeout=_AUDIO_TEST_TIMEOUT,
                wait_for_speech=True,
            ):
                return False
        return True
    finally:
        engine._audio_test_route = None
        close_stream(engine, abort=True)
