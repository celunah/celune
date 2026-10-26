# SPDX-License-Identifier: Apache-2.0
"""Engine-level speech entry points coordinated with the pipeline."""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..pipeline import buffering, submission
from ..typing.pipeline import SpeechStreamQueue

if TYPE_CHECKING:
    from ..celune import Celune


def say(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Submit text for ordinary buffered speech generation."""
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
