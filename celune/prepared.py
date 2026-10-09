# SPDX-License-Identifier: Apache-2.0
"""Playback helpers for speech that has already been generated."""

from __future__ import annotations

import time
import queue
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .i18n import string, tagged_string
from .utils import format_error_message
from .constants import APP_NAME, BASE_SR
from .exceptions import NotAvailableError
from .audio.dsp import is_silent_utterance
from .typing.aliases import AudioChunk, AudioChunks
from .dataclasses.pipeline import PreparedSpeechAudio, SpeechRequest, SpeechTiming

from .playback import (
    release_pipeline,
    _queue_playback_done,
    _next_playback_source_id,
    _register_playback_source,
    _flush_buffered_speech_chunks,
)

if TYPE_CHECKING:
    from .celune import Celune
    from .captions import CaptionAlignmentWorker


def _submit_prepared_caption_sections(
    engine: Celune,
    item: SpeechRequest,
    audio: AudioChunk,
    source_id: int,
    caption_worker: CaptionAlignmentWorker,
    speech_timing: SpeechTiming,
    words: tuple[str, ...],
    language: Optional[str],
) -> None:
    """Queue one alignment job per prepared tutorial passage.

    Args:
        engine: Runtime that owns the shared playback source.
        item: Prepared request containing passage boundaries and display text.
        audio: Complete prepared waveform, including inter-passage pauses.
        source_id: Playback source registered for the full waveform.
        caption_worker: Worker that aligns passages and queues audio in order.
        speech_timing: Timing data shared by all queued audio ranges.
        words: Display words for mapping each passage onto caption text.
        language: Resolved language passed to the forced aligner.
    """
    alignment_failed = [False]
    pushed_audio = [False]
    audio_cursor = 0
    for section in item.prepared_caption_sections:
        if section.audio_start_frame > audio_cursor:
            caption_worker.submit_audio(
                engine,
                source_id,
                [audio[audio_cursor : section.audio_start_frame]],
                speech_timing,
                item.stream_queue,
                pushed_audio,
            )
        section_audio = audio[section.audio_start_frame : section.audio_end_frame]
        caption_worker.submit_chunk(
            engine,
            source_id,
            [section_audio],
            speech_timing,
            item.stream_queue,
            item.display_text,
            section.text,
            words[section.word_start : section.word_end],
            language,
            section.word_start,
            section.word_end,
            tuple(section.text.split()),
            alignment_failed,
            pushed_audio,
            BASE_SR,
        )
        audio_cursor = section.audio_end_frame

    if audio_cursor < len(audio):
        caption_worker.submit_audio(
            engine,
            source_id,
            [audio[audio_cursor:]],
            speech_timing,
            item.stream_queue,
            pushed_audio,
        )


def finish_tutorial_audio_capture(
    engine: Celune,
    audio_capture_queue: queue.Queue[Union[PreparedSpeechAudio, Exception]],
    speech_len: float,
    full_audio: AudioChunks,
    prepared_sections: list[AudioChunk],
    silent_retry_count: int,
    max_silent_retries: int,
) -> bool:
    """Finish captured tutorial generation and report whether it should retry."""
    engine.reverb.reset()
    full_audio_array = np.concatenate(full_audio) if full_audio else None
    is_silent = False
    silence_tier = 0
    if full_audio_array is not None:
        is_silent, silence_tier = is_silent_utterance(full_audio_array)
    if is_silent and silence_tier == 2 and silent_retry_count < max_silent_retries:
        engine.regenerate = True
        engine.log(
            string(
                "pipeline.silent_regenerating",
                retry_count=silent_retry_count + 1,
                max_retries=max_silent_retries,
            ),
            "warning",
        )
        return True
    if is_silent and silence_tier == 2:
        engine.log(
            string(
                "pipeline.silent_regeneration_limit_reached",
                max_retries=max_silent_retries,
            ),
            "warning",
        )
    if is_silent and silence_tier == 1:
        engine.log(string("pipeline.may_be_silent"), "warning")
    engine.total_generated_speech_seconds += speech_len
    audio_capture_queue.put(PreparedSpeechAudio(tuple(prepared_sections)))
    release_pipeline(engine)
    return False


def play_prepared_speech(
    engine: Celune,
    item: SpeechRequest,
    caption_worker: Optional[CaptionAlignmentWorker],
) -> None:
    """Queue prepared speech through the shared playback and caption path."""
    source_id: Optional[int] = None
    try:
        audio = item.prepared_audio
        if audio is None or not audio.size:
            raise NotAvailableError("prepared speech has no audio")
        audio = np.asarray(audio, dtype=np.float32)
        source_id = _next_playback_source_id(engine)
        caption_enabled = (
            engine.config.get("captions") is True and caption_worker is not None
        )
        words = tuple(item.display_text.split())
        _register_playback_source(
            engine,
            source_id,
            kind="speech",
            async_caption_audio=caption_enabled,
            caption_word_total=len(words) if caption_enabled else 0,
        )
        engine.log(f"[GEN] {item.display_text}")
        speech_timing = SpeechTiming(start_time=time.monotonic())
        if caption_enabled and caption_worker is not None:
            language_resolver = getattr(
                engine.backend,
                "resolve_generation_language",
                None,
            )
            language = (
                language_resolver(item.language)
                if callable(language_resolver)
                else item.language
            )
            if item.prepared_caption_sections:
                _submit_prepared_caption_sections(
                    engine,
                    item,
                    audio,
                    source_id,
                    caption_worker,
                    speech_timing,
                    words,
                    language if isinstance(language, str) else None,
                )
            else:
                caption_worker.submit_chunk(
                    engine,
                    source_id,
                    [audio],
                    speech_timing,
                    item.stream_queue,
                    item.display_text,
                    item.text,
                    words,
                    language if isinstance(language, str) else None,
                    0,
                    len(words),
                    tuple(item.text.split()),
                    [False],
                    [False],
                    BASE_SR,
                )
            caption_worker.submit_done(
                engine,
                source_id,
                item.stream_queue,
                release_pipeline_when_finished=False,
                analysis_audio=audio.copy(),
            )
            release_pipeline(engine, playback_idle=False)
        else:
            _flush_buffered_speech_chunks(
                engine,
                source_id,
                [audio],
                speech_timing,
                False,
                item.stream_queue,
            )
            _queue_playback_done(
                engine,
                source_id,
                release_pipeline_when_finished=True,
                analysis_audio=audio.copy(),
            )
        if item.playback_source_queue is not None:
            item.playback_source_queue.put(source_id)
    except Exception as error:
        if source_id is not None and caption_worker is not None:
            caption_worker.cancel_source(source_id)
        if item.playback_source_queue is not None:
            item.playback_source_queue.put(error)
        engine.log(
            format_error_message(
                tagged_string("pipeline.gen_error", "GEN ERROR"),
                error,
                engine.log_level,
            ),
            "error",
        )
        engine.cur_state = "error"
        release_pipeline(engine)
        engine.progress_callback(0, 1)
        engine.error_callback(string("pipeline.could_not_generate", app_name=APP_NAME))
