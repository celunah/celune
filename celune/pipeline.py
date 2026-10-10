# SPDX-License-Identifier: Apache-2.0
"""Speech pipeline helpers."""

from __future__ import annotations

import os
import re
import time
import queue
import ctypes
import random
import asyncio
import pathlib
import datetime
import threading
import contextlib
import collections.abc
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, cast

import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import pyrubberband as rb

from .paths import (
    outputs_dir,
)
from .utils import (
    discard,
    format_number,
    format_error_message,
)
from .metadata import (
    _write_celune_flac,
    _write_flac_metadata,
    _flac_metadata_blocks,
    _celune_metadata_payload,
    _parse_vorbis_comment_block,
)
from .playback import (
    acquire_pipeline,
    release_pipeline,
    _apply_source_gain,
    _youtube_sfx_title,
    _pipeline_cpu_config,
    _queue_playback_done,
    _download_youtube_sfx,
    _playback_source_meta,
    _queue_playback_chunk,
    _dequeue_playback_item,
    current_playback_status,
    _next_playback_source_id,
    _playback_caption_states,
    _playback_source_statuses,
    _register_playback_source,
    _update_playback_progress,
    _caption_chunk_word_ranges,
    _register_overlay_playback,
    _set_playback_source_status,
    _remember_smart_buffer_speed,
    _smart_buffer_target_seconds,
    _clear_playback_source_status,
    _flush_buffered_speech_chunks,
    _notify_speech_playback_finished,
    _record_caption_playback_segment,
)
from .audio.dsp import (
    soften,
    to_48khz,
    error_signal,
    working_signal,
    sleeping_signal,
    readiness_signal,
    is_silent_utterance,
)
from .constants import (
    BASE_SR,
    APP_NAME,
    APP_SLUG,
)
from .pipelinecore import (
    _monotonic_time,
    _PIPELINE_CPU_YIELD_SECONDS,
    _PLAYBACK_BUFFER_STARTUP_GRACE_SECONDS,
)
from .dataclasses.pipeline import (
    SpeechTiming,
    PlaybackChunk,
    SpeechRequest,
    PlaybackSourceDone,
)
from .i18n import string, tagged_string
from .config import resolve_audio_device
from .typing.common import JSONSerializable
from .binding import install_class_functions
from .analysis import _schedule_voice_analysis
from .typing.pipeline import SpeechStreamQueue
from .contention import _PlaybackContentionMonitor
from .typing.aliases import AudioChunk, AudioChunks
from .exceptions import BackendError, NotAvailableError
from .vram import is_cuda_out_of_memory, release_cuda_after_oom
from .conversation import _think_persona, _effective_voice_prompt
from .threads import run_in_daemon_thread as _run_in_daemon_thread
from .prepared import finish_tutorial_audio_capture, play_prepared_speech

if TYPE_CHECKING:
    from .celune import Celune
    from .speech import (
        say,
        play,
        close,
        say_async,
        queue_speech,
        queue_sfx_audio,
        handle_audio_input,
        queue_speech_async,
        convert_audio_input,
        stop_live_audio_input,
        prepare_playback_audio,
        deliver_persona_response,
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from .captions import CaptionAlignmentWorker
    from .playback import (
        _config_text,
        _config_lines,
        _notify_component_busy,
        acquire_pipeline_result,
    )
    from .conversation import (
        think,
        _extract_persona_text,
        _persona_memory_store,
        build_persona_context,
        build_persona_request,
        build_persona_messages,
        _persona_manifest_files,
        build_persona_character_card,
        build_agent_classification_request,
    )

    _PIPELINE_TYPE_EXPORTS = (
        _effective_voice_prompt,
        _extract_persona_text,
        _persona_manifest_files,
        _persona_memory_store,
        build_agent_classification_request,
        build_persona_character_card,
        build_persona_context,
        build_persona_messages,
        build_persona_request,
        think,
        _apply_source_gain,
        _clear_playback_source_status,
        _config_lines,
        _config_text,
        _dequeue_playback_item,
        _download_youtube_sfx,
        _flush_buffered_speech_chunks,
        _next_playback_source_id,
        _notify_component_busy,
        _notify_speech_playback_finished,
        _pipeline_cpu_config,
        _playback_source_meta,
        _playback_source_statuses,
        _queue_playback_chunk,
        _queue_playback_done,
        _register_overlay_playback,
        _register_playback_source,
        _remember_smart_buffer_speed,
        _set_playback_source_status,
        _smart_buffer_target_seconds,
        _update_playback_progress,
        _youtube_sfx_title,
        acquire_pipeline,
        acquire_pipeline_result,
        current_playback_status,
        release_pipeline,
        close,
        convert_audio_input,
        deliver_persona_response,
        finish_streaming_sfx_audio,
        handle_audio_input,
        play,
        prepare_playback_audio,
        queue_sfx_audio,
        queue_speech,
        queue_speech_async,
        queue_streaming_sfx_audio,
        say,
        say_async,
        stop_live_audio_input,
        SpeechStreamQueue,
    )


_SHORT_INPUT_VOCODER_ERROR = (
    "Calculated padded input size per channel: (6). Kernel size: (7)."
)
_EMPTY_INPUT_VOCODER_ERROR = (
    "min(): Expected reduction dim to be specified for input.numel() == 0"
)


def _is_short_input_generation_error(error: BaseException) -> bool:
    """Return whether a backend rejected the known too-short input shape."""
    message = str(error)
    return (
        _SHORT_INPUT_VOCODER_ERROR in message
        or _EMPTY_INPUT_VOCODER_ERROR in message
        or "LuxTTS produced no acoustic frames" in message
    )


def _stop_pipeline_jobs(self: Celune) -> None:
    """Stop startup pipeline workers after an initialization failure."""
    with self.queue_lock:
        self.text_queue.put(self.sentinel)
        if self._is_voice_conversion_mode():
            self.audio_queue.put(self.sentinel)

    if self._playback_thread is not None:
        self._playback_thread.join()


@dataclass(frozen=True)
class _PlaybackWriteItem:
    """Describe one mixed block waiting for the persistent output writer."""

    audio: AudioChunk
    source_ids: tuple[int, ...]
    duration_seconds: float
    submitted_at: float


def _prioritize_playback_thread() -> None:
    """Raise the output writer above normal priority on supported Windows hosts."""
    if os.name != "nt":
        return

    try:
        windll = getattr(ctypes, "WinDLL", None)
        if not callable(windll):
            return
        windll = cast(collections.abc.Callable[..., ctypes.CDLL], windll)
        win_dll = windll("kernel32", use_last_error=True)
        current_thread = win_dll.GetCurrentThread()
        if not win_dll.SetThreadPriority(current_thread, 1):
            return
    except (AttributeError, OSError, TypeError):
        return


class _PlaybackWriter:
    """Write mixed audio on one persistent thread with reserve accounting."""

    def __init__(self, engine: Celune, monitor: _PlaybackContentionMonitor) -> None:
        self._engine = engine
        self._monitor = monitor
        self._queue: queue.Queue[Optional[_PlaybackWriteItem]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending_frames = 0
        self._pending_sources: dict[int, int] = {}
        self._error: Optional[BaseException] = None
        self._last_write_finished_at: Optional[float] = None
        self._next_playback_at: Optional[float] = None

    def start(self) -> None:
        """Start the persistent writer when it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        with self._lock:
            if self._error is not None:
                raise self._error
        self._thread = threading.Thread(
            target=self._run,
            name="CelunePlaybackWriter",
            daemon=True,
        )
        self._thread.start()

    def _decrement_pending(self, item: _PlaybackWriteItem) -> None:
        """Remove one completed or discarded item from reserve accounting."""
        with self._lock:
            self._pending_frames = max(
                0,
                self._pending_frames - len(item.audio),
            )
            for source_id in item.source_ids:
                count = self._pending_sources.get(source_id, 0) - 1
                if count > 0:
                    self._pending_sources[source_id] = count
                else:
                    self._pending_sources.pop(source_id, None)

    def _discard_pending(self) -> None:
        """Discard queued blocks after a forced stop or writer failure."""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                self._decrement_pending(item)
            self._queue.task_done()

    def _record_played_frames(self, item: _PlaybackWriteItem) -> None:
        """Advance source positions after one mixed block reaches the output."""
        source_meta = _playback_source_meta(self._engine)
        for source_id in item.source_ids:
            metadata = source_meta.get(source_id)
            if isinstance(metadata, dict):
                metadata["played_frames"] = float(
                    metadata.get("played_frames", 0.0)
                ) + float(len(item.audio))
        _update_playback_progress(self._engine)

    def _pace_output(self, item: _PlaybackWriteItem, started_at: float) -> None:
        """Keep accepted output frames from advancing faster than real time."""
        caption_states = _playback_caption_states(self._engine)
        if not any(source_id in caption_states for source_id in item.source_ids):
            return

        playback_start = max(
            started_at,
            self._next_playback_at or started_at,
        )
        self._next_playback_at = playback_start + item.duration_seconds
        wait_seconds = self._next_playback_at - time.monotonic()
        if wait_seconds > 0.0:
            time.sleep(wait_seconds)

    def _run(self) -> None:
        """Consume the output queue until a stop marker is received."""
        _prioritize_playback_thread()
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return

            started_at = time.monotonic()
            writer_wait = max(0.0, started_at - item.submitted_at)
            if self._last_write_finished_at is None:
                writer_gap = 0.0
            else:
                writer_gap = max(0.0, started_at - self._last_write_finished_at)
            self._engine.playback_writer_wait_seconds = writer_wait
            self._engine.playback_writer_gap_seconds = writer_gap
            underflowed = False
            failed: Optional[BaseException] = None
            write_started_at = time.monotonic()
            write_finished_at = write_started_at
            try:
                underflowed = bool(_write_playback_block(self._engine, item.audio))
                write_finished_at = time.monotonic()
                self._pace_output(item, started_at)
                self._record_played_frames(item)
            except BaseException as error:  # pylint: disable=broad-exception-caught
                failed = error
                with self._lock:
                    self._error = error
            finally:
                finished_at = time.monotonic()
                self._decrement_pending(item)
                self._monitor.observe_write(
                    finished_at - started_at,
                    item.duration_seconds,
                    underflowed,
                )
                self._engine.playback_writer_write_seconds = max(
                    0.0,
                    write_finished_at - write_started_at,
                )
                self._last_write_finished_at = finished_at
                self._queue.task_done()

            if failed is not None:
                self._discard_pending()
                return

    def submit(self, audio: AudioChunk, source_ids: tuple[int, ...]) -> None:
        """Queue one mixed block and account for its playback reserve."""
        submitted_at = time.monotonic()
        with self._lock:
            if self._error is not None:
                raise self._error
            duration_seconds = len(audio) / BASE_SR
            self._pending_frames += len(audio)
            for source_id in source_ids:
                self._pending_sources[source_id] = (
                    self._pending_sources.get(source_id, 0) + 1
                )
        self._queue.put(
            _PlaybackWriteItem(
                audio=np.asarray(audio, dtype=np.float32),
                source_ids=source_ids,
                duration_seconds=duration_seconds,
                submitted_at=submitted_at,
            )
        )

    def wait_empty(self) -> None:
        """Wait until all submitted audio has reached the output stream."""
        self._queue.join()

    def stop(self, clear: bool = False) -> None:
        """Stop the writer, optionally discarding queued audio first."""
        thread = self._thread
        if thread is None:
            return
        if clear:
            self._discard_pending()
            with self._lock:
                self._error = None
        self._queue.put(None)
        thread.join(timeout=2.0)
        if not thread.is_alive():
            self._thread = None

    def has_pending_source(self, source_id: int) -> bool:
        """Return whether output still contains audio for one source."""
        with self._lock:
            return self._pending_sources.get(source_id, 0) > 0

    @property
    def pending_seconds(self) -> float:
        """Return the seconds queued for the output writer."""
        with self._lock:
            return self._pending_frames / BASE_SR

    @property
    def error(self) -> Optional[BaseException]:
        """Return the first asynchronous output error, if any."""
        with self._lock:
            return self._error


class _PlaybackInputReader:
    """Move playback queue waits onto one persistent daemon thread."""

    def __init__(self, engine: Celune) -> None:
        self._engine = engine
        source_maxsize = getattr(engine.audio_queue, "maxsize", 0)
        self._queue: queue.Queue[object] = queue.Queue(
            maxsize=source_maxsize if source_maxsize > 0 else 8
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event: Optional[asyncio.Event] = None

    def start(self) -> None:
        """Start the one queue reader used for the lifetime of playback."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="CelunePlaybackInput",
            daemon=True,
        )
        self._thread.start()

    def _notify(self) -> None:
        """Wake the playback coroutine after a queue item is forwarded."""
        event = self._event
        if event is not None:
            event.set()

    def _run(self) -> None:
        """Forward source items without creating a thread per polling cycle."""
        while not self._stop.is_set():
            try:
                item = self._engine.audio_queue.get(True, 0.1)
            except queue.Empty:
                continue

            while not self._stop.is_set():
                try:
                    self._queue.put(item, True, 0.1)
                    break
                except queue.Full:
                    continue
            else:
                return

            loop = self._loop
            if loop is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._notify)

            if item is self._engine.sentinel:
                return

    async def get(self) -> object:
        """Wait for one forwarded item without blocking the event loop."""
        event = self._event
        if event is None:
            raise RuntimeError("playback input reader has not started")

        while True:
            try:
                return self._queue.get_nowait()
            except queue.Empty:
                event.clear()
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    await event.wait()

    def get_nowait(self) -> object:
        """Remove one already-forwarded item for priority-aware draining."""
        return self._queue.get_nowait()

    @property
    def queue(self) -> queue.Queue[object]:
        """Return the forwarded queue for priority-aware playback draining."""
        return self._queue

    def empty(self) -> bool:
        """Return whether the forwarded playback queue has no pending item."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the number of items waiting in the forwarded queue."""
        return self._queue.qsize()

    def clear(self) -> None:
        """Discard forwarded items after a generation force-stop."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        """Stop the persistent input reader and release its thread."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            self._thread = None


_AGENT_CLASSIFICATION_INSTRUCTIONS = (
    "This is an internal routing request, not a character response. The routing "
    "output rules below take priority over any character-response, speech, or "
    "conversation-style guidance in the Persona context. Classify the latest user "
    "input for conversation-first routing. The latest user message is the primary "
    "intent signal; use earlier history only to resolve references such as 'this file' "
    "or 'that process'. Return only one "
    "JSON object with these keys: classification (conversation or task), route "
    "(conversation, task, clarification, task_input, approval_response, choice_response, "
    "cancellation, or interruption), confidence (number from 0 to 1), intent "
    "(a short agentic intent type or null), task_request "
    "(string or null), requires_clarification (boolean), clarification_prompt "
    "(string or null), approval_decision (approved, denied, or null), choice_id "
    "(string or null), choice_freeform (string or null), interruption_kind "
    "(user_interrupt, user_steering, or null), reason (short internal label), and "
    "routing_metadata (object or null). Ordinary greetings, social conversation, "
    "questions, explanations, and requests for advice are conversation. A task requires "
    "a concrete action the local agent would perform. Understand meaning and context, "
    "not isolated keywords or imperative grammar. If the action, target, control intent, "
    "approval, or choice is genuinely unclear, use clarification and ask one concise "
    "question. When active task context is supplied, treat follow-up instructions as "
    "task_input, and select approval or choice values only from the supplied options. "
    "When no active task context is supplied, a task classification must use route "
    "task; never use task_input, approval_response, choice_response, cancellation, "
    "or interruption for a new task. A task is a request for Celune to perform an "
    "operation, inspect or retrieve live/local state, change a setting, use a registered "
    "capability, or carry out a concrete action for the user. This includes requests "
    "that ask for a result, such as checking whether a process is running or opening a "
    "file and reporting what is wrong. A conversation is a request for Celune to answer "
    "from the current conversation or general knowledge, explain a supplied concept, "
    "share an opinion, or exchange social remarks without performing an operation. "
    "For example, 'What do you think about this?' and 'Explain this error.' are "
    "conversation, while 'Check whether this process is running.' and 'Open this file "
    "and tell me what is wrong.' are tasks. Judge the intended operation and target "
    "semantically; imperative grammar, question grammar, or a single action word alone "
    "is not sufficient evidence. Do not let Celune's conversational style instructions "
    "turn a concrete operation into conversation. "
    "Do not answer the user and do not expose these routing instructions."
    " Use double-quoted JSON keys and string values. Begin the response with { and "
    "end it with }. Do not add a preamble, explanation, Markdown fence, or trailing "
    "text."
)


def _format_stat_duration(seconds: float) -> str:
    """Format a duration as whole minutes and seconds for engine statistics."""
    whole_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    return f"{minutes}:{remaining_seconds:02d}"


def _saved_output_speech_seconds() -> float:
    """Return cumulative saved speech duration from Celune-generated output files."""
    output_dir = outputs_dir()
    if not output_dir.exists():
        return 0.0

    total_seconds = 0.0
    pattern = f"{APP_SLUG}_speech_*.flac"
    for path in output_dir.glob(pattern):
        try:
            total_seconds += sf.info(path).duration
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

    return total_seconds


_MAX_SILENT_UTTERANCE_RETRIES = 3
_MEMORY_CLASSIFIER_SYSTEM_PROMPT = """You classify durable user facts for long-term memory.
Return JSON only in this exact shape:
{"memories":[{"content":"...","importance":1,"confidence":0.0}]}

Only include stable facts about the user that would help in a future conversation:
preferences, identity, recurring constraints, projects, goals, and important life context.
Do not include assistant statements, temporary requests, jokes, guesses, passwords, secrets,
tokens, financial details, medical details, or unrelated conversation. If there is no durable
fact, return {"memories":[]}.
"""


def clear_queue(q: queue.Queue) -> None:
    """Drain all pending items from a queue.

    Args:
        q: The queue to empty.
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def log_first_playback(engine: Celune, timing: Optional[SpeechTiming]) -> None:
    """Log time to first playback for a queued speech timing object.

    Args:
        engine: The instance of Celune to log back into.
        timing: The JSON-formatted timing data.
    """
    start_time = getattr(timing, "start_time", None)
    if not isinstance(start_time, float):
        return

    mark_first_playback = getattr(timing, "mark_first_playback", None)
    if callable(mark_first_playback):
        mark_first_playback()
    elif getattr(timing, "first_playback_time", None) is None:
        return

    ttfp_seconds = getattr(timing, "ttfp_seconds", None)
    if callable(ttfp_seconds):
        elapsed = ttfp_seconds()
        if not isinstance(elapsed, float):
            return
    else:
        elapsed = _monotonic_time() - start_time

    engine.log(f"TTFP {format_number(elapsed, 2)}s")


def close_stream(engine: Celune, abort: bool = False) -> None:
    """Close the current audio stream if one exists.

    Args:
        engine: The Celune engine that owns the audio stream.
        abort: Whether to abort immediately instead of stopping gracefully.
    """
    stream_lock = getattr(engine, "stream_lock", None)
    if stream_lock is None:
        _close_stream_unlocked(engine, abort)
        return

    with stream_lock:
        _close_stream_unlocked(engine, abort)


def _close_stream_unlocked(engine: Celune, abort: bool = False) -> None:
    """Close the audio stream while the stream lifecycle lock is held."""
    if engine.stream is None:
        return

    with contextlib.suppress(Exception):
        if abort:
            engine.stream.abort()
        else:
            engine.stream.stop()

    with contextlib.suppress(Exception):
        engine.stream.close()

    engine._stream = None
    engine._current_sr = None


def _write_playback_block(engine: Celune, audio: AudioChunk) -> Optional[bool]:
    """Write one playback block while serialized against stream teardown."""
    stream_lock = getattr(engine, "stream_lock", None)
    if stream_lock is None:
        return _write_playback_block_unlocked(engine, audio)

    with stream_lock:
        return _write_playback_block_unlocked(engine, audio)


def _write_playback_block_unlocked(
    engine: Celune,
    audio: AudioChunk,
) -> Optional[bool]:
    """Write one playback block while the stream lifecycle lock is held."""
    stream = engine.stream
    if stream is None:
        raise NotAvailableError("audio stream is not available")
    return stream.write(audio)


def _reset_glow_audio_reactivity(engine: Celune) -> None:
    """Clear any pending audio-reactive glow state after abrupt playback stops."""
    reset_audio_reactivity = getattr(engine.glow, "reset_audio_reactivity", None)
    if callable(reset_audio_reactivity):
        reset_audio_reactivity()


def force_stop_speech(engine: Celune) -> bool:
    """Forcefully stop Celune from speaking or playing audio.

    Args:
        engine: The Celune engine whose queues and playback should be interrupted.

    Returns:
        bool: ``True`` when active speech or playback was stopped, otherwise ``False``.
    """
    with engine.say_lock:
        if engine.utterance_force_stop.is_set():
            return False
        is_active = (
            engine.locked
            or engine.cur_state in {"generating", "speaking"}
            or not engine.playback_done.is_set()
            or bool(_playback_source_meta(engine))
            or bool(_playback_source_statuses(engine))
        )

    if not is_active:
        engine.utterance_force_stop.clear()
        return False

    engine.log(string("pipeline.forcefully_stopping_speech"))
    _invalidate_speech_work(engine)

    return True


def _invalidate_speech_work(engine: Celune) -> None:
    """Invalidate queued and in-flight speech generations immediately."""
    engine.utterance_force_stop.set()

    cancel_active_request = getattr(engine.backend, "cancel_active_request", None)
    if callable(cancel_active_request):
        with contextlib.suppress(Exception):
            cancel_active_request(wait_for_ack=False)

    with engine.queue_lock:
        engine._speech_generation = getattr(engine, "_speech_generation", 0) + 1
        engine._playback_generation = getattr(engine, "_playback_generation", 0) + 1
        clear_queue(engine.text_queue)
        clear_queue(engine.persona_queue)
        clear_queue(engine.audio_queue)
        engine.kept_sfx_audio = None
        engine.audio_queue.put(engine.force_stop_marker)


def split_text(engine: Celune, text: str) -> list[str]:
    """Adaptively split text into chunks. Short text is unaffected, while long text is chunked effectively.

    Args:
        engine: The Celune engine to report output back to.
        text: The input text to split.

    Returns:
        list[str]: The generated text chunks.
    """
    text = text.strip()
    if not text:
        return []

    chunk_length = 150
    max_length = 400

    # detect sentences
    unit_checker = re.compile(r"\S.*?(?:[.!?]+[\"')\]]*(?=\s+|$)|$)", re.DOTALL)

    # detected quoted text with a boundary
    quote_checker = re.compile(r'"[^"]*[.!?]"')

    if len(text) <= max_length and not quote_checker.search(text):
        # input is short, return as is
        return [text]

    def split_long_unit(value: str) -> list[str]:
        pieces = [piece.strip() for piece in value.splitlines() if piece.strip()]
        if not pieces:
            pieces = value.split()

        unit_chunks = []
        unit_current = ""

        for piece in pieces:
            if len(piece) > max_length:
                if unit_current:
                    unit_chunks.append(unit_current)
                    unit_current = ""
                unit_chunks.extend(split_words(piece))
                continue

            if (unit_current and len(unit_current) + 1 + len(piece) > max_length) or (
                unit_current and len(unit_current) >= chunk_length
            ):
                unit_chunks.append(unit_current)
                unit_current = piece
            elif unit_current:
                unit_current = f"{unit_current} {piece}"
            else:
                unit_current = piece

        if unit_current:
            unit_chunks.append(unit_current)

        return unit_chunks

    def split_words(value: str) -> list[str]:
        word_chunks = []
        word_current = ""

        for word in value.split():
            if word_current and len(word_current) + 1 + len(word) > max_length:
                word_chunks.append(word_current)
                word_current = word
            elif word_current:
                word_current = f"{word_current} {word}"
            else:
                word_current = word

        if word_current:
            word_chunks.append(word_current)

        return word_chunks

    def split_sentences(value: str) -> list[str]:
        units = []
        for rmatch in unit_checker.finditer(value):
            unit = rmatch.group(0).strip()
            if len(unit) > max_length:
                units.extend(split_long_unit(unit))
            elif unit:
                units.append(unit)
        return units

    def split_units(value: str) -> list[str]:
        units = []
        start = 0

        for qmatch in quote_checker.finditer(value):
            units.extend(split_sentences(value[start : qmatch.start()]))
            units.append(qmatch.group(0).strip())
            start = qmatch.end()

        units.extend(split_sentences(value[start:]))
        return [unit for unit in units if unit]

    all_units = split_units(text)
    if not all_units:
        return []

    chunks = []
    current = ""

    for u in all_units:
        if quote_checker.fullmatch(u):
            if current:
                chunks.append(current)
                current = ""
            chunks.append(u)
            continue

        if (current and len(current) + 1 + len(u) > max_length) or (
            current and len(current) >= chunk_length
        ):
            chunks.append(current)
            current = u
        elif current:
            current = f"{current} {u}"
        else:
            current = u

    if current:
        chunks.append(current)

    engine.log(f"Chunks: {len(chunks)}")
    return chunks


def play_signal(engine: Celune, signal_type: str) -> bool:
    """Queue a readiness signal to be played.

    Args:
        engine: The instance of Celune to do this with.
        signal_type: The signal type to be played.

    Returns:
        bool: Whether the readiness signal was processed successfully.

    Raises:
        ValueError: An invalid signal name was requested.
    """
    if signal_type == "readiness":
        signal = readiness_signal()
    elif signal_type == "working":
        signal = working_signal()
    elif signal_type == "sleeping":
        signal = sleeping_signal()
    elif signal_type == "error":
        signal = error_signal()
    else:
        raise ValueError("no such signal")

    # if a pipeline lock is already held or was not initialized this can cause
    # Celune to become deadlocked, or it won't have an effect, so please call
    # Celune._try_play_signal() instead of calling this method directly
    if signal_type == "readiness":
        source_id = _next_playback_source_id(engine)
        _register_overlay_playback(engine)
        _register_playback_source(engine, source_id, kind="sfx")
        _queue_playback_chunk(engine, source_id, signal, BASE_SR)
        _queue_playback_done(
            engine,
            source_id,
            notify_idle_when_finished=True,
        )
        return True

    if acquire_pipeline(engine, f"play {signal_type} signal"):
        release_to_idle = False
        if engine.cur_state != "error":
            if signal_type == "sleeping":
                engine.cur_state = "sleeping"
            elif signal_type != "working":
                engine.cur_state = "speaking"
        source_id = _next_playback_source_id(engine)
        _register_playback_source(engine, source_id, kind="sfx")
        _queue_playback_chunk(engine, source_id, signal, BASE_SR)
        _queue_playback_done(
            engine,
            source_id,
            release_pipeline_when_finished=release_to_idle,
            notify_idle_when_finished=signal_type == "readiness",
        )
        release_pipeline(engine, playback_idle=False)
        return True
    return False


def _recover_quantized_tts(engine: Celune, language: Optional[str]) -> None:
    """Reload a failed quantized TTS runtime in BF16.

    Args:
        engine: Celune runtime that owns the active TTS backend.
        language: Language requested by the failed generation.

    Raises:
        BackendError: The active voice cannot be resolved or BF16 recovery fails.
    """
    backend = engine.backend
    try:
        backend.disable_runtime_quantization()
        backend.unload_model()
        active_voice = engine.current_voice or backend.default_voice
        if active_voice is None:
            raise BackendError(
                "cannot recover a quantized model without an active voice",
                error_code="tts_quantization_recovery_failed",
            )
        model_id = backend.model_id_for_voice(active_voice)
        model = backend.load_model(model_id, lang=language)
        backend.model = model
        engine.model = model
        engine.model_name = model_id
    except Exception as error:
        engine.fatal()
        if isinstance(error, BackendError):
            raise
        raise BackendError(
            "quantized TTS recovery failed",
            error_code="tts_quantization_recovery_failed",
            error_type=type(error).__name__,
        ) from error


def _process_generation_request(
    engine: Celune,
    item: SpeechRequest,
    caption_worker: Optional[CaptionAlignmentWorker],
) -> None:
    """Process one queued speech request on a blocking worker thread."""
    if item.prepared_audio is not None:
        play_prepared_speech(engine, item, caption_worker)
        return

    text = item.text
    display_text = item.display_text
    request_language = item.language
    save_output = item.save
    stream_queue = item.stream_queue
    capture_tutorial_audio = item.audio_capture_queue is not None
    kept_sfx_audio = engine.kept_sfx_audio
    engine.kept_sfx_audio = None

    if engine.exit_requested:
        if stream_queue is not None:
            stream_queue.put(NotAvailableError("stream queue interrupted"))
            stream_queue.put(None)
        if item.audio_capture_queue is not None:
            item.audio_capture_queue.put(
                NotAvailableError("tutorial speech preparation was interrupted")
            )
        release_pipeline(engine)
        return

    while True:
        source_id = _next_playback_source_id(engine)
        prepared_sections: list[AudioChunk] = []
        try:
            engine.model_ready.wait()

            if not engine.loaded and not engine.backend.is_fake:
                engine.log(string("ui.core_engine_not_loaded"), "warning")
                engine.locked = False
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("model is not ready")
                    )
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("model is not ready"))
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            start_time = _monotonic_time()
            engine.log(f"[GEN] {display_text}")
            engine.log(
                "[GEN] start "
                f"generation={item.generation} text_chars={len(text)} "
                f"backend={getattr(engine.backend, 'name', type(engine.backend).__name__)} "
                f"model={getattr(engine, 'model_name', '')} language={request_language}",
                loglevel="debug",
            )
            speech_len = 0.0
            buffered_speech_len = 0.0
            smart_buffer_target_seconds = _smart_buffer_target_seconds(
                engine,
                0.0,
                0.0,
            )
            engine.smart_buffer_target_seconds = smart_buffer_target_seconds
            speech_timing = SpeechTiming(start_time)
            pushed_audio = False
            stream_frame_count = 0
            accepted_stream_frame_count = 0
            caption_language: Optional[str] = None
            caption_alignment_failed: list[bool] = [False]
            pushed_audio_state: list[bool] = [False]

            # these generation parameters are fixed and do not change
            # this only applies to Qwen3-TTS, other backends discard this
            generation_params: collections.abc.Mapping[str, JSONSerializable] = {
                "temperature": 0.15,
                "top_k": 20,
                "top_p": 0.7,
                "repetition_penalty": 1.1,
            }

            chunks = split_text(engine, text)
            if capture_tutorial_audio:
                chunks = list(item.synthesis_sections)
            if not chunks:
                engine.progress_callback(0, 1)
                engine.error_callback(string("pipeline.nothing_to_say"))
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech has no sections")
                    )
                release_pipeline(engine)
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("nothing to say"))
                    stream_queue.put(None)
                break

            buffer: AudioChunks = []
            full_audio: AudioChunks = []
            generated_text_parts: list[str] = []
            request_generation = item.generation
            caption_word_total = len(display_text.split())
            caption_alignment_enabled = (
                not capture_tutorial_audio
                and engine.config.get("captions") is True
                and caption_worker is not None
            )
            if not capture_tutorial_audio:
                _register_playback_source(
                    engine,
                    source_id,
                    kind="speech",
                    async_caption_audio=caption_alignment_enabled,
                    caption_word_total=(
                        caption_word_total if caption_alignment_enabled else 0
                    ),
                )
            caption_word_ranges = _caption_chunk_word_ranges(display_text, chunks)

            for chunk_index, chunk_text in enumerate(chunks):
                if engine.exit_requested:
                    break

                if engine.utterance_force_stop.is_set():
                    break

                chunk_word_start, chunk_word_end = caption_word_ranges[chunk_index]
                chunk_start_frames = 0
                if source_id is not None and source_id in _playback_caption_states(
                    engine
                ):
                    with engine.queue_lock:
                        source_meta = _playback_source_meta(engine).get(source_id)
                        if isinstance(source_meta, dict):
                            chunk_start_frames = int(
                                float(source_meta.get("total_frames", 0.0))
                            )

                if item.normalize:
                    engine.status_callback(string("status.normalizing"))
                    engine.progress_callback(None, None)
                    normalized = engine.normalize(chunk_text)
                    if normalized is not None:
                        if normalized == chunk_text:
                            engine.log(
                                "This input is already normalized.",
                                "warning",
                                loglevel="verbose",
                            )
                        else:
                            differences = sum(
                                x != y for x, y in zip(normalized, chunk_text)
                            ) + abs(len(normalized) - len(chunk_text))

                            if differences > max(5, int(len(chunk_text) * 0.05)):
                                chunk_text = normalized

                generated_text_parts.append(chunk_text)
                is_first_chunk = chunk_index == 0
                last_timing: Optional[dict] = None

                with engine.model_lock:
                    if engine.model is None:
                        raise NotAvailableError(
                            "cannot generate without a model reference"
                        )

                    resolve_generation_language = getattr(
                        engine.backend,
                        "resolve_generation_language",
                        None,
                    )
                    if callable(resolve_generation_language):
                        target_language = resolve_generation_language(request_language)
                    else:
                        target_language = request_language
                    caption_language = (
                        target_language if isinstance(target_language, str) else None
                    )

                    should_reload_for_language = getattr(
                        engine.backend,
                        "should_reload_for_language",
                        None,
                    )
                    if callable(should_reload_for_language) and (
                        should_reload_for_language(target_language)
                    ):
                        active_voice = (
                            engine.current_voice or engine.backend.default_voice
                        )
                        if active_voice is None:
                            raise NotAvailableError(
                                "cannot switch language without an active voice"
                            )

                        model_id = engine.backend.model_id_for_voice(active_voice)
                        engine.log(
                            f"[RELOAD] Loading {model_id} for language: {target_language}",
                            loglevel="verbose",
                        )
                        engine.backend.unload_model()
                        engine.model = engine.backend.load_model(
                            model_id,
                            lang=target_language,
                        )
                        engine.model_name = model_id

                    for (
                        audio_chunk,
                        sr,  # 24 kHz if Qwen3 or Celune Mini, 48 kHz if VoxCPM2
                        timing,
                    ) in engine.backend.generate_stream(
                        engine.model,
                        text=chunk_text,
                        language=target_language,
                        chunk_size=engine.chunk_size,
                        instruct=_effective_voice_prompt(engine),
                        voice=engine.current_voice,
                        temperature=generation_params["temperature"],
                        top_k=generation_params["top_k"],
                        top_p=generation_params["top_p"],
                        repetition_penalty=generation_params["repetition_penalty"],
                    ):
                        stream_frame_count += 1
                        if timing is not None:
                            last_timing = timing
                        if engine.exit_requested:
                            break

                        if (
                            engine.utterance_force_stop.is_set()
                            or request_generation
                            != getattr(engine, "_speech_generation", request_generation)
                        ):
                            break

                        first_chunk_time = None
                        if timing is not None:
                            raw_first_chunk_time = timing.get("first_chunk_time")
                            if isinstance(raw_first_chunk_time, float):
                                first_chunk_time = raw_first_chunk_time

                        speech_timing.mark_first_chunk(first_chunk_time)

                        if isinstance(audio_chunk, torch.Tensor):
                            audio_chunk = audio_chunk.cpu().numpy()

                        audio_chunk = to_48khz(
                            np.asarray(audio_chunk, dtype=np.float32), sr
                        )
                        has_audio = bool(audio_chunk.size and np.any(audio_chunk))
                        if (
                            stream_frame_count <= 3
                            or audio_chunk.size == 0
                            or not has_audio
                        ):
                            engine.log(
                                "[STREAM] core received "
                                f"frame={stream_frame_count} samples={audio_chunk.size} "
                                f"sample_rate={sr} nonzero={has_audio}",
                                loglevel="debug",
                            )
                        if not has_audio:
                            engine.log(
                                f"[STREAM] core discarded frame={stream_frame_count}",
                                loglevel="debug",
                            )
                            continue
                        accepted_stream_frame_count += 1

                        if engine.speed != 1.0 and engine.can_use_rubberband:
                            try:
                                audio_chunk = rb.time_stretch(
                                    audio_chunk, BASE_SR, engine.speed
                                )
                            except RuntimeError:
                                engine.log(
                                    string("pipeline.rubber_band_unavailable"),
                                    "warning",
                                )
                                engine.can_use_rubberband = False
                            else:
                                audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                        if engine.reverb.strength > 0.0:
                            audio_chunk = engine.reverb.process(audio_chunk, BASE_SR)
                            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                        if is_first_chunk:
                            audio_chunk = soften(audio_chunk, BASE_SR, end=False)
                            is_first_chunk = False

                        if (
                            engine.exit_requested
                            or engine.utterance_force_stop.is_set()
                            or request_generation
                            != getattr(engine, "_speech_generation", request_generation)
                        ):
                            break

                        buffer.append(audio_chunk)
                        full_audio.append(audio_chunk)
                        chunk_dur = len(audio_chunk) / BASE_SR
                        speech_len += chunk_dur
                        buffered_speech_len += chunk_dur
                        generation_elapsed = max(
                            _monotonic_time() - start_time,
                            1e-6,
                        )
                        smart_buffer_target_seconds = _smart_buffer_target_seconds(
                            engine,
                            speech_len,
                            generation_elapsed,
                        )
                        engine.smart_buffer_target_seconds = smart_buffer_target_seconds

                        if (
                            not capture_tutorial_audio
                            and not caption_alignment_enabled
                            and (
                                smart_buffer_target_seconds <= 0.0
                                or buffered_speech_len >= smart_buffer_target_seconds
                            )
                        ):
                            pushed_audio = _flush_buffered_speech_chunks(
                                engine,
                                source_id,
                                buffer,
                                speech_timing,
                                pushed_audio,
                                stream_queue,
                                caption_text=None,
                            )
                            buffered_speech_len = 0.0

                    if (
                        not engine.exit_requested
                        and not engine.utterance_force_stop.is_set()
                        and last_timing is not None
                        and last_timing.get("is_final")
                        and bool(last_timing.get("missing_eos"))
                    ):
                        engine.log(
                            string("pipeline.token_limit_reached"),
                            "warning",
                        )

                if capture_tutorial_audio:
                    if engine.reverb.strength > 0.0:
                        tail = engine.reverb.flush()
                        if len(tail) > 0:
                            buffer.append(np.asarray(tail, dtype=np.float32))
                            full_audio.append(np.asarray(tail, dtype=np.float32))
                        engine.reverb.reset()
                    prepared_sections.append(
                        np.concatenate(buffer).astype(np.float32, copy=False)
                        if buffer
                        else np.empty(0, dtype=np.float32)
                    )
                    buffer.clear()
                elif (
                    not engine.exit_requested
                    and not engine.utterance_force_stop.is_set()
                    and request_generation
                    == getattr(engine, "_speech_generation", request_generation)
                ):
                    word_start_frames: tuple[int, ...] = ()
                    if (
                        caption_alignment_enabled
                        and caption_worker is not None
                        and buffer
                    ):
                        caption_worker.submit_chunk(
                            engine,
                            source_id,
                            buffer,
                            speech_timing,
                            stream_queue,
                            display_text,
                            chunk_text,
                            tuple(display_text.split())[
                                chunk_word_start:chunk_word_end
                            ],
                            caption_language,
                            chunk_word_start,
                            chunk_word_end,
                            tuple(chunk_text.split()),
                            caption_alignment_failed,
                            pushed_audio_state,
                            BASE_SR,
                        )
                        buffer = []
                    else:
                        pushed_audio = _flush_buffered_speech_chunks(
                            engine,
                            source_id,
                            buffer,
                            speech_timing,
                            pushed_audio,
                            stream_queue,
                            caption_text=None,
                        )
                    buffered_speech_len = 0.0
                    if (
                        not caption_alignment_enabled or caption_worker is None
                    ) and source_id in _playback_caption_states(engine):
                        with engine.queue_lock:
                            source_meta = _playback_source_meta(engine).get(source_id)
                            chunk_end_frames = (
                                int(float(source_meta.get("total_frames", 0.0)))
                                if isinstance(source_meta, dict)
                                else chunk_start_frames
                            )
                        _record_caption_playback_segment(
                            engine,
                            source_id,
                            chunk_start_frames,
                            chunk_end_frames,
                            chunk_word_start,
                            chunk_word_end,
                            tuple(chunk_text.split()),
                            word_start_frames,
                        )

            timing_text = (
                " ".join(generated_text_parts) if generated_text_parts else text
            )
            if generated_text_parts:
                text = timing_text

            generation_time = max(_monotonic_time() - start_time, 1e-6)

            engine.log(
                "[GEN] stream complete "
                f"generation={request_generation} seconds={speech_len:.3f} "
                f"elapsed={generation_time:.3f} buffered={buffered_speech_len:.3f}",
                loglevel="debug",
            )

            engine.log(
                f"[STREAM] core totals received={stream_frame_count} "
                f"accepted={accepted_stream_frame_count}",
                loglevel="debug",
            )

            if engine.exit_requested:
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech preparation was interrupted")
                    )
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                elif stream_queue is not None:
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            if engine.utterance_force_stop.is_set() or request_generation != getattr(
                engine, "_speech_generation", request_generation
            ):
                if item.audio_capture_queue is not None:
                    item.audio_capture_queue.put(
                        NotAvailableError("tutorial speech preparation was interrupted")
                    )
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                elif stream_queue is not None:
                    stream_queue.put(None)
                engine.reverb.reset()
                break

            generation_speed = speech_len / generation_time
            engine.log(
                f"{_format_stat_duration(speech_len)}, "
                f"{format_number(generation_time, 2)}s, "
                f"{format_number(generation_speed, 2)}x"
            )
            _remember_smart_buffer_speed(engine, generation_speed)
            engine.smart_buffer_target_seconds = _smart_buffer_target_seconds(
                engine,
                speech_len,
                generation_time,
            )
            engine.log(f"TTFC {format_number(speech_timing.ttfc_ms(), 1)}ms")

            if buffer and not capture_tutorial_audio:
                if caption_alignment_enabled and caption_worker is not None:
                    caption_worker.submit_audio(
                        engine,
                        source_id,
                        buffer,
                        speech_timing,
                        stream_queue,
                        pushed_audio_state,
                    )
                else:
                    _flush_buffered_speech_chunks(
                        engine,
                        source_id,
                        buffer,
                        speech_timing,
                        pushed_audio,
                        stream_queue,
                        caption_text=None,
                    )

            engine.log("[GEN] done")

            saved_path = None
            analysis_audio = None
            if capture_tutorial_audio:
                audio_capture_queue = item.audio_capture_queue
                if audio_capture_queue is None:
                    raise NotAvailableError("tutorial capture queue is unavailable")
                if finish_tutorial_audio_capture(
                    engine,
                    audio_capture_queue,
                    speech_len,
                    full_audio,
                    prepared_sections,
                    item.silent_retry_count,
                    _MAX_SILENT_UTTERANCE_RETRIES,
                ):
                    item = replace(
                        item,
                        silent_retry_count=item.silent_retry_count + 1,
                    )
                    continue
                break

            if not engine.exit_requested:
                if engine.reverb.strength > 0.0:
                    tail = engine.reverb.flush()
                    if len(tail) > 0:
                        if caption_alignment_enabled and caption_worker is not None:
                            caption_worker.submit_audio(
                                engine,
                                source_id,
                                [tail],
                                speech_timing,
                                stream_queue,
                                pushed_audio_state,
                            )
                            full_audio.append(tail)
                        else:
                            queued_tail = _queue_playback_chunk(
                                engine,
                                source_id,
                                tail,
                                BASE_SR,
                            )
                            if queued_tail and stream_queue is not None:
                                stream_queue.put(tail.copy())
                            if queued_tail:
                                buffer.append(tail)
                                full_audio.append(tail)

                engine.reverb.reset()
                is_silent = False
                silence_tier = 0
                full_audio_array = np.concatenate(full_audio) if full_audio else None
                if full_audio_array is not None:
                    is_silent, silence_tier = is_silent_utterance(full_audio_array)

                if is_silent and silence_tier == 2:
                    if item.silent_retry_count < _MAX_SILENT_UTTERANCE_RETRIES:
                        engine.regenerate = True
                        if caption_alignment_enabled and caption_worker is not None:
                            caption_worker.submit_done(
                                engine,
                                source_id,
                                stream_queue,
                                release_pipeline_when_finished=False,
                                notify_idle_when_finished=False,
                                finish_stream=False,
                            )
                        else:
                            _queue_playback_done(
                                engine,
                                source_id,
                                release_pipeline_when_finished=False,
                                notify_idle_when_finished=False,
                            )
                        item = replace(
                            item,
                            silent_retry_count=item.silent_retry_count + 1,
                        )
                        engine.log(
                            string(
                                "pipeline.silent_regenerating",
                                retry_count=item.silent_retry_count,
                                max_retries=_MAX_SILENT_UTTERANCE_RETRIES,
                            ),
                            "warning",
                        )
                        continue
                    engine.log(
                        string(
                            "pipeline.silent_regeneration_limit_reached",
                            max_retries=_MAX_SILENT_UTTERANCE_RETRIES,
                        ),
                        "warning",
                    )
                if is_silent and silence_tier == 1:
                    engine.log(string("pipeline.may_be_silent"), "warning")

                engine.total_generated_speech_seconds += speech_len

                if save_output and full_audio_array is not None:
                    wav = full_audio_array
                    analysis_audio = wav.copy()
                    if kept_sfx_audio is not None:
                        wav = np.concatenate([*kept_sfx_audio, wav])
                    timestamp = datetime.datetime.now(datetime.UTC).strftime(
                        "%Y%m%d%H%M%S"
                    )

                    first_words = "_".join(text.split()[:3]).lower()
                    first_words = re.sub(r"[^a-zA-Z0-9_]", "", first_words)

                    output_dir = outputs_dir()
                    if not output_dir.exists():
                        engine.log(string("pipeline.outputs_path_creating"), "warning")
                        try:
                            output_dir.mkdir(parents=True)
                        except OSError as e:
                            engine.log(
                                format_error_message(
                                    string("pipeline.outputs_create_failed"),
                                    e,
                                    engine.log_level,
                                ),
                                "warning",
                            )

                    if output_dir.exists():
                        file_name = f"{APP_SLUG}_speech_{timestamp}_{first_words}.flac"
                        saved_path = str(pathlib.Path("outputs") / file_name)
                        actual_saved_path = str(output_dir / file_name)
                        sample_rate = BASE_SR
                        subtype = "PCM_24"
                        metadata = _celune_metadata_payload(
                            engine,
                            text=text,
                            display_text=display_text,
                            generation_params=generation_params,
                            sample_rate=sample_rate,
                            subtype=subtype,
                            included_kept_sfx=kept_sfx_audio is not None,
                        )
                        try:
                            _write_celune_flac(
                                engine,
                                actual_saved_path,
                                wav,
                                sample_rate,
                                subtype=subtype,
                                metadata=metadata,
                            )
                        except Exception as e:
                            engine.log(
                                format_error_message(
                                    string("pipeline.flac_save_failed"),
                                    e,
                                    engine.log_level,
                                ),
                                "warning",
                            )
                            saved_path = None

                engine.recently_saved = saved_path
                if caption_alignment_enabled and caption_worker is not None:
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        saved_path=saved_path,
                        analysis_audio=analysis_audio,
                    )
                    release_pipeline(engine, playback_idle=False)
                else:
                    _queue_playback_done(
                        engine,
                        source_id,
                        release_pipeline_when_finished=True,
                        saved_path=saved_path,
                        analysis_audio=analysis_audio,
                    )
                    if stream_queue is not None:
                        stream_queue.put(None)
            break
        except Exception as original_error:
            if engine.exit_requested:
                if caption_worker is not None and not capture_tutorial_audio:
                    caption_worker.cancel_source(source_id)
                    caption_worker.submit_done(
                        engine,
                        source_id,
                        stream_queue,
                        release_pipeline_when_finished=False,
                        notify_idle_when_finished=False,
                    )
                release_pipeline(engine)
                break

            if caption_worker is not None and not capture_tutorial_audio:
                caption_worker.cancel_source(source_id)
                caption_worker.submit_done(
                    engine,
                    source_id,
                    stream_queue,
                    release_pipeline_when_finished=False,
                    notify_idle_when_finished=False,
                    finish_stream=False,
                )

            cuda_oom = is_cuda_out_of_memory(original_error)
            recovery_error: Optional[Exception] = None
            if not cuda_oom and getattr(engine.backend, "quantization_active", False):
                try:
                    _recover_quantized_tts(engine, request_language)
                except Exception as caught_recovery_error:
                    recovery_error = caught_recovery_error

            error = recovery_error or original_error
            if item.audio_capture_queue is not None:
                item.audio_capture_queue.put(error)
            short_input_error = _is_short_input_generation_error(error)
            input_too_short_message = string("pipeline.input_too_short")
            if cuda_oom:
                release_cuda_after_oom()
                engine.log(
                    format_error_message(
                        tagged_string("pipeline.gen_error", "GEN ERROR"),
                        error,
                        engine.log_level,
                    ),
                    "warning",
                )
                engine.status_callback(
                    string("pipeline.generation_oom", app_name=APP_NAME), "warning"
                )
            elif short_input_error:
                engine.log(input_too_short_message, "warning")
            else:
                engine.log(
                    format_error_message(
                        tagged_string("pipeline.gen_error", "GEN ERROR"),
                        error,
                        engine.log_level,
                    ),
                    "error",
                )
            if stream_queue is not None and caption_worker is not None:
                caption_worker.submit_stream_error(stream_queue, error)
            elif stream_queue is not None:
                stream_queue.put(error)
                stream_queue.put(None)
            engine.cur_state = "idle" if short_input_error or cuda_oom else "error"
            release_pipeline(engine)
            engine.progress_callback(0, 1)
            if short_input_error:
                engine.status_callback(input_too_short_message, "warning")
            else:
                engine.error_callback(
                    string("pipeline.could_not_generate", app_name=APP_NAME)
                )
            break


async def generation_worker_job(engine: Celune) -> None:
    """Generate audio tokens and send them to the audio pipeline as an async job.

    Args:
        engine: Runtime that owns the generation queue and playback state.
    """
    caption_worker = None
    if engine.config.get("captions") is True:
        from .captions import CaptionAlignmentWorker

        caption_worker = CaptionAlignmentWorker(engine.log, engine.log_level)

    while True:
        item = await _run_in_daemon_thread(engine.text_queue.get)
        engine.regenerate = False

        if item is engine.sentinel:
            if caption_worker is not None:
                await _run_in_daemon_thread(
                    lambda: caption_worker.close(
                        lambda: engine.audio_queue.put(engine.sentinel)
                    )
                )
            else:
                try:
                    engine.audio_queue.put_nowait(engine.sentinel)
                except queue.Full:
                    await _run_in_daemon_thread(
                        lambda: engine.audio_queue.put(engine.sentinel)
                    )
            break

        request = cast(SpeechRequest, item)
        if request.generation != getattr(
            engine, "_speech_generation", request.generation
        ):
            if request.stream_queue is not None:
                request.stream_queue.put(None)
            if request.audio_capture_queue is not None:
                request.audio_capture_queue.put(
                    NotAvailableError("speech generation was superseded")
                )
            if request.playback_source_queue is not None:
                request.playback_source_queue.put(
                    NotAvailableError("prepared playback was superseded")
                )
            continue

        engine.utterance_force_stop.clear()
        engine._active_speech_generation = request.generation
        try:
            await _run_in_daemon_thread(
                lambda request=request: _process_generation_request(
                    engine,
                    request,
                    caption_worker,
                )
            )
        finally:
            engine._active_speech_generation = None


def _playback_blocks(
    chunk: PlaybackChunk,
    block_seconds: float = 0.05,
) -> deque[tuple[AudioChunk, Optional[SpeechTiming]]]:
    """Split one queued source chunk into short blocks for the mixer."""
    blocks = deque[tuple[AudioChunk, Optional[SpeechTiming]]]()
    audio = np.asarray(chunk.audio, dtype=np.float32)
    frames_per_block = max(1, round(chunk.sample_rate * block_seconds))
    for start in range(0, len(audio), frames_per_block):
        piece = np.asarray(audio[start : start + frames_per_block], dtype=np.float32)
        blocks.append((piece, chunk.timing if start == 0 else None))
    return blocks


def _ensure_playback_stream(engine: Celune, sample_rate: int) -> bool:
    """Ensure the shared playback stream exists for the requested sample rate."""
    stream_lock = getattr(engine, "stream_lock", None)
    if stream_lock is None:
        return _ensure_playback_stream_unlocked(engine, sample_rate)

    with stream_lock:
        return _ensure_playback_stream_unlocked(engine, sample_rate)


def _ensure_playback_stream_unlocked(engine: Celune, sample_rate: int) -> bool:
    """Ensure the playback stream exists while its lifecycle lock is held."""
    if engine.stream is not None and getattr(engine, "current_sr", None) == sample_rate:
        return True

    if engine.stream is not None and getattr(engine, "current_sr", None) != sample_rate:
        close_stream(engine, abort=True)

    try:
        output_device_key = (
            "output_recording_device"
            if "output_recording_device" in engine.config
            else "output_device"
        )
        output_device = resolve_audio_device(
            engine.config,
            output_device_key,
            "output",
        )
        engine.log(
            "[PLAY] resolved "
            f"{output_device_key}={engine.config.get(output_device_key)!r} "
            f"audio_api={engine.config.get('audio_api')!r} -> {output_device!r}",
            loglevel="verbose",
        )
        engine.current_sr = sample_rate
        engine.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=2,
            dtype="float32",
            blocksize=0,
            latency="high",
            device=output_device,
        )
        if engine.stream is None:
            raise NotAvailableError("audio stream is not available")
        engine.stream.start()
        engine._audio_unavailable = False
        engine.log(f"[PLAY] started stream at {sample_rate} Hz", loglevel="verbose")
        return True
    except ValueError as error:
        if not getattr(engine, "audio_unavailable", False):
            engine.log(str(error), "warning")
            engine.error_callback(string("pipeline.no_audio_devices_short"))
        engine._audio_unavailable = True
        return False
    except sd.PortAudioError as error:
        if not getattr(engine, "audio_unavailable", False):
            engine.log(
                format_error_message(
                    string("pipeline.audio_stream_init_failed", app_name=APP_NAME),
                    error,
                    getattr(engine, "log_level", "info"),
                ),
                "error",
            )
            engine.log(string("pipeline.no_audio_device"), "error")
            engine.error_callback(string("pipeline.no_audio_devices_short"))
        engine._audio_unavailable = True
        return False


def _finalize_playback_idle(
    engine: Celune,
    saved_path: Optional[str] = None,
    analysis_audio: Optional[AudioChunk] = None,
) -> None:
    """Handle post-playback reactions when the mixer becomes fully idle."""
    _reset_glow_audio_reactivity(engine)
    engine.progress_callback(1, 1)
    engine.playback_done.set()

    if engine.cur_state in {"error", "reloading"}:
        return

    if getattr(engine, "locked", False):
        return

    engine.cur_state = "idle"
    engine.idle_callback()

    if getattr(engine, "backend_mode", None) == "agent_test":
        engine._ready_announced = True
    elif random.random() < 0.01:
        flavor_texts = [
            "I will speak.",
            "I'll answer.",
            "I'm always listening.",
            "I'm all ears.",
            "You shall hear.",
        ]

        choice = random.choice(flavor_texts)
        if choice == getattr(engine, "_last_flavor", None):
            choice = random.choice(flavor_texts)
        engine._last_flavor = choice
        engine.log(string("pipeline.just_type", choice=choice))
    else:
        if (
            engine.log_level != "info"
            and saved_path is not None
            and analysis_audio is not None
        ):
            engine.log(string("pipeline.analyzing"), loglevel="verbose")
            saved = pathlib.Path(saved_path)
            _schedule_voice_analysis(
                analysis_audio,
                BASE_SR,
                saved.name,
                saved.parent,
                saved.stem,
                engine.current_voice,
                on_error=lambda error: engine.log(
                    format_error_message(
                        string("pipeline.analysis_start_failed"),
                        error,
                        engine.log_level,
                    ),
                    "warning",
                ),
            )

        if (
            engine.cur_state == "idle"
            and getattr(engine, "loaded", False)
            and not getattr(engine, "_ready_announced", False)
        ):
            is_vc_mode = False
            discard(is_vc_mode)
            mode_check = getattr(engine, "_is_voice_conversion_mode", None)
            if callable(mode_check):
                is_vc_mode = bool(mode_check())
            else:
                is_vc_mode = (
                    getattr(engine, "input_mode", "text_to_speech")
                    == "voice_conversion"
                )
            if is_vc_mode:
                engine.log(string("pipeline.ready_to_vc"))
            else:
                engine.log(string("pipeline.ready_to_speak"))
            engine._ready_announced = True

    if torch.cuda.is_available():
        avail, total = tuple(v / 1024**3 for v in torch.cuda.mem_get_info(0))
        if avail <= total * 0.1:
            engine.log(
                string("pipeline.vram_low", app_name=APP_NAME),
                "warning",
            )
            engine.log(
                string("pipeline.close_memory_apps"),
                "warning",
            )


async def playback_worker_job(engine: Celune) -> None:
    """Receive audio chunks from multiple sources, mix them, and play them.

    Args:
        engine: Runtime that owns the playback queue and output stream.

    Raises:
        NotAvailableError: Raised when no suitable output audio device is available.
    """
    source_buffers: dict[int, deque[tuple[AudioChunk, Optional[SpeechTiming]]]] = {}
    source_done: dict[int, PlaybackSourceDone] = {}
    stop_requested = False
    stop_cleanup_generation: Optional[int] = None
    (
        cpu_guard_enabled,
        max_buffer_seconds,
        max_drain_items,
        yield_seconds,
    ) = _pipeline_cpu_config(engine)
    buffered_seconds = 0.0
    contention = _PlaybackContentionMonitor(engine)
    writer = _PlaybackWriter(engine, contention)
    input_reader = _PlaybackInputReader(engine)
    input_reader.start()
    last_monitor_at = _monotonic_time()
    buffering_started_at: Optional[float] = None
    rebuffer_wait_started_at: Optional[float] = None
    pending_idle_marker: Optional[PlaybackSourceDone] = None
    last_idle_wait_state: Optional[tuple[int, bool, int, int, int, int, int, bool]] = (
        None
    )

    def playback_queue_empty() -> bool:
        """Return whether both stages of the playback input queue are empty."""
        return engine.audio_queue.empty() and input_reader.empty()

    def publish_buffered_seconds() -> None:
        """Publish the complete application-side output reserve."""
        engine.playback_buffer_seconds = max(
            0.0,
            buffered_seconds + writer.pending_seconds,
        )

    def finish_rebuffer_wait(now: Optional[float] = None) -> None:
        """Record the active reserve-gate wait, if one is in progress."""
        nonlocal rebuffer_wait_started_at
        if rebuffer_wait_started_at is None:
            return
        finished_at = _monotonic_time() if now is None else now
        engine.playback_rebuffer_wait_seconds += max(
            0.0,
            finished_at - rebuffer_wait_started_at,
        )
        rebuffer_wait_started_at = None

    def playback_is_idle() -> bool:
        """Return whether every queued playback stage has drained."""
        with engine.queue_lock:
            registered_sources = bool(_playback_source_meta(engine))
        return (
            not source_buffers
            and not source_done
            and not registered_sources
            and playback_queue_empty()
            and engine.text_queue.empty()
            and writer.pending_seconds <= 0.0
        )

    def finalize_pending_idle() -> None:
        """Deliver a deferred idle callback after the final queue stage drains."""
        nonlocal last_idle_wait_state, pending_idle_marker
        marker = pending_idle_marker
        if marker is None:
            return
        if engine.cur_state in {"error", "reloading", "stopped"}:
            pending_idle_marker = None
            last_idle_wait_state = None
            return
        if engine.locked or not playback_is_idle():
            wait_state = (
                marker.source_id,
                engine.locked,
                len(source_buffers),
                len(source_done),
                input_reader.qsize(),
                engine.audio_queue.qsize(),
                engine.text_queue.qsize(),
                writer.pending_seconds > 0.0,
            )
            if wait_state != last_idle_wait_state:
                engine.log(
                    "[IDLE] completion deferred "
                    f"source={marker.source_id} locked={engine.locked} "
                    f"source_buffers={len(source_buffers)} "
                    f"source_done={len(source_done)} "
                    f"reader_queue={input_reader.qsize()} "
                    f"audio_queue={engine.audio_queue.qsize()} "
                    f"text_queue={engine.text_queue.qsize()} "
                    f"writer_pending={writer.pending_seconds:.6f}s",
                    loglevel="debug",
                )
                last_idle_wait_state = wait_state
            return
        pending_idle_marker = None
        last_idle_wait_state = None
        engine.log(
            f"[IDLE] playback complete source={marker.source_id}",
            loglevel="debug",
        )
        _finalize_playback_idle(
            engine,
            saved_path=marker.saved_path,
            analysis_audio=marker.analysis_audio,
        )

    async def handle_playback_error(error: BaseException) -> None:
        """Report an output failure and clear the playback pipeline."""
        nonlocal buffered_seconds, buffering_started_at, pending_idle_marker
        pending_idle_marker = None
        finish_rebuffer_wait()
        engine.log(
            format_error_message(
                "[PLAY ERROR]",
                error,
                engine.log_level,
            ),
            "error",
        )
        engine.error_callback(string("pipeline.playback_error"))
        await _run_in_daemon_thread(lambda: writer.stop(clear=True))
        await _run_in_daemon_thread(lambda: close_stream(engine, True))
        engine._stream = None
        engine._current_sr = None
        source_buffers.clear()
        source_done.clear()
        input_reader.clear()
        buffered_seconds = 0.0
        buffering_started_at = None
        publish_buffered_seconds()
        for source_id in tuple(_playback_source_meta(engine)):
            _notify_speech_playback_finished(engine, source_id)
        _playback_source_statuses(engine).clear()
        _playback_source_meta(engine).clear()
        _playback_caption_states(engine).clear()
        engine._caption_source_id = None
        engine.playback_done.set()
        release_pipeline(engine)

    async def force_stop_playback() -> None:
        nonlocal buffered_seconds, buffering_started_at, stop_cleanup_generation
        nonlocal pending_idle_marker
        pending_idle_marker = None
        finish_rebuffer_wait()
        current_generation = getattr(engine, "_playback_generation", 0)
        if stop_cleanup_generation == current_generation:
            return
        stop_cleanup_generation = current_generation
        source_buffers.clear()
        source_done.clear()
        buffered_seconds = 0.0
        buffering_started_at = None
        await _run_in_daemon_thread(lambda: writer.stop(clear=True))
        publish_buffered_seconds()
        _playback_source_statuses(engine).clear()
        _playback_source_meta(engine).clear()
        _playback_caption_states(engine).clear()
        engine._caption_source_id = None
        _reset_glow_audio_reactivity(engine)
        await _run_in_daemon_thread(lambda: close_stream(engine, True))
        engine.playback_done.set()
        release_pipeline(engine)
        if getattr(engine, "_active_speech_generation", None) is None:
            engine.utterance_force_stop.clear()
        if engine.cur_state not in {"error", "stopped"} and not getattr(
            engine, "test_finished", False
        ):
            engine.idle_callback()

    async def drain_pending_items() -> bool:
        nonlocal buffered_seconds, buffering_started_at, last_monitor_at, stop_requested

        now = _monotonic_time()
        contention.sample_cpu(now)
        if source_buffers or writer.pending_seconds > 0.0:
            contention.observe_scheduler_lag(
                max(0.0, now - last_monitor_at - _PIPELINE_CPU_YIELD_SECONDS)
            )
        last_monitor_at = now
        drained_items = 0
        while drained_items < max_drain_items:
            buffer_capacity_seconds = max(
                max_buffer_seconds,
                contention.capacity_seconds(),
            )
            if (
                cpu_guard_enabled
                and buffered_seconds + writer.pending_seconds >= buffer_capacity_seconds
                and not engine.utterance_force_stop.is_set()
            ):
                break
            try:
                pending = _dequeue_playback_item(
                    engine,
                    prioritize_speech=True,
                    audio_queue=input_reader.queue,
                )
            except queue.Empty:
                break

            drained_items += 1

            if pending is engine.sentinel:
                stop_requested = True
                break

            if pending is engine.force_stop_marker:
                await force_stop_playback()
                return False

            if isinstance(pending, PlaybackChunk):
                if pending.generation != getattr(engine, "_playback_generation", 0):
                    continue
                blocking = _playback_blocks(pending)
                if blocking:
                    source_buffers.setdefault(pending.source_id, deque()).extend(
                        blocking
                    )
                    buffered_seconds += len(pending.audio) / max(1, pending.sample_rate)
                    if buffering_started_at is None:
                        buffering_started_at = _monotonic_time()
            elif isinstance(pending, PlaybackSourceDone):
                if pending.generation != getattr(engine, "_playback_generation", 0):
                    continue
                source_done[pending.source_id] = pending

            publish_buffered_seconds()

        if yield_seconds > 0.0 and not input_reader.empty():
            await asyncio.sleep(yield_seconds)
        return True

    while True:
        if engine.exit_requested:
            finish_rebuffer_wait()
            with engine.queue_lock:
                clear_queue(engine.audio_queue)
            input_reader.clear()

            await _run_in_daemon_thread(lambda: writer.stop(clear=True))
            await _run_in_daemon_thread(lambda: close_stream(engine, True))
            publish_buffered_seconds()
            input_reader.stop()
            release_pipeline(engine)
            if engine.cur_state not in {"error", "stopped"} and not getattr(
                engine, "test_finished", False
            ):
                engine.idle_callback()
            return

        try:
            timeout = (
                0.01
                if (
                    source_buffers
                    or source_done
                    or writer.pending_seconds > 0.0
                    or pending_idle_marker is not None
                )
                else None
            )
            if timeout is None:
                item = await input_reader.get()
            else:
                try:
                    item = await asyncio.wait_for(input_reader.get(), timeout)
                except TimeoutError:
                    raise queue.Empty from None
        except queue.Empty:
            item = None

        if item is engine.sentinel:
            stop_requested = True
            item = None

        if item is engine.force_stop_marker:
            await force_stop_playback()
            continue

        if isinstance(item, PlaybackChunk):
            if item.generation != getattr(engine, "_playback_generation", 0):
                continue
            stop_cleanup_generation = None
            blocks = _playback_blocks(item)
            if blocks:
                source_buffers.setdefault(item.source_id, deque()).extend(blocks)
                buffered_seconds += len(item.audio) / max(1, item.sample_rate)
                if buffering_started_at is None:
                    buffering_started_at = _monotonic_time()
        elif isinstance(item, PlaybackSourceDone):
            if item.generation != getattr(engine, "_playback_generation", 0):
                continue
            stop_cleanup_generation = None
            source_done[item.source_id] = item

        if not await drain_pending_items():
            continue

        if (writer_error := writer.error) is not None:
            await handle_playback_error(writer_error)
            continue

        if engine.exit_requested:
            continue

        while source_buffers:
            if not await drain_pending_items():
                break

            if (writer_error := writer.error) is not None:
                await handle_playback_error(writer_error)
                break

            contention.sample_cpu(_monotonic_time())
            now = _monotonic_time()
            if (
                not source_done
                and contention.requires_rebuffer()
                and buffered_seconds + writer.pending_seconds
                < contention.target_seconds()
            ):
                if rebuffer_wait_started_at is None:
                    rebuffer_wait_started_at = now
                await asyncio.sleep(0.01)
                continue

            finish_rebuffer_wait(now)

            if (
                buffered_seconds + writer.pending_seconds < contention.target_seconds()
                and not source_done
                and buffering_started_at is not None
                and _monotonic_time() - buffering_started_at
                < _PLAYBACK_BUFFER_STARTUP_GRACE_SECONDS
            ):
                await asyncio.sleep(0.005)
                continue

            buffering_started_at = None

            if not await _run_in_daemon_thread(
                lambda: _ensure_playback_stream(engine, BASE_SR)
            ):
                source_buffers.clear()
                source_done.clear()
                buffered_seconds = 0.0
                finish_rebuffer_wait()
                _playback_source_statuses(engine).clear()
                _playback_source_meta(engine).clear()
                _playback_caption_states(engine).clear()
                engine._caption_source_id = None
                release_pipeline(engine)
                if engine.cur_state not in {"error", "stopped"} and not getattr(
                    engine, "test_finished", False
                ):
                    engine.idle_callback()
                break

            writer.start()

            ready_ids = [
                source_id for source_id, blocks in source_buffers.items() if blocks
            ]
            if not ready_ids:
                break

            playback_meta = _playback_source_meta(engine)
            speech_active = any(
                playback_meta.get(source_id, {}).get("kind") == "speech"
                for source_id in ready_ids
            )
            block_len = min(
                len(source_buffers[source_id][0][0]) for source_id in ready_ids
            )
            mixed = np.zeros((block_len, 2), dtype=np.float32)
            timing_to_log: Optional[SpeechTiming] = None

            for source_id in ready_ids:
                block, timing = source_buffers[source_id][0]
                block_audio = _apply_source_gain(
                    np.asarray(block[:block_len], dtype=np.float32),
                    source_id,
                    speech_active=speech_active,
                    block_seconds=block_len / BASE_SR,
                    engine=engine,
                    source_meta=playback_meta.get(source_id),
                )
                mixed += block_audio
                if timing_to_log is None and timing is not None:
                    timing_to_log = timing

                if len(block) == block_len:
                    source_buffers[source_id].popleft()
                else:
                    source_buffers[source_id][0] = (
                        np.asarray(block[block_len:], dtype=np.float32),
                        None,
                    )

                if not source_buffers[source_id]:
                    del source_buffers[source_id]

            buffered_seconds = max(
                0.0,
                buffered_seconds - (block_len / BASE_SR) * len(ready_ids),
            )
            publish_buffered_seconds()

            mixed = np.clip(mixed, -1.0, 1.0)

            try:
                if engine.utterance_force_stop.is_set():
                    await force_stop_playback()
                    break
                log_first_playback(engine, timing_to_log)
                engine.glow.schedule(mixed)
                writer.submit(mixed, tuple(ready_ids))
                _update_playback_progress(engine, source_buffers)
                if yield_seconds > 0.0:
                    await asyncio.sleep(yield_seconds)
            except Exception as e:
                await handle_playback_error(e)
                break

            while True:
                newly_complete = [
                    source_id
                    for source_id, marker in source_done.items()
                    if source_id not in source_buffers
                    and not writer.has_pending_source(source_id)
                ]
                if not newly_complete:
                    break

                for source_id in newly_complete:
                    marker = source_done.pop(source_id)
                    engine.recently_saved = marker.saved_path
                    _notify_speech_playback_finished(engine, source_id)
                    _clear_playback_source_status(engine, source_id)
                    if marker.release_pipeline:
                        release_pipeline(
                            engine,
                            playback_idle=playback_is_idle(),
                        )
                    if marker.notify_idle:
                        pending_idle_marker = marker
                    elif playback_is_idle():
                        engine.playback_done.set()
                        _reset_glow_audio_reactivity(engine)
                        engine.progress_callback(1, 1)
                    finalize_pending_idle()

        while True:
            orphaned = [
                source_id
                for source_id, marker in source_done.items()
                if source_id not in source_buffers
                and not writer.has_pending_source(source_id)
            ]
            if not orphaned:
                break

            for source_id in orphaned:
                marker = source_done.pop(source_id)
                engine.recently_saved = marker.saved_path
                _notify_speech_playback_finished(engine, source_id)
                _clear_playback_source_status(engine, source_id)
                if marker.release_pipeline:
                    release_pipeline(
                        engine,
                        playback_idle=playback_is_idle(),
                    )
                if marker.notify_idle:
                    pending_idle_marker = marker
                elif playback_is_idle():
                    engine.playback_done.set()
                    _reset_glow_audio_reactivity(engine)
                    engine.progress_callback(1, 1)
                finalize_pending_idle()

        finalize_pending_idle()
        publish_buffered_seconds()
        finish_rebuffer_wait()
        if (
            stop_requested
            and not source_buffers
            and not source_done
            and writer.pending_seconds <= 0.0
        ):
            await _run_in_daemon_thread(writer.wait_empty)
            if (writer_error := writer.error) is not None:
                await handle_playback_error(writer_error)
            await _run_in_daemon_thread(writer.stop)
            input_reader.stop()
            finalize_pending_idle()
            break


def _install_pipeline_facade() -> None:
    """Install the split pipeline modules after their circular imports settle."""
    from . import speech, playback, conversation

    globals().update(
        {
            "conversation": conversation,
            "playback": playback,
            "speech": speech,
            **{
                value.__name__: value
                for value in (
                    _effective_voice_prompt,
                    _think_persona,
                    _apply_source_gain,
                    _clear_playback_source_status,
                    _dequeue_playback_item,
                    _download_youtube_sfx,
                    _flush_buffered_speech_chunks,
                    _next_playback_source_id,
                    _notify_speech_playback_finished,
                    _pipeline_cpu_config,
                    _playback_source_meta,
                    _playback_source_statuses,
                    _queue_playback_chunk,
                    _queue_playback_done,
                    _register_overlay_playback,
                    _register_playback_source,
                    _remember_smart_buffer_speed,
                    _set_playback_source_status,
                    _smart_buffer_target_seconds,
                    _update_playback_progress,
                    _youtube_sfx_title,
                    acquire_pipeline,
                    current_playback_status,
                    release_pipeline,
                )
            },
            "think": _think_persona,
        }
    )
    playback.install(globals())
    conversation.install_pipeline(globals())
    speech.install(globals())


_install_pipeline_facade()


def install_engine(target):
    """Install pipeline lifecycle entrypoints on ``Celune``."""
    install_class_functions(target, {"_stop_pipeline_jobs": _stop_pipeline_jobs})


celune_metadata_payload = _celune_metadata_payload
parse_vorbis_comment_block = _parse_vorbis_comment_block
flac_metadata_blocks = _flac_metadata_blocks
write_flac_metadata = _write_flac_metadata
write_celune_flac = _write_celune_flac
saved_output_speech_seconds = _saved_output_speech_seconds
register_playback_source = _register_playback_source
set_playback_source_status = _set_playback_source_status
get_current_playback_status = current_playback_status
queue_playback_chunk = _queue_playback_chunk
queue_playback_done = _queue_playback_done
youtube_sfx_title = _youtube_sfx_title
download_youtube_sfx = _download_youtube_sfx
finalize_playback_idle = _finalize_playback_idle
