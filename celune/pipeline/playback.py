# SPDX-License-Identifier: Apache-2.0
"""Shared playback worker, mixing, and audio stream lifecycle."""

from __future__ import annotations
import os
import time
import queue
import ctypes
import random
import asyncio
import pathlib
import threading
import contextlib
import collections.abc
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Optional, cast

import torch
import numpy as np
import sounddevice as sd

from .timing import (
    SFX_DUCK_GAIN,
    SFX_DUCK_FADE_SECONDS,
    PIPELINE_CPU_YIELD_SECONDS,
    PLAYBACK_BUFFER_STARTUP_GRACE_SECONDS,
    monotonic_time,
)

from ..i18n import string
from ..typing.aliases import AudioChunk
from ..config import resolve_audio_device
from ..constants import BASE_SR, APP_NAME
from ..exceptions import NotAvailableError
from ..threads import run_in_daemon_thread
from ..analysis import schedule_voice_analysis
from ..utils import discard, clear_queue, format_number, format_error_message
from ..dataclasses.pipeline import SpeechTiming, PlaybackChunk, PlaybackSourceDone
from ..audio.dsp import error_signal, working_signal, sleeping_signal, readiness_signal

from .buffering import (
    PlaybackContentionMonitor,
    acquire_pipeline,
    release_pipeline,
    pipeline_cpu_config,
    queue_playback_done,
    queue_playback_chunk,
    dequeue_playback_item,
    should_apply_contention_buffering,
)
from .captions import (
    playback_source_meta,
    next_playback_source_id,
    playback_caption_states,
    playback_source_statuses,
    register_playback_source,
    update_playback_progress,
    register_overlay_playback,
    clear_playback_source_status,
    notify_speech_playback_finished,
)

if TYPE_CHECKING:
    from ..celune import Celune


def prioritize_playback_writer_thread() -> None:
    """Raise the playback writer above normal priority on supported Windows hosts."""
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


def _active_speech_source_ids(
    source_buffers: dict[int, deque[tuple[AudioChunk, Optional[SpeechTiming]]]],
    engine: Celune,
) -> set[int]:
    """Return active speech-source ids that should trigger SFX ducking."""
    meta = playback_source_meta(engine)
    return {
        source_id
        for source_id in source_buffers
        if meta.get(source_id, {}).get("kind") == "speech"
    }


def apply_source_gain(
    audio: AudioChunk,
    source_id: int,
    *,
    speech_active: bool,
    block_seconds: float,
    engine: Celune,
    source_meta: Optional[dict[str, Union[str, float]]] = None,
) -> AudioChunk:
    """Apply ducking and smooth gain ramps for one mixer source block."""
    meta = (
        playback_source_meta(engine).get(source_id)
        if source_meta is None
        else source_meta
    )
    if not isinstance(meta, dict):
        return audio

    kind = str(meta.get("kind", "sfx"))
    base_gain = float(meta.get("base_gain", 1.0))
    current_gain = float(meta.get("current_gain", base_gain))
    if kind == "sfx":
        target_gain = base_gain * (SFX_DUCK_GAIN if speech_active else 1.0)
    else:
        target_gain = base_gain

    if abs(target_gain - current_gain) < 1e-6:
        meta["current_gain"] = target_gain
        return np.asarray(audio * target_gain, dtype=np.float32)

    fade_ratio = min(1.0, block_seconds / SFX_DUCK_FADE_SECONDS)
    next_gain = current_gain + (target_gain - current_gain) * fade_ratio
    ramp = np.linspace(current_gain, next_gain, len(audio), dtype=np.float32)
    meta["current_gain"] = next_gain
    return np.asarray(audio * ramp[:, None], dtype=np.float32)


@dataclass
class _PlaybackWriteItem:
    """Describe one mixed block waiting for the persistent output writer."""

    audio: AudioChunk
    source_ids: tuple[int, ...]
    duration_seconds: float
    submitted_at: float


class _PlaybackWriter:
    """Write mixed audio on one persistent thread with reserve accounting."""

    def __init__(self, engine: Celune, monitor: PlaybackContentionMonitor) -> None:
        self._engine = engine
        self._monitor = monitor
        self._queue: queue.Queue[Optional[_PlaybackWriteItem]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending_frames = 0
        self._pending_sources: dict[int, int] = {}
        self._error: Optional[BaseException] = None
        self._last_write_finished_at: Optional[float] = None

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
        source_meta = playback_source_meta(self._engine)
        for source_id in item.source_ids:
            metadata = source_meta.get(source_id)
            if isinstance(metadata, dict):
                metadata["played_frames"] = float(
                    metadata.get("played_frames", 0.0)
                ) + float(len(item.audio))
        update_playback_progress(self._engine)

    def _run(self) -> None:
        """Consume the output queue until a stop marker is received."""
        prioritize_playback_writer_thread()
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
            try:
                underflowed = bool(_write_playback_block(self._engine, item.audio))
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
                    finished_at - started_at,
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
        elapsed = monotonic_time() - start_time

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
            or bool(playback_source_meta(engine))
            or bool(playback_source_statuses(engine))
        )

    if not is_active:
        engine.utterance_force_stop.clear()
        return False

    engine.log(string("pipeline.forcefully_stopping_speech"))
    invalidate_speech_work(engine)

    return True


def invalidate_speech_work(engine: Celune) -> None:
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
    if engine.exit_requested:
        return False

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
    # engine.try_play_signal() instead of calling this method directly.
    if signal_type == "readiness":
        source_id = next_playback_source_id(engine)
        state_event = register_overlay_playback(engine)
        register_playback_source(engine, source_id, kind="sfx")
        if not queue_playback_chunk(engine, source_id, signal, BASE_SR):
            return False
        if not queue_playback_done(
            engine,
            source_id,
            notify_idle_when_finished=True,
        ):
            return False
        if state_event is not None:
            engine.emit_event("state_changed", state_event)
        return True

    if acquire_pipeline(engine, f"play {signal_type} signal"):
        release_to_idle = False
        if engine.cur_state != "error":
            if signal_type == "sleeping":
                engine.cur_state = "sleeping"
            elif signal_type != "working":
                engine.cur_state = "speaking"
        source_id = next_playback_source_id(engine)
        register_playback_source(engine, source_id, kind="sfx")
        if not queue_playback_chunk(engine, source_id, signal, BASE_SR):
            release_pipeline(engine)
            return False
        if not queue_playback_done(
            engine,
            source_id,
            release_pipeline_when_finished=release_to_idle,
            notify_idle_when_finished=signal_type == "readiness",
        ):
            release_pipeline(engine)
            return False
        release_pipeline(engine, playback_idle=False)
        return True
    return False


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
            schedule_voice_analysis(
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
            is_singing = False

            discard(is_singing)
            discard(is_vc_mode)
            mode_check = getattr(engine, "is_voice_conversion_mode", None)

            if callable(mode_check):
                is_vc_mode = bool(mode_check())
            else:
                is_vc_mode = (
                    getattr(engine, "input_mode", "text_to_speech")
                    == "voice_conversion"
                )

            is_singing = getattr(engine.backend, "singing", False)

            if is_vc_mode:
                engine.log(
                    string("pipeline.ready", action=string("pipeline.action_vc"))
                )
            elif is_singing:
                engine.log(
                    string("pipeline.ready", action=string("pipeline.action_sing"))
                )
            else:
                engine.log(
                    string("pipeline.ready", action=string("pipeline.action_speak"))
                )

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
    ) = pipeline_cpu_config(engine)
    buffered_seconds = 0.0
    contention = PlaybackContentionMonitor(engine)
    writer = _PlaybackWriter(engine, contention)
    input_reader = _PlaybackInputReader(engine)
    input_reader.start()
    last_monitor_at = monotonic_time()
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
        finished_at = monotonic_time() if now is None else now
        engine.playback_rebuffer_wait_seconds += max(
            0.0,
            finished_at - rebuffer_wait_started_at,
        )
        rebuffer_wait_started_at = None

    def playback_is_idle() -> bool:
        """Return whether every queued playback stage has drained."""
        with engine.queue_lock:
            registered_sources = bool(playback_source_meta(engine))
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
        if engine.cur_state in {"error", "stopped"}:
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
        await run_in_daemon_thread(lambda: writer.stop(clear=True))
        await run_in_daemon_thread(lambda: close_stream(engine, True))
        engine._stream = None
        engine._current_sr = None
        source_buffers.clear()
        source_done.clear()
        input_reader.clear()
        buffered_seconds = 0.0
        buffering_started_at = None
        publish_buffered_seconds()
        for source_id in tuple(playback_source_meta(engine)):
            notify_speech_playback_finished(engine, source_id)
        playback_source_statuses(engine).clear()
        playback_source_meta(engine).clear()
        playback_caption_states(engine).clear()
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
        await run_in_daemon_thread(lambda: writer.stop(clear=True))
        publish_buffered_seconds()
        playback_source_statuses(engine).clear()
        playback_source_meta(engine).clear()
        playback_caption_states(engine).clear()
        engine._caption_source_id = None
        _reset_glow_audio_reactivity(engine)
        await run_in_daemon_thread(lambda: close_stream(engine, True))
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

        now = monotonic_time()
        contention.sample_cpu(now)
        if source_buffers or writer.pending_seconds > 0.0:
            contention.observe_scheduler_lag(
                max(0.0, now - last_monitor_at - PIPELINE_CPU_YIELD_SECONDS)
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
                pending = dequeue_playback_item(
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
                        buffering_started_at = monotonic_time()
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

            await run_in_daemon_thread(lambda: writer.stop(clear=True))
            await run_in_daemon_thread(lambda: close_stream(engine, True))
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
                    buffering_started_at = monotonic_time()
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

            apply_contention_buffering = should_apply_contention_buffering(
                engine,
                source_buffers,
                bool(source_done),
            )

            if (writer_error := writer.error) is not None:
                await handle_playback_error(writer_error)
                break

            contention.sample_cpu(monotonic_time())
            now = monotonic_time()
            if (
                apply_contention_buffering
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
                apply_contention_buffering
                and buffered_seconds + writer.pending_seconds
                < contention.target_seconds()
                and buffering_started_at is not None
                and monotonic_time() - buffering_started_at
                < PLAYBACK_BUFFER_STARTUP_GRACE_SECONDS
            ):
                await asyncio.sleep(0.005)
                continue

            buffering_started_at = None

            if not await run_in_daemon_thread(
                lambda: _ensure_playback_stream(engine, BASE_SR)
            ):
                source_buffers.clear()
                source_done.clear()
                buffered_seconds = 0.0
                finish_rebuffer_wait()
                playback_source_statuses(engine).clear()
                playback_source_meta(engine).clear()
                playback_caption_states(engine).clear()
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

            playback_meta = playback_source_meta(engine)
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
                block_audio = apply_source_gain(
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
                update_playback_progress(engine, source_buffers)
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
                    notify_speech_playback_finished(engine, source_id)
                    clear_playback_source_status(engine, source_id)
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
                notify_speech_playback_finished(engine, source_id)
                clear_playback_source_status(engine, source_id)
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
            await run_in_daemon_thread(writer.wait_empty)
            if (writer_error := writer.error) is not None:
                await handle_playback_error(writer_error)
            await run_in_daemon_thread(writer.stop)
            input_reader.stop()
            finalize_pending_idle()
            break
