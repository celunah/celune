# SPDX-License-Identifier: Apache-2.0
"""Pipeline locks, buffering, and playback contention policy."""

from __future__ import annotations
import os
import uuid
import queue
import threading
import contextlib
import collections.abc
from typing import TYPE_CHECKING, Union, Optional, cast

import psutil
import numpy as np

from .captions import (
    playback_source_meta,
    set_playback_source_status,
)
from .timing import (
    LEGACY_BUFFER_SECONDS,
    SMART_BUFFER_SMOOTHING,
    SMART_BUFFER_MAX_SECONDS,
    SMART_BUFFER_MIN_SECONDS,
    PIPELINE_CPU_YIELD_SECONDS,
    PLAYBACK_BUFFER_MAX_SECONDS,
    PLAYBACK_BUFFER_MIN_SECONDS,
    SMART_BUFFER_REALTIME_SPEED,
    PIPELINE_CPU_MAX_DRAIN_ITEMS,
    PLAYBACK_CONTENTION_CPU_START,
    PIPELINE_CPU_MAX_BUFFER_SECONDS,
    PLAYBACK_TRACE_INTERVAL_SECONDS,
    PLAYBACK_CONTENTION_CPU_CRITICAL,
    PLAYBACK_CONTENTION_STABLE_DECAY,
    SMART_BUFFER_COMPLETE_BELOW_SPEED,
    PLAYBACK_CONTENTION_REBUFFER_LEVEL,
    PLAYBACK_CONTENTION_SAMPLE_SECONDS,
    PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS,
    PLAYBACK_CONTENTION_LAG_START_SECONDS,
    SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS,
    SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS,
    PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
    monotonic_time,
)

from ..i18n import string
from ..utils import discard
from ..core.concurrency import ComponentLockManager
from ..typing.common import JSONSerializable
from ..typing.pipeline import SpeechStreamQueue
from ..dataclasses.events import StateChangedEvent
from ..typing.aliases import AudioChunk, AudioChunks
from ..constants import BASE_SR, APP_NAME, PipelineStates
from ..dataclasses.pipeline import (
    SpeechTiming,
    PlaybackChunk,
    PlaybackSourceDone,
)
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
    ComponentLockAcquisition,
    ComponentLockRequirement,
)

if TYPE_CHECKING:
    from ..celune import Celune


def _pipeline_requirements(action: str) -> tuple[ComponentLockRequirement, ...]:
    """Return the existing pipeline resources required by one playback action."""
    components = (
        (ComponentLockName.TTS, ComponentLockName.SPEECH_QUEUE)
        if action == "speak"
        else (ComponentLockName.SPEECH_QUEUE,)
    )
    return tuple(
        ComponentLockRequirement(component)
        for component in (*components, ComponentLockName.AUDIO_PLAYBACK)
    )


def _component_busy_message(busy: ComponentBusyResult) -> str:
    """Return a localized-friendly component list for busy diagnostics."""
    return ", ".join(component.value.upper() for component in busy.components)


def notify_component_busy(
    engine: Celune,
    action: str,
    busy: ComponentBusyResult,
) -> None:
    """Report one typed component conflict through the existing engine callbacks."""
    engine._last_component_busy = busy
    engine.log(
        string(
            "pipeline.busy_components",
            components=_component_busy_message(busy),
        ),
        "warning",
    )
    engine.log(
        string("pipeline.busy_action", action=action, app_name=APP_NAME),
        "warning",
    )
    engine.error_callback(string("celune.app_busy", app_name=APP_NAME))


def _reload_busy_result(
    requirements: tuple[ComponentLockRequirement, ...],
    owner: ComponentLockOwner,
    manager: Optional[ComponentLockManager],
) -> ComponentLockAcquisition:
    """Build a typed speech conflict while model lifecycle work is active."""
    model_owner = (
        manager.owner_for(ComponentLockName.MODEL_LOADING)
        if manager is not None
        else None
    )
    return ComponentLockAcquisition(
        owner,
        tuple(requirement.component for requirement in requirements),
        ComponentBusyResult(
            components=(ComponentLockName.MODEL_LOADING,),
            owners=((ComponentLockName.MODEL_LOADING, model_owner),),
        ),
    )


def acquire_pipeline_result(
    engine: Celune,
    action: str,
    owner: Optional[ComponentLockOwner] = None,
) -> ComponentLockAcquisition:
    """Atomically claim the legacy pipeline and typed component resources."""
    resolved_owner = owner or ComponentLockOwner(
        operation_id=f"pipeline:{action}:{uuid.uuid4().hex}",
    )
    requirements = _pipeline_requirements(action)
    with engine.say_lock:
        if engine.exit_requested:
            busy = ComponentBusyResult(
                components=tuple(requirement.component for requirement in requirements),
                owners=tuple(
                    (requirement.component, None) for requirement in requirements
                ),
            )
            return ComponentLockAcquisition(
                resolved_owner,
                tuple(requirement.component for requirement in requirements),
                busy,
            )
        engine.log(
            f"[LOCK] acquire requested by {action}, locked={engine.locked}",
            loglevel="verbose",
        )
        manager = cast(
            Optional[ComponentLockManager],
            getattr(engine, "component_locks", None),
        )
        reload_pending = (
            bool(getattr(engine, "_reload_pending", False))
            or getattr(engine, "cur_state", None) == "reloading"
        )
        if action == "speak" and reload_pending:
            acquisition = _reload_busy_result(requirements, resolved_owner, manager)
        elif engine.locked:
            owners = (
                tuple(
                    manager.snapshot().get(component)
                    for component in (
                        requirement.component for requirement in requirements
                    )
                )
                if manager is not None
                else (None,) * len(requirements)
            )
            busy = ComponentBusyResult(
                components=tuple(requirement.component for requirement in requirements),
                owners=tuple(
                    (requirement.component, owner_value)
                    for requirement, owner_value in zip(requirements, owners)
                ),
            )
            acquisition = ComponentLockAcquisition(
                resolved_owner,
                tuple(requirement.component for requirement in requirements),
                busy,
            )
        elif manager is None:
            acquisition = ComponentLockAcquisition(
                resolved_owner,
                tuple(requirement.component for requirement in requirements),
            )
        else:
            acquisition = manager.try_acquire(requirements, resolved_owner)

        if not acquisition.acquired:
            busy = acquisition.busy
            assert busy is not None
            notify_component_busy(engine, action, busy)
            return acquisition

        engine._pipeline_lock_owner = resolved_owner
        engine._last_component_busy = None
        if action != "play readiness signal":
            engine._ready_announced = False
        engine.locked = True
        engine.playback_done.clear()
        engine.log(
            f"[LOCK] acquired by {action} text_queue={engine.text_queue.qsize()} "
            f"audio_queue={engine.audio_queue.qsize()}",
            loglevel="debug",
        )
        return acquisition


def acquire_pipeline(engine: Celune, action: str) -> bool:
    """Atomically claim Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.
        action: A short label describing the action requesting the lock.

    Returns:
        bool: ``True`` when the pipeline was claimed, otherwise ``False``.
    """
    return acquire_pipeline_result(engine, action).acquired


def release_pipeline(engine: Celune, playback_idle: bool = True) -> None:
    """Release Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.
        playback_idle: Whether playback should be marked fully idle now.
    """
    state_event: Optional[StateChangedEvent] = None
    with engine.say_lock:
        manager = getattr(engine, "component_locks", None)
        owner = getattr(engine, "_pipeline_lock_owner", None)
        if manager is not None and owner is not None:
            manager.release(owner)
        engine._pipeline_lock_owner = None
        engine.locked = False
        if playback_idle:
            engine.playback_done.set()
            if engine.cur_state not in {"error", "stopped"} and not getattr(
                engine, "test_finished", False
            ):
                state_event = engine.update_cur_state("idle")
        engine.log("[LOCK] released", loglevel="verbose")
        engine.log(
            "[LOCK] release complete "
            f"playback_done={engine.playback_done.is_set()} "
            f"text_queue={engine.text_queue.qsize()} "
            f"audio_queue={engine.audio_queue.qsize()}",
            loglevel="debug",
        )
    if state_event is not None:
        engine.emit_event("state_changed", state_event)


def should_apply_contention_buffering(
    engine: Celune,
    source_ids: collections.abc.Iterable[int],
    has_completed_source: bool,
) -> bool:
    """Return whether active sources should wait for contention reserve."""
    if has_completed_source:
        return False

    source_meta = playback_source_meta(engine)
    for source_id in source_ids:
        metadata = source_meta.get(source_id)
        if isinstance(metadata, dict) and metadata.get("low_latency") is True:
            return False
    return True


def queue_playback_chunk(
    engine: Celune,
    source_id: int,
    audio: AudioChunk,
    sample_rate: int,
    timing: Optional[SpeechTiming] = None,
    generation: Optional[int] = None,
) -> bool:
    """Queue one chunk for the shared DSP playback mixer."""
    with engine.queue_lock:
        if engine.exit_requested:
            return False
        active_playback_generation = getattr(engine, "_playback_generation", 0)
        expected_generation = (
            active_playback_generation if generation is None else generation
        )
        if expected_generation != active_playback_generation:
            return False

        meta = playback_source_meta(engine).get(source_id)
        active_generation = getattr(engine, "_active_speech_generation", None)
        if (
            active_generation is not None
            and active_generation
            != getattr(engine, "_speech_generation", active_generation)
            and (not isinstance(meta, dict) or not meta.get("async_caption_audio"))
        ):
            return False
        if engine.utterance_force_stop.is_set():
            return False

        if isinstance(meta, dict):
            if float(meta.get("generation", 0.0)) != float(
                getattr(engine, "_playback_generation", 0)
            ):
                return False
            meta["total_frames"] = float(meta.get("total_frames", 0.0)) + float(
                len(audio)
            )

    queue_wait_started_at = monotonic_time()
    engine.audio_queue.put(
        PlaybackChunk(
            source_id=source_id,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            timing=timing,
            generation=expected_generation,
        )
    )
    enqueued_at = monotonic_time()
    queue_wait_seconds = max(0.0, enqueued_at - queue_wait_started_at)
    with engine.queue_lock:
        last_queued_by_source = engine.playback_chunk_last_queued_at
        last_queued_at = last_queued_by_source.get(source_id)
        if last_queued_at is None:
            generation_gap_seconds = 0.0
        else:
            generation_gap_seconds = max(0.0, enqueued_at - last_queued_at)
        last_queued_by_source[source_id] = enqueued_at
        engine.playback_queue_wait_seconds = queue_wait_seconds
        engine.playback_generation_gap_seconds = generation_gap_seconds
    _playback_trace(engine, enqueued_at)
    return True


def dequeue_playback_item(
    engine: Celune,
    prioritize_speech: bool = False,
    audio_queue: Optional[queue.Queue[object]] = None,
) -> Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]:
    """Remove one playback item, prioritizing speech overlays when requested."""
    playback_queue = engine.audio_queue if audio_queue is None else audio_queue
    if not prioritize_speech:
        return cast(
            Union[PlaybackChunk, PlaybackSourceDone, PipelineStates],
            playback_queue.get_nowait(),
        )

    with playback_queue.mutex:
        if not playback_queue.queue:
            raise queue.Empty

        speech_chunk_index: Optional[int] = None
        speech_done_index: Optional[int] = None
        for index, pending in enumerate(playback_queue.queue):
            if isinstance(pending, PlaybackChunk):
                source_meta = playback_source_meta(engine).get(pending.source_id)
                if (
                    isinstance(source_meta, dict)
                    and source_meta.get("kind") == "speech"
                ):
                    speech_chunk_index = index
                    break
            elif isinstance(pending, PlaybackSourceDone):
                source_meta = playback_source_meta(engine).get(pending.source_id)
                if (
                    speech_done_index is None
                    and isinstance(source_meta, dict)
                    and source_meta.get("kind") == "speech"
                ):
                    speech_done_index = index

        selected_index = (
            speech_chunk_index
            if speech_chunk_index is not None
            else speech_done_index
            if speech_done_index is not None
            else 0
        )
        playback_queue.queue.rotate(-selected_index)
        pending = playback_queue.queue.popleft()
        playback_queue.queue.rotate(selected_index)
        playback_queue.not_full.notify()
        return pending


def _playback_trace(engine: Celune, now: Optional[float] = None) -> None:
    """Emit the consolidated playback timing trace at a safe low frequency."""
    timestamp = monotonic_time() if now is None else now
    last_trace_at = float(getattr(engine, "_playback_trace_last_logged_at", 0.0))
    if timestamp - last_trace_at < PLAYBACK_TRACE_INTERVAL_SECONDS:
        return

    engine._playback_trace_last_logged_at = timestamp
    engine.log(
        "[PLAY] playback timing "
        f"reserve={float(getattr(engine, 'playback_buffer_seconds', 0.0)):.2f}s "
        f"contention={float(getattr(engine, 'playback_contention_level', 0.0)):.2f} "
        f"underflows={int(getattr(engine, 'playback_underflows', 0))} "
        f"queue_wait={float(getattr(engine, 'playback_queue_wait_seconds', 0.0)):.3f}s "
        f"generation_gap={float(getattr(engine, 'playback_generation_gap_seconds', 0.0)):.3f}s "
        f"writer_wait={float(getattr(engine, 'playback_writer_wait_seconds', 0.0)):.3f}s "
        f"writer_gap={float(getattr(engine, 'playback_writer_gap_seconds', 0.0)):.3f}s "
        f"writer_write={float(getattr(engine, 'playback_writer_write_seconds', 0.0)):.3f}s "
        f"rebuffer_wait={float(getattr(engine, 'playback_rebuffer_wait_seconds', 0.0)):.3f}s",
        loglevel="debug",
    )


def queue_playback_done(
    engine: Celune,
    source_id: int,
    *,
    release_pipeline_when_finished: bool = False,
    notify_idle_when_finished: bool = True,
    saved_path: Optional[str] = None,
    analysis_audio: Optional[AudioChunk] = None,
    generation: Optional[int] = None,
) -> bool:
    """Queue a completion marker for one playback source."""
    with engine.queue_lock:
        if engine.exit_requested:
            return False
        active_playback_generation = getattr(engine, "_playback_generation", 0)
        expected_generation = (
            active_playback_generation if generation is None else generation
        )
        if expected_generation != active_playback_generation:
            return False

        source_meta = playback_source_meta(engine).get(source_id)
        active_generation = getattr(engine, "_active_speech_generation", None)
        if (
            active_generation is not None
            and active_generation
            != getattr(engine, "_speech_generation", active_generation)
            and (
                not isinstance(source_meta, dict)
                or not source_meta.get("async_caption_audio")
            )
        ):
            return False
        if engine.utterance_force_stop.is_set():
            return False

        if isinstance(source_meta, dict) and float(
            source_meta.get("generation", 0.0)
        ) != float(getattr(engine, "_playback_generation", 0)):
            return False
        if isinstance(source_meta, dict):
            source_meta["total_frames_final"] = 1.0

    engine.audio_queue.put(
        PlaybackSourceDone(
            source_id=source_id,
            release_pipeline=release_pipeline_when_finished,
            notify_idle=notify_idle_when_finished,
            saved_path=saved_path,
            analysis_audio=analysis_audio,
            generation=expected_generation,
        )
    )
    with engine.queue_lock:
        last_queued_at = getattr(engine, "playback_chunk_last_queued_at", None)
        if isinstance(last_queued_at, dict):
            last_queued_at.pop(source_id, None)
    engine.log(
        "[QUEUE] playback done "
        f"source={source_id} generation={expected_generation} "
        f"release_pipeline={release_pipeline_when_finished} "
        f"notify_idle={notify_idle_when_finished} "
        f"audio_queue={engine.audio_queue.qsize()}",
        loglevel="debug",
    )
    return True


def flush_buffered_speech_chunks(
    engine: Celune,
    source_id: int,
    buffer: AudioChunks,
    speech_timing: SpeechTiming,
    pushed_audio: bool,
    stream_queue: Optional[SpeechStreamQueue],
    caption_text: Optional[str] = None,
) -> bool:
    """Queue buffered speech chunks without merging them into a larger copy."""
    if not buffer:
        return pushed_audio

    first_buffer_chunk = True
    for queued_audio in buffer:
        queued = queue_playback_chunk(
            engine,
            source_id,
            queued_audio,
            BASE_SR,
            speech_timing if not pushed_audio and first_buffer_chunk else None,
        )
        if not queued:
            buffer.clear()
            return pushed_audio
        if stream_queue is not None:
            stream_queue.put(queued_audio.copy())
        first_buffer_chunk = False

    buffer.clear()
    if not pushed_audio:
        caption_callback = getattr(engine, "caption_callback", None)
        if caption_text is not None and callable(caption_callback):
            engine._caption_source_id = source_id
            caption_callback(caption_text)
        set_playback_source_status(engine, source_id, string("status.speaking"))
        engine.cur_state = "speaking"
        engine.queue_avail_callback()
        return True

    return pushed_audio


def config_text(engine: Celune, key: str, default: str) -> str:
    """Read a string configuration value with a fallback."""
    value = engine.config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    return default


def config_lines(engine: Celune, key: str) -> tuple[str, ...]:
    """Read a text or text-list configuration value as non-empty lines."""
    return config_value_lines(engine.config.get(key))


def config_value_lines(value: JSONSerializable) -> tuple[str, ...]:
    """Normalize one scalar or list configuration value into text lines."""
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        lines = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        return tuple(lines)
    return ()


def _config_float(
    source: collections.abc.Mapping[str, JSONSerializable], key: str, default: float
) -> float:
    """Read one numeric config field as a float with a fallback."""
    value = source.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except ValueError:
            return default
    return default


def _safe_config_int(
    source: collections.abc.Mapping[str, JSONSerializable], key: str, default: int
) -> int:
    """Read a bounded integer configuration value without raising on bad input."""
    value = _config_float(source, key, float(default))
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _smart_buffer_config(
    engine: Celune,
) -> tuple[bool, float, float, float, float, float, float]:
    """Return Celune's fixed adaptive speech-buffer settings."""
    discard(engine)
    return (
        True,
        SMART_BUFFER_REALTIME_SPEED,
        SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS,
        SMART_BUFFER_MIN_SECONDS,
        SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS,
        SMART_BUFFER_MAX_SECONDS,
        SMART_BUFFER_COMPLETE_BELOW_SPEED,
    )


def pipeline_cpu_config(engine: Celune) -> tuple[bool, float, int, float]:
    """Return Celune's bounded cooperative playback-pressure controls."""
    discard(engine)
    return (
        True,
        PIPELINE_CPU_MAX_BUFFER_SECONDS,
        PIPELINE_CPU_MAX_DRAIN_ITEMS,
        PIPELINE_CPU_YIELD_SECONDS,
    )


def _smart_buffer_speed_estimate(
    engine: Celune,
    speech_len: float,
    generation_elapsed: float,
    min_speed_sample_seconds: float,
) -> Optional[float]:
    """Estimate current generation speed in audio-seconds per wall-second."""
    if generation_elapsed > 0.0 and speech_len >= min_speed_sample_seconds:
        return speech_len / generation_elapsed

    previous = getattr(engine, "smart_buffer_generation_speed", None)
    if isinstance(previous, (int, float)) and previous > 0.0:
        return float(previous)
    return None


def smart_buffer_target_seconds(
    engine: Celune,
    speech_len: float,
    generation_elapsed: float,
) -> float:
    """Return the current adaptive pre-playback buffer target in seconds."""
    (
        enabled,
        realtime_speed,
        protected_playback_seconds,
        minimum_seconds,
        min_speed_sample_seconds,
        max_seconds,
        complete_below_speed,
    ) = _smart_buffer_config(engine)

    if not enabled:
        return LEGACY_BUFFER_SECONDS

    speed_estimate = _smart_buffer_speed_estimate(
        engine,
        speech_len,
        generation_elapsed,
        min_speed_sample_seconds,
    )
    if speed_estimate is None:
        return min(max_seconds, max(1.0, minimum_seconds))

    if speed_estimate >= realtime_speed:
        return 0.0

    if speed_estimate <= complete_below_speed:
        return float("inf")

    speed_deficit = max(0.0, 1.0 - speed_estimate)
    target_seconds = minimum_seconds + (protected_playback_seconds * speed_deficit)
    return min(max_seconds, max(minimum_seconds, target_seconds))


def remember_smart_buffer_speed(engine: Celune, generation_speed: float) -> None:
    """Update the engine's rolling generation-speed estimate."""
    if generation_speed <= 0.0:
        return

    previous = getattr(engine, "smart_buffer_generation_speed", None)
    if isinstance(previous, (int, float)) and previous > 0.0:
        generation_speed = (float(previous) * (1.0 - SMART_BUFFER_SMOOTHING)) + (
            generation_speed * SMART_BUFFER_SMOOTHING
        )
    engine.smart_buffer_generation_speed = generation_speed


class PlaybackContentionMonitor:
    """Estimate playback contention from CPU pressure and output timing."""

    def __init__(self, engine: Celune) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._last_sample_at = 0.0
        self._level = 0.0
        self._underflows = 0
        with contextlib.suppress(psutil.Error, OSError):
            self._process.cpu_percent(interval=None)

    @staticmethod
    def _pressure(value: float, start: float, critical: float) -> float:
        """Normalize one observed pressure value to the inclusive 0..1 range."""
        if value <= start:
            return 0.0
        if value >= critical:
            return 1.0
        return (value - start) / (critical - start)

    def _publish_locked(self) -> None:
        """Publish lightweight diagnostics for logs and status views."""
        self._engine.playback_contention_level = self._level
        self._engine.playback_underflows = self._underflows

    def sample_cpu(self, now: float) -> None:
        """Sample system and process CPU without blocking the playback loop."""
        with self._lock:
            if now - self._last_sample_at < PLAYBACK_CONTENTION_SAMPLE_SECONDS:
                return
            self._last_sample_at = now

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except (psutil.Error, OSError, TypeError, ValueError):
            cpu_percent = 0.0

        try:
            process_percent = self._process.cpu_percent(interval=None)
        except (psutil.Error, OSError, TypeError, ValueError):
            process_percent = 0.0

        logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
        normalized_process_percent = process_percent / logical_cpus
        pressure = max(
            self._pressure(
                cpu_percent,
                PLAYBACK_CONTENTION_CPU_START,
                PLAYBACK_CONTENTION_CPU_CRITICAL,
            ),
            self._pressure(
                normalized_process_percent,
                PLAYBACK_CONTENTION_CPU_START,
                PLAYBACK_CONTENTION_CPU_CRITICAL,
            ),
        )

        with self._lock:
            if pressure > 0.0:
                self._level = max(self._level * 0.9, pressure)
            else:
                self._level *= PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def observe_scheduler_lag(self, delay_seconds: float) -> None:
        """Record a delayed playback-loop wakeup as contention evidence."""
        self._observe_pressure(
            self._pressure(
                delay_seconds,
                PLAYBACK_CONTENTION_LAG_START_SECONDS,
                PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
            )
        )

    def _observe_pressure(self, pressure: float) -> None:
        """Update the smoothed contention level from one pressure sample."""
        pressure = max(0.0, min(1.0, pressure))
        with self._lock:
            if pressure > 0.0:
                self._level = max(self._level * 0.9, pressure)
            else:
                self._level *= PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def observe_write(
        self,
        elapsed_seconds: float,
        block_seconds: float,
        underflowed: bool,
    ) -> None:
        """Record one output write and any PortAudio-reported underflow."""
        with self._lock:
            if underflowed:
                self._underflows += 1
                self._level = 1.0
            else:
                write_pressure = self._pressure(
                    max(0.0, elapsed_seconds - block_seconds),
                    PLAYBACK_CONTENTION_LAG_START_SECONDS,
                    PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
                )
                if write_pressure > 0.0:
                    self._level = max(self._level * 0.9, write_pressure)
                else:
                    self._level *= PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def target_seconds(self) -> float:
        """Return the current reserve target, rising as contention increases."""
        with self._lock:
            max_seconds = PLAYBACK_BUFFER_MAX_SECONDS + (
                (PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS - PLAYBACK_BUFFER_MAX_SECONDS)
                * self._level
            )
            return PLAYBACK_BUFFER_MIN_SECONDS + (
                (max_seconds - PLAYBACK_BUFFER_MIN_SECONDS) * self._level
            )

    def capacity_seconds(self) -> float:
        """Return the maximum reserve allowed at the current contention level."""
        with self._lock:
            return PLAYBACK_BUFFER_MAX_SECONDS + (
                (PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS - PLAYBACK_BUFFER_MAX_SECONDS)
                * self._level
            )

    def requires_rebuffer(self) -> bool:
        """Return whether contention is high enough to pause for more reserve."""
        with self._lock:
            return self._level >= PLAYBACK_CONTENTION_REBUFFER_LEVEL
