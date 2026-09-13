# SPDX-License-Identifier: Apache-2.0
"""Playback pipeline helpers."""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import pathlib
import queue
import subprocess
import sys
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional, Union, cast
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
from uuid import uuid4

import numpy as np

from .constants import APP_NAME, BASE_SR, PipelineStates
from .dataclasses.pipeline import PlaybackChunk, PlaybackSourceDone, SpeechTiming
from .i18n import string
from .paths import project_root, running_compiled, temp_data_dir
from .pipelinecore import (
    _LEGACY_BUFFER_SECONDS,
    _MAX_YOUTUBE_DOWNLOAD_RETRIES,
    _PIPELINE_CPU_MAX_BUFFER_SECONDS,
    _PIPELINE_CPU_MAX_DRAIN_ITEMS,
    _PIPELINE_CPU_YIELD_SECONDS,
    _PLAYBACK_TRACE_INTERVAL_SECONDS,
    _SFX_DUCK_FADE_SECONDS,
    _SFX_DUCK_GAIN,
    _SMART_BUFFER_COMPLETE_BELOW_SPEED,
    _SMART_BUFFER_MAX_SECONDS,
    _SMART_BUFFER_MIN_SECONDS,
    _SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS,
    _SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS,
    _SMART_BUFFER_REALTIME_SPEED,
    _SMART_BUFFER_SMOOTHING,
    _monotonic_time,
)
from .typing.aliases import AudioChunk, AudioChunks
from .typing.common import JSONSerializable
from .typing.locks import (
    ComponentBusyResult,
    ComponentLockAcquisition,
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from .typing.pipeline import SpeechStreamQueue
from .binding import install_class_functions, install_module_functions
from .utils import available

if TYPE_CHECKING:
    from .celune import Celune

__all__ = (
    "_acquire_pipeline",
    "_active_speech_source_ids",
    "_apply_source_gain",
    "_clear_playback_source_status",
    "_component_busy_message",
    "_config_float",
    "_config_lines",
    "_config_text",
    "_config_value_lines",
    "_dequeue_playback_item",
    "_download_youtube_sfx",
    "_flush_buffered_speech_chunks",
    "_is_youtube_sfx_url",
    "_next_playback_source_id",
    "_notify_caption_timing",
    "_notify_component_busy",
    "_notify_speech_playback_finished",
    "_pipeline_cpu_config",
    "_pipeline_requirements",
    "_playback_source_meta",
    "_playback_source_statuses",
    "_playback_trace",
    "_queue_playback_chunk",
    "_queue_playback_done",
    "_register_overlay_playback",
    "_register_overlay_playback_state",
    "_register_playback_source",
    "_release_pipeline",
    "_remember_smart_buffer_speed",
    "_safe_config_int",
    "_set_playback_source_status",
    "_smart_buffer_config",
    "_smart_buffer_speed_estimate",
    "_smart_buffer_target_seconds",
    "_summarize_youtube_download_error",
    "_update_playback_progress",
    "_youtube_download_options",
    "_youtube_sfx_temp_path",
    "_youtube_sfx_title",
    "acquire_pipeline",
    "acquire_pipeline_result",
    "current_playback_status",
    "release_pipeline",
)


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


def _notify_component_busy(
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


def acquire_pipeline_result(
    engine: Celune,
    action: str,
    owner: Optional[ComponentLockOwner] = None,
) -> ComponentLockAcquisition:
    """Atomically claim the legacy pipeline and typed component resources."""
    resolved_owner = owner or ComponentLockOwner(
        operation_id=f"pipeline:{action}:{uuid4().hex}",
    )
    requirements = _pipeline_requirements(action)
    with engine.say_lock:
        engine.log(
            f"[LOCK] acquire requested by {action}, locked={engine.locked}",
            loglevel="verbose",
        )
        manager = getattr(engine, "component_locks", None)
        if engine.locked:
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
            _notify_component_busy(engine, action, busy)
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
                engine.cur_state = "idle"
        engine.log("[LOCK] released", loglevel="verbose")
        engine.log(
            "[LOCK] release complete "
            f"playback_done={engine.playback_done.is_set()} "
            f"text_queue={engine.text_queue.qsize()} "
            f"audio_queue={engine.audio_queue.qsize()}",
            loglevel="debug",
        )


def _acquire_pipeline(self: Celune, action: str) -> bool:
    """Atomically claim Celune's shared playback pipeline."""
    return acquire_pipeline(self, action)


def _release_pipeline(self: Celune) -> None:
    """Release Celune's shared playback pipeline."""
    release_pipeline(self)


def _next_playback_source_id(engine: Celune) -> int:
    """Return the next monotonically increasing playback source id."""
    source_id = getattr(engine, "_next_playback_source_id", 0) + 1
    engine._next_playback_source_id = source_id
    return source_id


def _register_overlay_playback(engine: Celune) -> None:
    """Mark the mixer busy for a newly queued non-speech playback source."""
    _register_overlay_playback_state(engine, reset_ready_announcement=True)


def _register_overlay_playback_state(
    engine: Celune,
    *,
    reset_ready_announcement: bool,
) -> None:
    """Mark the mixer busy for overlay playback with optional ready reset."""
    with engine.say_lock:
        if not engine.locked:
            engine.cur_state = "speaking"
        engine.playback_done.clear()
        if reset_ready_announcement:
            engine._ready_announced = False


def _playback_source_statuses(engine: Celune) -> dict[int, str]:
    """Return the mutable per-source playback status map."""
    statuses = getattr(engine, "_playback_source_statuses", None)
    if isinstance(statuses, dict):
        return statuses

    statuses = {}
    engine._playback_source_statuses = statuses
    return statuses


def current_playback_status(engine: Celune) -> Optional[str]:
    """Return the most recently registered status for an active playback source."""
    statuses = _playback_source_statuses(engine)
    try:
        return next(reversed(statuses.values()), None)
    except RuntimeError:
        return None


def _playback_source_meta(
    engine: Celune,
) -> dict[int, dict[str, Union[str, float]]]:
    """Return per-source mixer metadata such as kind, gain state, and progress."""
    meta = getattr(engine, "_playback_source_meta", None)
    if isinstance(meta, dict):
        return meta

    meta = {}
    engine._playback_source_meta = meta
    return meta


def _register_playback_source(
    engine: Celune,
    source_id: int,
    *,
    kind: str,
    base_gain: float = 1.0,
) -> None:
    """Register one playback source for status and gain management."""
    clipped = float(np.clip(base_gain, 0.0, 1.0))
    _playback_source_meta(engine)[source_id] = {
        "kind": kind,
        "base_gain": clipped,
        "current_gain": clipped,
        "total_frames": 0.0,
        "played_frames": 0.0,
        "generation": float(getattr(engine, "_playback_generation", 0)),
    }


def _set_playback_source_status(engine: Celune, source_id: int, status: str) -> None:
    """Record and surface the current status for one active playback source."""
    statuses = _playback_source_statuses(engine)
    statuses[source_id] = status
    engine.status_callback(status)


def _clear_playback_source_status(engine: Celune, source_id: int) -> None:
    """Forget one playback-source status and restore the next active status."""
    statuses = _playback_source_statuses(engine)
    statuses.pop(source_id, None)
    if statuses:
        engine.status_callback(next(reversed(statuses.values())))
    _playback_source_meta(engine).pop(source_id, None)


def _notify_speech_playback_finished(engine: Celune, source_id: int) -> None:
    """Complete caption progress when one speech source drains independently."""
    source_meta = _playback_source_meta(engine).get(source_id)
    if not isinstance(source_meta, dict) or source_meta.get("kind") != "speech":
        return

    caption_progress_callback = getattr(engine, "caption_progress_callback", None)
    if callable(caption_progress_callback):
        total_frames = max(1.0, float(source_meta.get("total_frames", 0.0)))
        caption_progress_callback(total_frames, total_frames)


def _queue_playback_chunk(
    engine: Celune,
    source_id: int,
    audio: AudioChunk,
    sample_rate: int,
    timing: Optional[SpeechTiming] = None,
    generation: Optional[int] = None,
) -> bool:
    """Queue one chunk for the shared DSP playback mixer."""
    with engine.queue_lock:
        active_playback_generation = getattr(engine, "_playback_generation", 0)
        expected_generation = (
            active_playback_generation if generation is None else generation
        )
        if expected_generation != active_playback_generation:
            return False

        active_generation = getattr(engine, "_active_speech_generation", None)
        if active_generation is not None and (
            active_generation
            != getattr(engine, "_speech_generation", active_generation)
            or engine.utterance_force_stop.is_set()
        ):
            return False

        meta = _playback_source_meta(engine).get(source_id)
        if isinstance(meta, dict):
            if float(meta.get("generation", 0.0)) != float(
                getattr(engine, "_playback_generation", 0)
            ):
                return False
            meta["total_frames"] = float(meta.get("total_frames", 0.0)) + float(
                len(audio)
            )

    queue_wait_started_at = _monotonic_time()
    engine.audio_queue.put(
        PlaybackChunk(
            source_id=source_id,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            timing=timing,
            generation=expected_generation,
        )
    )
    enqueued_at = _monotonic_time()
    queue_wait_seconds = max(0.0, enqueued_at - queue_wait_started_at)
    with engine.queue_lock:
        last_queued_by_source = engine._playback_chunk_last_queued_at
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


def _dequeue_playback_item(
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
                source_meta = _playback_source_meta(engine).get(pending.source_id)
                if (
                    isinstance(source_meta, dict)
                    and source_meta.get("kind") == "speech"
                ):
                    speech_chunk_index = index
                    break
            elif isinstance(pending, PlaybackSourceDone):
                source_meta = _playback_source_meta(engine).get(pending.source_id)
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
    timestamp = _monotonic_time() if now is None else now
    last_trace_at = float(getattr(engine, "_playback_trace_last_logged_at", 0.0))
    if timestamp - last_trace_at < _PLAYBACK_TRACE_INTERVAL_SECONDS:
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


def _update_playback_progress(
    engine: Celune,
    source_buffers: dict[int, deque[tuple[AudioChunk, Optional[SpeechTiming]]]],
) -> None:
    """Reflect the active playback source position in the shared progress bar."""
    if not source_buffers:
        return

    meta = _playback_source_meta(engine)
    active_ids = [source_id for source_id in source_buffers if source_id in meta]
    if not active_ids:
        return

    source_id = max(active_ids)
    source_meta = meta.get(source_id)
    if not isinstance(source_meta, dict):
        return

    total_frames = float(source_meta.get("total_frames", 0.0))
    played_frames = float(source_meta.get("played_frames", 0.0))
    if total_frames <= 0.0:
        return

    now = _monotonic_time()
    last_emit_at = float(getattr(engine, "_playback_progress_last_emit_at", 0.0))
    last_source_id = getattr(engine, "_playback_progress_last_source_id", None)
    emit_interval = 0.08
    if last_source_id == source_id and (now - last_emit_at) < emit_interval:
        return

    engine._playback_progress_last_emit_at = now
    engine._playback_progress_last_source_id = source_id
    engine.progress_callback(min(played_frames, total_frames), total_frames)

    speech_ids = [
        active_source_id
        for active_source_id in active_ids
        if meta.get(active_source_id, {}).get("kind") == "speech"
    ]
    if not speech_ids:
        return
    speech_meta = meta.get(max(speech_ids))
    if not isinstance(speech_meta, dict):
        return
    caption_progress_callback = getattr(engine, "caption_progress_callback", None)
    if callable(caption_progress_callback):
        caption_progress_callback(
            min(
                float(speech_meta.get("played_frames", 0.0)),
                float(speech_meta.get("total_frames", 0.0)),
            ),
            float(speech_meta.get("total_frames", 0.0)),
        )


def _active_speech_source_ids(
    source_buffers: dict[int, deque[tuple[AudioChunk, Optional[SpeechTiming]]]],
    engine: Celune,
) -> set[int]:
    """Return active speech-source ids that should trigger SFX ducking."""
    meta = _playback_source_meta(engine)
    return {
        source_id
        for source_id in source_buffers
        if meta.get(source_id, {}).get("kind") == "speech"
    }


def _apply_source_gain(
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
        _playback_source_meta(engine).get(source_id)
        if source_meta is None
        else source_meta
    )
    if not isinstance(meta, dict):
        return audio

    kind = str(meta.get("kind", "sfx"))
    base_gain = float(meta.get("base_gain", 1.0))
    current_gain = float(meta.get("current_gain", base_gain))
    if kind == "sfx":
        target_gain = base_gain * (_SFX_DUCK_GAIN if speech_active else 1.0)
    else:
        target_gain = base_gain

    if abs(target_gain - current_gain) < 1e-6:
        meta["current_gain"] = target_gain
        return np.asarray(audio * target_gain, dtype=np.float32)

    fade_ratio = min(1.0, block_seconds / _SFX_DUCK_FADE_SECONDS)
    next_gain = current_gain + (target_gain - current_gain) * fade_ratio
    ramp = np.linspace(current_gain, next_gain, len(audio), dtype=np.float32)
    meta["current_gain"] = next_gain
    return np.asarray(audio * ramp[:, None], dtype=np.float32)


def _queue_playback_done(
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
        active_playback_generation = getattr(engine, "_playback_generation", 0)
        expected_generation = (
            active_playback_generation if generation is None else generation
        )
        if expected_generation != active_playback_generation:
            return False

        active_generation = getattr(engine, "_active_speech_generation", None)
        if active_generation is not None and (
            active_generation
            != getattr(engine, "_speech_generation", active_generation)
            or engine.utterance_force_stop.is_set()
        ):
            return False

        source_meta = _playback_source_meta(engine).get(source_id)
        if isinstance(source_meta, dict) and float(
            source_meta.get("generation", 0.0)
        ) != float(getattr(engine, "_playback_generation", 0)):
            return False

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
        last_queued_at = getattr(engine, "_playback_chunk_last_queued_at", None)
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


def _flush_buffered_speech_chunks(
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
        queued = _queue_playback_chunk(
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
            caption_callback(caption_text)
        _set_playback_source_status(engine, source_id, string("status.speaking"))
        engine.cur_state = "speaking"
        engine.queue_avail_callback()
        return True

    return pushed_audio


def _notify_caption_timing(
    engine: Celune,
    caption: str,
    audio: AudioChunk,
    sample_rate: int,
    timing_text: str,
) -> None:
    """Send display and synthesis text to caption timing callbacks compatibly."""
    caption_timing_callback = getattr(engine, "caption_timing_callback", None)
    if not callable(caption_timing_callback):
        return

    try:
        parameters = tuple(
            inspect.signature(caption_timing_callback).parameters.values()
        )
    except (TypeError, ValueError):
        parameters = ()

    supports_timing_text = (
        not parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        or sum(
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        )
        >= 4
    )
    if supports_timing_text:
        caption_timing_callback(caption, audio, sample_rate, timing_text)
        return
    caption_timing_callback(caption, audio, sample_rate)


def _youtube_sfx_temp_path() -> pathlib.Path:
    """Return the fixed temporary WAV path used for URL-backed SFX playback."""
    return temp_data_dir(create=True) / "temporary_audio.wav"


def _is_youtube_sfx_url(value: str) -> bool:
    """Return whether ``value`` looks like a supported YouTube URL."""
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.netloc or "").lower().removeprefix("www.")
    return host in {"youtube.com", "youtu.be", "music.youtube.com"}


def _youtube_sfx_title(url: str) -> str:
    """Return a friendly title for one YouTube URL when available."""
    query = urlencode({"url": url, "format": "json"})
    endpoint = f"https://www.youtube.com/oembed?{query}"
    # noinspection PyBroadException
    try:
        with urlopen(endpoint, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return "YouTube audio"

    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return "YouTube audio"


def _summarize_youtube_download_error(output: str) -> str:
    """Extract the actionable reason from yt-dlp output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    ignored_markers = (
        "no supported javascript runtime",
        "only deno is enabled",
        "youtube extraction without a js runtime",
        "github.com/yt-dlp/yt-dlp/wiki/ejs",
    )

    for line in reversed(lines):
        if line.upper().startswith("ERROR:"):
            reason = line.split(":", 1)[1].strip()
            if reason:
                return reason

    for line in reversed(lines):
        if line.lower().startswith("warning:"):
            continue
        if not any(marker in line.lower() for marker in ignored_markers):
            return line

    return string("pipeline.download_unknown_error")


def _download_youtube_sfx(
    engine: Celune, url: str
) -> Optional[tuple[pathlib.Path, str]]:
    """Download one YouTube URL as a temporary WAV file for SFX playback."""
    yt_dlp_module = "yt_dlp"
    if not available(yt_dlp_module):
        engine.log(string("pipeline.yt_dlp_missing"), "warning")
        engine.error_callback(string("pipeline.yt_dlp_required"))
        return None

    output_path = _youtube_sfx_temp_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        output_path.unlink(missing_ok=True)

    title = _youtube_sfx_title(url)
    out_tmpl = str(output_path.with_suffix(".%(ext)s"))
    engine.status_callback(string("status.downloading_audio"))
    engine.log(f"[SFX] Downloading audio from {url}...")
    python_executable = sys.executable
    if running_compiled():
        if os.name == "nt":
            python_executable = str(project_root() / ".venv" / "Scripts" / "python.exe")
        else:
            python_executable = str(project_root() / ".venv" / "bin" / "python")
    command = [
        python_executable,
        "-m",
        yt_dlp_module,
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--no-progress",
        "--force-overwrites",
        *_youtube_download_options(engine),
        "--output",
        out_tmpl,
        url,
    ]
    failure_reason = string("pipeline.download_unknown_error")
    total_attempts = _MAX_YOUTUBE_DOWNLOAD_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        with contextlib.suppress(OSError):
            output_path.unlink(missing_ok=True)

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            failure_reason = string("pipeline.download_timeout")
        else:
            output = "\n".join(
                part for part in (completed.stderr, completed.stdout) if part
            )
            failure_reason = _summarize_youtube_download_error(output)
            if completed.returncode == 0 and output_path.exists():
                return output_path, title
            if completed.returncode == 0 and not output_path.exists():
                failure_reason = string("pipeline.downloader_no_file")

        if attempt < total_attempts:
            engine.log(
                f"{string('pipeline.download_failed')}: {failure_reason}; "
                f"{string('pipeline.download_retry', retry_count=attempt, max_retries=_MAX_YOUTUBE_DOWNLOAD_RETRIES)}",
                "warning",
            )
            continue

        engine.log(f"{string('pipeline.download_failed')}: {failure_reason}", "warning")
        engine.error_callback(string("pipeline.download_youtube_failed_short"))
        return None

    return None


def _config_text(engine: Celune, key: str, default: str) -> str:
    """Read a string configuration value with a fallback."""
    value = engine.config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()

    return default


def _config_lines(engine: Celune, key: str) -> tuple[str, ...]:
    """Read a text or text-list configuration value as non-empty lines."""
    return _config_value_lines(engine.config.get(key))


def _config_value_lines(value: JSONSerializable) -> tuple[str, ...]:
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


def _youtube_download_options(engine: Celune) -> list[str]:
    """Build optional yt-dlp arguments from the nested YouTube configuration."""
    value = engine.config.get("youtube")
    if not isinstance(value, dict):
        return []

    options: list[str] = []
    cookies_file = _config_value_lines(value.get("cookies_file"))
    cookies_from_browser = _config_value_lines(value.get("cookies_from_browser"))
    if cookies_file:
        options.extend(("--cookies", cookies_file[0]))
    elif cookies_from_browser:
        options.extend(("--cookies-from-browser", cookies_from_browser[0]))

    for key, option in (
        ("js_runtimes", "--js-runtimes"),
        ("remote_components", "--remote-components"),
    ):
        for configured_value in _config_value_lines(value.get(key)):
            options.extend((option, configured_value))

    po_tokens = _config_value_lines(value.get("po_token"))
    if po_tokens:
        options.extend(("--extractor-args", f"youtube:po_token={','.join(po_tokens)}"))

    player_clients = _config_value_lines(value.get("player_client"))
    if player_clients:
        options.extend(
            ("--extractor-args", f"youtube:player_client={','.join(player_clients)}")
        )

    for extractor_arg in _config_value_lines(value.get("extractor_args")):
        options.extend(("--extractor-args", extractor_arg))
    return options


def _config_float(
    source: Mapping[str, JSONSerializable], key: str, default: float
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
    source: Mapping[str, JSONSerializable], key: str, default: int
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
    del engine
    return (
        True,
        _SMART_BUFFER_REALTIME_SPEED,
        _SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS,
        _SMART_BUFFER_MIN_SECONDS,
        _SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS,
        _SMART_BUFFER_MAX_SECONDS,
        _SMART_BUFFER_COMPLETE_BELOW_SPEED,
    )


def _pipeline_cpu_config(engine: Celune) -> tuple[bool, float, int, float]:
    """Return Celune's bounded cooperative playback-pressure controls."""
    del engine
    return (
        True,
        _PIPELINE_CPU_MAX_BUFFER_SECONDS,
        _PIPELINE_CPU_MAX_DRAIN_ITEMS,
        _PIPELINE_CPU_YIELD_SECONDS,
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


def _smart_buffer_target_seconds(
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
        return _LEGACY_BUFFER_SECONDS

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


def _remember_smart_buffer_speed(engine: Celune, generation_speed: float) -> None:
    """Update the engine's rolling generation-speed estimate."""
    if generation_speed <= 0.0:
        return

    previous = getattr(engine, "smart_buffer_generation_speed", None)
    if isinstance(previous, (int, float)) and previous > 0.0:
        generation_speed = (float(previous) * (1.0 - _SMART_BUFFER_SMOOTHING)) + (
            generation_speed * _SMART_BUFFER_SMOOTHING
        )
    engine.smart_buffer_generation_speed = generation_speed


def install(target):
    """Install extracted definitions in the original module."""
    install_module_functions(target, {name: globals()[name] for name in __all__})


def install_engine(target):
    """Install playback-lock entrypoints on ``Celune``."""
    install_class_functions(
        target,
        {
            "_acquire_pipeline": _acquire_pipeline,
            "_release_pipeline": _release_pipeline,
        },
    )
