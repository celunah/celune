# SPDX-License-Identifier: MIT
"""Speech pipeline helpers."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import pathlib
import queue
import random
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from importlib import util as importlib_util
from typing import TYPE_CHECKING, Optional, Union, cast
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import numpy as np
import pyrubberband as rb
import sounddevice as sd
import soundfile as sf
import torch
from iso639 import Lang
from iso639.exceptions import DeprecatedLanguageValue, InvalidLanguageValue

from . import __version__
from .analysis import analyze_voice_audio
from .cevoice import (
    bundle_character_name,
    default_loader,
    persona_files_from_bundle,
    persona_metadata_from_manifest,
)
from .config import resolve_audio_device
from .constants import (
    APP_NAME,
    APP_SLUG,
    BASE_SR,
    PERSONA_EMOTION_MODEL,
    PERSONA_MEMORY_EMBEDDING_MODEL,
    PipelineStates,
)
from .dataclasses.pipeline import (
    AudioInputRequest,
    AudioOutput,
    PlaybackChunk,
    PlaybackSourceDone,
    SpeechRequest,
    SpeechTiming,
    VoiceConversionRequest,
)
from .dsp import (
    error_signal,
    is_silent_utterance,
    pitch_shift_audio,
    readiness_signal,
    resample_audio,
    sleeping_signal,
    soften,
    split,
    to_48khz,
    working_signal,
)
from .exceptions import NotAvailableError
from .i18n import string
from .paths import (
    app_data_dir,
    outputs_dir,
    project_root,
    running_compiled,
)
from .persona.emotion import PersonaEmotionAnalyzer
from .persona.impl import (
    compact_persona_history,
    default_persona_age,
    default_persona_context,
    default_persona_gender,
    default_persona_persona,
    pack_identity_text,
    pack_persona_lines,
    pack_persona_text,
    persona_active_character_name,
    persona_config,
    persona_debug_overrides_enabled,
    persona_history_messages,
    persona_model_id,
    persona_pending_attachments,
    persona_quantization,
    persona_session_summary,
    persona_style_traits,
)
from .persona.memory import PersonaMemoryStore, classifier_memory_candidates
from .persona.paths import persona_override_files
from .persona.prompts import (
    CharacterProfile,
    PersonaCard,
    PersonaContext,
    PersonaPromptBuilder,
    PersonaSourceMaterial,
    RetrievedMemoryBundle,
)
from .typing.aliases import AudioChunk, AudioChunks
from .typing.common import JSON, JSONSerializable
from .typing.pipeline import SpeechStreamQueue
from .utils import (
    detect_language,
    discard,
    format_error,
    format_number,
    is_april_fools,
    rng_replace,
    run_async,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .celune import Celune
    from .typing.persona import PersonaClientResponse

_FLAC_MAGIC = b"fLaC"
_FLAC_STREAMINFO_BLOCK = 0
_FLAC_VORBIS_COMMENT_BLOCK = 4
_MAX_FLAC_METADATA_BLOCK_SIZE = 0xFFFFFF
_SFX_DUCK_GAIN = 0.25
_SFX_DUCK_FADE_SECONDS = 0.15
_LEGACY_BUFFER_SECONDS = 10.0
_SMART_BUFFER_REALTIME_SPEED = 1.05


def _monotonic_time() -> float:
    """Return the current monotonic clock value for pipeline timing."""
    return time.monotonic()


_SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS = 20.0
_SMART_BUFFER_MIN_SECONDS = 0.35
_SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS = 0.75
_SMART_BUFFER_MAX_SECONDS = 20.0
_SMART_BUFFER_COMPLETE_BELOW_SPEED = 0.5
_SMART_BUFFER_SMOOTHING = 0.35
_MAX_SILENT_UTTERANCE_RETRIES = 3
_MAX_YOUTUBE_DOWNLOAD_RETRIES = 3
_MEMORY_CLASSIFIER_SYSTEM_PROMPT = """You classify durable user facts for long-term memory.
Return JSON only in this exact shape:
{"memories":[{"content":"...","importance":1,"confidence":0.0}]}

Only include stable facts about the user that would help in a future conversation:
preferences, identity, recurring constraints, projects, goals, and important life context.
Do not include assistant statements, temporary requests, jokes, guesses, passwords, secrets,
tokens, financial details, medical details, or unrelated conversation. If there is no durable
fact, return {"memories":[]}.
"""
_PIPELINE_CPU_MAX_BUFFER_SECONDS = 4.0
_PIPELINE_CPU_MAX_DRAIN_ITEMS = 1
_PIPELINE_CPU_YIELD_SECONDS = 0.001


def _json_value(value: JSONSerializable) -> JSONSerializable:
    """Return a value only when it is already JSON-compatible."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list) and all(_is_json_value(item) for item in value):
        return cast(list[JSONSerializable], value)
    if isinstance(value, dict) and all(
        isinstance(key, str) and _is_json_value(item) for key, item in value.items()
    ):
        return cast(dict[str, JSONSerializable], value)
    return None


def _is_json_value(value: JSONSerializable) -> bool:
    """Return whether a value can be stored in Celune JSON metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _celune_metadata_payload(
    engine: Celune,
    *,
    text: str,
    display_text: str,
    generation_params: Mapping[str, JSONSerializable],
    sample_rate: int,
    subtype: str,
    included_kept_sfx: bool,
) -> JSON:
    """Build the Celune generation metadata payload."""
    return {
        "format": "CEMETA",
        "format_version": 1,
        "celune_version": __version__,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "text": text,
        "display_text": display_text,
        "backend": _json_value(getattr(engine, "tts_backend", None)),
        "qwen3_x_vector_only": _json_value(
            getattr(engine.backend, "x_vector_only", None)
        ),
        "model_name": _json_value(getattr(engine, "model_name", "")),
        "voice": _json_value(getattr(engine, "current_voice", None)),
        "voice_prompt": _json_value(getattr(engine, "voice_prompt", None)),
        "language": _json_value(getattr(engine, "language", None)),
        "chunk_size": _json_value(getattr(engine, "chunk_size", None)),
        "speed": _json_value(getattr(engine, "speed", None)),
        "reverb_strength": _json_value(getattr(engine.reverb, "strength", None)),
        "use_normalizer": _json_value(getattr(engine, "use_normalization", None)),
        "sample_rate": sample_rate,
        "subtype": subtype,
        "included_kept_sfx": included_kept_sfx,
        "generation": dict(generation_params),
    }


def _valid_vorbis_comment_key(key: str) -> bool:
    """Return whether ``key`` is a valid Vorbis comment field name."""
    return (
        bool(key) and "=" not in key and all(0x20 <= ord(char) <= 0x7D for char in key)
    )


def _read_vorbis_string(payload: bytes, offset: int) -> tuple[bytes, int]:
    """Read one little-endian length-prefixed Vorbis comment string."""
    if offset + 4 > len(payload):
        raise ValueError("truncated Vorbis comment")

    length = int.from_bytes(payload[offset : offset + 4], "little")
    offset += 4
    end = offset + length
    if end > len(payload):
        raise ValueError("truncated Vorbis comment")

    return payload[offset:end], end


def _parse_vorbis_comment_block(payload: bytes) -> tuple[bytes, list[tuple[str, str]]]:
    """Parse a Vorbis comment block into a vendor string and field pairs."""
    vendor, offset = _read_vorbis_string(payload, 0)
    if offset + 4 > len(payload):
        raise ValueError("truncated Vorbis comment list")

    comment_count = int.from_bytes(payload[offset : offset + 4], "little")
    offset += 4
    comments: list[tuple[str, str]] = []
    for _ in range(comment_count):
        raw_comment, offset = _read_vorbis_string(payload, offset)
        decoded = raw_comment.decode("utf-8", errors="replace")
        key, separator, value = decoded.partition("=")
        if separator and _valid_vorbis_comment_key(key):
            comments.append((key, value))

    return vendor, comments


def _encode_vorbis_comment_block(
    vendor: bytes, comments: list[tuple[str, str]]
) -> bytes:
    """Encode Vorbis comments into a FLAC metadata block payload."""
    payload = bytearray()
    payload.extend(len(vendor).to_bytes(4, "little"))
    payload.extend(vendor)
    payload.extend(len(comments).to_bytes(4, "little"))
    for key, value in comments:
        raw_comment = f"{key}={value}".encode()
        payload.extend(len(raw_comment).to_bytes(4, "little"))
        payload.extend(raw_comment)

    return bytes(payload)


def _flac_metadata_blocks(data: bytes) -> tuple[list[tuple[int, bytes]], int]:
    """Return FLAC metadata blocks and the byte offset where audio frames start."""
    if not data.startswith(_FLAC_MAGIC):
        raise ValueError("not a FLAC file")

    offset = len(_FLAC_MAGIC)
    blocks: list[tuple[int, bytes]] = []
    while True:
        if offset + 4 > len(data):
            raise ValueError("truncated FLAC metadata")

        header = data[offset]
        block_type = header & 0x7F
        block_length = int.from_bytes(data[offset + 1 : offset + 4], "big")
        offset += 4
        end = offset + block_length
        if end > len(data):
            raise ValueError("truncated FLAC metadata")

        blocks.append((block_type, data[offset:end]))
        offset = end
        if header & 0x80:
            return blocks, offset


def _encode_flac_metadata_blocks(blocks: list[tuple[int, bytes]]) -> bytes:
    """Encode FLAC metadata blocks with the final-block flag repaired."""
    encoded = bytearray(_FLAC_MAGIC)
    for index, (block_type, payload) in enumerate(blocks):
        if len(payload) > _MAX_FLAC_METADATA_BLOCK_SIZE:
            raise ValueError("FLAC metadata block is too large")

        final_flag = 0x80 if index == len(blocks) - 1 else 0
        encoded.append(final_flag | block_type)
        encoded.extend(len(payload).to_bytes(3, "big"))
        encoded.extend(payload)

    return bytes(encoded)


def _stringify_flac_metadata(value: JSONSerializable) -> str:
    """Convert an arbitrary metadata value into a Vorbis comment value."""
    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_flac_metadata(path: str, tags: JSON) -> None:
    """Write arbitrary valid FLAC Vorbis comment tags to ``path``."""
    valid_tags = {
        key: _stringify_flac_metadata(value)
        for key, value in tags.items()
        if _valid_vorbis_comment_key(key)
    }
    if not valid_tags:
        return

    path_obj = pathlib.Path(path)
    data = path_obj.read_bytes()
    blocks, audio_offset = _flac_metadata_blocks(data)
    audio_data = data[audio_offset:]

    comment_index: Optional[int] = None
    vendor = f"{APP_NAME} {__version__}".encode()
    comments: list[tuple[str, str]] = []
    for index, (block_type, payload) in enumerate(blocks):
        if block_type == _FLAC_VORBIS_COMMENT_BLOCK:
            comment_index = index
            vendor, comments = _parse_vorbis_comment_block(payload)
            break

    replaced_keys = {key.casefold() for key in valid_tags}
    comments = [
        (key, value) for key, value in comments if key.casefold() not in replaced_keys
    ]
    comments.extend(valid_tags.items())
    vorbis_payload = _encode_vorbis_comment_block(vendor, comments)

    if comment_index is None:
        insert_index = 1 if blocks and blocks[0][0] == _FLAC_STREAMINFO_BLOCK else 0
        blocks.insert(insert_index, (_FLAC_VORBIS_COMMENT_BLOCK, vorbis_payload))
    else:
        blocks[comment_index] = (_FLAC_VORBIS_COMMENT_BLOCK, vorbis_payload)

    path_obj.write_bytes(_encode_flac_metadata_blocks(blocks) + audio_data)


def _write_celune_flac(
    engine: Celune,
    path: str,
    audio: AudioChunk,
    sample_rate: int,
    subtype: str,
    metadata: JSON,
) -> None:
    """Write a FLAC file with Celune metadata in Vorbis comments."""
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    with sf.SoundFile(
        path,
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        format="FLAC",
        subtype=subtype,
    ) as audio_file:
        audio_file.write(audio)

    created_at = metadata.get(
        "created_at", datetime.datetime.now(datetime.UTC).isoformat()
    )
    display_text = metadata.get("display_text")

    if not isinstance(display_text, str):
        display_text = f"{APP_NAME} speech from {created_at}"

    prompt = display_text.split()
    words = " ".join(prompt[:5])
    if len(prompt) > 5:
        words += "..."

    tags: JSON = {
        "encoder": f"{APP_NAME} {__version__}",
        "artist": engine.current_character or APP_NAME,
        "album": f"{APP_NAME} via {engine.backend.name}",
        "title": words,
        "comment": encoded,
        "created_at": created_at,
        "date": datetime.datetime.now(datetime.UTC).year,
    }
    _write_flac_metadata(path, tags)


def _saved_output_speech_seconds() -> float:
    """Return cumulative saved speech duration from Celune-generated output files."""
    output_dir = outputs_dir()
    if not output_dir.exists():
        return 0.0

    total_seconds = 0.0
    pattern = f"{APP_SLUG}_speech_*.flac"
    for path in output_dir.glob(pattern):
        try:
            total_seconds += float(sf.info(path).duration)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

    return total_seconds


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

    engine.log(string("pipeline.ttfp_seconds", seconds=format_number(elapsed, 2)))


def close_stream(engine: Celune, abort: bool = False) -> None:
    """Close the current audio stream if one exists.

    Args:
        engine: The Celune engine that owns the audio stream.
        abort: Whether to abort immediately instead of stopping gracefully.
    """
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
    engine.utterance_force_stop.set()

    with engine.queue_lock:
        engine._speech_generation = getattr(engine, "_speech_generation", 0) + 1
        engine._playback_generation = getattr(engine, "_playback_generation", 0) + 1
        clear_queue(engine.text_queue)
        clear_queue(engine.persona_queue)
        clear_queue(engine.audio_queue)
        engine.kept_sfx_audio = None
        engine.audio_queue.put(engine.force_stop_marker)

    return True


def acquire_pipeline(engine: Celune, action: str) -> bool:
    """Atomically claim Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.
        action: A short label describing the action requesting the lock.

    Returns:
        bool: ``True`` when the pipeline was claimed, otherwise ``False``.
    """
    with engine.say_lock:
        engine.log_dev(f"[LOCK] acquire requested by {action}, locked={engine.locked}")
        if engine.locked:
            engine.log(
                string("pipeline.busy_action", action=action, app_name=APP_NAME),
                "warning",
            )
            engine.error_callback(string("celune.app_busy", app_name=APP_NAME))
            return False

        engine.locked = True
        if action != "play readiness signal":
            engine._ready_announced = False
        engine.playback_done.clear()
        engine.log_dev(f"[LOCK] acquired by {action}")
        return True


def release_pipeline(engine: Celune, playback_idle: bool = True) -> None:
    """Release Celune's shared playback pipeline.

    Args:
        engine: The Celune engine that owns the playback pipeline.
        playback_idle: Whether playback should be marked fully idle now.
    """
    with engine.say_lock:
        engine.locked = False
        if playback_idle:
            engine.playback_done.set()
            if engine.cur_state != "error":
                engine.cur_state = "idle"
        engine.log_dev("[LOCK] released")


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

    engine.audio_queue.put(
        PlaybackChunk(
            source_id=source_id,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            timing=timing,
            generation=expected_generation,
        )
    )
    return True


def _dequeue_playback_item(
    engine: Celune,
    prioritize_speech: bool = False,
) -> Union[PlaybackChunk, PlaybackSourceDone, PipelineStates]:
    """Remove one playback item, prioritizing speech overlays when requested."""
    audio_queue = engine.audio_queue
    if not prioritize_speech:
        return audio_queue.get_nowait()

    with audio_queue.mutex:
        if not audio_queue.queue:
            raise queue.Empty

        speech_chunk_index: Optional[int] = None
        speech_done_index: Optional[int] = None
        for index, pending in enumerate(audio_queue.queue):
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
        audio_queue.queue.rotate(-selected_index)
        pending = audio_queue.queue.popleft()
        audio_queue.queue.rotate(selected_index)
        audio_queue.not_full.notify()
        return pending


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
) -> AudioChunk:
    """Apply ducking and smooth gain ramps for one mixer source block."""
    meta = _playback_source_meta(engine).get(source_id)
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
    return True


def _flush_buffered_speech_chunks(
    engine: Celune,
    source_id: int,
    buffer: AudioChunks,
    speech_timing: SpeechTiming,
    pushed_audio: bool,
    stream_queue: Optional[SpeechStreamQueue],
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
        _set_playback_source_status(engine, source_id, string("status.speaking"))
        engine.cur_state = "speaking"
        engine.queue_avail_callback()
        return True

    return pushed_audio


def _youtube_sfx_temp_path() -> pathlib.Path:
    """Return the fixed temporary WAV path used for URL-backed SFX playback."""
    return app_data_dir(create=True) / "temp" / "temporary_audio.wav"


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
    if importlib_util.find_spec(yt_dlp_module) is None:
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
    engine.log(string("pipeline.youtube_download_start", url=url))
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
                string(
                    "pipeline.download_retrying",
                    error=failure_reason,
                    retry_count=attempt,
                    max_retries=_MAX_YOUTUBE_DOWNLOAD_RETRIES,
                ),
                "warning",
            )
            continue

        engine.log(string("pipeline.download_failed", error=failure_reason), "warning")
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
    value = engine.config.get(key)
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
    """Resolve the adaptive speech buffer settings for the current engine."""
    value = engine.config.get("smart_buffer", {})
    config = value if isinstance(value, dict) else {}
    enabled = bool(config.get("enabled", True))
    realtime_speed = max(
        0.1,
        _config_float(config, "realtime_speed", _SMART_BUFFER_REALTIME_SPEED),
    )
    protected_playback_seconds = max(
        0.0,
        _config_float(
            config,
            "protected_playback_seconds",
            _SMART_BUFFER_PROTECTED_PLAYBACK_SECONDS,
        ),
    )
    minimum_seconds = max(
        0.0,
        _config_float(config, "minimum_seconds", _SMART_BUFFER_MIN_SECONDS),
    )
    min_speed_sample_seconds = max(
        0.0,
        _config_float(
            config,
            "min_speed_sample_seconds",
            _SMART_BUFFER_MIN_SPEED_SAMPLE_SECONDS,
        ),
    )
    max_seconds = max(
        0.0,
        _config_float(config, "max_seconds", _SMART_BUFFER_MAX_SECONDS),
    )
    complete_below_speed = max(
        0.0,
        _config_float(
            config,
            "complete_below_speed",
            _SMART_BUFFER_COMPLETE_BELOW_SPEED,
        ),
    )
    return (
        enabled,
        realtime_speed,
        protected_playback_seconds,
        minimum_seconds,
        min_speed_sample_seconds,
        max_seconds,
        complete_below_speed,
    )


def _pipeline_cpu_config(engine: Celune) -> tuple[bool, float, int, float]:
    """Resolve cooperative CPU-pressure controls for the playback pipeline."""
    value = engine.config.get("pipeline_cpu", {})
    config = value if isinstance(value, dict) else {}
    enabled = config.get("enabled", True)
    if isinstance(enabled, bool) and not enabled:
        return False, float("inf"), 128, 0.0

    max_buffer_seconds = max(
        0.25,
        _config_float(
            config,
            "max_buffer_seconds",
            _PIPELINE_CPU_MAX_BUFFER_SECONDS,
        ),
    )
    max_drain_items = max(
        1,
        _safe_config_int(
            config,
            "max_drain_items",
            _PIPELINE_CPU_MAX_DRAIN_ITEMS,
        ),
    )
    yield_seconds = max(
        0.0,
        _config_float(config, "yield_seconds", _PIPELINE_CPU_YIELD_SECONDS),
    )
    return True, max_buffer_seconds, max_drain_items, yield_seconds


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


def build_persona_character_card(engine: Celune) -> str:
    """Build the compact character and persona summary sent with requests.

    Args:
        engine: The instance of Celune to use.

    Returns:
        str: The formatted Persona character card and summary.
    """
    context = build_persona_context(engine, "")
    return f"{context.character_profile.render()}\n\n{context.persona_card.render()}"


def _persona_emotion_analyzer(engine: Celune) -> Optional[PersonaEmotionAnalyzer]:
    """Return the configured Persona emotion analyzer for this engine."""
    existing = getattr(engine, "persona_emotion_analyzer", None)
    if isinstance(existing, PersonaEmotionAnalyzer):
        return existing

    emotion_config = persona_config(engine.config).get("emotion")
    if isinstance(emotion_config, dict):
        enabled = emotion_config.get("enabled", True)
        if isinstance(enabled, bool) and not enabled:
            return None
        model_name = emotion_config.get("model")
        user_weight = emotion_config.get("user_weight", 0.75)
        assistant_weight = emotion_config.get("assistant_weight", 0.25)
        decay_power = emotion_config.get("history_decay_power", 3.0)
        analyzer = PersonaEmotionAnalyzer(
            model_name=model_name.strip()
            if isinstance(model_name, str) and model_name.strip()
            else PERSONA_EMOTION_MODEL,
            user_weight=float(user_weight)
            if isinstance(user_weight, (int, float))
            and not isinstance(user_weight, bool)
            else 0.75,
            assistant_weight=float(assistant_weight)
            if isinstance(assistant_weight, (int, float))
            and not isinstance(assistant_weight, bool)
            else 0.25,
            history_decay_power=float(decay_power)
            if isinstance(decay_power, (int, float))
            and not isinstance(decay_power, bool)
            else 3.0,
        )
    else:
        analyzer = PersonaEmotionAnalyzer()

    setattr(engine, "persona_emotion_analyzer", analyzer)
    return analyzer


def _persona_mood_or_state(
    engine: Celune,
    request: str,
) -> str:
    """Return the Persona state string for the current request."""
    configured_state = _config_text(engine, "persona_state", "")
    if configured_state:
        return configured_state

    analyzer = _persona_emotion_analyzer(engine)
    if analyzer is None:
        return "Neutral."

    summary = analyzer.summarize_history(persona_history_messages(engine), request)
    if summary is None or not summary.target_state.strip():
        emotion_warning = (
            f"Persona emotion analysis fell back to Neutral: {analyzer.last_error}"
            if analyzer.last_error.strip()
            else "Persona emotion analysis fell back to Neutral."
        )
        log_dev = getattr(engine, "log_dev", None)
        if callable(log_dev):
            log_dev(emotion_warning, "warning")
        return "Neutral."
    return summary.target_state


def _persona_memory_store(engine: Celune) -> Optional[PersonaMemoryStore]:
    """Return the configured Persona memory store for this engine."""
    existing = getattr(engine, "persona_memory_store", None)
    if isinstance(existing, PersonaMemoryStore):
        return existing

    memory_config = persona_config(engine.config).get("memory")
    normalized_memory = memory_config if isinstance(memory_config, dict) else {}
    enabled = normalized_memory.get("enabled", True)
    if isinstance(enabled, bool) and not enabled:
        return None

    similarity_threshold = normalized_memory.get("semantic_similarity_threshold", 0.62)
    overlap_threshold = normalized_memory.get("fallback_token_overlap_threshold", 1)
    embedding_model = normalized_memory.get("semantic_embedding_model")
    embedding_model_name = (
        embedding_model.strip()
        if isinstance(embedding_model, str) and embedding_model.strip()
        else None
    )
    configured_storage = normalized_memory.get("storage_dir")
    storage_dir = (
        configured_storage.strip()
        if isinstance(configured_storage, str) and configured_storage.strip()
        else None
    )
    store = PersonaMemoryStore(
        storage_dir=storage_dir,
        semantic_similarity_threshold=float(similarity_threshold)
        if isinstance(similarity_threshold, (int, float))
        and not isinstance(similarity_threshold, bool)
        else 0.62,
        fallback_token_overlap_threshold=int(overlap_threshold)
        if isinstance(overlap_threshold, (int, float))
        and not isinstance(overlap_threshold, bool)
        else 1,
        embedding_model=embedding_model_name or PERSONA_MEMORY_EMBEDDING_MODEL,
    )

    setattr(engine, "persona_memory_store", store)
    return store


def _store_persona_memories(engine: Celune, request: str) -> None:
    """Persist long-term memory candidates extracted from the user request."""
    store = _persona_memory_store(engine)
    if store is None:
        return

    character_name = persona_active_character_name(engine)
    if not character_name.strip():
        return

    store.remember_from_user_message(character_name, request)


def _persona_memory_classifier_context(engine: Celune) -> str:
    """Build the bounded conversation context sent to the memory classifier."""
    sections: list[str] = []
    summary = persona_session_summary(engine)
    if summary:
        sections.append(f"Conversation summary:\n{summary}")

    messages = persona_history_messages(engine)
    if messages:
        sections.append(
            "Recent conversation:\n"
            + "\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
        )
    return "\n\n".join(sections)


def _classify_persona_memories(engine: Celune, request: str) -> None:
    """Classify and persist unmatched durable user facts without blocking reply logic."""
    store = _persona_memory_store(engine)
    if store is None or store.collect_candidates(request):
        return

    memory_config = persona_config(engine.config).get("memory")
    normalized_memory = memory_config if isinstance(memory_config, dict) else {}
    enabled = normalized_memory.get("auto_classifier", True)
    if isinstance(enabled, bool) and not enabled:
        return

    classifier = cast(
        "Optional[Callable[[JSON], PersonaClientResponse]]",
        getattr(getattr(engine, "vision", None), "classify_memory", None),
    )
    if not callable(classifier):
        return

    minimum_confidence = normalized_memory.get("auto_classifier_min_confidence", 0.82)
    if isinstance(minimum_confidence, bool) or not isinstance(
        minimum_confidence, (int, float)
    ):
        minimum_confidence = 0.82
    maximum_candidates = normalized_memory.get("auto_classifier_max_candidates", 3)
    if isinstance(maximum_candidates, bool) or not isinstance(
        maximum_candidates, (int, float)
    ):
        maximum_candidates = 3

    context = _persona_memory_classifier_context(engine)
    if request.strip():
        context = f"{context}\n\nCurrent user message:\n{request.strip()}".strip()

    payload: JSON = {
        "format": "celune_memory_classifier",
        "format_version": 1,
        "model": persona_model_id(engine.config),
        "quantization": persona_quantization(engine.config),
        "quantized": True,
        "system": _MEMORY_CLASSIFIER_SYSTEM_PROMPT,
        "user": context,
        "request": context,
        "messages": [
            {"role": "system", "content": _MEMORY_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        "max_new_tokens": 180,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }

    try:
        response = classifier(payload)
        response.raise_for_status()
        candidates = classifier_memory_candidates(
            _extract_persona_text(response.json()),
            minimum_confidence=float(minimum_confidence),
            maximum_candidates=max(1, int(maximum_candidates)),
        )
        character_name = persona_active_character_name(engine)
        if not character_name.strip():
            return
        for candidate in candidates:
            store.remember(
                character_name,
                candidate.content,
                importance=candidate.importance,
                explicit=False,
            )
    except Exception as error:
        log_dev = getattr(engine, "log_dev", None)
        if callable(log_dev):
            log_dev(
                f"Persona memory classifier failed: {format_error(error, engine.dev)}"
            )


def _build_retrieved_memory_bundle(
    engine: Celune, request: str
) -> RetrievedMemoryBundle:
    """Return retrieved long-term memory for the current request."""
    direct_memories = getattr(engine, "retrieved_long_term_memory", None)
    if isinstance(direct_memories, list):
        memories = [
            memory.strip()
            for memory in direct_memories
            if isinstance(memory, str) and memory.strip()
        ]
        return RetrievedMemoryBundle(memories=tuple(memories))

    store = _persona_memory_store(engine)
    if store is not None:
        character_name = persona_active_character_name(engine)
        memories = tuple(
            record.content
            for record in store.retrieve(character_name, request.strip())
            if record.content.strip()
        )
        if memories:
            return RetrievedMemoryBundle(memories=memories)

    return RetrievedMemoryBundle(
        memories=_config_lines(engine, "persona_long_term_memory")
    )


def _persona_manifest_files(engine: Celune) -> dict[str, str]:
    """Return whitelisted persona Markdown files for the active engine persona."""
    loader = default_loader()
    if loader is None:
        return {}
    pack_persona = persona_metadata_from_manifest(loader.bundle.metadata)
    current_persona = getattr(engine, "current_character_persona", None)
    if pack_persona is not None:
        if current_persona != pack_persona:
            return {}
    else:
        current_character = getattr(engine, "current_character", None)
        bundle_name = bundle_character_name(loader.bundle)
        if not (
            isinstance(current_character, str)
            and isinstance(bundle_name, str)
            and current_character.strip()
            and current_character.strip() == bundle_name.strip()
        ):
            return {}
    files = persona_files_from_bundle(loader.bundle)
    if persona_debug_overrides_enabled(engine.config):
        files.update(persona_override_files(persona_active_character_name(engine)))
    return files


def _legacy_identity_source(profile: CharacterProfile) -> str:
    """Render legacy identity metadata into CECHAR v3-style source material."""
    lines: list[str] = []
    if profile.name.strip():
        lines.append(f"Name: {profile.name.strip()}")
    if profile.age.strip():
        lines.append(f"Age: {profile.age.strip()}")
    if profile.gender.strip():
        lines.append(f"Gender: {profile.gender.strip()}")
    if profile.profile.strip():
        if lines:
            lines.append("")
        lines.append(profile.profile.strip())
    return "\n".join(lines).strip()


def _legacy_personality_source(engine: Celune) -> str:
    """Render legacy persona settings into the v3 personality source slot."""
    blocks: list[str] = []
    persona_text = _config_text(
        engine,
        "persona_persona",
        default_persona_persona(),
    )
    if persona_text:
        blocks.append(persona_text)

    prompt_rules = pack_persona_lines(engine, "prompt_rules")
    if prompt_rules:
        blocks.append("\n".join(f"- {line}" for line in prompt_rules))

    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _legacy_speech_style_source(engine: Celune) -> str:
    """Render legacy speech-style metadata into the v3 speech-style slot."""
    blocks: list[str] = []
    speaking_style = pack_persona_text(engine, "speaking_style")
    if speaking_style:
        blocks.append(speaking_style)

    traits = persona_style_traits(engine)
    trait_lines = [
        f"- Warmth: {traits['warmth']}",
        f"- Directness: {traits['directness']}",
        f"- Humor: {traits['humor']}",
        f"- Detail: {traits['detail']}",
        f"- Formality: {traits['formality']}",
        f"- Enthusiasm: {traits['enthusiasm']}",
    ]
    blocks.append("\n".join(trait_lines))
    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _legacy_boundaries_source(engine: Celune) -> str:
    """Render legacy boundary lines into the v3 boundaries source slot."""
    lines = pack_persona_lines(engine, "boundaries")
    return "\n".join(f"- {line}" for line in lines)


def _legacy_examples_source(engine: Celune) -> str:
    """Render legacy example dialogue into the v3 examples source slot."""
    return "\n".join(pack_persona_lines(engine, "example_dialogue")).strip()


def _build_persona_source_material(
    engine: Celune,
    character_profile: CharacterProfile,
) -> PersonaSourceMaterial:
    """Build v3 prompt source material from package files with legacy fallback."""
    persona_files = _persona_manifest_files(engine)
    return PersonaSourceMaterial(
        identity=persona_files.get("identity.md", "")
        or _legacy_identity_source(character_profile),
        soul=persona_files.get("soul.md", ""),
        personality=persona_files.get("personality.md", "")
        or _legacy_personality_source(engine),
        speech_style=persona_files.get("speech_style.md", "")
        or _legacy_speech_style_source(engine),
        boundaries=persona_files.get("boundaries.md", "")
        or _legacy_boundaries_source(engine),
        examples=persona_files.get("examples.md", "")
        or _legacy_examples_source(engine),
    )


def build_persona_context(engine: Celune, request: str) -> PersonaContext:
    """Build structured Persona context for one user request.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.

    Returns:
        PersonaContext: The built RAG context for Persona.
    """
    name = persona_active_character_name(engine)
    voice = getattr(engine, "current_voice", None) or "balanced"
    voice_prompt = _effective_voice_prompt(engine)
    traits = persona_style_traits(engine)

    voice_notes = f"Selected voice: {voice}."
    if isinstance(voice_prompt, str) and voice_prompt.strip():
        voice_notes = f"{voice_notes}\nVoice prompt: {voice_prompt.strip()}"

    character_profile = CharacterProfile(
        name=name,
        age=pack_identity_text(engine, "age")
        or _config_text(engine, "persona_character_age", default_persona_age(engine)),
        gender=pack_identity_text(engine, "gender")
        or _config_text(
            engine, "persona_character_gender", default_persona_gender(engine)
        ),
        profile=pack_identity_text(engine, "profile")
        or _config_text(engine, "persona_character_profile", ""),
    )
    persona_card = PersonaCard(
        persona=_config_text(
            engine,
            "persona_persona",
            default_persona_persona(),
        ),
        warmth=traits["warmth"],
        directness=traits["directness"],
        humor=traits["humor"],
        detail=traits["detail"],
        formality=traits["formality"],
        enthusiasm=traits["enthusiasm"],
        context=_config_text(
            engine,
            "persona_context",
            default_persona_context(),
        ),
        voice=voice_notes,
        speaking_style=pack_persona_text(engine, "speaking_style"),
        boundaries=pack_persona_lines(engine, "boundaries"),
        prompt_rules=pack_persona_lines(engine, "prompt_rules"),
        example_dialogue=pack_persona_lines(engine, "example_dialogue"),
    )
    persona_source_material = _build_persona_source_material(engine, character_profile)
    mood_or_state = _persona_mood_or_state(engine, request)

    return PersonaContext(
        character_profile=character_profile,
        persona_card=persona_card,
        persona_source_material=persona_source_material,
        mood_or_state=mood_or_state,
        conversation_summary=persona_session_summary(engine),
        retrieved_long_term_memory=_build_retrieved_memory_bundle(engine, request),
    )


def _effective_voice_prompt(engine: Celune) -> Optional[str]:
    """Return the active voice prompt only when the engine supports it."""
    supported = getattr(engine, "voice_prompt_supported", None)
    if callable(supported) and not supported():
        return None
    if supported is False:
        return None

    voice_prompt = getattr(engine, "voice_prompt", None)
    return voice_prompt if isinstance(voice_prompt, str) else None


def build_persona_messages(engine: Celune, request: str) -> list[JSON]:
    """Build OpenAI-style messages for the Persona model.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.

    Returns:
        list[JSON]: A list of JSON objects containing current message history.
    """
    context = build_persona_context(engine, request)
    attachments = persona_pending_attachments(engine)
    user_content: JSONSerializable = request.strip()
    if attachments:
        user_content = [
            *attachments,
            {"type": "text", "text": request.strip()},
        ]

    messages: list[JSON] = [
        cast(JSON, {"role": "system", "content": PersonaPromptBuilder.build(context)})
    ]
    for message in persona_history_messages(engine):
        messages.append(message)  # noqa: PERF402
    messages.append(cast(JSON, {"role": "user", "content": user_content}))
    return messages


def build_persona_request(engine: Celune, request: str) -> JSON:
    """Build the JSON payload sent to the Persona model.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.

    Returns:
        JSON: The JSON payload to be sent to the Persona model.
    """
    context = build_persona_context(engine, request)
    character_card = (
        f"{context.character_profile.render()}\n\n{context.persona_card.render()}"
    )
    system_prompt = PersonaPromptBuilder.build(context)
    clean_request = request.strip()
    return {
        "format": "celune_persona_request",
        "format_version": 1,
        "model": persona_model_id(engine.config),
        "quantization": persona_quantization(engine.config),
        "quantized": True,
        "character": getattr(engine, "current_character", None) or "Unknown",
        "voice": getattr(engine, "current_voice", None) or "balanced",
        "character_card": character_card,
        "system": system_prompt,
        "user": clean_request,
        "request": clean_request,
        "messages": cast(
            JSONSerializable, build_persona_messages(engine, clean_request)
        ),
    }


def _extract_persona_text(payload: JSONSerializable) -> str:
    """Extract spoken text from common Persona response payload shapes."""
    if isinstance(payload, str):
        return payload.strip()

    if not isinstance(payload, dict):
        return ""

    for key in ("text", "response", "reply", "output", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return _extract_persona_text(first)

    return ""


def think(engine: Celune, request: str) -> bool:
    """Let Celune think about the input given, and speak back.

    Args:
        engine: The Celune engine that should speak the output.
        request: The input request that will be sent to Persona.

    Returns:
        bool: Whether Celune completed the thinking action successfully or not.
    """
    _store_persona_memories(engine, request)
    payload = build_persona_request(engine, request)
    attachments = getattr(engine, "persona_attachments", None)

    try:
        vision = engine.vision
        if vision is None:
            engine.log(string("pipeline.persona_not_connected"), "warning")
            return False

        response = vision.post(json=payload)
        response.raise_for_status()
        spoken_text = _extract_persona_text(response.json())
    except Exception as e:
        engine.log(
            string(
                "pipeline.persona_request_failed",
                error=format_error(e, engine.dev),
            ),
            "warning",
        )
        return False
    finally:
        if isinstance(attachments, list):
            attachments.clear()

    if not spoken_text:
        engine.log(string("pipeline.persona_empty_response"), "warning")
        return False

    history = getattr(engine, "persona_history", None)
    if isinstance(history, list):
        history.extend(
            [
                {"role": "user", "content": request.strip()},
                {"role": "assistant", "content": spoken_text},
            ]
        )
        compact_persona_history(engine)

    queued = queue_speech(
        engine,
        spoken_text,
        save=False,
        display_text=spoken_text,
    )
    _classify_persona_memories(engine, request)
    return queued


def say(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say.

    Args:
        engine: The Celune engine that should speak the text.
        text: The input text to queue for synthesis.
        save: Whether to save generated output artifacts.
        display_text: Optional text to show in logs instead of the synthesis text.

    Returns:
        bool: ``True`` when the text was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if queueing fails.
    """
    return queue_speech(
        engine, text, save=save, stream_queue=None, display_text=display_text
    )


async def say_async(
    engine: Celune,
    text: str,
    save: bool = True,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say without blocking an async caller.

    Args:
        engine: Runtime that owns the speech queues.
        text: The text to synthesize.
        save: Whether the generated utterance should be persisted to disk.
        display_text: Optional UI-facing text to associate with the request.

    Returns:
        bool: ``True`` when the request was queued successfully, otherwise ``False``.
    """
    return await queue_speech_async(
        engine,
        text,
        save=save,
        stream_queue=None,
        display_text=display_text,
    )


def handle_audio_input(engine: Celune, request: AudioInputRequest) -> bool:
    """Accept engine-level audio input and route it according to the active mode.

    Args:
        engine: The Celune engine receiving the audio input.
        request: The submitted audio input request.

    Returns:
        bool: ``True`` when the request was accepted, otherwise ``False``.
    """
    audio = np.asarray(request.audio, dtype=np.float32)
    if getattr(engine, "input_mode", "text_to_speech") == "voice_conversion":
        output = convert_audio_input(engine, request)
        if output is None:
            return False
        return queue_sfx_audio(
            engine,
            output.audio,
            output.sample_rate,
            output.label,
            status_label_key="pipeline.revoicing_label",
            log_length=request.log_playback,
            reset_ready_announcement=request.reset_ready_announcement,
        )

    engine.log_dev(
        "Audio input was accepted but ignored because the current mode is text-to-speech only."
        f" label={request.label!r} sample_rate={request.sample_rate} shape={audio.shape!r}"
    )
    return True


def convert_audio_input(
    engine: Celune, request: AudioInputRequest
) -> Optional[AudioOutput]:
    """Run one VC conversion request and return the converted audio output.

    Args:
        engine: The Celune engine receiving the audio input.
        request: The submitted audio input request.

    Returns:
        Optional[AudioOutput]: The converted audio output, or ``None`` when voice conversion is unavailable.
    """
    backend = getattr(engine, "vc_backend", None)
    if backend is None:
        engine.log(string("pipeline.vc_backend_unconfigured"), "warning")
        engine.error_callback(string("pipeline.vc_backend_unconfigured"))
        engine.progress_callback(0, 1)
        return None

    target_references: tuple[pathlib.Path, ...] = ()
    current_voice = getattr(engine, "current_voice", None)
    if isinstance(current_voice, str) and current_voice.strip():
        loader = default_loader()
        if loader is not None:
            try:
                target_references = (loader.materialize(current_voice, "wav"),)
            except Exception as e:
                engine.log(
                    string(
                        "pipeline.vc_reference_load_failed",
                        error=format_error(e, getattr(engine, "dev", False)),
                    ),
                    "warning",
                )
                target_references = ()

    output = backend.convert(
        VoiceConversionRequest(
            source_audio=np.asarray(request.audio, dtype=np.float32),
            sample_rate=request.sample_rate,
            target_voice=getattr(engine, "current_voice", None),
            target_character=getattr(engine, "current_character", None),
            target_references=target_references,
            label=request.label,
            pitch_shift=0,
            f0_condition=(
                request.f0_condition
                if isinstance(request.f0_condition, bool)
                else getattr(engine, "vc_f0_condition", False)
            ),
        )
    )
    resolved_pitch_shift = (
        request.pitch_shift
        if isinstance(request.pitch_shift, int)
        else getattr(engine, "vc_pitch_shift", 0)
    )
    if resolved_pitch_shift == 0:
        return output

    return AudioOutput(
        audio=pitch_shift_audio(
            np.asarray(output.audio, dtype=np.float32),
            output.sample_rate,
            resolved_pitch_shift,
        ),
        sample_rate=output.sample_rate,
        label=output.label,
    )


def queue_speech(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say and optionally mirror audio chunks.

    Args:
        engine: The Celune engine that should speak the text.
        text: The input text to queue for synthesis.
        save: Whether to save generated output artifacts.
        stream_queue: Optional queue receiving generated 48 kHz float32 chunks.
        display_text: Optional text to show in logs instead of the synthesis text.

    Returns:
        bool: ``True`` when the text was queued successfully, otherwise ``False``.

    Raises:
        Exception: An exception was caught and subsequently raised to propagate it to Celune.
    """
    if not _prepare_speech_readiness(engine):
        return False

    engine.model_ready.wait()
    if not _finish_speech_readiness(engine):
        return False

    return _queue_speech_after_ready(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
    )


def _prepare_speech_readiness(engine: Celune) -> bool:
    """Run the pre-wait checks shared by synchronous and async speech queueing."""
    if engine.is_in_tutorial:
        engine.log(string("celune.speech_input_disabled_tutorial"), "warning")
        return False

    if getattr(engine, "sleeping", False):
        engine.log(
            string("pipeline.cannot_speak_sleeping", app_name=APP_NAME),
            "warning",
        )
        engine.error_callback(string("celune.app_sleeping", app_name=APP_NAME))
        engine.progress_callback(0, 1)
        return False

    if not engine.model_ready.is_set():
        engine.status_callback(string("status.waiting_for_model"))
        engine.progress_callback(None, None)
        engine.log(string("pipeline.speak_waiting_reload"), "info")

    return True


def _finish_speech_readiness(engine: Celune) -> bool:
    """Run the post-wait model checks shared by speech queueing paths."""
    if not engine.loaded and not getattr(engine.backend, "is_fake", False):
        engine.log(string("ui.core_engine_not_loaded"), "warning")
        engine.error_callback(string("pipeline.not_ready_app", app_name=APP_NAME))
        engine.progress_callback(0, 1)
        return False

    return True


def _queue_speech_after_ready(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue one speech request after reload readiness is satisfied."""

    language_meta = detect_language(text, list(engine.backend.supported_languages))
    requested_language = engine.language
    backend_name = str(getattr(engine.backend, "name", "")).strip().lower()
    if (
        not isinstance(requested_language, str)
        or not requested_language.strip()
        or requested_language.strip().lower() == "auto"
    ):
        # Qwen3 handles automatic language selection internally, so keep the
        # backend-facing value as "Auto" instead of passing a detected language code.
        requested_language = (
            "Auto" if backend_name == "qwen3" else language_meta["language"]
        )

    if not language_meta["supported"]:
        # "zh-cn" has to be clipped to just "zh" to be a valid language code
        try:
            language = Lang(language_meta["language"][:2]).name
        except (InvalidLanguageValue, DeprecatedLanguageValue):
            language = language_meta["language"]

        engine.log(
            string("pipeline.received_unsupported_language", language=language),
            "warning",
        )
        engine.log(
            string("pipeline.may_not_say_properly", app_name=APP_NAME), "warning"
        )

    if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
        "1",
        "true",
        "on",
        "yes",
        "enabled",
    }:
        engine.log(string("pipeline.april_fools"))
        text = rng_replace(text, targets=["celune"], replacements=["celine"])

    if not acquire_pipeline(engine, "speak"):
        engine.progress_callback(0, 1)
        return False

    try:
        if not engine.loaded and not engine.backend.is_fake:
            engine.log(string("ui.core_engine_not_loaded"), "warning")
            engine.error_callback(string("pipeline.not_ready_app", app_name=APP_NAME))
            release_pipeline(engine)
            engine.progress_callback(0, 1)
            return False

        engine.cur_state = "generating"
        with engine.queue_lock:
            engine._speech_generation = getattr(engine, "_speech_generation", 0) + 1
            engine.utterance_force_stop.clear()
            engine.text_queue.put(
                SpeechRequest(
                    text,
                    display_text=display_text if display_text is not None else text,
                    language=requested_language,
                    save=save,
                    stream_queue=stream_queue,
                    normalize=engine.use_normalization,
                    generation=engine.speech_generation,
                )
            )
        engine.status_callback(string("status.generating"))
        engine.progress_callback(None, None)
        return True
    except Exception:
        release_pipeline(engine)
        raise


async def queue_speech_async(
    engine: Celune,
    text: str,
    save: bool = True,
    stream_queue: Optional[SpeechStreamQueue] = None,
    display_text: Optional[str] = None,
) -> bool:
    """Queue text for Celune to say without blocking the caller's event loop.

    Args:
        engine: Runtime that owns the speech queues.
        text: The text to synthesize.
        save: Whether the generated utterance should be persisted to disk.
        stream_queue: Optional queue receiving generated playback chunks.
        display_text: Optional UI-facing text to associate with the request.

    Returns:
        bool: ``True`` when the request was queued successfully, otherwise ``False``.
    """
    if not _prepare_speech_readiness(engine):
        return False

    await asyncio.to_thread(engine.model_ready.wait)

    if not _finish_speech_readiness(engine):
        return False

    return _queue_speech_after_ready(
        engine,
        text,
        save=save,
        stream_queue=stream_queue,
        display_text=display_text,
    )


def queue_sfx_audio(
    engine: Celune,
    audio: AudioChunk,
    sample_rate: int,
    label: str,
    keep: bool = False,
    volume: float = 1.0,
    status_label_key: str = "pipeline.playing_label",
    log_length: bool = True,
    reset_ready_announcement: bool = True,
) -> bool:
    """Queue decoded SFX audio through Celune's playback pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.
        label: Human-readable label for logs and status.
        keep: Whether to prepend this SFX to the next saved utterance.
        volume: Gain multiplier applied before the clip is queued for playback.
        status_label_key: Localization key used for the surfaced playback status.
        log_length: Whether to log the prepared playback sample rate and length.
        reset_ready_announcement: Whether this source should trigger a later ready announcement.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup fails.
    """
    try:
        audio = prepare_playback_audio(audio, sample_rate)
        playback_sample_rate = BASE_SR
        audio_len = len(audio) / playback_sample_rate
        if log_length:
            engine.log(
                string(
                    "pipeline.sample_rate_length",
                    sample_rate=playback_sample_rate,
                    seconds=format_number(audio_len, 2),
                )
            )

        if keep:
            engine.kept_sfx_audio = [chunk.copy() for chunk in split(audio, BASE_SR, 1)]

        source_id = _next_playback_source_id(engine)
        _register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        _register_playback_source(engine, source_id, kind="sfx", base_gain=volume)
        engine.cur_state = "speaking"
        playback_generation = getattr(engine, "_playback_generation", 0)
        _set_playback_source_status(
            engine,
            source_id,
            string(status_label_key, label=label),
        )
        # push the smallest possible chunks for responsive stopping
        for chunk in split(audio, playback_sample_rate, 1):
            if not _queue_playback_chunk(
                engine,
                source_id,
                chunk,
                playback_sample_rate,
                generation=playback_generation,
            ):
                _clear_playback_source_status(engine, source_id)
                return False
        if not _queue_playback_done(
            engine,
            source_id,
            generation=playback_generation,
        ):
            _clear_playback_source_status(engine, source_id)
            return False
        return True
    except Exception:
        engine.playback_done.set()
        raise


def queue_streaming_sfx_audio(
    engine: Celune,
    audio: AudioChunk,
    sample_rate: int,
    label: str,
    *,
    source_id: Optional[int] = None,
    generation: Optional[int] = None,
    volume: float = 1.0,
    status_label_key: str = "pipeline.playing_label",
    log_length: bool = False,
    reset_ready_announcement: bool = False,
) -> Optional[int]:
    """Queue one SFX segment onto a persistent playback source.

    Args:
        engine: The Celune engine that should play the sound.
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.
        label: Human-readable label for logs and status.
        source_id: Existing playback source to append to, or ``None`` to create one.
        generation: Playback generation captured by the producer session.
        volume: Gain multiplier applied before the clip is queued for playback.
        status_label_key: Localization key used for the surfaced playback status.
        log_length: Whether to log the prepared playback sample rate and length.
        reset_ready_announcement: Whether a newly created source should reset readiness.

    Returns:
        Optional[int]: The persistent playback source id, or ``None`` when the
            producer belongs to a cancelled playback generation.
    """
    audio = prepare_playback_audio(audio, sample_rate)
    playback_sample_rate = BASE_SR
    audio_len = len(audio) / playback_sample_rate
    if log_length:
        engine.log(
            string(
                "pipeline.sample_rate_length",
                sample_rate=playback_sample_rate,
                seconds=format_number(audio_len, 2),
            )
        )

    active_generation = getattr(engine, "_playback_generation", 0)
    if generation is not None and generation != active_generation:
        return None
    source_generation = active_generation if generation is None else generation

    meta = _playback_source_meta(engine)
    if source_id is None or source_id not in meta:
        source_id = _next_playback_source_id(engine)
        _register_overlay_playback_state(
            engine,
            reset_ready_announcement=reset_ready_announcement,
        )
        _register_playback_source(engine, source_id, kind="sfx", base_gain=volume)
        engine.cur_state = "speaking"
    elif float(meta[source_id].get("generation", 0.0)) != float(active_generation):
        return None

    _set_playback_source_status(
        engine,
        source_id,
        string(status_label_key, label=label),
    )

    if len(audio) > 0:
        if not _queue_playback_chunk(
            engine,
            source_id,
            audio,
            playback_sample_rate,
            generation=source_generation,
        ):
            return None

    return source_id


def finish_streaming_sfx_audio(engine: Celune, source_id: Optional[int]) -> None:
    """Mark one persistent SFX playback source as complete.

    Args:
        engine: Runtime that owns the playback queues.
        source_id: Persistent playback source to finish.
    """
    if source_id is None:
        return
    source_meta = _playback_source_meta(engine).get(source_id)
    if not isinstance(source_meta, dict):
        return
    _queue_playback_done(
        engine,
        source_id,
        generation=int(float(source_meta.get("generation", 0.0))),
    )


def prepare_playback_audio(
    audio: AudioChunk,
    sample_rate: int,
) -> AudioChunk:
    """Normalize audio to Celune's shared playback format.

    Args:
        audio: Decoded mono or stereo audio.
        sample_rate: Source sample rate for the decoded audio.

    Returns:
        AudioChunk: Audio resampled into Celune's playback format.
    """
    return resample_audio(np.asarray(audio, dtype=np.float32), sample_rate)


def play(
    engine: Celune, sound_path: str, keep: bool = False, volume: float = 1.0
) -> bool:
    """Play a sound via Celune's pipeline.

    Args:
        engine: The Celune engine that should play the sound.
        sound_path: The path to the audio file to play.
        keep: Whether to prepend this SFX to the next saved utterance.
        volume: How loud should the SFX be played at, limited to half of max volume to protect headphone users.

    Returns:
        bool: ``True`` when playback was queued successfully, otherwise ``False``.

    Raises:
        Exception: Re-raised after releasing the pipeline if SFX playback setup fails.
    """
    downloaded_from_url = _is_youtube_sfx_url(sound_path)
    if downloaded_from_url:
        downloaded_info = _download_youtube_sfx(engine, sound_path)
        if downloaded_info is None:
            return False
        downloaded, playback_label = downloaded_info
        sound_path = str(downloaded)
    else:
        playback_label = sound_path

    if not os.path.exists(sound_path):
        engine.log(
            string("pipeline.cannot_find_sound", app_name=APP_NAME, path=sound_path),
            "warning",
        )
        return False

    supported_formats = ("wav", "flac", "ogg", "mp3", "aiff")

    if not any(sound_path.endswith(audio_format) for audio_format in supported_formats):
        engine.log(
            string("pipeline.sfx_format_unsupported", app_name=APP_NAME),
            "warning",
        )
        engine.log(
            string(
                "pipeline.supported_formats",
                formats=", ".join(supported_formats),
            ),
            "warning",
        )
        return False

    audio, sr = sf.read(sound_path, dtype="float32")

    queued = queue_sfx_audio(
        engine,
        np.asarray(audio, dtype=np.float32),
        sr,
        playback_label,
        keep,
        volume=volume * 0.5,
    )
    if queued and downloaded_from_url:
        engine.status_callback(
            string("pipeline.playing_label", label=playback_label),
        )
    return queued


def close(engine: Celune) -> None:
    """Shut off Celune and exit.

    Args:
        engine: The Celune engine to shut down.
    """
    engine.log(string("pipeline.exiting"))
    engine._exit_requested = True

    with engine.queue_lock:
        clear_queue(engine.text_queue)
        clear_queue(engine.audio_queue)

    engine.text_queue.put(engine.sentinel)
    engine.audio_queue.put(engine.sentinel)

    if engine.generation_thread is not None:
        engine.generation_thread.join(timeout=2)

    if engine.playback_thread is not None:
        engine.playback_thread.join(timeout=2)

    close_stream(engine, abort=True)
    engine.glow.leave()
    engine.glow.finished.wait(timeout=5)


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


def _process_generation_request(engine: Celune, item: SpeechRequest) -> None:
    """Process one queued speech request on a blocking worker thread."""
    text = item.text
    display_text = item.display_text
    request_language = item.language
    save_output = item.save
    stream_queue = item.stream_queue
    kept_sfx_audio = engine.kept_sfx_audio
    engine.kept_sfx_audio = None

    if engine.exit_requested:
        if stream_queue is not None:
            stream_queue.put(NotAvailableError("stream queue interrupted"))
            stream_queue.put(None)
        release_pipeline(engine)
        return

    while True:
        try:
            engine.model_ready.wait()

            if not engine.loaded and not engine.backend.is_fake:
                engine.log(string("ui.core_engine_not_loaded"), "warning")
                engine.locked = False
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("model is not ready"))
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            start_time = _monotonic_time()
            engine.log(f"[GEN] {display_text}")
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

            # these generation parameters are fixed and do not change
            # this only applies to Qwen3-TTS, other backends discard this
            generation_params: Mapping[str, JSONSerializable] = {
                "temperature": 0.15,
                "top_k": 20,
                "top_p": 0.7,
                "repetition_penalty": 1.1,
            }

            chunks = split_text(engine, text)
            if not chunks:
                engine.progress_callback(0, 1)
                engine.error_callback(string("pipeline.nothing_to_say"))
                release_pipeline(engine)
                if stream_queue is not None:
                    stream_queue.put(NotAvailableError("nothing to say"))
                    stream_queue.put(None)
                break

            buffer: AudioChunks = []
            full_audio: AudioChunks = []
            generated_text_parts: list[str] = []
            request_generation = item.generation
            source_id = _next_playback_source_id(engine)
            _register_playback_source(engine, source_id, kind="speech")

            for chunk_index, chunk_text in enumerate(chunks):
                if engine.exit_requested:
                    break

                if engine.utterance_force_stop.is_set():
                    break

                if item.normalize:
                    engine.status_callback(string("status.normalizing"))
                    engine.progress_callback(None, None)
                    normalized = engine.normalize(chunk_text)
                    if normalized is not None:
                        if normalized == chunk_text:
                            engine.log_dev(
                                "This input is already normalized.", "warning"
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
                        engine.log_dev(
                            f"[RELOAD] Loading {model_id} for language: {target_language!s}"
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
                        if audio_chunk.size == 0 or not np.any(audio_chunk):
                            continue

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
                            smart_buffer_target_seconds <= 0.0
                            or buffered_speech_len >= smart_buffer_target_seconds
                        ):
                            pushed_audio = _flush_buffered_speech_chunks(
                                engine,
                                source_id,
                                buffer,
                                speech_timing,
                                pushed_audio,
                                stream_queue,
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

            if generated_text_parts:
                text = " ".join(generated_text_parts)

            generation_time = _monotonic_time() - start_time

            if engine.exit_requested:
                if stream_queue is not None:
                    stream_queue.put(None)
                release_pipeline(engine)
                break

            if engine.utterance_force_stop.is_set() or request_generation != getattr(
                engine, "_speech_generation", request_generation
            ):
                if stream_queue is not None:
                    stream_queue.put(None)
                engine.reverb.reset()
                break

            engine.log(
                string(
                    "pipeline.generation_summary",
                    speech_seconds=format_number(speech_len, 2),
                    generation_seconds=format_number(generation_time, 2),
                )
            )
            generation_speed = speech_len / generation_time
            engine.log(
                string(
                    "pipeline.generation_speed",
                    speed=format_number(generation_speed, 2),
                )
            )
            _remember_smart_buffer_speed(engine, generation_speed)
            engine.smart_buffer_target_seconds = _smart_buffer_target_seconds(
                engine,
                speech_len,
                generation_time,
            )
            engine.log(
                string(
                    "pipeline.ttfc_ms",
                    milliseconds=format_number(speech_timing.ttfc_ms(), 1),
                )
            )

            if buffer:
                _flush_buffered_speech_chunks(
                    engine,
                    source_id,
                    buffer,
                    speech_timing,
                    pushed_audio,
                    stream_queue,
                )

            engine.log(string("pipeline.generation_done"))

            saved_path = None
            analysis_audio = None
            if not engine.exit_requested:
                if engine.reverb.strength > 0.0:
                    tail = engine.reverb.flush()
                    if len(tail) > 0:
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
                        if save_output:
                            full_audio.append(tail)

                engine.reverb.reset()
                is_silent = False
                silence_tier = 0
                if full_audio:
                    is_silent, silence_tier = is_silent_utterance(
                        np.concatenate(full_audio)
                    )

                if is_silent and silence_tier == 2:
                    if item.silent_retry_count < _MAX_SILENT_UTTERANCE_RETRIES:
                        engine.regenerate = True
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

                if save_output and full_audio:
                    wav = np.concatenate(full_audio)
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
                                string(
                                    "pipeline.outputs_create_failed",
                                    error=format_error(e, engine.dev),
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
                                string(
                                    "pipeline.flac_save_failed",
                                    error=format_error(e, engine.dev),
                                ),
                                "warning",
                            )
                            saved_path = None

                engine.recently_saved = saved_path
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
        except Exception as e:
            if engine.exit_requested:
                release_pipeline(engine)
                break

            engine.log(
                string(
                    "pipeline.gen_error",
                    error=format_error(e, engine.dev),
                ),
                "error",
            )
            if stream_queue is not None:
                stream_queue.put(e)
                stream_queue.put(None)
            engine.cur_state = "error"
            engine.locked = False
            engine.playback_done.set()
            engine.progress_callback(0, 1)
            engine.error_callback(
                string("pipeline.could_not_generate", app_name=APP_NAME)
            )
            break


async def generation_worker_job(engine: Celune) -> None:
    """Generate audio tokens and send them to the audio pipeline as an async job.

    Args:
        engine: Runtime that owns the generation queue and playback state.
    """
    while True:
        item = await asyncio.to_thread(engine.text_queue.get)
        engine.regenerate = False

        if item is engine.sentinel:
            try:
                engine.audio_queue.put_nowait(engine.sentinel)
            except queue.Full:
                await asyncio.to_thread(engine.audio_queue.put, engine.sentinel)
            break

        request = cast(SpeechRequest, item)
        if request.generation != getattr(
            engine, "_speech_generation", request.generation
        ):
            if request.stream_queue is not None:
                request.stream_queue.put(None)
            continue

        engine.utterance_force_stop.clear()
        setattr(engine, "_active_speech_generation", request.generation)
        try:
            await asyncio.to_thread(_process_generation_request, engine, request)
        finally:
            setattr(engine, "_active_speech_generation", None)


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
        engine.log_dev(
            f"[PLAY] resolved {output_device_key}={engine.config.get(output_device_key)!r} "
            f"audio_api={engine.config.get('audio_api')!r} -> {output_device!r}"
        )
        engine.current_sr = sample_rate
        engine.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=2,
            dtype="float32",
            blocksize=0,
            device=output_device,
        )
        if engine.stream is None:
            raise NotAvailableError("audio stream is not available")
        engine.stream.start()
        engine._audio_unavailable = False
        engine.log_dev(f"[PLAY] started stream at {sample_rate} Hz")
        return True
    except ValueError as error:
        if not getattr(engine, "audio_unavailable", False):
            engine.log(str(error), "warning")
            engine.error_callback(string("pipeline.no_audio_devices_short"))
        engine._audio_unavailable = True
        return False
    except sd.PortAudioError:
        if not getattr(engine, "audio_unavailable", False):
            engine.log(
                string("pipeline.audio_stream_init_failed", app_name=APP_NAME),
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

    if random.random() < 0.01:
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
        if engine.dev and saved_path is not None and analysis_audio is not None:
            engine.log_dev("Analyzing...")
            saved = pathlib.Path(saved_path)
            run_async(
                analyze_voice_audio,
                analysis_audio,
                BASE_SR,
                saved.name,
                saved.parent,
                saved.stem,
                engine.current_voice,
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

    async def force_stop_playback() -> None:
        nonlocal buffered_seconds, stop_cleanup_generation
        current_generation = getattr(engine, "_playback_generation", 0)
        if stop_cleanup_generation == current_generation:
            return
        stop_cleanup_generation = current_generation
        source_buffers.clear()
        source_done.clear()
        buffered_seconds = 0.0
        _playback_source_statuses(engine).clear()
        _playback_source_meta(engine).clear()
        _reset_glow_audio_reactivity(engine)
        await asyncio.to_thread(close_stream, engine, True)
        engine.playback_done.set()
        release_pipeline(engine)
        if getattr(engine, "_active_speech_generation", None) is None:
            engine.utterance_force_stop.clear()
        if engine.cur_state != "error":
            engine.idle_callback()

    async def drain_pending_items() -> bool:
        nonlocal buffered_seconds, stop_requested

        drained_items = 0
        while drained_items < max_drain_items:
            if (
                cpu_guard_enabled
                and buffered_seconds >= max_buffer_seconds
                and not engine.utterance_force_stop.is_set()
            ):
                break
            try:
                pending = _dequeue_playback_item(engine, prioritize_speech=True)
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
            elif isinstance(pending, PlaybackSourceDone):
                if pending.generation != getattr(engine, "_playback_generation", 0):
                    continue
                source_done[pending.source_id] = pending

        if yield_seconds > 0.0 and not engine.audio_queue.empty():
            await asyncio.sleep(yield_seconds)
        return True

    while True:
        if engine.exit_requested:
            with engine.queue_lock:
                clear_queue(engine.audio_queue)

            await asyncio.to_thread(close_stream, engine, True)
            release_pipeline(engine)
            if engine.cur_state != "error":
                engine.idle_callback()
            return

        try:
            timeout = 0.01 if source_buffers else None
            if timeout is None:
                item = await asyncio.to_thread(engine.audio_queue.get)
            else:
                item = await asyncio.to_thread(engine.audio_queue.get, True, timeout)
        except queue.Empty:
            item = None

        if item is engine.sentinel:
            break

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
        elif isinstance(item, PlaybackSourceDone):
            if item.generation != getattr(engine, "_playback_generation", 0):
                continue
            stop_cleanup_generation = None
            source_done[item.source_id] = item

        if not await drain_pending_items():
            continue

        if engine.exit_requested:
            continue

        while source_buffers:
            if not await drain_pending_items():
                break

            if not await asyncio.to_thread(_ensure_playback_stream, engine, BASE_SR):
                source_buffers.clear()
                source_done.clear()
                buffered_seconds = 0.0
                _playback_source_statuses(engine).clear()
                _playback_source_meta(engine).clear()
                release_pipeline(engine)
                if engine.cur_state != "error":
                    engine.idle_callback()
                break

            ready_ids = [
                source_id for source_id, blocks in source_buffers.items() if blocks
            ]
            if not ready_ids:
                break

            speech_active = bool(_active_speech_source_ids(source_buffers, engine))
            block_len = min(
                len(source_buffers[source_id][0][0]) for source_id in ready_ids
            )
            mixed = np.zeros((block_len, 2), dtype=np.float32)
            timing_to_log: Optional[SpeechTiming] = None
            completed_now: list[int] = []

            for source_id in ready_ids:
                block, timing = source_buffers[source_id][0]
                block_audio = _apply_source_gain(
                    np.asarray(block[:block_len], dtype=np.float32),
                    source_id,
                    speech_active=speech_active,
                    block_seconds=block_len / BASE_SR,
                    engine=engine,
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

                source_meta = _playback_source_meta(engine).get(source_id)
                if isinstance(source_meta, dict):
                    source_meta["played_frames"] = float(
                        source_meta.get("played_frames", 0.0)
                    ) + float(block_len)

                if not source_buffers[source_id]:
                    if source_id in source_done:
                        completed_now.append(source_id)
                    del source_buffers[source_id]

            buffered_seconds = max(
                0.0,
                buffered_seconds - (block_len / BASE_SR) * len(ready_ids),
            )

            mixed = np.clip(mixed, -1.0, 1.0)

            try:
                if engine.utterance_force_stop.is_set():
                    await force_stop_playback()
                    break
                stream = engine.stream
                if stream is None:
                    raise NotAvailableError("audio stream is not available")
                log_first_playback(engine, timing_to_log)
                engine.glow.schedule(mixed)
                await asyncio.to_thread(stream.write, mixed)
                _update_playback_progress(engine, source_buffers)
            except Exception as e:
                engine.log(f"[PLAY ERROR] {format_error(e, engine.dev)}", "error")
                engine.error_callback(string("pipeline.playback_error"))
                await asyncio.to_thread(close_stream, engine, True)
                engine._stream = None
                engine._current_sr = None
                source_buffers.clear()
                source_done.clear()
                buffered_seconds = 0.0
                _playback_source_statuses(engine).clear()
                _playback_source_meta(engine).clear()
                break

            while True:
                newly_complete = [
                    source_id
                    for source_id, marker in source_done.items()
                    if source_id not in source_buffers
                ]
                if not newly_complete:
                    break

                for source_id in newly_complete:
                    marker = source_done.pop(source_id)
                    engine.recently_saved = marker.saved_path
                    _clear_playback_source_status(engine, source_id)
                    if marker.release_pipeline:
                        release_pipeline(
                            engine,
                            playback_idle=not source_buffers
                            and engine.audio_queue.empty()
                            and engine.text_queue.empty(),
                        )
                    if (
                        marker.notify_idle
                        and not source_buffers
                        and engine.audio_queue.empty()
                        and engine.text_queue.empty()
                    ):
                        _finalize_playback_idle(
                            engine,
                            saved_path=marker.saved_path,
                            analysis_audio=marker.analysis_audio,
                        )
                    elif (
                        not source_buffers
                        and engine.audio_queue.empty()
                        and engine.text_queue.empty()
                    ):
                        engine.playback_done.set()
                        _reset_glow_audio_reactivity(engine)
                        engine.progress_callback(1, 1)

        while True:
            orphaned = [
                source_id
                for source_id, marker in source_done.items()
                if source_id not in source_buffers
            ]
            if not orphaned:
                break

            for source_id in orphaned:
                marker = source_done.pop(source_id)
                engine.recently_saved = marker.saved_path
                _clear_playback_source_status(engine, source_id)
                if marker.release_pipeline:
                    release_pipeline(
                        engine,
                        playback_idle=not source_buffers
                        and engine.audio_queue.empty()
                        and engine.text_queue.empty(),
                    )
                if (
                    marker.notify_idle
                    and not source_buffers
                    and engine.audio_queue.empty()
                    and engine.text_queue.empty()
                ):
                    _finalize_playback_idle(
                        engine,
                        saved_path=marker.saved_path,
                        analysis_audio=marker.analysis_audio,
                    )
                elif (
                    not source_buffers
                    and engine.audio_queue.empty()
                    and engine.text_queue.empty()
                ):
                    engine.playback_done.set()
                    _reset_glow_audio_reactivity(engine)
                    engine.progress_callback(1, 1)

        if stop_requested and not source_buffers and not source_done:
            break
