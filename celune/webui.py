# SPDX-License-Identifier: Apache-2.0
"""WebUI lifecycle helpers."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import inspect
import io
import re
import time
from collections.abc import Callable, Iterator
from html import escape
from typing import Optional, Union, cast

import gradio as gr
import numpy as np
import numpy.typing as npt
import soundfile as sf
from fastapi import HTTPException

from . import api as _api
from .ui import resources as ui_resources
from .constants import APP_NAME
from .dataclasses.events import (
    AgentApprovalRequestedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskFinishedEvent,
    AgentTaskStateChangedEvent,
)
from .extensions.events import EventDispatcher
from .i18n import string
from .paths import main_window_log_path
from .persona.impl import persona_enabled, persona_talkback_enabled
from .playback import current_playback_status
from .typing.aliases import AudioChunk, AudioChunks, LogLevel
from .typing.api import TaskStatus, WebUiUpdate
from .typing.events import EventCallback, EventName
from .typing.pipeline import SpeechStreamQueue
from .cedts.ui import UiTimedUpdate, ui_timed_update_channel
from .theme import colors
from .utils import available, format_error_message
from .ui.app import CeluneUI
from .binding import install_module_functions
from .constants import BASE_SR
from .typing.api import WebUiUnset as _WebUiUnset, WEBUI_UNSET as _WEBUI_UNSET

__all__ = (
    "_append_webui_error",
    "_append_webui_log",
    "_collect_speech_job",
    "_delete_expired_speech_jobs",
    "_flac_bytes",
    "_forget_speech_job",
    "_input_update",
    "_normalized_audio",
    "_probe_webui_runtime",
    "_probed_status_text",
    "_publish_active_task_log",
    "_publish_active_task_progress",
    "_publish_task_event",
    "_publish_task_progress",
    "_publish_webui_agent_status",
    "_receive_webui_timed_update",
    "_remember_speech_job",
    "_seed_webui_logs",
    "_send_button_update",
    "_set_active_speech_task",
    "_set_webui_status",
    "_speech_job_snapshot",
    "_strip_webui_log_prefix",
    "_subscribe_to_speech_job",
    "_subscribe_webui_events",
    "_subscribe_webui_timed_updates",
    "_sync_webui_runtime_locks",
    "_task_event_status",
    "_task_status",
    "_unsubscribe_from_speech_job",
    "_update_speech_job",
    "_voice_button_update",
    "_webui_agent_approval_requested",
    "_webui_agent_choice_requested",
    "_webui_agent_status_message",
    "_webui_agent_task_finished",
    "_webui_agent_task_state_changed",
    "_webui_audio_waveform_options",
    "_webui_input_placeholder",
    "_webui_log_line_html",
    "_webui_logs_html",
    "_webui_persona_input_available",
    "_webui_persona_loaded",
    "_webui_recording_hint",
    "_webui_resources_html",
    "_webui_snapshot",
    "_webui_status_color",
    "_webui_status_html",
    "_webui_submit_snapshot",
    "_webui_theme_html",
    "_webui_vc_controls_update",
    "_webui_vc_mode_active",
    "_wrap_celune_callbacks",
    "api_log",
    "audio_bytes",
    "require_celune",
    "stream_headers",
)


def _subscribe_webui_events(celune: _api.Celune) -> None:
    """Subscribe the browser UI to the shared typed agent lifecycle events."""
    dispatcher = getattr(celune, "_event_dispatcher", None)
    if not isinstance(dispatcher, EventDispatcher):
        return

    callbacks: tuple[tuple[EventName, EventCallback], ...] = (
        ("agent_task_state_changed", _webui_agent_task_state_changed),
        ("agent_approval_requested", _webui_agent_approval_requested),
        ("agent_choice_requested", _webui_agent_choice_requested),
        ("agent_task_finished", _webui_agent_task_finished),
    )
    for event_name, callback in callbacks:
        dispatcher.subscribe(event_name, callback, "WebUI")
    _api.webui_event_dispatcher = dispatcher
    _api.webui_event_callbacks = callbacks


def _subscribe_webui_timed_updates() -> None:
    """Subscribe the browser UI to the shared CEDTS timed-update channel."""
    if _api.webui_timed_update_unsubscribe is not None:
        _api.webui_timed_update_unsubscribe()
    _api.webui_timed_update_unsubscribe = ui_timed_update_channel.subscribe(
        _receive_webui_timed_update
    )


def _receive_webui_timed_update(update: UiTimedUpdate) -> None:
    """Apply one newer TUI timed update to browser-owned state."""
    celune = _api.bound_celune
    if celune is None or update.runtime_id != str(id(celune)):
        return
    if update.sequence <= _api.webui_timed_update_sequence:
        return
    _api.webui_timed_update_sequence = update.sequence
    _api.webui_resource_page = update.resource_page
    _api.webui_active_theme_name = update.theme_name
    _api.webui_timed_update_received_at = time.monotonic()
    _api.webui_timed_update_source = "cedts"
    if update.status_text:
        _api.webui_status_text = update.status_text
        _api.webui_status_severity = update.status_severity


def _webui_agent_status_message(
    task_id: str,
    fallback_state: str,
    fallback_key: Optional[str] = None,
) -> Optional[str]:
    """Resolve the same compact agent status text used by the TUI."""
    celune = _api.bound_celune
    if celune is None:
        return None
    runtime = getattr(celune, "agent_runtime", None)
    task = None
    get_task = getattr(runtime, "get_task", None)
    if callable(get_task):
        get_task_call = cast(Callable[[str], object], get_task)
        with contextlib.suppress(KeyError, ValueError):
            task = get_task_call(task_id)  # pylint: disable=not-callable
    state = getattr(task, "state", fallback_state)
    state_value = getattr(state, "value", state)
    if not isinstance(state_value, str):
        state_value = str(state_value)
    if task is not None and state_value in {
        "planning",
        "working",
        "executing_tool",
        "responding",
    }:
        config = getattr(task, "config", None)
        maximum = getattr(config, "max_loops", 0)
        return string(
            "agent.status.working",
            iteration=getattr(task, "iterations", 0),
            maximum=maximum,
        )
    key = fallback_key or f"agent.status.{state_value}"
    message = string(key)
    return None if message == key else message


def _publish_webui_agent_status(
    task_id: str,
    fallback_state: str,
    fallback_key: Optional[str] = None,
) -> None:
    """Mirror one typed agent event into the browser status line."""
    message = _webui_agent_status_message(task_id, fallback_state, fallback_key)
    if message is not None:
        severity = "warning" if "awaiting" in message.casefold() else "info"
        _set_webui_status(message, severity, source="agent")


def _webui_agent_task_state_changed(event: AgentTaskStateChangedEvent) -> None:
    """Mirror agent task state changes into the browser UI."""
    state = getattr(event.new_state, "value", str(event.new_state))
    _publish_webui_agent_status(event.task_id, state)


def _webui_agent_approval_requested(event: AgentApprovalRequestedEvent) -> None:
    """Mirror agent approval pauses into the browser UI."""
    _publish_webui_agent_status(
        event.task_id,
        "awaiting_approval",
        "agent.status.awaiting_approval",
    )


def _webui_agent_choice_requested(event: AgentChoiceRequestedEvent) -> None:
    """Mirror agent choice pauses into the browser UI."""
    _publish_webui_agent_status(
        event.task_id,
        "awaiting_choice",
        "agent.status.awaiting_choice",
    )


def _webui_agent_task_finished(event: AgentTaskFinishedEvent) -> None:
    """Mirror terminal agent task states into the browser UI."""
    state = getattr(event.state, "value", str(event.state))
    _publish_webui_agent_status(event.task_id, state)


def _webui_status_color(severity: str) -> str:
    """Return the browser UI color for a given severity."""
    palette = colors.SEVERITY_COLORS.get(
        _api.webui_active_theme_name,
        colors.SEVERITY_COLORS["celune"],
    )
    return palette.get(severity, palette["info"])


def _webui_theme_html() -> str:
    """Render runtime CSS variables for the browser UI theme."""
    severity = "info"
    accent = _webui_status_color(severity)
    theme = colors.THEME
    if _api.webui_active_theme_name == "celune_light":
        theme = colors.THEME_LIGHT
    elif _api.webui_active_theme_name == "celune_april_fools":
        theme = colors.THEME_APRIL_FOOLS
    background = theme.background or colors.DEFAULT_BACKGROUND
    input_bg = colors.blend(accent, background, 0.78)
    return (
        "<style>:root {"
        f"--celune-ui-accent: {accent};"
        f"--celune-ui-bg: {background};"
        f"--celune-ui-input-bg: {input_bg};"
        "}</style>"
    )


def _webui_log_line_html(message: str, severity: str = "info") -> str:
    """Render one browser log line with severity-aware coloring."""
    color = _webui_status_color(severity)
    return f'<span style="color: {color};">{escape(message)}</span>'


def _webui_audio_waveform_options() -> gr.WaveformOptions:
    """Return theme-bound waveform colors for Gradio audio components."""
    primary = colors.SEVERITY_COLORS["celune"]["info"]
    secondary = colors.THEME.secondary or colors.FADED_ACCENT
    return gr.WaveformOptions(
        waveform_color=secondary,
        waveform_progress_color=primary,
        trim_region_color=primary,
    )


def _strip_webui_log_prefix(line: str) -> str:
    """Remove persisted timestamp and severity prefixes from one log line."""
    stripped = line.strip()
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    return stripped


def _seed_webui_logs() -> None:
    """Populate the browser log view from the persisted desktop log when available."""
    if _api.webui_logs_seeded:
        return

    _api.webui_logs_seeded = True
    path = main_window_log_path()
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-180:]
    except OSError:
        return

    record: list[str] = []
    record_severity = "info"
    record_pattern = re.compile(
        r"^\[[^\]]+\]\s+\[(?P<severity>[^\]]+)\]\s?(?P<message>.*)$"
    )

    def append_record() -> None:
        if record:
            _append_webui_log("\n".join(record), record_severity)

    for line in lines:
        match = record_pattern.match(line)
        if match is not None:
            append_record()
            record.clear()
            severity = match.group("severity").casefold()
            record_severity = (
                severity
                if severity in {"debug", "info", "warning", "error"}
                else "info"
            )
            record.append(match.group("message"))
        elif record:
            record.append(line)
        else:
            record.append(_strip_webui_log_prefix(line))
    append_record()


def _append_webui_log(msg: str, severity: str = "info") -> None:
    """Store one browser log line."""
    if _api.webui_log_lines and _api.webui_log_lines[-1] == (msg, severity):
        return
    _api.webui_log_lines.append((msg, severity))


def _append_webui_error(
    message: str,
    error: BaseException,
    severity: str = "error",
    celune: Optional[_api.Celune] = None,
) -> None:
    """Store a WebUI error with detail selected by the active log level."""
    runtime = celune or _api.bound_celune
    _append_webui_log(
        format_error_message(
            message,
            error,
            getattr(runtime, "log_level", "info"),
        ),
        severity,
    )


def _set_webui_status(
    msg: str,
    severity: str = "info",
    *,
    source: str = "callback",
    updated_at: Optional[float] = None,
) -> None:
    """Update the browser UI status line."""
    _api.webui_status_text = msg
    _api.webui_status_severity = severity
    _api.webui_status_source = source
    _api.webui_status_updated_at = (
        time.monotonic() if updated_at is None else updated_at
    )


def _probed_status_text(celune: _api.Celune) -> tuple[str, str]:
    """Return the best-effort footer status derived from Celune's live state."""
    if not celune.current_voice and not celune.voices:
        return string("status.could_not_start", app_name=APP_NAME), "error"

    state = (celune.cur_state or "").strip().lower()
    return {
        "idle": (string("status.idle"), "info"),
        "speaking": (string("status.speaking"), "info"),
        "thinking": (string("status.thinking"), "info"),
        "waking": (string("status.waking_up"), "info"),
        "reloading": (string("status.reloading"), "info"),
        "sleeping": (string("status.sleeping"), "sleeping"),
        "init": (string("status.initializing"), "info"),
        "generating": (string("status.generating"), "info"),
        "error": (string("status.could_not_continue", app_name=APP_NAME), "error"),
    }.get(
        state,
        (state.title() if state else string("status.initializing"), "info"),
    )


def _probe_webui_runtime() -> None:
    """Poll the live runtime so the WebUI footer updates even without new log lines."""

    celune = _api.bound_celune
    if celune is None:
        return

    now = time.monotonic()
    current_state = (celune.cur_state or "").strip().lower()
    playback_status = current_playback_status(celune)
    if playback_status is not None:
        if (
            _api.webui_status_text != playback_status
            or _api.webui_status_source != "playback"
        ):
            _set_webui_status(
                playback_status,
                "info",
                source="playback",
                updated_at=now,
            )
        _api.webui_last_probed_state = current_state
    elif current_state == "sleeping":
        sleeping_log = string("webui.sleeping_log", app_name=APP_NAME)
        if not any(message == sleeping_log for message, _ in _api.webui_log_lines):
            _append_webui_log(sleeping_log, "sleeping")
        sleeping_status, sleeping_severity = _probed_status_text(celune)
        if (
            _api.webui_status_text != sleeping_status
            or _api.webui_status_severity != sleeping_severity
            or _api.webui_status_source in {"callback", "agent", "cedts"}
        ):
            _set_webui_status(
                sleeping_status,
                sleeping_severity,
                source="probe",
                updated_at=now,
            )
    if playback_status is None and current_state != _api.webui_last_probed_state:
        status_text, severity = _probed_status_text(celune)
        should_override_status = (
            _api.webui_last_probed_state is None
            or _api.webui_status_text == string("status.api_starting")
            or _api.webui_status_source not in {"callback", "agent", "cedts"}
            or now - _api.webui_status_updated_at
            >= _api.WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS
            or current_state in {"idle", "sleeping", "error"}
        )
        if should_override_status:
            _set_webui_status(
                status_text,
                severity,
                source="probe",
                updated_at=now,
            )
        _api.webui_last_probed_state = current_state

    pages = ui_resources.resource_pages(celune, _api.webui_active_theme_name)
    if not pages:
        return

    if _api.webui_last_resource_advance <= 0:
        _api.webui_last_resource_advance = now
        return

    timed_update_is_fresh = (
        _api.webui_timed_update_source == "cedts"
        and now - _api.webui_timed_update_received_at
        < _api.WEBUI_TIMED_UPDATE_STALE_SECONDS
    )
    if (
        not timed_update_is_fresh
        and now - _api.webui_last_resource_advance >= _api.WEBUI_RESOURCE_ROTATE_SECONDS
    ):
        _api.webui_resource_page = (_api.webui_resource_page + 1) % len(pages)
        _api.webui_last_resource_advance = now


def _wrap_celune_callbacks(celune: _api.Celune) -> None:
    """Mirror Celune callbacks into browser UI state without replacing existing handlers."""
    if getattr(celune, "_webui_callbacks_wrapped", False):
        return

    original_log = celune.log_callback
    original_status = celune.status_callback
    original_error = cast(
        Callable[[str], None],
        getattr(celune, "error_callback", lambda _message: None),
    )
    original_progress = cast(
        Callable[[Optional[float], Optional[float]], None],
        getattr(celune, "progress_callback", lambda _progress, _total: None),
    )
    original_idle = cast(
        Callable[[], None], getattr(celune, "idle_callback", lambda: None)
    )
    original_queue_available = cast(
        Callable[[], None], getattr(celune, "queue_avail_callback", lambda: None)
    )
    original_caption_progress = cast(
        Callable[[Optional[float], Optional[float]], None],
        getattr(celune, "caption_progress_callback", lambda _progress, _total: None),
    )
    original_caption = cast(
        Callable[[Optional[str]], None],
        getattr(celune, "caption_callback", lambda _caption: None),
    )
    original_caption_timing = cast(
        Callable[..., None],
        getattr(celune, "caption_timing_callback", lambda *_args: None),
    )
    original_voice_changed = celune.voice_changed_callback
    original_input_state = celune.change_input_state_callback
    original_voice_lock_state = celune.change_voice_lock_state_callback

    def wrapped_log(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        _publish_active_task_log(msg, severity)
        _append_webui_log(msg, severity)
        _api._invoke_message_callback(original_log, msg, severity, loglevel)

    def wrapped_status(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        _publish_active_task_log(msg, severity)
        _set_webui_status(msg, severity, source="callback")
        _api._invoke_message_callback(original_status, msg, severity, loglevel)

    def wrapped_error(msg: str) -> None:
        _publish_active_task_log(
            string("status.could_not_continue", app_name=APP_NAME),
            "error",
        )
        _append_webui_log(msg, "error")
        _set_webui_status(msg, "error", source="callback")
        original_error(msg)

    def wrapped_progress(
        progress: Optional[float],
        total: Optional[float],
    ) -> None:
        _api.webui_progress_current = progress
        _api.webui_progress_total = total
        _publish_active_task_progress(progress, total)
        original_progress(progress, total)

    def wrapped_idle() -> None:
        original_idle()
        _api.webui_caption_active = False
        _api.webui_caption_text = ""
        _api.webui_caption_progress = 0.0
        _api.webui_progress_current = None
        _api.webui_progress_total = None
        _sync_webui_runtime_locks(celune, locked=getattr(celune, "locked", False))
        if getattr(celune, "sleeping", False):
            _set_webui_status(string("status.sleeping"), "sleeping", source="callback")
        elif getattr(celune, "cur_state", "") not in {"reloading", "waking"}:
            _set_webui_status(string("status.idle"), source="callback")

    def wrapped_queue_available() -> None:
        original_queue_available()
        locked = bool(getattr(celune, "is_in_tutorial", False))
        _sync_webui_runtime_locks(celune, locked=locked)
        _set_webui_status(string("status.speaking"), source="callback")

    def wrapped_caption_progress(
        progress: Optional[float],
        total: Optional[float],
    ) -> None:
        if total is not None and total > 0:
            _api.webui_caption_progress = max(0.0, min(1.0, (progress or 0.0) / total))
        original_caption_progress(progress, total)

    def wrapped_caption(caption: Optional[str]) -> None:
        if caption:
            _api.webui_caption_active = True
            _api.webui_caption_text = caption
            _api.webui_caption_progress = 0.0
        else:
            _api.webui_caption_active = False
            _api.webui_caption_text = ""
            _api.webui_caption_progress = 0.0
        original_caption(caption)

    def wrapped_caption_timing(
        caption: str,
        audio: AudioChunk,
        sample_rate: int,
        timing_text: Optional[str] = None,
    ) -> None:
        _api.webui_caption_active = True
        _api.webui_caption_text = caption
        _api.webui_caption_progress = 0.0
        try:
            signature = inspect.signature(original_caption_timing)
            signature.bind(caption, audio, sample_rate, timing_text)
        except (TypeError, ValueError):
            original_caption_timing(caption, audio, sample_rate)
        else:
            original_caption_timing(caption, audio, sample_rate, timing_text)

    def wrapped_voice_changed(name: str) -> None:
        _append_webui_log(string("webui.voice_changed", voice=name))
        original_voice_changed(name)

    def wrapped_input_state(locked: bool) -> None:
        has_voice = bool(celune.current_voice) or bool(celune.voices)
        _api.webui_input_locked = locked or not has_voice
        _api.webui_input_placeholder = _webui_input_placeholder(
            celune,
            _api.webui_input_locked,
            has_voice,
        )
        original_input_state(locked)

    def wrapped_voice_lock_state(locked: bool) -> None:
        _api.webui_voice_locked = (
            locked
            or len(celune.voices) < 2
            or not bool(celune.current_voice or celune.voices)
        )
        original_voice_lock_state(locked)

    glow = getattr(celune, "glow", None)
    if glow is not None and available("fatal", obj=glow):
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            original_fatal()
            _api._shutdown_api_for_fatal_error()

        glow.fatal = wrapped_fatal

    celune.log_callback = wrapped_log
    celune.status_callback = wrapped_status
    celune.error_callback = wrapped_error
    celune.idle_callback = wrapped_idle
    celune.queue_avail_callback = wrapped_queue_available
    celune.progress_callback = wrapped_progress
    celune.caption_progress_callback = wrapped_caption_progress
    celune.caption_callback = wrapped_caption
    celune.caption_timing_callback = wrapped_caption_timing
    celune.voice_changed_callback = wrapped_voice_changed
    celune.change_input_state_callback = wrapped_input_state
    celune.change_voice_lock_state_callback = wrapped_voice_lock_state
    celune._webui_callbacks_wrapped = True


def _sync_webui_runtime_locks(celune: _api.Celune, *, locked: bool) -> None:
    """Synchronize browser input controls from one runtime transition."""
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    _api.webui_input_locked = locked or not has_voice
    _api.webui_input_placeholder = _webui_input_placeholder(
        celune,
        _api.webui_input_locked,
        has_voice,
    )
    _api.webui_voice_locked = (
        locked or len(celune.voices) < 2 or celune.is_in_tutorial or not has_voice
    )


def require_celune() -> _api.Celune:
    """Return the bound Celune instance or fail the request.

    Returns:
        Celune: The bound Celune instance set for the request.

    Raises:
        HTTPException: The user has requested an API route that required Celune, but Celune wasn't available.
    """
    if _api.bound_celune is None:
        raise HTTPException(
            status_code=503,
            detail=string("webui.not_available"),
        )
    return _api.bound_celune


def api_log(action: str, content: str, suffix: str = "") -> None:
    """Print the API control log line.

    Args:
        action: The request made by the user.
        content: The request body sent by the user.
        suffix: The suffix to append to the log line.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    preview = content.replace("\n", "\\n").replace("\r", "\\r")[:64]
    if len(content) > 64:
        preview += "..."
    ui = CeluneUI._instance
    if ui is None or not getattr(ui, "_runtime_log_capture_enabled", False):
        _append_webui_log(f"{action} {preview!r}{suffix}")
    try:
        print(f"[{timestamp}] {action} {preview!r}{suffix}", flush=True)
    except ValueError:
        # Some embedded launch paths can close stdout while the WebUI stays alive.
        pass


def _normalized_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return stereo audio in frame-major form for file encoding."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[0] == 2 and normalized.shape[1] != 2:
        return normalized.transpose()
    return normalized


def _flac_bytes(audio: npt.NDArray[np.float32], sample_rate: int = BASE_SR) -> bytes:
    """Encode audio as PCM24 FLAC bytes."""
    buffer = io.BytesIO()
    sf.write(
        buffer,
        _normalized_audio(audio),
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )
    return buffer.getvalue()


def audio_bytes(
    chunks: SpeechStreamQueue,
    on_chunk: Optional[Callable[[int], None]] = None,
) -> Iterator[bytes]:
    """Yield one FLAC payload from queued 48 kHz stereo float32 chunks.

    Args:
        chunks: A queue of audio chunks.
        on_chunk: Optional callback invoked after each audio chunk is received.

    Returns:
        Iterator[bytes]: The audio chunk from the queue as raw bytes.

    Raises:
        item: An exception class was raised, causing the stream to be interrupted.
        Exception: The stream was interrupted by Celune.
    """
    audio_chunks: AudioChunks = []
    chunk_count = 0
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        audio_chunks.append(_normalized_audio(item))
        chunk_count += 1
        if on_chunk is not None:
            on_chunk(chunk_count)

    if audio_chunks:
        yield _flac_bytes(np.concatenate(audio_chunks))
    else:
        yield _flac_bytes(np.empty((0, 2), dtype=np.float32))


def stream_headers(sample_rate: int = BASE_SR) -> dict[str, str]:
    """Return headers describing the FLAC response.

    Args:
        sample_rate: Sample rate advertised in the response headers.

    Returns:
        dict[str, str]: Response headers for a FLAC response.
    """
    return {
        "X-Audio-Format": "flac-pcm24",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "2",
    }


def _remember_speech_job(job_id: str, job: _api.SpeechJob) -> None:
    """Store one speech job and remove expired entries."""
    with _api.speech_jobs_lock:
        _delete_expired_speech_jobs(time.time())
        _api.speech_jobs[job_id] = job


def _forget_speech_job(job_id: str) -> None:
    """Remove one speech job that was rejected before it could be observed."""
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.pop(job_id, None)
        if job is not None:
            for subscription in job.subscriptions:
                subscription.close()
            job.subscriptions.clear()


def _delete_expired_speech_jobs(now: float) -> None:
    """Remove jobs older than the in-memory job TTL."""
    expired_ids = [
        job_id
        for job_id, job in _api.speech_jobs.items()
        if now - job.created_at > _api.speech_job_ttl_seconds
    ]
    for job_id in expired_ids:
        _api.speech_jobs.pop(job_id, None)


def _update_speech_job(
    job_id: str,
    *,
    status: TaskStatus,
    audio: Optional[bytes] = None,
    error: Optional[str] = None,
) -> None:
    """Update one speech job if it still exists."""
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.audio = audio
        job.error = error


def _publish_task_event(job_id: str, event: _api.TaskEvent) -> None:
    """Append one task event and fan it out to current subscriptions."""
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.get(job_id)
        if job is None:
            return
        job.events.append(event)
        subscriptions = tuple(job.subscriptions)

    for subscription in subscriptions:
        subscription.put(event)


def _task_status(job_id: str) -> Optional[TaskStatus]:
    """Return the current task status for API event association."""
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.get(job_id)
        return None if job is None else job.status


def _task_event_status(job_id: str) -> Optional[TaskStatus]:
    """Return a non-terminal task status suitable for callback events."""
    status = _task_status(job_id)
    if status in {"completed", "failed", "cancelled"}:
        return None
    return status


def _publish_task_progress(
    job_id: str,
    *,
    current: Optional[float] = None,
    total: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """Publish a safe progress or status update for one task."""
    status = _task_event_status(job_id)
    if status is None:
        return
    _publish_task_event(
        job_id,
        _api.TaskEvent(
            task_id=job_id,
            event="progress",
            status=status,
            current=current,
            total=total,
            message=message,
        ),
    )


def _publish_active_task_log(message: str, severity: str = "info") -> None:
    """Mirror one safe Core status callback into the active speech task."""
    task_id = _api.active_speech_task_id
    if task_id is None:
        return
    if severity != "error" and message.startswith("["):
        return
    status = _task_event_status(task_id)
    if status is None:
        return
    safe_message = (
        string("status.could_not_continue", app_name=APP_NAME)
        if severity == "error"
        else message
    )
    _publish_task_event(
        task_id,
        _api.TaskEvent(
            task_id=task_id,
            event="log",
            status=status,
            message=safe_message,
            severity=severity,
        ),
    )


def _publish_active_task_progress(
    current: Optional[float],
    total: Optional[float],
) -> None:
    """Mirror one Core progress callback into the active speech task."""
    task_id = _api.active_speech_task_id
    if task_id is not None:
        _publish_task_progress(task_id, current=current, total=total)


def _set_active_speech_task(task_id: Optional[str]) -> None:
    """Set the API task receiving Core speech callbacks."""
    _api.active_speech_task_id = task_id


def _subscribe_to_speech_job(job_id: str) -> Optional[_api.TaskSubscription]:
    """Subscribe to one speech job and replay its retained event history."""
    subscription = _api.TaskSubscription(loop=asyncio.get_running_loop())
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.get(job_id)
        if job is None:
            return None
        for event in job.events:
            subscription.put(event)
        job.subscriptions.append(subscription)
    return subscription


def _unsubscribe_from_speech_job(
    job_id: str,
    subscription: _api.TaskSubscription,
) -> None:
    """Detach one WebSocket subscription without changing task execution."""
    with _api.speech_jobs_lock:
        job = _api.speech_jobs.get(job_id)
        if job is not None:
            try:
                job.subscriptions.remove(subscription)
            except ValueError:
                pass
    subscription.close()


def _speech_job_snapshot(job_id: str) -> Optional[_api.SpeechJob]:
    """Return a copy of one speech job for response handling."""
    with _api.speech_jobs_lock:
        _delete_expired_speech_jobs(time.time())
        job = _api.speech_jobs.get(job_id)
        if job is None:
            return None
        return _api.SpeechJob(
            status=job.status,
            created_at=job.created_at,
            audio=job.audio,
            error=job.error,
        )


def _collect_speech_job(job_id: str, chunks: SpeechStreamQueue) -> None:
    """Consume a speech stream queue and store its final FLAC payload."""
    try:
        audio = b"".join(
            audio_bytes(
                chunks,
                on_chunk=lambda count: _publish_task_progress(
                    job_id,
                    current=float(count),
                ),
            )
        )
    except Exception as e:
        if _task_status(job_id) == "cancelled":
            _set_active_speech_task(None)
            return
        _update_speech_job(
            job_id,
            status="failed",
            error=format_error_message(
                string("pipeline.gen_error"),
                e,
                getattr(_api.bound_celune, "log_level", "info"),
            ),
        )
        _publish_task_event(
            job_id,
            _api.TaskEvent(
                task_id=job_id,
                event="failed",
                status="failed",
                error="generation_failed",
            ),
        )
        _set_active_speech_task(None)
        return

    if _task_status(job_id) == "cancelled":
        _set_active_speech_task(None)
        return

    _update_speech_job(job_id, status="completed", audio=audio)
    _publish_task_event(
        job_id,
        _api.TaskEvent(
            task_id=job_id,
            event="completed",
            status="completed",
            location=f"/v1/speak/jobs/{job_id}",
        ),
    )
    _set_active_speech_task(None)


def _webui_logs_html() -> str:
    """Render the mirrored log buffer as terminal-like HTML."""
    if not _api.webui_log_lines:
        content = _webui_log_line_html("Waiting for response...")
    else:
        content = "\n".join(
            _webui_log_line_html(line, severity)
            for line, severity in _api.webui_log_lines
        )
    return f'<div id="celune-log-panel"><pre>{content}</pre></div>'


def _webui_status_html() -> str:
    """Render the footer status cell."""
    color = _webui_status_color(_api.webui_status_severity)
    details: list[str] = []
    if _api.webui_caption_active and _api.webui_caption_text:
        details.append(
            f'<div class="webui-caption">{escape(_api.webui_caption_text)}</div>'
        )
        if _api.webui_caption_progress > 0.0:
            details.append(
                f'<div class="webui-caption-progress">{round(_api.webui_caption_progress * 100):d}%</div>'
            )
    detail_html = "".join(details)
    return (
        f"{_webui_theme_html()}"
        '<div class="footer-block" '
        f'style="color: {color};">{escape(_api.webui_status_text)}{detail_html}</div>'
    )


def _webui_resources_html() -> str:
    """Render the footer resource cell."""
    celune = _api.bound_celune
    resource = ""
    if celune is not None:
        pages = ui_resources.resource_pages(celune, _api.webui_active_theme_name)
        if pages:
            resource = pages[_api.webui_resource_page % len(pages)]
    recording_hint = _webui_recording_hint(celune)
    hint_html = (
        f'<div class="webui-recording-hint">{escape(recording_hint)}</div>'
        if recording_hint
        else ""
    )
    if "CTRL+" in resource:
        return (
            '<div class="footer-block">'
            f"{hint_html}"
            f'<span class="webui-desktop-only">{escape(resource)}</span>'
            '<span class="webui-mobile-only">Use buttons for controls</span>'
            "</div>"
        )
    return f'<div class="footer-block">{hint_html}{escape(resource)}</div>'


def _voice_button_update() -> WebUiUpdate:
    """Return the current browser voice-button state."""
    celune = _api.bound_celune
    if celune is None:
        return gr.update(value="Loading", interactive=False)

    has_voice = bool(celune.current_voice) or bool(celune.voices)
    voice_name = celune.current_voice or (
        celune.voices[0] if celune.voices else "No Voice Set"
    )
    interactive = (
        not _api.webui_voice_locked and len(celune.voices) >= 2 and has_voice
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else len(celune.voices) >= 2 and not celune.is_in_tutorial
    )
    return gr.update(
        value=voice_name.capitalize(),
        interactive=interactive,
    )


def _webui_vc_mode_active(celune: Optional[_api.Celune]) -> bool:
    """Return whether the browser UI should expose active VC controls."""
    if celune is None:
        return False
    predicate = getattr(celune, "is_voice_conversion_mode", None)
    if callable(predicate):
        return bool(predicate())
    return bool(
        getattr(celune, "input_mode", "text_to_speech") == "voice_conversion"
        or getattr(celune, "vc_backend", None) is not None
    )


def _webui_persona_loaded(celune: _api.Celune) -> bool:
    """Return whether the attached runtime has loaded Persona."""
    persona_ready = getattr(celune, "persona_ready", None)
    if persona_ready is None:
        return bool(getattr(celune, "vision", None))
    return bool(persona_ready)


def _webui_persona_input_available(celune: _api.Celune) -> bool:
    """Return whether browser text input can use Persona talkback."""
    config = getattr(celune, "config", {})
    return (
        _webui_persona_loaded(celune)
        and isinstance(config, dict)
        and persona_enabled(config)
        and persona_talkback_enabled(config)
    )


def _webui_recording_hint(celune: Optional[_api.Celune]) -> str:
    """Return the live-recording shortcut shown in the browser footer."""
    if (
        celune is None
        or CeluneUI._instance is None
        or _api.webui_input_locked
        or getattr(celune, "is_in_tutorial", False)
    ):
        return ""
    if _webui_vc_mode_active(celune):
        return string("webui.recording_toggle_hint")
    if _webui_persona_input_available(celune):
        return string("webui.recording_voice_hint")
    return ""


def _webui_input_placeholder(
    celune: _api.Celune,
    locked: bool,
    has_voice: bool,
) -> str:
    """Return the current browser input placeholder string."""
    if celune.is_in_tutorial:
        return string("ui.tutorial_placeholder")
    if locked or not has_voice:
        return string("ui.wait_placeholder")
    if _webui_vc_mode_active(celune):
        return string("ui.voice_changer_placeholder")
    if _webui_persona_input_available(celune):
        return string("ui.say_placeholder")
    return string("ui.input_placeholder")


def _input_update(
    value: Union[Optional[str], _WebUiUnset] = _WEBUI_UNSET,
) -> WebUiUpdate:
    """Return the current browser input state."""
    has_value = value is not _WEBUI_UNSET
    celune = _api.bound_celune
    if celune is None:
        if has_value:
            return gr.update(
                value=value,
                interactive=False,
                placeholder=string("ui.wait_placeholder"),
            )
        return gr.update(
            interactive=False,
            placeholder=string("ui.wait_placeholder"),
        )
    if celune.is_in_tutorial:
        if has_value:
            return gr.update(
                value=value,
                interactive=False,
                placeholder=string("ui.tutorial_placeholder"),
            )
        return gr.update(
            interactive=False,
            placeholder=string("ui.tutorial_placeholder"),
        )
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    vc_mode = _webui_vc_mode_active(celune)
    if getattr(celune, "_webui_callbacks_wrapped", False):
        interactive = not _api.webui_input_locked and has_voice and not vc_mode
        placeholder = _webui_input_placeholder(
            celune, _api.webui_input_locked, has_voice
        )
    else:
        interactive = not celune.locked and has_voice and not vc_mode
        placeholder = _webui_input_placeholder(celune, celune.locked, has_voice)
    if has_value:
        return gr.update(
            value=value,
            interactive=interactive,
            placeholder=placeholder,
        )
    return gr.update(
        interactive=interactive,
        placeholder=placeholder,
    )


def _send_button_update() -> WebUiUpdate:
    """Return the current browser send-button state."""
    celune = _api.bound_celune
    if celune is None:
        return gr.update(interactive=False)
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    vc_mode = _webui_vc_mode_active(celune)
    interactive = (
        not _api.webui_input_locked and has_voice and not vc_mode
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else not celune.is_in_tutorial
        and not celune.locked
        and has_voice
        and not vc_mode
    )
    return gr.update(interactive=interactive)


def _webui_vc_controls_update() -> tuple[
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser VC control state."""
    celune = _api.bound_celune
    vc_enabled = _webui_vc_mode_active(celune)
    control_interactive = celune is not None and vc_enabled
    return (
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
    )


def _webui_snapshot() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser UI snapshot."""
    _seed_webui_logs()
    _probe_webui_runtime()
    return (
        _webui_logs_html(),
        _webui_status_html(),
        _webui_resources_html(),
        _voice_button_update(),
        _send_button_update(),
        _input_update(),
    )


def _webui_submit_snapshot(
    input_value: Optional[str],
) -> tuple[
    WebUiUpdate,
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return a browser snapshot shaped for submit/click handlers."""
    (
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
        _input,
    ) = _webui_snapshot()
    return (
        _input_update(input_value),
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
    )


def install(target):
    """Install extracted definitions in the original module."""
    install_module_functions(target, {name: globals()[name] for name in __all__})
