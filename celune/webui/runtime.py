# SPDX-License-Identifier: Apache-2.0
"""WebUI lifecycle helpers."""

from __future__ import annotations
import io
import re
import time
import inspect
import datetime
import contextlib
from html import escape
from pathlib import Path
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING, Union, Optional, cast
from collections.abc import Mapping, Callable, Iterator, Sequence

import gradio as gr
import numpy as np
import soundfile as sf
from fastapi import HTTPException

from ..i18n import string
from ..theme import colors
from ..server.jobs import (
    TaskEvent,
    task_status,
    update_speech_job,
    publish_task_event,
    publish_task_progress,
    set_active_speech_task,
    publish_active_task_log,
    publish_active_task_progress,
)
from ..ui.app import CeluneUI
from ..server.constants import (
    WEBUI_GENERATION_API_NAMES,
    WEBUI_RESOURCE_ROTATE_SECONDS,
    WEBUI_TIMED_UPDATE_STALE_SECONDS,
    WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS,
)
from ..dataclasses.events import (
    AgentTaskFinishedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)
from ..cevoice import default_loader
from ..typing.api import WebUiUpdate
from ..constants import BASE_SR, APP_NAME
from ..ui import resources as ui_resources
from ..extensions.events import EventDispatcher
from ..typing.pipeline import SpeechStreamQueue
from ..typing.aliases import LogLevel, AudioChunks
from ..typing.api import WebUiUnset as _WebUiUnset
from ..utils import available, format_error_message
from ..typing.api import WEBUI_UNSET as _WEBUI_UNSET
from ..typing.events import EventName, EventCallback
from ..paths import project_root, main_window_log_path
from ..pipeline.captions import current_playback_status
from ..server.session import ServerSession, server_session
from ..cedts.ui import UiTimedUpdate, ui_timed_update_channel
from ..persona.impl import persona_enabled, persona_talkback_enabled
from ..server.runtime import invoke_message_callback, shutdown_api_for_fatal_error

npt = np.typing

if TYPE_CHECKING:
    from ..celune import Celune

__all__ = (
    "_probe_webui_runtime",
    "_probed_status_text",
    "_publish_webui_agent_status",
    "_sync_webui_runtime_locks",
    "_voice_button_update",
    "_webui_agent_approval_requested",
    "_webui_agent_choice_requested",
    "_webui_agent_status_message",
    "_webui_agent_task_finished",
    "_webui_agent_task_state_changed",
    "_webui_log_line_html",
    "_webui_persona_input_available",
    "_webui_persona_loaded",
    "_webui_recording_hint",
    "_webui_status_color",
    "api_log",
    "append_webui_error",
    "append_webui_log",
    "audio_bytes",
    "collect_speech_job",
    "configure_webui_generation_fn_indices",
    "configure_webui_theme",
    "flac_bytes",
    "get_webui_input_placeholder",
    "input_update",
    "normalized_audio",
    "receive_webui_timed_update",
    "require_celune",
    "seed_webui_logs",
    "send_button_update",
    "set_webui_status",
    "stream_headers",
    "strip_webui_log_prefix",
    "subscribe_webui_events",
    "subscribe_webui_timed_updates",
    "webui_audio_waveform_options",
    "webui_logs_html",
    "webui_resources_html",
    "webui_snapshot",
    "webui_status_html",
    "webui_submit_snapshot",
    "webui_vc_controls_update",
    "webui_vc_mode_active",
    "wrap_celune_callbacks",
)


def configure_webui_theme(session: ServerSession = server_session) -> None:
    """Sync the browser palette with the active CEVOICE-derived theme."""
    colors.configure_theme()
    loader = default_loader()
    if loader is not None:
        theme = loader.bundle.metadata.get("theme")
        if isinstance(theme, dict):
            background = theme.get("background")
            accent = theme.get("accent")
            faded_accent = theme.get("faded_accent")
            if faded_accent is None:
                faded_accent = theme.get("sleeping_color")
            if (
                isinstance(background, str)
                and isinstance(accent, str)
                and (faded_accent is None or isinstance(faded_accent, str))
            ):
                colors.configure_theme(background, accent, faded_accent)

    background = colors.THEME.background or "#1d1826"
    palette = colors.SEVERITY_COLORS["celune"]
    primary = palette["info"]
    error = palette["error"]
    foreground = colors.THEME.foreground or "#ffffff"
    secondary = colors.THEME.secondary or primary
    accent = colors.THEME.accent or primary
    sleeping = palette["sleeping"]
    button_bg = colors.blend(primary, background, 0.72)
    button_hover = colors.blend(primary, background, 0.6)
    input_bg = colors.blend(primary, background, 0.78)
    theme_css = read_webui_asset("theme.css")
    theme_values = {
        "BACKGROUND": background,
        "PRIMARY": primary,
        "ERROR": error,
        "FOREGROUND": foreground,
        "SECONDARY": secondary,
        "ACCENT": accent,
        "INPUT_BG": input_bg,
        "SLEEPING": sleeping,
        "BUTTON_BG": button_bg,
        "BUTTON_HOVER": button_hover,
    }
    for key, value in theme_values.items():
        theme_css = theme_css.replace(f"__{key}__", value)
    session.webui_theme_style = theme_css


def configure_webui_generation_fn_indices(
    dependencies: Sequence[Mapping[str, object]],
    session: ServerSession = server_session,
) -> None:
    """Store Gradio action IDs used by the API generation-rate limiter."""
    session.webui_generation_fn_indices = frozenset(
        dependency_id
        for dependency in dependencies
        if isinstance(dependency.get("api_name"), str)
        and dependency.get("api_name") in WEBUI_GENERATION_API_NAMES
        and isinstance((dependency_id := dependency.get("id")), int)
    )


def subscribe_webui_events(
    celune: Celune,
    session: ServerSession = server_session,
) -> None:
    """Subscribe the browser UI to the shared typed agent lifecycle events."""
    dispatcher = getattr(celune, "_event_dispatcher", None)
    if not isinstance(dispatcher, EventDispatcher):
        return

    callbacks: tuple[tuple[EventName, EventCallback], ...] = (
        (
            "agent_task_state_changed",
            partial(_webui_agent_task_state_changed, session=session),
        ),
        (
            "agent_approval_requested",
            partial(_webui_agent_approval_requested, session=session),
        ),
        (
            "agent_choice_requested",
            partial(_webui_agent_choice_requested, session=session),
        ),
        (
            "agent_task_finished",
            partial(_webui_agent_task_finished, session=session),
        ),
    )
    for event_name, callback in callbacks:
        dispatcher.subscribe(event_name, callback, "WebUI")
    session.webui_event_dispatcher = dispatcher
    session.webui_event_callbacks = callbacks


def subscribe_webui_timed_updates(session: ServerSession = server_session) -> None:
    """Subscribe the browser UI to the shared CEDTS timed-update channel."""
    if session.webui_timed_update_unsubscribe is not None:
        session.webui_timed_update_unsubscribe()
    session.webui_timed_update_unsubscribe = ui_timed_update_channel.subscribe(
        partial(receive_webui_timed_update, session=session)
    )


def receive_webui_timed_update(
    update: UiTimedUpdate,
    session: ServerSession = server_session,
) -> None:
    """Apply one newer TUI timed update to browser-owned state."""
    celune = session.bound_celune
    if celune is None or update.runtime_id != str(id(celune)):
        return
    if update.sequence <= session.webui_timed_update_sequence:
        return
    session.webui_timed_update_sequence = update.sequence
    session.webui_resource_page = update.resource_page
    session.webui_active_theme_name = update.theme_name
    session.webui_timed_update_received_at = time.monotonic()
    session.webui_timed_update_source = "cedts"
    if update.status_text:
        session.webui_status_text = update.status_text
        session.webui_status_severity = update.status_severity


def _webui_agent_status_message(
    task_id: str,
    fallback_state: str,
    fallback_key: Optional[str] = None,
    session: ServerSession = server_session,
) -> Optional[str]:
    """Resolve the same compact agent status text used by the TUI."""
    celune = session.bound_celune
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
    session: ServerSession = server_session,
) -> None:
    """Mirror one typed agent event into the browser status line."""
    message = _webui_agent_status_message(
        task_id,
        fallback_state,
        fallback_key,
        session,
    )
    if message is not None:
        severity = "warning" if "awaiting" in message.casefold() else "info"
        set_webui_status(message, severity, source="agent", session=session)


def _webui_agent_task_state_changed(
    event: AgentTaskStateChangedEvent,
    session: ServerSession = server_session,
) -> None:
    """Mirror agent task state changes into the browser UI."""
    state = getattr(event.new_state, "value", str(event.new_state))
    _publish_webui_agent_status(event.task_id, state, session=session)


def _webui_agent_approval_requested(
    event: AgentApprovalRequestedEvent,
    session: ServerSession = server_session,
) -> None:
    """Mirror agent approval pauses into the browser UI."""
    _publish_webui_agent_status(
        event.task_id,
        "awaiting_approval",
        "agent.status.awaiting_approval",
        session,
    )


def _webui_agent_choice_requested(
    event: AgentChoiceRequestedEvent,
    session: ServerSession = server_session,
) -> None:
    """Mirror agent choice pauses into the browser UI."""
    _publish_webui_agent_status(
        event.task_id,
        "awaiting_choice",
        "agent.status.awaiting_choice",
        session,
    )


def _webui_agent_task_finished(
    event: AgentTaskFinishedEvent,
    session: ServerSession = server_session,
) -> None:
    """Mirror terminal agent task states into the browser UI."""
    state = getattr(event.state, "value", str(event.state))
    _publish_webui_agent_status(event.task_id, state, session=session)


def _webui_status_color(
    severity: str,
    session: ServerSession = server_session,
) -> str:
    """Return the browser UI color for a given severity."""
    palette = colors.SEVERITY_COLORS.get(
        session.webui_active_theme_name,
        colors.SEVERITY_COLORS["celune"],
    )
    return palette.get(severity, palette["info"])


def _webui_log_line_html(
    message: str,
    severity: str = "info",
    session: ServerSession = server_session,
) -> str:
    """Render one browser log line with severity-aware coloring."""
    color = _webui_status_color(severity, session)
    return f'<span style="color: {color};">{escape(message)}</span>'


def webui_audio_waveform_options() -> gr.WaveformOptions:
    """Return theme-bound waveform colors for Gradio audio components."""
    primary = colors.SEVERITY_COLORS["celune"]["info"]
    secondary = colors.THEME.secondary or colors.FADED_ACCENT
    return gr.WaveformOptions(
        waveform_color=secondary,
        waveform_progress_color=primary,
        trim_region_color=primary,
    )


def strip_webui_log_prefix(line: str) -> str:
    """Remove persisted timestamp and severity prefixes from one log line."""
    stripped = line.strip()
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    return stripped


def seed_webui_logs(session: ServerSession = server_session) -> None:
    """Populate the browser log view from the persisted desktop log when available."""
    if session.webui_logs_seeded:
        return

    session.webui_logs_seeded = True
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
            append_webui_log("\n".join(record), record_severity, session)

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
            record.append(strip_webui_log_prefix(line))
    append_record()


def append_webui_log(
    msg: str,
    severity: str = "info",
    session: ServerSession = server_session,
) -> None:
    """Store one browser log line."""
    if session.webui_log_lines and session.webui_log_lines[-1] == (msg, severity):
        return
    session.webui_log_lines.append((msg, severity))


def append_webui_error(
    message: str,
    error: BaseException,
    severity: str = "error",
    celune: Optional[Celune] = None,
    session: ServerSession = server_session,
) -> None:
    """Store a WebUI error with detail selected by the active log level."""
    runtime = celune or session.bound_celune
    append_webui_log(
        format_error_message(
            message,
            error,
            getattr(runtime, "log_level", "info"),
        ),
        severity,
        session,
    )


def set_webui_status(
    msg: str,
    severity: str = "info",
    *,
    source: str = "callback",
    updated_at: Optional[float] = None,
    session: ServerSession = server_session,
) -> None:
    """Update the browser UI status line."""
    session.webui_status_text = msg
    session.webui_status_severity = severity
    session.webui_status_source = source
    session.webui_status_updated_at = (
        time.monotonic() if updated_at is None else updated_at
    )


def _probed_status_text(celune: Celune) -> tuple[str, str]:
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


def _probe_webui_runtime(session: ServerSession = server_session) -> None:
    """Poll the live runtime so the WebUI footer updates even without new log lines."""

    celune = session.bound_celune
    if celune is None:
        return

    now = time.monotonic()
    current_state = (celune.cur_state or "").strip().lower()
    playback_status = current_playback_status(celune)
    if playback_status is not None:
        if (
            session.webui_status_text != playback_status
            or session.webui_status_source != "playback"
        ):
            set_webui_status(
                playback_status,
                "info",
                source="playback",
                updated_at=now,
                session=session,
            )
        session.webui_last_probed_state = current_state
    elif current_state == "sleeping":
        sleeping_log = string("webui.sleeping_log", app_name=APP_NAME)
        if not any(message == sleeping_log for message, _ in session.webui_log_lines):
            append_webui_log(sleeping_log, "sleeping", session)
        sleeping_status, sleeping_severity = _probed_status_text(celune)
        if (
            session.webui_status_text != sleeping_status
            or session.webui_status_severity != sleeping_severity
            or session.webui_status_source in {"callback", "agent", "cedts"}
        ):
            set_webui_status(
                sleeping_status,
                sleeping_severity,
                source="probe",
                updated_at=now,
                session=session,
            )
    if playback_status is None and current_state != session.webui_last_probed_state:
        status_text, severity = _probed_status_text(celune)
        should_override_status = (
            session.webui_last_probed_state is None
            or session.webui_status_text == string("status.api_starting")
            or session.webui_status_source not in {"callback", "agent", "cedts"}
            or now - session.webui_status_updated_at
            >= WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS
            or current_state in {"idle", "sleeping", "error"}
        )
        if should_override_status:
            set_webui_status(
                status_text,
                severity,
                source="probe",
                updated_at=now,
                session=session,
            )
        session.webui_last_probed_state = current_state

    pages = ui_resources.resource_pages(celune, session.webui_active_theme_name)
    if not pages:
        return

    if session.webui_last_resource_advance <= 0:
        session.webui_last_resource_advance = now
        return

    timed_update_is_fresh = (
        session.webui_timed_update_source == "cedts"
        and now - session.webui_timed_update_received_at
        < WEBUI_TIMED_UPDATE_STALE_SECONDS
    )
    if (
        not timed_update_is_fresh
        and now - session.webui_last_resource_advance >= WEBUI_RESOURCE_ROTATE_SECONDS
    ):
        session.webui_resource_page = (session.webui_resource_page + 1) % len(pages)
        session.webui_last_resource_advance = now


def _invoke_caption_progress_callback(
    callback: Callable[..., None],
    progress: Optional[float],
    total: Optional[float],
    visible_words: Optional[int],
) -> None:
    """Forward aligned caption progress to callbacks that support word counts.

    Args:
        callback: Existing callback to preserve when WebUI wraps Celune.
        progress: Played speech frames.
        total: Total speech frames, when known.
        visible_words: Number of caption words aligned to played audio.
    """
    if visible_words is None:
        callback(progress, total)
        return

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(progress, total)
        return

    try:
        signature.bind(progress, total, visible_words)
    except TypeError:
        try:
            signature.bind(progress, total, visible_words=visible_words)
        except TypeError:
            callback(progress, total)
        else:
            callback(progress, total, visible_words=visible_words)
    else:
        callback(progress, total, visible_words)


def wrap_celune_callbacks(
    celune: Celune,
    session: ServerSession = server_session,
) -> None:
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
        Callable[..., None],
        getattr(celune, "caption_progress_callback", lambda _progress, _total: None),
    )
    original_caption = cast(
        Callable[[Optional[str]], None],
        getattr(celune, "caption_callback", lambda _caption: None),
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
        publish_active_task_log(msg, severity, session)
        append_webui_log(msg, severity, session)
        invoke_message_callback(original_log, msg, severity, loglevel)

    def wrapped_status(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        publish_active_task_log(msg, severity, session)
        set_webui_status(msg, severity, source="callback", session=session)
        invoke_message_callback(original_status, msg, severity, loglevel)

    def wrapped_error(msg: str) -> None:
        publish_active_task_log(
            string("status.could_not_continue", app_name=APP_NAME),
            "error",
            session,
        )
        append_webui_log(msg, "error", session)
        set_webui_status(msg, "error", source="callback", session=session)
        original_error(msg)

    def wrapped_progress(
        progress: Optional[float],
        total: Optional[float],
    ) -> None:
        session.webui_progress_current = progress
        session.webui_progress_total = total
        publish_active_task_progress(progress, total, session)
        original_progress(progress, total)

    def wrapped_idle() -> None:
        original_idle()
        session.webui_caption_active = False
        session.webui_caption_text = ""
        session.webui_caption_progress = 0.0
        session.webui_progress_current = None
        session.webui_progress_total = None
        _sync_webui_runtime_locks(
            celune,
            locked=getattr(celune, "locked", False),
            session=session,
        )
        if getattr(celune, "sleeping", False):
            set_webui_status(
                string("status.sleeping"),
                "sleeping",
                source="callback",
                session=session,
            )
        elif getattr(celune, "cur_state", "") not in {"reloading", "waking"}:
            set_webui_status(string("status.idle"), source="callback", session=session)

    def wrapped_queue_available() -> None:
        original_queue_available()
        locked = bool(getattr(celune, "is_in_tutorial", False))
        _sync_webui_runtime_locks(celune, locked=locked, session=session)
        set_webui_status(
            string("status.speaking"),
            source="callback",
            session=session,
        )

    def wrapped_caption_progress(
        progress: Optional[float],
        total: Optional[float],
        visible_words: Optional[int] = None,
    ) -> None:
        if visible_words is not None:
            total_words = len(session.webui_caption_text.split())
            if total_words > 0:
                session.webui_caption_progress = max(
                    0.0,
                    min(1.0, visible_words / total_words),
                )
        elif total is not None and total > 0:
            session.webui_caption_progress = max(
                0.0, min(1.0, (progress or 0.0) / total)
            )
        _invoke_caption_progress_callback(
            original_caption_progress,
            progress,
            total,
            visible_words,
        )

    def wrapped_caption(caption: Optional[str]) -> None:
        if caption:
            session.webui_caption_active = True
            session.webui_caption_text = caption
            session.webui_caption_progress = 0.0
        else:
            session.webui_caption_active = False
            session.webui_caption_text = ""
            session.webui_caption_progress = 0.0
        original_caption(caption)

    def wrapped_voice_changed(name: str) -> None:
        append_webui_log(string("webui.voice_changed", voice=name), session=session)
        original_voice_changed(name)

    def wrapped_input_state(locked: bool) -> None:
        has_voice = bool(celune.current_voice) or bool(celune.voices)
        session.webui_input_locked = locked or not has_voice
        session.webui_input_placeholder = get_webui_input_placeholder(
            celune,
            session.webui_input_locked,
            has_voice,
        )
        original_input_state(locked)

    def wrapped_voice_lock_state(locked: bool) -> None:
        session.webui_voice_locked = (
            locked
            or len(celune.voices) < 2
            or not bool(celune.current_voice or celune.voices)
            or getattr(getattr(celune, "backend", None), "singing", False)
        )
        original_voice_lock_state(locked)

    glow = getattr(celune, "glow", None)
    if glow is not None and available("fatal", obj=glow):
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            original_fatal()
            shutdown_api_for_fatal_error(session)

        glow.fatal = wrapped_fatal

    celune.log_callback = wrapped_log
    celune.status_callback = wrapped_status
    celune.error_callback = wrapped_error
    celune.idle_callback = wrapped_idle
    celune.queue_avail_callback = wrapped_queue_available
    celune.progress_callback = wrapped_progress
    celune.caption_progress_callback = wrapped_caption_progress
    celune.caption_callback = wrapped_caption
    celune.voice_changed_callback = wrapped_voice_changed
    celune.change_input_state_callback = wrapped_input_state
    celune.change_voice_lock_state_callback = wrapped_voice_lock_state
    celune._webui_callbacks_wrapped = True


def _sync_webui_runtime_locks(
    celune: Celune,
    *,
    locked: bool,
    session: ServerSession = server_session,
) -> None:
    """Synchronize browser input controls from one runtime transition."""
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    session.webui_input_locked = locked or not has_voice
    session.webui_input_placeholder = get_webui_input_placeholder(
        celune,
        session.webui_input_locked,
        has_voice,
    )
    session.webui_voice_locked = (
        locked
        or len(celune.voices) < 2
        or celune.is_in_tutorial
        or not has_voice
        or getattr(getattr(celune, "backend", None), "singing", False)
    )


def require_celune(session: ServerSession = server_session) -> Celune:
    """Return the bound Celune instance or fail the request.

    Returns:
        Celune: The bound Celune instance set for the request.

    Raises:
        HTTPException: The user has requested an API route that required Celune, but Celune wasn't available.
    """
    if session.bound_celune is None:
        raise HTTPException(
            status_code=503,
            detail=string("webui.not_available"),
        )
    return session.bound_celune


def api_log(
    action: str,
    content: str,
    suffix: str = "",
    session: ServerSession = server_session,
) -> None:
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
    ui = CeluneUI.get_instance()
    if ui is None or not getattr(ui, "_runtime_log_capture_enabled", False):
        append_webui_log(f"{action} {preview!r}{suffix}", session=session)
    try:
        print(f"[{timestamp}] {action} {preview!r}{suffix}", flush=True)
    except ValueError:
        # Celune has no stdout attached, but WebUI wants to display a message
        pass


def normalized_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return stereo audio in frame-major form for file encoding."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[0] == 2 and normalized.shape[1] != 2:
        return normalized.transpose()
    return normalized


def flac_bytes(audio: npt.NDArray[np.float32], sample_rate: int = BASE_SR) -> bytes:
    """Encode audio as PCM24 FLAC bytes."""
    buffer = io.BytesIO()
    sf.write(
        buffer,
        normalized_audio(audio),
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

        audio_chunks.append(normalized_audio(item))
        chunk_count += 1
        if on_chunk is not None:
            on_chunk(chunk_count)

    if audio_chunks:
        yield flac_bytes(np.concatenate(audio_chunks))
    else:
        yield flac_bytes(np.empty((0, 2), dtype=np.float32))


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


def collect_speech_job(
    job_id: str,
    chunks: SpeechStreamQueue,
    session: ServerSession = server_session,
) -> None:
    """Consume a speech stream queue and store its final FLAC payload."""
    try:
        audio = b"".join(
            audio_bytes(
                chunks,
                on_chunk=lambda count: publish_task_progress(
                    job_id,
                    current=float(count),
                    session=session,
                ),
            )
        )
    except Exception as e:
        if task_status(job_id, session) == "cancelled":
            set_active_speech_task(None, session)
            return
        update_speech_job(
            job_id,
            status="failed",
            error=format_error_message(
                string("pipeline.gen_error"),
                e,
                getattr(session.bound_celune, "log_level", "info"),
            ),
            session=session,
        )
        publish_task_event(
            job_id,
            TaskEvent(
                task_id=job_id,
                event="failed",
                status="failed",
                error="generation_failed",
            ),
            session,
        )
        set_active_speech_task(None, session)
        return

    if task_status(job_id, session) == "cancelled":
        set_active_speech_task(None, session)
        return

    update_speech_job(job_id, status="completed", audio=audio, session=session)
    publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="completed",
            status="completed",
            location=f"/v1/speak/jobs/{job_id}",
        ),
        session,
    )
    set_active_speech_task(None, session)


def webui_logs_html(session: ServerSession = server_session) -> str:
    """Render the mirrored log buffer as terminal-like HTML."""
    if not session.webui_log_lines:
        content = _webui_log_line_html("Waiting for response...", session=session)
    else:
        content = "\n".join(
            _webui_log_line_html(line, severity, session)
            for line, severity in session.webui_log_lines
        )
    return f'<div id="celune-log-panel"><pre>{content}</pre></div>'


def webui_status_html(session: ServerSession = server_session) -> str:
    """Render the footer status cell."""
    color = _webui_status_color(session.webui_status_severity, session)
    return (
        '<div class="footer-block" '
        f'style="color: {color};">{escape(session.webui_status_text)}</div>'
    )


def webui_resources_html(session: ServerSession = server_session) -> str:
    """Render the footer resource cell."""
    celune = session.bound_celune
    resource = ""
    if celune is not None:
        pages = ui_resources.resource_pages(celune, session.webui_active_theme_name)
        if pages:
            resource = pages[session.webui_resource_page % len(pages)]
    recording_hint = _webui_recording_hint(celune, session)
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


def _voice_button_update(session: ServerSession = server_session) -> WebUiUpdate:
    """Return the current browser voice-button state."""
    celune = session.bound_celune
    if celune is None:
        return gr.update(value="Loading", interactive=False)

    has_voice = bool(celune.current_voice) or bool(celune.voices)
    voice_name = celune.current_voice or (
        celune.voices[0] if celune.voices else "No Voice Set"
    )
    interactive = (
        not session.webui_voice_locked
        and len(celune.voices) >= 2
        and has_voice
        and not getattr(getattr(celune, "backend", None), "singing", False)
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else len(celune.voices) >= 2
        and not celune.is_in_tutorial
        and not getattr(getattr(celune, "backend", None), "singing", False)
    )
    return gr.update(
        value=voice_name.capitalize(),
        interactive=interactive,
    )


def webui_vc_mode_active(celune: Optional[Celune]) -> bool:
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


def _webui_persona_loaded(celune: Celune) -> bool:
    """Return whether the attached runtime has loaded Persona."""
    persona_ready = getattr(celune, "persona_ready", None)
    if persona_ready is None:
        return bool(getattr(celune, "vision", None))
    return bool(persona_ready)


def _webui_persona_input_available(celune: Celune) -> bool:
    """Return whether browser text input can use Persona talkback."""
    config = getattr(celune, "config", {})
    return (
        _webui_persona_loaded(celune)
        and isinstance(config, dict)
        and persona_enabled(config)
        and persona_talkback_enabled(config)
    )


def _webui_recording_hint(
    celune: Optional[Celune],
    session: ServerSession = server_session,
) -> str:
    """Return the live-recording shortcut shown in the browser footer."""
    if (
        celune is None
        or CeluneUI.get_instance() is None
        or session.webui_input_locked
        or getattr(celune, "is_in_tutorial", False)
    ):
        return ""
    if webui_vc_mode_active(celune):
        return string("webui.recording_toggle_hint")
    if _webui_persona_input_available(celune):
        return string("webui.recording_voice_hint")
    return ""


def get_webui_input_placeholder(
    celune: Celune,
    locked: bool,
    has_voice: bool,
) -> str:
    """Return the current browser input placeholder string."""
    if celune.is_in_tutorial:
        return string("ui.tutorial_placeholder")
    if locked or not has_voice:
        return string("ui.wait_placeholder")
    if webui_vc_mode_active(celune):
        return string("ui.voice_changer_placeholder")
    if getattr(getattr(celune, "backend", None), "singing", False):
        return string("ui.cesn_input_placeholder")
    if _webui_persona_input_available(celune):
        return string("ui.say_placeholder")
    return string("ui.input_placeholder")


def input_update(
    value: Union[Optional[str], _WebUiUnset] = _WEBUI_UNSET,
    session: ServerSession = server_session,
) -> WebUiUpdate:
    """Return the current browser input state."""
    has_value = value is not _WEBUI_UNSET
    celune = session.bound_celune
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
    vc_mode = webui_vc_mode_active(celune)
    if getattr(celune, "_webui_callbacks_wrapped", False):
        interactive = not session.webui_input_locked and has_voice and not vc_mode
        placeholder = get_webui_input_placeholder(
            celune, session.webui_input_locked, has_voice
        )
    else:
        interactive = not celune.locked and has_voice and not vc_mode
        placeholder = get_webui_input_placeholder(celune, celune.locked, has_voice)
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


def send_button_update(session: ServerSession = server_session) -> WebUiUpdate:
    """Return the current browser send-button state."""
    celune = session.bound_celune
    if celune is None:
        return gr.update(interactive=False)
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    vc_mode = webui_vc_mode_active(celune)
    interactive = (
        not session.webui_input_locked and has_voice and not vc_mode
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else not celune.is_in_tutorial
        and not celune.locked
        and has_voice
        and not vc_mode
    )
    return gr.update(interactive=interactive)


def webui_vc_controls_update(
    session: ServerSession = server_session,
) -> tuple[
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser VC control state."""
    celune = session.bound_celune
    vc_enabled = webui_vc_mode_active(celune)
    control_interactive = celune is not None and vc_enabled
    return (
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
    )


def webui_snapshot(
    session: ServerSession = server_session,
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser UI snapshot."""
    seed_webui_logs(session)
    _probe_webui_runtime(session)
    return (
        webui_logs_html(session),
        webui_status_html(session),
        webui_resources_html(session),
        _voice_button_update(session),
        send_button_update(session),
        input_update(session=session),
    )


def webui_submit_snapshot(
    input_value: Optional[str],
    session: ServerSession = server_session,
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
    ) = webui_snapshot(session)
    return (
        input_update(input_value, session),
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
    )


_ASSET_MEDIA_TYPES = {
    "api_key.html": "text/html; charset=utf-8",
    "api_key.css": "text/css; charset=utf-8",
    "api_key.js": "text/javascript; charset=utf-8",
    "head.html": "text/html; charset=utf-8",
    "theme.css": "text/css; charset=utf-8",
    "webui.css": "text/css; charset=utf-8",
    "webui.js": "text/javascript; charset=utf-8",
}
_FONT_ASSET_NAMES = {
    "fonts/michroma-400.ttf": "michroma-400.ttf",
    "fonts/outfit-400.ttf": "outfit-400.ttf",
    "fonts/outfit-700.ttf": "outfit-700.ttf",
}


def read_webui_asset(name: str) -> str:
    """Read a packaged WebUI text asset by its fixed name."""
    if name not in _ASSET_MEDIA_TYPES:
        raise FileNotFoundError(name)
    return files("celune").joinpath("assets", name).read_text(encoding="utf-8")


def webui_asset_response_data(name: str) -> Optional[tuple[bytes, str]]:
    """Return safe packaged WebUI or bundled font data for an asset route."""
    media_type = _ASSET_MEDIA_TYPES.get(name)
    if media_type is not None:
        data = files("celune").joinpath("assets", name).read_bytes()
        return data, media_type

    font_name = _FONT_ASSET_NAMES.get(name)
    if font_name is None:
        return None
    font_path: Path = project_root() / "resources" / "fonts" / font_name
    try:
        return font_path.read_bytes(), "font/ttf"
    except OSError:
        return None
