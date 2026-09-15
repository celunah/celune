# SPDX-License-Identifier: Apache-2.0
"""Extracted CeluneUI methods group 1."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import inspect
import logging
import math
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Callable
from io import TextIOWrapper
from typing import Optional, cast

from . import app as _app
from ..binding import install_class_functions

__all__ = (
    "_agent_status_text",
    "_agent_task_for_display",
    "_apply_test_finished_state",
    "_apply_theme",
    "_bind_agent_events",
    "_bind_runtime_callbacks",
    "_cancel_sleep_timer",
    "_caption_word_timing_ranges",
    "_caption_words_for_progress",
    "_chain_runtime_callback",
    "_check_launcher_loss",
    "_clear_border_pulses",
    "_dismiss_loading_screen",
    "_emit_startup_diagnostic",
    "_ensure_startup_error_themes_registered",
    "_ensure_themes_registered",
    "_enter_sleep_mode",
    "_finish_test_startup",
    "_forward_low_level_stderr",
    "_handle_deferred_runtime_error",
    "_has_celune",
    "_install_low_level_stderr_capture",
    "_is_agent_test_mode",
    "_is_textual_terminal_frame",
    "_is_ui_test_mode",
    "_load_deferred_runtime",
    "_on_agent_approval_requested",
    "_on_agent_choice_requested",
    "_on_agent_task_finished",
    "_on_agent_task_state_changed",
    "_persist_log_entry",
    "_prepare_loading_theme",
    "_prepare_terminal_output_stream",
    "_publish_webui_timed_update",
    "_redirect_dunder_stdio",
    "_refresh_agent_status",
    "_refresh_logs",
    "_refresh_status",
    "_refresh_theme_text",
    "_remove_low_level_stderr_capture",
    "_remove_runtime_log_redirects",
    "_render_status_text",
    "_restore_dunder_stdio",
    "_restore_progress_bar",
    "_run_on_ui_thread",
    "_runtime_theme_name",
    "_schedule_sleep_timer",
    "_set_progress_row_display",
    "_set_terminal_status",
    "_severity_color",
    "_show_loading_error",
    "_show_loading_screen",
    "_start_deferred_runtime",
    "_startup_terminal_status_for",
    "_status_view_width",
    "_terminal_status_for",
    "_unbind_agent_events",
    "_update_loading_log",
    "_update_status_label",
    "_write_terminal_escape",
    "_write_terminal_title",
    "advance_resources",
    "attach_celune",
    "compose",
    "on_mount",
    "on_resize",
    "prepare_theme",
    "receive_startup_diagnostic",
    "run",
    "safe_progress",
    "start_background_init",
    "tts_caption_timing",
    "update_resources",
)


def _run_on_ui_thread(self, callback: Callable[[], None]) -> None:
    if threading.current_thread() is threading.main_thread():
        callback()
    else:
        try:
            self.call_from_thread(callback)
        except RuntimeError:
            pass


def _severity_color(self, severity: str = "info") -> str:
    """Return the current theme color for a log severity."""
    return _app.severity_color(self.active_theme_name, severity)


def _runtime_theme_name(self) -> str:
    """Return the current Textual theme name, including runtime error overrides."""
    if self._fatal_error_active:
        if self.active_theme_name == "celune_light":
            return "celune_light_error"
        return "celune_error"
    return self.active_theme_name


def _ensure_themes_registered(self) -> None:
    """Register Celune's built-in themes when the app is not fully mounted yet."""
    if _app.colors.THEME.name not in self.available_themes:
        self.register_theme(_app.colors.THEME)
    if _app.colors.THEME_LIGHT.name not in self.available_themes:
        self.register_theme(_app.colors.THEME_LIGHT)
    if _app.colors.THEME_APRIL_FOOLS.name not in self.available_themes:
        self.register_theme(_app.colors.THEME_APRIL_FOOLS)
    self._register_runtime_error_themes()


def _ensure_startup_error_themes_registered(self) -> None:
    """Register error themes without importing full runtime dependencies."""
    for theme in _app.default_error_theme_family():
        if theme.name not in self.available_themes:
            self.register_theme(theme)


def _prepare_loading_theme(self) -> None:
    """Apply Celune's palette before the first loading frame is rendered."""
    dark_theme, light_theme = _app.default_theme_family()
    self.register_theme(dark_theme)
    self.register_theme(light_theme)
    self.theme = self.active_theme_name
    self.refresh_css(animate=False)


def _apply_theme(self, theme_name: str) -> None:
    """Apply theme and repaint theme-sensitive widgets."""
    self._clear_border_pulses()
    self.active_theme_name = theme_name
    self.theme = self._runtime_theme_name()
    self._refresh_status()
    self._refresh_theme_text()
    self._refresh_logs(recolor=True)


def _has_celune(self) -> bool:
    """Is the app attached to this UI instance?"""
    return self.celune is not None


def prepare_theme(self) -> None:
    """Prepare the selected Celune theme before the first rendered frame."""
    if not _app._RUNTIME_DEPENDENCIES_LOADED:
        _app._load_ui_runtime_dependencies()
    _app.colors.configure_theme()

    if self._has_celune():
        loader_factory = _app.default_loader
        if loader_factory is None:
            _app._load_ui_runtime_dependencies()
            loader_factory = _app.default_loader
        loader = loader_factory() if loader_factory is not None else None
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
                    _app.colors.configure_theme(
                        background,
                        accent,
                        faded_accent,
                    )

    self._ensure_themes_registered()
    if _app.is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
        "1",
        "true",
        "on",
        "yes",
        "enabled",
    }:
        self.active_theme_name = "celune_april_fools"
    else:
        theme = os.getenv("CELUNE_THEME") or (
            self.celune.config.get("theme", "dark")
            if self.celune is not None
            else "dark"
        )

        if theme == "dark":
            self.active_theme_name = "celune"
        elif theme == "light":
            self.active_theme_name = "celune_light"
        else:
            self.active_theme_name = "celune"

    self.theme = self.active_theme_name
    self.refresh_css(animate=False)


def _clear_border_pulses(self) -> None:
    """Remove temporary border pulse overrides so CSS can theme them."""
    for widget_key, widget in list(self._border_pulse_widgets.items()):
        self._border_pulse_tokens[widget_key] = (
            self._border_pulse_tokens.get(widget_key, 0) + 1
        )
        widget.styles.border = None
        widget.refresh(layout=False)

    self._border_pulse_widgets.clear()
    self._border_pulse_tokens.clear()


def _refresh_theme_text(self) -> None:
    """Refresh widgets after a runtime theme change."""
    from ..utils import available

    def repaint(widget: _app.Widget) -> None:
        refresh = getattr(widget, "refresh", None)
        if refresh is None:
            return
        try:
            refresh(layout=False)
        except TypeError:
            refresh()

    runtime_theme_name = self._runtime_theme_name()
    self._ensure_themes_registered()
    if self.theme != runtime_theme_name:
        self.theme = runtime_theme_name
    try:
        screen = self.screen
    except _app.ScreenStackError:
        screen = None
    if screen is not None and available("styles", obj=screen):
        screen.styles.background = None
        repaint(screen)
    if self.logs is not None:
        self.logs.styles.color = None
        self.logs.styles.border = None
        self.logs.styles.background = None
        self.logs.styles.scrollbar_color = None
        self.logs.styles.scrollbar_color_hover = None
        self.logs.styles.scrollbar_color_active = None
        self.logs.styles.scrollbar_background = None
        self.logs.styles.scrollbar_background_hover = None
        self.logs.styles.scrollbar_background_active = None
        repaint(self.logs)
    if self.input_box is not None:
        self.input_box.styles.color = None
        self.input_box.styles.border = None
        self.input_box.styles.background = None
        self.input_box.styles.scrollbar_color = None
        self.input_box.styles.scrollbar_color_hover = None
        self.input_box.styles.scrollbar_color_active = None
        self.input_box.styles.scrollbar_background = None
        self.input_box.styles.scrollbar_background_hover = None
        self.input_box.styles.scrollbar_background_active = None
        repaint(self.input_box)
    if self.style_button is not None:
        self.style_button.styles.color = None
        self.style_button.styles.border = None
        self.style_button.styles.background = None
        repaint(self.style_button)
    if self.resources is not None:
        self.resources.styles.color = None
        repaint(self.resources)
    if self.header is not None:
        self.header.styles.color = None
        repaint(self.header)
    for line in self.header_lines:
        line.styles.border_top = None
        repaint(line)
    if self.progress_bar is not None and available("styles", obj=self.progress_bar):
        self.progress_bar.styles.color = None
        self.progress_bar.styles.background = None
        repaint(self.progress_bar)
    if self.caption is not None and available("styles", obj=self.caption):
        self.caption.styles.color = None
        self.caption.styles.background = None
        repaint(self.caption)


def _bind_runtime_callbacks(self) -> None:
    """Bind one attached Celune instance back into this UI."""
    if self.celune is None:
        return

    callbacks: tuple[tuple[str, Callable[..., None]], ...] = (
        ("log_callback", self.tts_log),
        ("status_callback", self.safe_status),
        ("error_callback", self.error),
        ("idle_callback", self.tts_idle),
        ("queue_avail_callback", self.tts_queue_avail),
        ("voice_changed_callback", self.tts_voice_changed),
        ("change_input_state_callback", self.change_input_state),
        ("change_voice_lock_state_callback", self.change_voice_lock_state),
        ("progress_callback", self.safe_progress),
        ("caption_progress_callback", self.safe_caption_progress),
        ("caption_callback", self.tts_caption),
        ("caption_timing_callback", self.tts_caption_timing),
    )
    for attribute, callback in callbacks:
        self._chain_runtime_callback(attribute, callback)


def _chain_runtime_callback(
    self,
    attribute: str,
    callback: Callable[..., None],
) -> None:
    """Add one UI callback without overwriting another frontend callback."""
    if self.celune is None:
        return

    current_value = getattr(self.celune, attribute, None)
    if not callable(current_value):
        setattr(self.celune, attribute, callback)
        return
    current = cast(Callable[..., None], current_value)
    if current == callback:
        return
    if attribute == "log_callback" and callback == getattr(
        self.celune, "_startup_log_sink", None
    ):
        return
    chained_callbacks = getattr(current, "_celune_callback_chain", ())
    if callback in chained_callbacks:
        return

    def invoke(
        target: Callable[..., None],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            target(*args, **kwargs)
            return
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            target(*args)
        else:
            target(*args, **kwargs)

    def chained(*args: object, **kwargs: object) -> None:
        invoke(callback, args, kwargs)
        invoke(current, args, kwargs)

    chained._celune_callback_chain = (  # type: ignore[attr-defined]
        *chained_callbacks,
        callback,
    )
    setattr(self.celune, attribute, chained)


def _publish_webui_timed_update(self) -> None:
    """Publish the current TUI timing state through the CEDTS UI channel."""
    if self.celune is None:
        return

    try:
        from ..cedts.ui import UiTimedUpdate, ui_timed_update_channel
    except ImportError:
        return

    sequence = getattr(self, "_webui_timed_update_sequence", 0) + 1
    self._webui_timed_update_sequence = sequence
    ui_timed_update_channel.publish(
        UiTimedUpdate(
            runtime_id=str(id(self.celune)),
            sequence=sequence,
            emitted_at=time.monotonic(),
            resource_page=self._resource_page,
            theme_name=self.active_theme_name,
            status_text=self._status_text,
            status_severity=self.status_severity,
            status_marquee_offset=self._status_marquee_offset,
        )
    )


def _bind_agent_events(self) -> None:
    """Subscribe the UI to the existing typed agent lifecycle events."""
    self._unbind_agent_events()
    if self.celune is None:
        return
    dispatcher = getattr(self.celune, "_event_dispatcher", None)
    if dispatcher is None:
        return

    dispatcher.subscribe(
        "agent_task_state_changed",
        self._on_agent_task_state_changed,
        "CeluneUI",
    )
    dispatcher.subscribe(
        "agent_approval_requested",
        self._on_agent_approval_requested,
        "CeluneUI",
    )
    dispatcher.subscribe(
        "agent_choice_requested",
        self._on_agent_choice_requested,
        "CeluneUI",
    )
    dispatcher.subscribe(
        "agent_task_finished",
        self._on_agent_task_finished,
        "CeluneUI",
    )
    self._agent_event_dispatcher = dispatcher


def _unbind_agent_events(self) -> None:
    """Unsubscribe UI lifecycle callbacks before replacing or closing Celune."""
    dispatcher = self._agent_event_dispatcher
    if dispatcher is None:
        return

    dispatcher.unsubscribe(
        "agent_task_state_changed", self._on_agent_task_state_changed
    )
    dispatcher.unsubscribe(
        "agent_approval_requested", self._on_agent_approval_requested
    )
    dispatcher.unsubscribe("agent_choice_requested", self._on_agent_choice_requested)
    dispatcher.unsubscribe("agent_task_finished", self._on_agent_task_finished)
    self._agent_event_dispatcher = None


def _on_agent_task_state_changed(
    self,
    event: _app.AgentTaskStateChangedEvent,
) -> None:
    """Refresh the UI after a typed agent task transition."""
    self._agent_task_id = event.task_id
    self._run_on_ui_thread(self._refresh_agent_status)


def _on_agent_approval_requested(
    self,
    event: _app.AgentApprovalRequestedEvent,
) -> None:
    """Refresh the UI when a task pauses for approval."""
    self._agent_task_id = event.task_id
    self._run_on_ui_thread(self._refresh_agent_status)


def _on_agent_choice_requested(
    self,
    event: _app.AgentChoiceRequestedEvent,
) -> None:
    """Refresh the UI when a task pauses for a user choice."""
    self._agent_task_id = event.task_id
    self._run_on_ui_thread(self._refresh_agent_status)


def _on_agent_task_finished(self, event: _app.AgentTaskFinishedEvent) -> None:
    """Refresh the UI once a task reaches its terminal lifecycle state."""
    self._agent_task_id = event.task_id
    self._run_on_ui_thread(self._refresh_agent_status)


def _agent_task_for_display(self) -> Optional[_app.AgentTask]:
    """Return the active or most recently evented task for status rendering."""
    celune = self.celune
    if celune is None:
        return None
    runtime = getattr(celune, "agent_runtime", None)
    if runtime is None:
        return None

    active_task = runtime.get_active_task("default")
    if active_task is not None:
        self._agent_task_id = active_task.task_id
        return active_task
    if self._agent_task_id is None:
        return None
    with contextlib.suppress(ValueError):
        return runtime.get_task(self._agent_task_id)
    return None


def _agent_status_text(
    task: Optional[_app.AgentTask],
    busy_components: tuple[_app.ComponentLockName, ...],
) -> Optional[str]:
    """Resolve one localized status message from typed task and lock state."""
    if busy_components:
        labels = ", ".join(
            _app.string(f"agent.component_{component.value}")
            for component in busy_components
        )
        return _app.string("agent.status.busy_components", components=labels)
    if task is None:
        return None
    if task.needs_context_compaction and task.state in {
        _app.AgentTaskState.PLANNING,
        _app.AgentTaskState.WORKING,
    }:
        return _app.string("agent.status.compacting")
    if task.state in {
        _app.AgentTaskState.WORKING,
        _app.AgentTaskState.PLANNING,
        _app.AgentTaskState.EXECUTING_TOOL,
        _app.AgentTaskState.RESPONDING,
    }:
        return _app.string(
            "agent.status.working",
            iteration=task.iterations,
            maximum=task.config.max_loops,
        )
    status_keys = {
        _app.AgentTaskState.QUEUED: "agent.status.queued",
        _app.AgentTaskState.IDLE: "agent.status.idle",
        _app.AgentTaskState.CLASSIFYING: "agent.status.classifying",
        _app.AgentTaskState.AWAITING_APPROVAL: "agent.status.awaiting_approval",
        _app.AgentTaskState.AWAITING_CHOICE: "agent.status.awaiting_choice",
        _app.AgentTaskState.PAUSED: "agent.status.paused",
        _app.AgentTaskState.INTERRUPTED: "agent.status.interrupted",
        _app.AgentTaskState.CANCELLING: "agent.status.cancelling",
        _app.AgentTaskState.COMPLETED: "agent.status.completed",
        _app.AgentTaskState.FAILED: "agent.status.failed",
        _app.AgentTaskState.CANCELLED: "agent.status.cancelled",
        _app.AgentTaskState.ABORTED: "agent.status.aborted",
    }
    key = status_keys.get(task.state)
    return _app.string(key) if key is not None else None


def _refresh_agent_status(self) -> None:
    """Project typed agent progress and component contention into the UI status."""
    celune = self.celune
    if celune is None or getattr(celune, "test_finished", False):
        return
    if getattr(celune, "cur_state", None) == "stopped":
        return

    task = self._agent_task_for_display()
    busy = getattr(celune, "last_component_busy", None)
    busy_components = tuple(getattr(busy, "components", ()))
    message = self._agent_status_text(task, busy_components)
    if message is None:
        return

    task_id = task.task_id if task is not None else ""
    task_state = task.state if task is not None else None
    iterations = task.iterations if task is not None else 0
    maximum = task.config.max_loops if task is not None else 0
    signature = (
        task_id,
        task_state.value if task_state is not None else "",
        str(iterations),
        str(maximum),
        *(component.value for component in busy_components),
        message,
    )
    self._agent_task_state = task_state
    self._agent_iterations = iterations
    self._agent_max_loops = maximum
    self._agent_busy_components = busy_components
    if self._agent_status_signature == signature:
        return
    self._agent_status_signature = signature
    self.safe_status(message, "warning" if busy_components else "info")


def _is_ui_test_mode(self) -> bool:
    """Return whether the attached runtime is the interactive fake-backend UI test mode."""
    if self.celune is None:
        return False

    backend_mode = getattr(self.celune, "backend_mode", None)
    if isinstance(backend_mode, str):
        return backend_mode == "ui_test"
    backend = getattr(self.celune, "backend", None)
    return bool(getattr(backend, "is_fake", False)) and "pytest" not in sys.modules


def _is_agent_test_mode(self) -> bool:
    """Return whether the attached runtime is the restricted agent test mode."""
    return bool(
        self.celune is not None
        and getattr(self.celune, "backend_mode", None) == "agent_test"
    )


def _finish_test_startup(
    self,
    success: bool,
    detail: Optional[str] = None,
) -> None:
    """Report explicit test-mode startup completion once to the runner."""
    callback = self._test_completion_callback
    if callback is None or self.celune is None:
        return
    try:
        callback(self.celune, success, detail)
    except Exception as error:
        self.safe_log(
            _app.format_error_message(
                _app.string("test.callback_failed"),
                error,
                _app.resolve_log_level(
                    getattr(self.celune, "log_level", None),
                    self._startup_log_level,
                ),
            ),
            "error",
        )
    self._run_on_ui_thread(self._apply_test_finished_state)


def _apply_test_finished_state(self) -> None:
    """Reconcile the visible UI with a completed explicit test."""
    celune = self.celune
    if celune is None or not getattr(celune, "test_finished", False):
        return

    self._hide_caption_widgets()
    self.change_input_state(locked=True)
    self.change_voice_lock_state(locked=True)

    voice = getattr(celune, "current_voice", None)
    if not isinstance(voice, str) or not voice:
        voices = getattr(celune, "voices", ())
        voice = voices[0] if voices else None
    if isinstance(voice, str) and voice:
        self.tts_voice_changed(voice)

    if self.input_box is not None:
        self.input_box.placeholder = _app.string("ui.stopped_placeholder")
    self.safe_status(_app.string("status.stopped"), "sleeping")
    self._refresh_logs()


def _refresh_status(self) -> None:
    """Refresh the status color for the active theme."""
    if self.status is None:
        return
    self.status.styles.color = self._severity_color(self.status_severity)


def _status_view_width(self) -> int:
    """Estimate how many status characters can fit without clipping."""
    if self.status is None:
        return 32

    size = getattr(self.status, "size", None)
    width = getattr(size, "width", 0) if size is not None else 0
    if isinstance(width, int) and width > 6:
        return max(8, width - 2)
    return 32


def _render_status_text(self) -> str:
    """Return the current status text, marqueeing when it exceeds the label width."""
    width = self._status_view_width()
    if len(self._status_text) <= width:
        self._status_marquee_offset = 0
        return _app.indent(self._status_text, spaces=2)

    loop = f"{self._status_text}{self._status_marquee_gap}"
    offset = self._status_marquee_offset % len(loop)
    window = (loop * 2)[offset : offset + width]
    return _app.indent(window, spaces=2)


def _update_status_label(self) -> None:
    """Push the current status text into the label."""
    if self.status is None:
        return
    self.status.update(self._render_status_text())
    self._refresh_status()


def on_resize(self, _event: _app.events.Resize) -> None:
    """Re-render width-sensitive widgets after the window size changes.

    Args:
        _event: Textual resize event that triggered the redraw.
    """
    if self.status is not None:
        self._update_status_label()


def _refresh_logs(self, *, recolor: bool = False) -> None:
    """Reconcile visible log entries with the retained UI log history.

    Args:
        recolor: Whether existing entries should be rebuilt with the active
            theme colors.
    """
    if self.logs is None:
        return

    with self._log_history_lock:
        history = tuple(self.log_history)

    if not recolor:
        if self._rendered_log_count > len(history):
            self.logs.clear()
            self._rendered_log_count = 0
        elif self._rendered_log_count == len(history):
            if not history or not getattr(self.logs, "_size_known", True):
                return
            if self.logs.lines:
                if self.logs.auto_scroll:
                    self.logs.scroll_end(
                        animate=False,
                        immediate=True,
                        force=True,
                    )
                return
            self.logs.clear()
            self._rendered_log_count = 0

        for message, severity in history[self._rendered_log_count :]:
            self.logs.write(
                _app.Text(message, style=self._severity_color(severity)),
            )
            self._rendered_log_count += 1
        return

    scroll_offset = self.logs.scroll_offset
    auto_scroll = self.logs.auto_scroll
    self.logs.auto_scroll = False
    self.logs.clear()

    for message, severity in history:
        self.logs.write(
            _app.Text(message, style=self._severity_color(severity)),
            scroll_end=False,
        )

    self.logs.auto_scroll = auto_scroll
    self.logs.scroll_to(
        scroll_offset.x,
        scroll_offset.y,
        animate=False,
        immediate=True,
        force=True,
    )
    self._rendered_log_count = len(history)


def _persist_log_entry(self, msg: str, severity: str) -> None:
    """Append one UI log entry to the persisted main-window log file."""
    with contextlib.suppress(OSError):
        if not self._log_file_initialized:
            self._log_file_path.write_text("", encoding="utf-8")
            self._log_file_initialized = True

        timestamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        with self._log_file_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{severity.upper()}] {msg}\n")


def _prepare_terminal_output_stream(self) -> Optional[TextIOWrapper]:
    """Give Textual an independent terminal stream before stderr capture starts."""
    if os.name == "nt" or self._terminal_output_stream is not None:
        return self._terminal_output_stream

    original_stderr = sys.__stderr__
    if original_stderr is None:
        return None

    fileno = getattr(original_stderr, "fileno", None)
    if not callable(fileno):
        return None

    output_fd: Optional[int] = None
    output_stream: Optional[TextIOWrapper] = None
    _app.discard(output_stream)
    try:
        output_fd = os.dup(cast(Callable[[], int], fileno)())
        output_stream = os.fdopen(
            output_fd,
            "w",
            encoding=getattr(original_stderr, "encoding", None) or "utf-8",
            errors=getattr(original_stderr, "errors", None) or "replace",
            buffering=1,
        )
    except (OSError, TypeError, ValueError):
        if output_fd is not None:
            with contextlib.suppress(OSError):
                os.close(output_fd)
        return None

    if output_stream is None:
        return None

    self._terminal_output_stream = output_stream
    sys.__stderr__ = output_stream
    return output_stream


def run(
    self,
    *,
    headless: bool = False,
    inline: bool = False,
    inline_no_clear: bool = False,
    mouse: bool = True,
    size: Optional[tuple[int, int]] = None,
    auto_pilot: Optional[_app.AutopilotCallbackType] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Optional[_app.ReturnType]:
    """Run Textual with an output stream independent of low-level stderr capture.

    Args:
        headless: Whether to run without interactive terminal input.
        inline: Whether to use Textual's inline terminal mode.
        inline_no_clear: Whether inline mode should preserve existing terminal output.
        mouse: Whether mouse input should be enabled.
        size: Optional terminal size override.
        auto_pilot: Optional callback used to drive automated interaction.
        loop: Optional event loop used by the Textual application.

    Returns:
        Optional return value produced by the Textual application.
    """
    original_stderr = sys.__stderr__
    output_stream = self._prepare_terminal_output_stream()

    try:
        return super(_app.CeluneUI, self).run(
            headless=headless,
            inline=inline,
            inline_no_clear=inline_no_clear,
            mouse=mouse,
            size=size,
            auto_pilot=auto_pilot,
            loop=loop,
        )
    finally:
        if output_stream is not None:
            sys.__stderr__ = original_stderr
            self._terminal_output_stream = None
            with contextlib.suppress(OSError, ValueError):
                output_stream.close()


def compose() -> _app.ComposeResult:
    """Define the UI.

    Returns:
        ComposeResult: The root widget tree for the interface.
    """
    with _app.Vertical(id="container"):
        with _app.Horizontal(id="header-container"):
            yield _app.Label("", classes="line")
            yield _app.Label(_app.APP_NAME, id="header")
            yield _app.Label("", classes="line")
        yield _app.RichLog(id="logs", wrap=True, markup=False)
        yield _app.Label("", id="caption", markup=False)
        with _app.Horizontal(id="progress-container"):
            yield _app.ProgressBar(
                id="progress", show_percentage=False, show_eta=False, total=1
            )
            yield _app.ProgressLabel(widget_id="progress-label")
        with _app.Horizontal(id="controls"):
            yield _app.TextArea(
                id="input", placeholder=_app.string("ui.wait_placeholder")
            )
            yield _app.VoiceButton(
                _app.string("ui.no_voice_set"),
                id="style",
                actions=_app.ButtonActions(press=False, hold=False),
            )
            yield _app.Button(
                _app.string("ui.vc_mode_talk"),
                id="vc-mode",
                actions=_app.ButtonActions(press=False),
            )
            yield _app.Button(
                _app.string("ui.vc_pitch_button", value="+0"),
                id="vc-pitch",
                actions=_app.ButtonActions(press=False),
            )
        with _app.Horizontal(id="bottom"):
            yield _app.Label("", id="status")
            yield _app.Label("", id="resources")
    yield _app.CeluneLoadingScreen(widget_id="loading-overlay")


def on_mount(self) -> None:
    """Prepare the UI and start deferred runtime initialization."""
    if os.name == "nt":
        self._install_windows_signal_handler()
    else:
        if _app.SIGTSTP is not None:
            signal.signal(_app.SIGTSTP, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    self.set_interval(0.1, self._check_launcher_loss)

    self._loading_screen = self.query_one("#loading-overlay", _app.CeluneLoadingScreen)
    self._loading_screen.set_startup_messages(self._startup_messages)
    self.logs = self.query_one("#logs", _app.RichLog)
    self.input_box = self.query_one("#input", _app.TextArea)
    self.status = self.query_one("#status", _app.Label)
    self.resources = self.query_one("#resources", _app.Label)
    self.caption = self.query_one("#caption", _app.Label)
    self.style_button = self.query_one("#style", _app.VoiceButton)
    self.vc_mode_button = self.query_one("#vc-mode", _app.Button)
    self.vc_pitch_button = self.query_one("#vc-pitch", _app.Button)
    self.progress_bar = self.query_one("#progress", _app.ProgressBar)
    self.progress_label = self.query_one("#progress-label", _app.ProgressLabel)
    self.header = self.query_one("#header", _app.Label)
    self.header_lines = tuple(
        cast(_app.Label, widget) for widget in self.query(".line")
    )
    self._refresh_logs()

    self.set_focus(None)
    self._prepare_loading_theme()
    self._show_loading_screen()
    if self._loading_screen is not None:
        self._loading_screen.set_status_message(_app.string("status.initializing"))
    self._status_text = _app.string("status.initializing")
    if self._startup_messages:
        terminal_status = self._startup_terminal_status_for(self._startup_messages[-1])
        if terminal_status is not None:
            self._set_terminal_status(*terminal_status)
        else:
            self._set_terminal_status(
                "initializing", _app.string("osc.action_starting")
            )
    else:
        self._set_terminal_status("initializing", _app.string("osc.action_starting"))
    if self._startup_loader is not None or self.celune is not None:
        if self._startup_loader is not None:
            self.call_after_refresh(self._start_deferred_runtime)
        else:
            self.attach_celune(self.celune)


def _start_deferred_runtime(self) -> None:
    """Start constructing Celune after the initial loading frame renders."""
    self.run_worker(self._load_deferred_runtime, thread=True, exclusive=True)


def _startup_terminal_status_for(
    message: str,
) -> Optional[tuple[str, str]]:
    """Resolve a loading-screen diagnostic to its terminal title transition."""
    startup_actions = {
        _app.string("ui.startup_checking_dependencies"): (
            "initializing",
            _app.string("osc.action_checking_dependencies"),
        ),
        _app.string("ui.startup_loading_core"): (
            "initializing",
            _app.string("osc.action_loading_core"),
        ),
        _app.string("ui.startup_initializing_core"): (
            "initializing",
            _app.string("osc.action_initializing_core"),
        ),
    }
    return startup_actions.get(message)


def receive_startup_diagnostic(self, message: str) -> None:
    """Display one early startup diagnostic on the loading screen.

    Args:
        message: Diagnostic emitted while the runtime is being prepared.
    """
    self._startup_messages.append(message)
    terminal_status = self._startup_terminal_status_for(message)
    if terminal_status is not None:
        self._set_terminal_status(*terminal_status)

    def update() -> None:
        if self._loading_screen is not None:
            self._loading_screen.append_startup_message(message)

    self._run_on_ui_thread(update)


def _emit_startup_diagnostic(self, message: str) -> None:
    """Display a verbose diagnostic for a stage of deferred startup.

    Args:
        message: Diagnostic text describing the current startup stage.
    """
    terminal_status = self._startup_terminal_status_for(message)
    if terminal_status is not None:
        self._set_terminal_status(*terminal_status)
    if self._startup_log_level != "info":
        self.receive_startup_diagnostic(message)


def _load_deferred_runtime(self) -> None:
    """Construct the engine and load optional UI integrations off the UI thread."""
    if self._startup_loader is None and self.celune is None:
        return
    try:
        celune = self.celune
        if self._startup_loader is not None:
            celune = self._startup_loader()
        _app._load_ui_runtime_dependencies()
    except BaseException as exc:
        self.call_from_thread(self._handle_deferred_runtime_error, exc)
        return
    if celune is not None:
        self.call_from_thread(self.attach_celune, celune)


def _handle_deferred_runtime_error(self, error: BaseException) -> None:
    """Show a deferred startup failure without tearing down the UI.

    Args:
        error: Exception raised while constructing the deferred runtime.
    """
    missing_dependency = isinstance(error, ModuleNotFoundError) or (
        isinstance(error, SystemExit)
        and error.code == _app.ExitCodes.EXIT_MISSING_DEPENDENCIES.value
    )
    self.cur_state = "error"
    self._startup_error_exit_code = (
        _app.ExitCodes.EXIT_MISSING_DEPENDENCIES.value
        if missing_dependency
        else _app.ExitCodes.EXIT_FAILURE.value
    )
    self._fatal_error_active = True
    self._ensure_startup_error_themes_registered()
    self.theme = self._runtime_theme_name()
    self.refresh_css(animate=False)
    if isinstance(error, ModuleNotFoundError) and error.name is not None:
        failure_status = _app.string("status.missing_dependency")
        terminal_action = _app.string("osc.action_missing_dependency")
    else:
        failure_status = _app.string("status.early_initialization_failed")
        terminal_action = _app.string("osc.action_early_initialization_failed")
    self._terminal_status = (
        _app.APP_NAME,
        _app.string("osc.state_error"),
        terminal_action,
    )
    self._write_terminal_title(self._terminal_status)
    message = _app.format_error_message(
        _app.tagged_string("ui.init_error", "INIT ERROR"),
        error,
        self._startup_log_level,
    )
    self._show_loading_error(
        message,
        status_message=_app.string("status.early_initialization_failed"),
        footer_message=failure_status,
    )


def attach_celune(self, celune: _app.Celune) -> None:
    """Attach the constructed engine and begin its normal initialization."""
    if threading.current_thread() is not threading.main_thread():
        self.call_from_thread(self.attach_celune, celune)
        return
    if self.cur_state == "exiting":
        celune.close()
        return

    if _app.default_loader is None or _app.ui_resources is None:
        _app._load_ui_runtime_dependencies()
    self._log_file_path = _app.main_window_log_path(create_parent=True)
    self._unbind_agent_events()
    self.celune = celune
    self._bind_runtime_callbacks()
    self._bind_agent_events()
    self.prepare_theme()
    self._wrap_runtime_fatal_glow()
    configured_theme = os.getenv("CELUNE_THEME") or self.celune.config.get(
        "theme", "dark"
    )
    if self.active_theme_name == "celune" and configured_theme not in {
        "dark",
        "light",
    }:
        self.safe_log(_app.string("ui.invalid_theme_defaulting_dark"), "warning")
    self._refresh_theme_text()
    self.refresh_vc_controls()
    if not self.celune.backend.is_fake or "pytest" in sys.modules:
        self._enable_runtime_log_capture()
    resources = _app.ui_resources
    if resources is None:
        return
    resources.prime_usage()
    resources.start_gpu_usage_worker()
    if not self._runtime_intervals_started:
        self.set_interval(_app.FOOTER_ROTATE_SECONDS, self.advance_resources)
        self._status_marquee_timer = self.set_interval(
            0.18, self._advance_status_marquee
        )
        self._runtime_intervals_started = True
    self.update_resources()
    self.call_after_refresh(self.start_background_init)


def _check_launcher_loss(self) -> None:
    """Run the normal UI shutdown path after the launcher disconnects."""
    if _app.launcher_loss_requested() and self.cur_state != "exiting":
        self._graceful_exit()


def update_resources(self) -> None:
    """Refresh the currently selected resource footer page."""
    if self.cur_state == "exiting" or self.resources is None or self.celune is None:
        return
    resources = _app.ui_resources
    if resources is None:
        return

    def update() -> None:
        pages = resources.resource_pages(self.celune, self.active_theme_name)
        text = pages[self._resource_page % len(pages)]

        self.resources.update(_app.indent(text, spaces=2, direction="right"))

    self._run_on_ui_thread(update)


def _write_terminal_escape(self, escape: str) -> None:
    """Write one ANSI escape sequence to the real terminal when available."""
    if self._log_stdout is not None:
        self._log_stdout.ansi(escape)
        return

    if self._old_stdout is not None:
        self._old_stdout.write(escape)
        self._old_stdout.flush()


def _write_terminal_title(self, status: tuple[str, str, str]) -> None:
    """Write one structured state title to the real terminal."""
    if self._log_stdout is not None:
        self._log_stdout.ansi(_app.terminal_title_escape(status))
        return

    if self._old_stdout is not None:
        _app.set_terminal_title(status, self._old_stdout)


def _set_terminal_status(self, state: str, action: str) -> None:
    """Publish a stable state and action in the terminal title."""
    status = (_app.APP_NAME, _app.string(f"osc.state_{state}"), action)
    if getattr(self, "_terminal_status", None) == status:
        return

    self._terminal_status = status

    def update() -> None:
        if _app.supports_ansi(self._old_stdout):
            self._write_terminal_title(status)

    self._run_on_ui_thread(update)


def _terminal_status_for(self, msg: str, severity: str) -> tuple[str, str]:
    """Resolve the terminal glossary state and action for one UI status."""
    status_actions = {
        _app.string("status.api_starting"): (
            "initializing",
            _app.string("osc.action_starting"),
        ),
        _app.string("status.could_not_continue"): (
            "error",
            _app.string("osc.action_failed"),
        ),
        _app.string("status.could_not_reload"): (
            "error",
            _app.string("osc.action_failed"),
        ),
        _app.string("status.could_not_start"): (
            "error",
            _app.string("osc.action_failed"),
        ),
        _app.string("status.could_not_wake"): (
            "error",
            _app.string("osc.action_failed"),
        ),
        _app.string("status.downloading_audio"): (
            "speaking",
            _app.string("osc.action_downloading"),
        ),
        _app.string("status.early_initialization_failed"): (
            "error",
            _app.string("osc.action_early_initialization_failed"),
        ),
        _app.string("status.failed_to_start"): (
            "error",
            _app.string("osc.action_failed"),
        ),
        _app.string("status.generating"): (
            "speaking",
            _app.string("osc.action_generating_audio"),
        ),
        _app.string("status.idle"): ("ready", _app.string("osc.action_idle")),
        _app.string("ui.idle_status"): ("ready", _app.string("osc.action_idle")),
        _app.string("status.initializing"): (
            "initializing",
            _app.string("osc.action_starting"),
        ),
        _app.string("status.missing_dependency"): (
            "error",
            _app.string("osc.action_missing_dependency"),
        ),
        _app.string("status.normalizing"): (
            "thinking",
            _app.string("osc.action_normalizing"),
        ),
        _app.string("status.reloading"): (
            "reloading",
            _app.string("osc.action_reloading"),
        ),
        _app.string("status.reloading_backend"): (
            "reloading",
            _app.string("osc.action_loading_backend"),
        ),
        _app.string("status.reloading_character"): (
            "reloading",
            _app.string("osc.action_loading_voice"),
        ),
        _app.string("status.restoring_backend"): (
            "reloading",
            _app.string("osc.action_restoring"),
        ),
        _app.string("status.sleeping"): ("sleeping", _app.string("osc.action_idle")),
        _app.string("ui.sleeping_status"): ("sleeping", _app.string("osc.action_idle")),
        _app.string("status.speaking"): (
            "speaking",
            _app.string("osc.action_playing_audio"),
        ),
        _app.string("status.stopped"): ("stopped", _app.string("osc.action_stopped")),
        _app.string("status.thinking"): (
            "thinking",
            _app.string("osc.action_thinking"),
        ),
        _app.string("status.waiting_for_model"): (
            "initializing",
            _app.string("osc.action_waiting_for_model"),
        ),
        _app.string("status.waking_up"): (
            "initializing",
            _app.string("osc.action_waking_up"),
        ),
        _app.string("status.warming_up"): (
            "initializing",
            _app.string("osc.action_warming_up"),
        ),
    }
    if severity == "error":
        if msg in status_actions:
            return status_actions[msg]
        return "error", _app.string("osc.action_error")
    if severity == "warning":
        return "warning", _app.string("osc.action_warning")

    if self._persona_recording_active() or self._vc_recording_active():
        if self._persona_recording_stop_requested:
            return "recording", _app.string("osc.action_transcribing_speech")
        return "recording", _app.string("osc.action_listening_microphone")

    runtime_state = getattr(self.celune, "cur_state", "idle")
    if runtime_state == "stopped":
        return "stopped", _app.string("osc.action_stopped")
    if msg in status_actions:
        return status_actions[msg]
    if msg.startswith(_app.string("pipeline.playing_label", label="")):
        return "speaking", _app.string("osc.action_playing_audio")
    if msg.startswith(_app.string("pipeline.revoicing_label", label="")):
        return "speaking", _app.string("osc.action_playing_audio")
    if not self.celune_ready:
        return "initializing", _app.string("osc.action_loading_voice_pack")

    if self._agent_task_state in _app._AGENT_AWAITING_STATES:
        return "awaiting", msg
    if self._agent_task_state in _app._AGENT_PAUSED_STATES:
        return "paused", msg
    if self._agent_task_state in _app._AGENT_ACTIVE_STATES:
        return "thinking", msg
    state_actions = {
        "idle": ("ready", _app.string("osc.action_idle")),
        "thinking": ("thinking", _app.string("osc.action_thinking")),
        "generating": ("speaking", _app.string("osc.action_generating_audio")),
        "speaking": ("speaking", _app.string("osc.action_playing_audio")),
        "sleeping": ("sleeping", _app.string("osc.action_idle")),
        "stopped": ("stopped", _app.string("osc.action_stopped")),
        "waking": ("initializing", _app.string("osc.action_waking_up")),
        "reloading": ("reloading", _app.string("osc.action_reloading")),
        "error": ("error", _app.string("osc.action_error")),
        "restarting": ("restarting", _app.string("osc.action_restarting")),
    }
    return state_actions.get(
        runtime_state,
        ("ready", _app.string("osc.action_idle")),
    )


def _redirect_dunder_stdio(self) -> None:
    """Redirect ``sys.__stdout__`` and ``sys.__stderr__`` when possible."""
    if self._original_dunder_stdout is None:
        self._original_dunder_stdout = sys.__stdout__
    if self._original_dunder_stderr is None:
        self._original_dunder_stderr = sys.__stderr__

    if self._log_stdout is not None:
        sys.__stdout__ = self._log_stdout
    if self._log_stderr is not None:
        sys.__stderr__ = self._log_stderr


def _restore_dunder_stdio(self) -> None:
    """Restore ``sys.__stdout__`` and ``sys.__stderr__`` after capture ends."""
    if self._original_dunder_stdout is not None:
        sys.__stdout__ = self._original_dunder_stdout
    if self._original_dunder_stderr is not None:
        sys.__stderr__ = self._original_dunder_stderr

    self._original_dunder_stdout = None
    self._original_dunder_stderr = None


def _install_low_level_stderr_capture(self) -> None:
    """Capture writes that bypass Python and go straight to stderr."""
    from ..utils import available

    if self._stderr_forward_thread is not None:
        return

    stderr_stream = self._old_stderr
    if stderr_stream is None or not available("fileno", obj=stderr_stream):
        return

    original_fd_dup: Optional[int] = None
    pipe_read_fd: Optional[int] = None
    pipe_write_fd: Optional[int] = None

    try:
        stderr_fd = stderr_stream.fileno()
        if not isinstance(stderr_fd, int):
            return
        original_fd_dup = os.dup(stderr_fd)
        pipe_read_fd, pipe_write_fd = os.pipe()
        os.dup2(pipe_write_fd, stderr_fd)
    except (AttributeError, OSError, TypeError, ValueError):
        with contextlib.suppress(OSError):
            if original_fd_dup is not None:
                os.close(original_fd_dup)
        with contextlib.suppress(OSError):
            if pipe_read_fd is not None:
                os.close(pipe_read_fd)
        with contextlib.suppress(OSError):
            if pipe_write_fd is not None:
                os.close(pipe_write_fd)
        return

    self._stderr_original_fd_dup = original_fd_dup
    self._stderr_pipe_read_fd = pipe_read_fd
    self._stderr_pipe_write_fd = pipe_write_fd

    forward_thread = threading.Thread(
        target=self._forward_low_level_stderr,
        name="celune-stderr-capture",
        daemon=True,
    )
    self._stderr_forward_thread = forward_thread
    forward_thread.start()


def _is_textual_terminal_frame(payload: bytes) -> bool:
    """Return whether bytes contain a Textual synchronized terminal frame."""
    return b"\x1b[?2026h" in payload or b"\x1b[?2026l" in payload


def _forward_low_level_stderr(self) -> None:
    """Forward low-level stderr bytes back to the terminal and UI log."""
    read_fd = self._stderr_pipe_read_fd
    original_fd_dup = self._stderr_original_fd_dup
    redirect = self._log_stderr

    if read_fd is None or original_fd_dup is None or redirect is None:
        return

    encoding = getattr(self._old_stderr, "encoding", None) or "utf-8"
    errors = getattr(self._old_stderr, "errors", None) or "replace"

    while True:
        try:
            payload = os.read(read_fd, 4096)
        except OSError:
            break

        if not payload:
            break

        if self._is_textual_terminal_frame(payload):
            try:
                os.write(original_fd_dup, payload)
            except OSError:
                break
            continue

        redirect.write(payload.decode(encoding, errors=errors))

    redirect.flush()


def _remove_low_level_stderr_capture(self) -> None:
    """Restore stderr after low-level capture was installed."""
    from ..utils import available

    stderr_stream = self._old_stderr
    original_fd_dup = self._stderr_original_fd_dup
    pipe_write_fd = self._stderr_pipe_write_fd
    pipe_read_fd = self._stderr_pipe_read_fd

    if (
        stderr_stream is not None
        and available("fileno", obj=stderr_stream)
        and original_fd_dup is not None
    ):
        with contextlib.suppress(OSError, ValueError):
            stderr_stream.flush()
        with contextlib.suppress(OSError, ValueError):
            os.dup2(original_fd_dup, stderr_stream.fileno())

    if pipe_write_fd is not None:
        with contextlib.suppress(OSError):
            os.close(pipe_write_fd)
    if pipe_read_fd is not None:
        with contextlib.suppress(OSError):
            os.close(pipe_read_fd)
    if original_fd_dup is not None:
        with contextlib.suppress(OSError):
            os.close(original_fd_dup)

    self._stderr_pipe_read_fd = None
    self._stderr_pipe_write_fd = None
    self._stderr_original_fd_dup = None
    self._stderr_forward_thread = None


def _remove_runtime_log_redirects(self) -> None:
    """Restore Python logging dispatch after UI shutdown."""
    handler = self._runtime_redirect_handler
    original_call_handlers = self._runtime_redirect_original_call_handlers
    if handler is not None and original_call_handlers is not None:
        logging.Logger.callHandlers = original_call_handlers
        logging.lastResort = self._runtime_redirect_original_last_resort
        if self._runtime_redirect_original_raise_exceptions is not None:
            logging.raiseExceptions = self._runtime_redirect_original_raise_exceptions
        handler.close()

    if self._warnings_capture_enabled:
        logging.captureWarnings(False)
        self._warnings_capture_enabled = False

    self._runtime_redirect_handler = None
    self._runtime_redirect_original_call_handlers = None
    self._runtime_redirect_original_last_resort = None
    self._runtime_redirect_original_raise_exceptions = None


def advance_resources(self) -> None:
    """Advance the resource footer to the next page and refresh it."""
    if self.cur_state == "exiting" or self.resources is None:
        return
    resources = _app.ui_resources
    if resources is None or self.celune is None:
        return

    self._resource_page = (self._resource_page + 1) % len(
        resources.resource_pages(self.celune, self.active_theme_name)
    )
    self.update_resources()
    self._publish_webui_timed_update()


def _cancel_sleep_timer(self) -> None:
    """Cancel a pending automatic sleep transition."""
    if threading.current_thread() is not threading.main_thread():
        self._run_on_ui_thread(self._cancel_sleep_timer)
        return

    if self._sleep_timer is not None:
        self._sleep_timer.stop()
        self._sleep_timer = None


def _schedule_sleep_timer(self) -> None:
    """Schedule automatic sleep after the configured idle timeout."""
    from ..utils import available

    if threading.current_thread() is not threading.main_thread():
        self._run_on_ui_thread(self._schedule_sleep_timer)
        return

    self._cancel_sleep_timer()
    if (
        self.cur_state == "exiting"
        or not self.celune_ready
        or not available("sleep_enabled", obj=self.celune)
        or not self.celune.sleep_enabled()
        or self.celune.sleeping
        or self.celune.is_in_tutorial
    ):
        return

    self._sleep_timer = self.set_timer(
        self.celune.sleep_timeout_seconds(),
        self._enter_sleep_mode,
    )


def _enter_sleep_mode(self) -> None:
    """Put the app to sleep from the UI idle timer."""
    self._sleep_timer = None
    if self.cur_state == "exiting" or self.celune is None:
        return

    self.enter_sleep_mode()


def start_background_init(self) -> None:
    """Run the initialization function."""
    self._show_loading_screen()
    self.load_tts()


def _show_loading_screen(self) -> None:
    """Reveal the startup overlay already mounted above the main UI."""
    if self._loading_screen is None:
        return
    try:
        main_container = self.query_one("#container", _app.Vertical)
    except (_app.NoMatches, _app.ScreenStackError):
        main_container = None
    if main_container is not None:
        main_container.styles.opacity = 0.0
        main_container.display = False
    self._loading_screen.styles.opacity = 1.0
    self._loading_screen.display = True


def _update_loading_log(self, message: str) -> None:
    """Forward one useful startup log line to the loading screen.

    Args:
        message: Non-verbose, non-debug log message to display.
    """
    if self._loading_screen is not None:
        self._loading_screen.set_latest_log_message(message)


def _show_loading_error(
    self,
    message: str,
    *,
    status_message: Optional[str] = None,
    footer_message: Optional[str] = None,
) -> None:
    """Keep the loading screen visible while showing an initialization error.

    Args:
        message: Initialization error to display.
        status_message: Optional replacement for the failure heading.
        footer_message: Optional status to show in the lower-left footer.
    """

    def update() -> None:
        if self._loading_screen is not None:
            self._loading_screen.show_error(
                message,
                status_message=status_message,
                footer_message=footer_message,
            )

    self._run_on_ui_thread(update)


def _dismiss_loading_screen(self) -> None:
    """Fade out and remove the startup screen after successful loading."""

    def dismiss() -> None:
        overlay = self._loading_screen
        if overlay is None:
            return
        main_container: Optional[_app.Vertical] = None

        def reveal_main_ui() -> None:
            nonlocal main_container
            if self._loading_screen is not overlay:
                return
            try:
                main_container = self.query_one("#container", _app.Vertical)
            except (_app.NoMatches, _app.ScreenStackError):
                self.call_after_refresh(fade_overlay)
                return
            main_container.styles.opacity = 0.0
            main_container.display = True
            main_container.refresh(layout=True, repaint=True)
            self.refresh(layout=True, repaint=True)
            self.call_after_refresh(fade_overlay)

        def show_main_ui() -> None:
            if self._loading_screen is not overlay:
                return
            overlay.display = False
            if main_container is not None:
                self._animate_opacity(
                    main_container,
                    1.0,
                    duration=_app._MAIN_UI_FADE_SECONDS,
                )
            self.call_after_refresh(self._refresh_logs)

        def fade_overlay() -> None:
            if self._loading_screen is not overlay:
                return
            overlay.animate(
                "opacity",
                0.0,
                duration=_app._LOADING_FADE_SECONDS,
                easing="out_cubic",
                on_complete=show_main_ui,
            )

        self.call_after_refresh(reveal_main_ui)

    self._run_on_ui_thread(dismiss)


def _caption_word_timing_ranges(
    words: tuple[str, ...],
    segments: tuple[_app.WhisperSegment, ...],
    audio_duration: float,
    timing_words: Optional[tuple[str, ...]] = None,
) -> tuple[tuple[float, float], ...]:
    """Map normalized speech timestamps onto displayed caption words."""
    if not words or not segments or audio_duration <= 0.0:
        return ()

    whisper_words = [
        word
        for segment in segments
        for word in segment.words
        if word.text and word.end >= word.start
    ]
    if not whisper_words:
        return ()

    def normalize(value: str) -> str:
        return re.sub(r"[^\w]+", "", value.casefold())

    matching_words = timing_words if timing_words else words
    caption_keys = [normalize(word) for word in matching_words]
    whisper_keys = [normalize(word.text) for word in whisper_words]
    assigned: list[Optional[int]] = [None] * len(matching_words)
    whisper_index = 0
    for caption_index, caption_key in enumerate(caption_keys):
        if not caption_key:
            continue
        for candidate in range(
            whisper_index,
            min(len(whisper_words), whisper_index + 5),
        ):
            if caption_key == whisper_keys[candidate]:
                assigned[caption_index] = candidate
                whisper_index = candidate + 1
                break

    timing_ranges: list[tuple[float, float]] = []
    for index, assigned_index in enumerate(assigned):
        if assigned_index is None:
            assigned_index = round(
                index * (len(whisper_words) - 1) / max(len(matching_words) - 1, 1)
            )
        assigned_index = max(0, min(len(whisper_words) - 1, assigned_index))
        word = whisper_words[assigned_index]
        start = max(0.0, min(audio_duration, word.start))
        end = max(start, min(audio_duration, word.end))
        timing_ranges.append((start, end))

    previous_end = 0.0
    normalized_ranges: list[tuple[float, float]] = []
    for start, end in timing_ranges:
        start = max(previous_end, start)
        end = max(start, end)
        normalized_ranges.append((start, end))
        previous_end = end
    if len(matching_words) == len(words):
        return tuple(normalized_ranges)

    displayed_ranges: list[tuple[float, float]] = []
    for index in range(len(words)):
        start_index = min(
            len(normalized_ranges) - 1,
            math.floor(index * len(normalized_ranges) / len(words)),
        )
        end_index = math.ceil((index + 1) * len(normalized_ranges) / len(words))
        end_index = max(start_index + 1, end_index)
        end_index = min(end_index, len(normalized_ranges))
        displayed_ranges.append(
            (
                normalized_ranges[start_index][0],
                normalized_ranges[end_index - 1][1],
            )
        )
    return tuple(displayed_ranges)


def tts_caption_timing(
    self,
    caption: str,
    audio: _app.AudioChunk,
    sample_rate: int,
    timing_text: Optional[str] = None,
) -> None:
    """Refine displayed caption timing from normalized speech timestamps."""
    if (
        self.cur_state == "exiting"
        or getattr(self.celune, "test_finished", False)
        or not caption
        or len(audio) <= 0
    ):
        return

    audio_copy = _app.np.asarray(audio, dtype=_app.np.float32).copy()
    token = self._caption_transition_token
    duration = len(audio_copy) / max(sample_rate, 1)
    normalized_timing_text = timing_text if timing_text is not None else caption
    timing_words = tuple(normalized_timing_text.split())

    def analyze() -> None:
        try:
            transcriber = (
                self._caption_transcriber or self._persona_recording_transcriber
            )
            if transcriber is None:
                if self.celune is None or not _app.persona_enabled(self.celune.config):
                    return
                model_id = getattr(self, "_persona_speech_model_id", None)
                language_getter = getattr(self, "_persona_speech_language", None)
                if not callable(model_id) or not callable(language_getter):
                    return
                model_id_getter = cast(Callable[[], str], model_id)
                language_value_getter = cast(
                    Callable[[], Optional[str]], language_getter
                )
                transcriber = _app.WhisperTranscriber(
                    model_id_getter(),
                    language=language_value_getter(),
                )
                self._caption_transcriber = transcriber
            segments = transcriber.transcribe_segments(audio_copy, sample_rate)
            word_timings = self._caption_word_timing_ranges(
                self._caption_words,
                segments,
                duration,
                timing_words,
            )
        except Exception as error:
            self.safe_log(
                _app.format_error_message(
                    _app.string("ui.caption_transcription_failed"),
                    error,
                    _app.resolve_log_level(
                        getattr(self.celune, "log_level", None),
                        self._startup_log_level,
                    ),
                ),
                "warning",
            )
            return
        if not word_timings:
            return

        def update() -> None:
            if (
                token != self._caption_transition_token
                or not self._caption_active
                or caption != self._caption_text
            ):
                return
            self._caption_word_timings = word_timings
            self._caption_audio_duration = duration
            visible_sentence, visible_words = self._caption_words_for_progress(
                self._caption_progress
            )
            rendered_text = " ".join(visible_sentence)
            self._caption_visible_words = visible_words
            self._caption_rendered_text = rendered_text
            if self.caption is not None:
                self.caption.update(rendered_text)

        with contextlib.suppress(LookupError, RuntimeError, _app.ScreenStackError):
            self._run_on_ui_thread(update)

    threading.Thread(target=analyze, daemon=True).start()


def _caption_words_for_progress(
    self,
    fraction: float,
) -> tuple[tuple[str, ...], int]:
    """Return the current sentence words and total revealed word count."""
    if (
        self._caption_word_timings
        and len(self._caption_word_timings) == len(self._caption_words)
        and self._caption_audio_duration > 0.0
    ):
        elapsed = fraction * self._caption_audio_duration
        revealed_words = sum(
            elapsed >= start for start, _end in self._caption_word_timings
        )
        revealed_words = max(self._caption_visible_words, revealed_words)
        remaining_words = revealed_words
        for sentence in self._caption_sentences:
            if remaining_words <= len(sentence):
                return sentence[:remaining_words], revealed_words
            remaining_words -= len(sentence)
        return (
            self._caption_sentences[-1] if self._caption_sentences else (),
            revealed_words,
        )

    visible_words = min(
        len(self._caption_words),
        math.ceil(fraction * len(self._caption_words)),
    )
    visible_words = max(self._caption_visible_words, visible_words)
    remaining_words = visible_words
    visible_sentence: tuple[str, ...] = ()
    for sentence in self._caption_sentences:
        if remaining_words <= len(sentence):
            visible_sentence = sentence[:remaining_words]
            break
        remaining_words -= len(sentence)
    return visible_sentence, visible_words


def safe_progress(
    self, progress: Optional[float], total: Optional[float] = None
) -> None:
    """Update current progress.

    Args:
        progress: Current progress, or ``None`` for an indeterminate bar.
        total: Total progress, or ``None`` for an indeterminate bar.
    """
    if self.cur_state == "exiting":
        return

    if self.progress_bar is None:
        return

    def update() -> None:
        celune = self.celune
        idle_after_startup = (
            progress is None
            and total is None
            and self.celune_ready
            and self.cur_state not in {"error", "exiting"}
            and celune is not None
            and getattr(celune, "cur_state", None) == "idle"
            and not getattr(celune, "persona_loading", False)
        )
        resolved_progress = 1 if idle_after_startup else progress
        resolved_total = 1 if idle_after_startup else total
        self.progress_bar.update(
            total=resolved_total,
            progress=0 if resolved_progress is None else resolved_progress,
        )
        if self.progress_label is not None:
            audio_playing = (
                celune is not None and getattr(celune, "cur_state", None) == "speaking"
            )
            self.progress_label.set_progress(
                resolved_progress,
                resolved_total,
                audio_playing=audio_playing,
                sample_rate=_app.BASE_SR,
            )
            if self._caption_active:
                self.progress_label.display = False
        if not self._caption_active and not self._caption_transitioning:
            self._restore_progress_bar()

    self._run_on_ui_thread(update)


def _set_progress_row_display(self, visible: bool) -> None:
    """Show or hide the progress row that contains the bar and readout."""
    if self.progress_bar is None:
        return
    progress_row = getattr(self.progress_bar, "parent", None)
    if progress_row is not None:
        progress_row.display = visible


def _restore_progress_bar(self, *, force: bool = False) -> None:
    """Restore the progress row and bar after a transient caption state."""
    if (
        self.progress_bar is None
        or self._caption_active
        or (self._caption_transitioning and not force)
    ):
        return
    self._set_progress_row_display(True)
    self.progress_bar.display = True
    self.progress_bar.styles.opacity = 1.0


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(target, {name: globals()[name] for name in __all__})
