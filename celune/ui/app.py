# SPDX-License-Identifier: MIT
"""Frontend layer."""

import asyncio
import contextlib
import ctypes
import datetime
import itertools
import logging
import os
import queue as queue_module
import shlex
import signal
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Optional, TextIO, Union, cast

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import yaml
from rich.text import Text
from textual import events, work
from textual.app import (
    App,
    AutopilotCallbackType,
    ComposeResult,
    ReturnType,
    ScreenStackError,
)
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.css.types import EdgeStyle
from textual.message import Message
from textual.theme import Theme
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Label, ProgressBar, RichLog, TextArea

from .. import colors
from ..celune import Celune
from ..cevoice import default_loader
from ..config import format_audio_device_name, resolve_audio_device_with_info
from ..constants import APP_NAME, CRASH_LINES, SIGTSTP
from ..i18n import string
from ..watchdog import launcher_loss_requested
from ..paths import config_path, main_window_log_path
from ..persona.asr import (
    DEFAULT_PERSONA_SPEECH_MODEL_ID,
    PERSONA_SPEECH_END_DELAY_SECONDS,
    WhisperTranscriber,
)
from ..persona.impl import (
    persona_config,
    persona_enabled,
    persona_talkback_enabled,
)
from ..pipeline import (
    current_playback_status,
    finish_streaming_sfx_audio,
    queue_streaming_sfx_audio,
)
from ..typing.aliases import (  # pylint: disable=W0611
    AudioChunk,
    AudioChunks,
    AudioDeviceScalar,
    _VCAudioCallback,  # noqa
)
from ..utils import (
    discard,
    format_error,
    indent,
    is_april_fools,
    replace_ipa,
    supports_ansi,
    typing_animation,
    typing_delay,
)
from ..vc import (
    VC_PITCH_SHIFT_MAX,
    VC_PITCH_SHIFT_MIN,
    clamp_vc_pitch_shift,
    create_live_voice_activity_detector,
    vc_input_has_voice,
    vc_input_rms,
    vc_live_chunk_frames,
    vc_live_chunk_overlap_frames,
    vc_vad_hangover_frames,
    vc_vad_preroll_frames,
)
from . import resources as ui_resources
from .commands import process_command as process_ui_command
from .resources import FOOTER_ROTATE_SECONDS
from .terminal import LogRedirect, UILogHandler, is_celune_log_record
from .theme import CELUNE_CSS, severity_color

_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES = frozenset(
    {
        "`torch_dtype` is deprecated! Use `dtype` instead!",
        "Skipped loading some keys due to shape mismatch:",
        "cfm loaded",
        "length_regulator loaded",
        "Removing weight norm...",
        "Loading weights from",
        "Loading Text2Semantic weights from",
        "Loading Text2Semantic Weights from",
        "min value is",
        "max value is",
        "it/s]",
        "s/it]",
    }
)


def _device_scalar_int(value: Optional[AudioDeviceScalar], default: int) -> int:
    """Return one audio-device metadata value as an integer when possible."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


class UILogMessage(Message):
    """Deliver one background log entry to the Textual application thread."""

    def __init__(self, message: str, severity: str) -> None:
        super().__init__()
        self.message = message
        self.severity = severity


@dataclass
class CeluneUIWidgetState:
    """Resolved widget references owned by the UI."""

    logs: Optional[RichLog] = None
    input_box: Optional[TextArea] = None
    style_button: Optional[Button] = None
    vc_mode_button: Optional[Button] = None
    vc_pitch_button: Optional[Button] = None
    status: Optional[Label] = None
    resources: Optional[Label] = None
    progress_bar: Optional[ProgressBar] = None
    header: Optional[Label] = None
    header_lines: tuple[Label, ...] = ()  # noqa


@dataclass
class CeluneUIThemeState:
    """Theme and status marquee state."""

    themes: tuple[str, str]
    active_theme_name: str
    fatal_error_active: bool = False
    log_history: list[tuple[str, str]] = field(default_factory=list)
    status_severity: str = "info"
    status_text: str = ""
    status_marquee_offset: int = 0
    status_marquee_gap: str = "   "
    status_marquee_timer: Optional[Timer] = None


@dataclass
class CeluneUIBindingState:
    """Bindings between the UI and the runtime."""

    celune: Optional[Celune] = None
    celune_ready: bool = False
    celune_styles: tuple[str, ...] = ()  # noqa
    celune_voices: Optional[Iterator[str]] = None
    style_index: int = 0
    cur_state: str = "active"
    consume_on_boundary: bool = False
    suppress_input_change: bool = False
    resource_page: int = 0
    input_locked: bool = True
    persona_available: bool = False
    persona_probe_running: bool = False


@dataclass
class CeluneUILogCaptureState:
    """Stdio/log redirection and persisted log state."""

    old_stdout: TextIO
    old_stderr: TextIO
    log_stdout: Optional[LogRedirect] = None
    log_stderr: Optional[LogRedirect] = None
    runtime_log_capture_enabled: bool = False
    runtime_redirect_handler: Optional[UILogHandler] = None
    runtime_redirect_original_call_handlers: Optional[
        Callable[[logging.Logger, logging.LogRecord], None]
    ] = None
    runtime_redirect_original_last_resort: Optional[logging.Handler] = None
    runtime_redirect_original_raise_exceptions: Optional[bool] = None
    original_dunder_stdout: Optional[TextIO] = None
    original_dunder_stderr: Optional[TextIO] = None
    terminal_output_stream: Optional[TextIOWrapper] = None
    stderr_pipe_read_fd: Optional[int] = None
    stderr_pipe_write_fd: Optional[int] = None
    stderr_original_fd_dup: Optional[int] = None
    stderr_forward_thread: Optional[threading.Thread] = None
    warnings_capture_enabled: bool = False
    log_file_path: Path = field(default_factory=Path)
    log_file_initialized: bool = False


@dataclass
class CeluneUIInteractionState:
    """Transient UI effects, sleep scheduling, and tutorial state."""

    border_pulse_tokens: dict[int, int] = field(default_factory=dict)
    border_pulse_widgets: dict[int, Widget] = field(default_factory=dict)
    tutorial_timers: list[Timer] = field(default_factory=list)
    vc_recording_buffered_frames: int = 0
    vc_recording_chunks: AudioChunks = field(default_factory=list)
    vc_recording_captured_frames: int = 0
    vc_recording_feedback_detected: bool = False
    vc_recording_feedback_spike_count: int = 0
    vc_recording_label: str = ""
    vc_recording_lock: threading.Lock = field(default_factory=threading.Lock)
    vc_recording_preroll_chunks: AudioChunks = field(default_factory=list)
    vc_recording_preroll_frames: int = 0
    vc_recording_previous_rms: float = 0.0
    vc_recording_sample_rate: int = 0
    vc_recording_silence_frames: int = 0
    vc_recording_submission_queue: Optional[
        queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]  # noqa
    ] = None
    vc_recording_stream: Optional[sd.InputStream] = None
    vc_recording_stop_thread: Optional[threading.Thread] = None
    vc_recording_worker: Optional[threading.Thread] = None
    persona_recording_chunks: AudioChunks = field(default_factory=list)
    persona_recording_lock: threading.Lock = field(default_factory=threading.Lock)
    persona_recording_queue: Optional[
        queue_module.Queue[tuple[AudioChunk, bool]]  # noqa
    ] = None
    persona_recording_sample_rate: int = 0
    persona_recording_silence_frames: int = 0
    persona_recording_speech_started: bool = False
    persona_recording_stop_requested: bool = False
    persona_recording_stream: Optional[sd.InputStream] = None
    persona_recording_text_prefix: str = ""
    persona_recording_transcriber: Optional[WhisperTranscriber] = None
    persona_recording_worker: Optional[threading.Thread] = None
    persona_recording_last_partial_at: float = 0.0
    sleep_timer: Optional[Timer] = None
    tutorial_token: int = 0
    tutorial_active: bool = False


def _forward_ui_property(container_name: str, field_name: str) -> property:
    """Create a property that forwards storage to a grouped UI state container."""

    def getter(instance):
        return getattr(getattr(instance, container_name), field_name)

    def setter(instance, value) -> None:
        setattr(getattr(instance, container_name), field_name, value)

    return property(getter, setter)


class CeluneUI(App):
    """User interface."""

    ENABLE_COMMAND_PALETTE = False
    CSS = CELUNE_CSS
    _instance: Optional["CeluneUI"] = None

    def __init__(self) -> None:
        super().__init__()

        if CeluneUI._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")

        if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }:
            themes = ("celune_april_fools", "celune_april_fools")
            active_theme_name = "celune_april_fools"
        else:
            themes = ("celune", "celune_light")
            active_theme_name = "celune"

        self._widgets = CeluneUIWidgetState()
        self._theme_state = CeluneUIThemeState(
            themes=themes,
            active_theme_name=active_theme_name,
        )
        self._binding_state = CeluneUIBindingState(celune_voices=itertools.cycle(()))
        self._log_capture_state = CeluneUILogCaptureState(
            old_stdout=sys.stdout,
            old_stderr=sys.stderr,
            log_file_path=main_window_log_path(create_parent=True),
        )
        self._interaction_state = CeluneUIInteractionState()
        self._shutdown_lock = threading.Lock()

        CeluneUI._instance = self

    logs = _forward_ui_property("_widgets", "logs")
    input_box = _forward_ui_property("_widgets", "input_box")
    style_button = _forward_ui_property("_widgets", "style_button")
    vc_mode_button = _forward_ui_property("_widgets", "vc_mode_button")
    vc_pitch_button = _forward_ui_property("_widgets", "vc_pitch_button")
    status = _forward_ui_property("_widgets", "status")
    resources = _forward_ui_property("_widgets", "resources")
    progress_bar = _forward_ui_property("_widgets", "progress_bar")
    header = _forward_ui_property("_widgets", "header")
    header_lines = _forward_ui_property("_widgets", "header_lines")

    themes = _forward_ui_property("_theme_state", "themes")
    active_theme_name = _forward_ui_property("_theme_state", "active_theme_name")
    _fatal_error_active = _forward_ui_property("_theme_state", "fatal_error_active")
    log_history = _forward_ui_property("_theme_state", "log_history")
    status_severity = _forward_ui_property("_theme_state", "status_severity")
    _status_text = _forward_ui_property("_theme_state", "status_text")
    _status_marquee_offset = _forward_ui_property(
        "_theme_state", "status_marquee_offset"
    )
    _status_marquee_gap = _forward_ui_property("_theme_state", "status_marquee_gap")
    _status_marquee_timer = _forward_ui_property("_theme_state", "status_marquee_timer")

    celune = _forward_ui_property("_binding_state", "celune")
    celune_ready = _forward_ui_property("_binding_state", "celune_ready")
    celune_styles = _forward_ui_property("_binding_state", "celune_styles")
    celune_voices = _forward_ui_property("_binding_state", "celune_voices")
    style_index = _forward_ui_property("_binding_state", "style_index")
    cur_state = _forward_ui_property("_binding_state", "cur_state")
    consume_on_boundary = _forward_ui_property("_binding_state", "consume_on_boundary")
    _suppress_input_change = _forward_ui_property(
        "_binding_state", "suppress_input_change"
    )
    _resource_page = _forward_ui_property("_binding_state", "resource_page")
    _input_locked = _forward_ui_property("_binding_state", "input_locked")
    _persona_available = _forward_ui_property("_binding_state", "persona_available")
    _persona_probe_running = _forward_ui_property(
        "_binding_state", "persona_probe_running"
    )

    _old_stdout = _forward_ui_property("_log_capture_state", "old_stdout")
    _old_stderr = _forward_ui_property("_log_capture_state", "old_stderr")
    _log_stdout = _forward_ui_property("_log_capture_state", "log_stdout")
    _log_stderr = _forward_ui_property("_log_capture_state", "log_stderr")
    _runtime_log_capture_enabled = _forward_ui_property(
        "_log_capture_state", "runtime_log_capture_enabled"
    )
    _runtime_redirect_handler = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_handler"
    )
    _runtime_redirect_original_call_handlers = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_call_handlers"
    )
    _runtime_redirect_original_last_resort = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_last_resort"
    )
    _runtime_redirect_original_raise_exceptions = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_raise_exceptions"
    )
    _original_dunder_stdout = _forward_ui_property(
        "_log_capture_state", "original_dunder_stdout"
    )
    _original_dunder_stderr = _forward_ui_property(
        "_log_capture_state", "original_dunder_stderr"
    )
    _terminal_output_stream = _forward_ui_property(
        "_log_capture_state", "terminal_output_stream"
    )
    _stderr_pipe_read_fd = _forward_ui_property(
        "_log_capture_state", "stderr_pipe_read_fd"
    )
    _stderr_pipe_write_fd = _forward_ui_property(
        "_log_capture_state", "stderr_pipe_write_fd"
    )
    _stderr_original_fd_dup = _forward_ui_property(
        "_log_capture_state", "stderr_original_fd_dup"
    )
    _stderr_forward_thread = _forward_ui_property(
        "_log_capture_state", "stderr_forward_thread"
    )
    _warnings_capture_enabled = _forward_ui_property(
        "_log_capture_state", "warnings_capture_enabled"
    )
    _log_file_path = _forward_ui_property("_log_capture_state", "log_file_path")
    _log_file_initialized = _forward_ui_property(
        "_log_capture_state", "log_file_initialized"
    )

    _border_pulse_tokens = _forward_ui_property(
        "_interaction_state", "border_pulse_tokens"
    )
    _border_pulse_widgets = _forward_ui_property(
        "_interaction_state", "border_pulse_widgets"
    )
    _tutorial_timers = _forward_ui_property("_interaction_state", "tutorial_timers")
    _vc_recording_chunks = _forward_ui_property(
        "_interaction_state", "vc_recording_chunks"
    )
    _vc_recording_buffered_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_buffered_frames"
    )
    _vc_recording_label = _forward_ui_property(
        "_interaction_state", "vc_recording_label"
    )
    _vc_recording_captured_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_captured_frames"
    )
    _vc_recording_feedback_detected = _forward_ui_property(
        "_interaction_state", "vc_recording_feedback_detected"
    )
    _vc_recording_feedback_spike_count = _forward_ui_property(
        "_interaction_state", "vc_recording_feedback_spike_count"
    )
    _vc_recording_lock = _forward_ui_property("_interaction_state", "vc_recording_lock")
    _vc_recording_preroll_chunks = _forward_ui_property(
        "_interaction_state", "vc_recording_preroll_chunks"
    )
    _vc_recording_preroll_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_preroll_frames"
    )
    _vc_recording_previous_rms = _forward_ui_property(
        "_interaction_state", "vc_recording_previous_rms"
    )
    _vc_recording_sample_rate = _forward_ui_property(
        "_interaction_state", "vc_recording_sample_rate"
    )
    _vc_recording_silence_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_silence_frames"
    )
    _vc_recording_submission_queue = _forward_ui_property(
        "_interaction_state", "vc_recording_submission_queue"
    )
    _vc_recording_stream = _forward_ui_property(
        "_interaction_state", "vc_recording_stream"
    )
    _vc_recording_stop_thread = _forward_ui_property(
        "_interaction_state", "vc_recording_stop_thread"
    )
    _vc_recording_worker = _forward_ui_property(
        "_interaction_state", "vc_recording_worker"
    )
    _persona_recording_chunks = _forward_ui_property(
        "_interaction_state", "persona_recording_chunks"
    )
    _persona_recording_lock = _forward_ui_property(
        "_interaction_state", "persona_recording_lock"
    )
    _persona_recording_queue = _forward_ui_property(
        "_interaction_state", "persona_recording_queue"
    )
    _persona_recording_sample_rate = _forward_ui_property(
        "_interaction_state", "persona_recording_sample_rate"
    )
    _persona_recording_silence_frames = _forward_ui_property(
        "_interaction_state", "persona_recording_silence_frames"
    )
    _persona_recording_speech_started = _forward_ui_property(
        "_interaction_state", "persona_recording_speech_started"
    )
    _persona_recording_stop_requested = _forward_ui_property(
        "_interaction_state", "persona_recording_stop_requested"
    )
    _persona_recording_stream = _forward_ui_property(
        "_interaction_state", "persona_recording_stream"
    )
    _persona_recording_text_prefix = _forward_ui_property(
        "_interaction_state", "persona_recording_text_prefix"
    )
    _persona_recording_transcriber = _forward_ui_property(
        "_interaction_state", "persona_recording_transcriber"
    )
    _persona_recording_worker = _forward_ui_property(
        "_interaction_state", "persona_recording_worker"
    )
    _persona_recording_last_partial_at = _forward_ui_property(
        "_interaction_state", "persona_recording_last_partial_at"
    )
    _sleep_timer = _forward_ui_property("_interaction_state", "sleep_timer")
    _tutorial_token = _forward_ui_property("_interaction_state", "tutorial_token")
    _tutorial_active = _forward_ui_property("_interaction_state", "tutorial_active")

    def _run_on_ui_thread(self, callback: Callable[[], None]) -> None:
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            self.call_from_thread(callback)

    def _severity_color(self, severity: str = "info") -> str:
        """Return the current theme color for a log severity."""
        return severity_color(self.active_theme_name, severity)

    def _runtime_theme_name(self) -> str:
        """Return the current Textual theme name, including runtime error overrides."""
        if self._fatal_error_active:
            if self.active_theme_name == "celune_light":
                return "celune_light_error"
            return "celune_error"
        return self.active_theme_name

    def _register_runtime_error_themes(self) -> None:
        """Register error themes used for runtime failure states."""
        dark_foreground = colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_BACKGROUND,
            7.0,
        )
        light_foreground = colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_LIGHT_BACKGROUND,
            7.0,
        )
        dark_theme = Theme(
            name="celune_error",
            primary=colors.ERROR_DARK_ACCENT,
            secondary=colors.ERROR_DARK_ACCENT,
            accent=colors.THEME.error,
            foreground=dark_foreground,
            background=colors.ERROR_BACKGROUND,
            surface=colors.ERROR_BACKGROUND,
            warning=colors.THEME.warning,
            error=colors.THEME.error,
            dark=True,
        )
        light_theme = Theme(
            name="celune_light_error",
            primary=colors.ERROR_DARK_ACCENT,
            secondary=colors.ERROR_DARK_ACCENT,
            accent=colors.THEME_LIGHT.error,
            foreground=light_foreground,
            background=colors.ERROR_LIGHT_BACKGROUND,
            surface=colors.ERROR_LIGHT_BACKGROUND,
            warning=colors.THEME_LIGHT.warning,
            error=colors.THEME_LIGHT.error,
            dark=False,
        )
        if dark_theme.name not in self.available_themes:
            self.register_theme(dark_theme)
        if light_theme.name not in self.available_themes:
            self.register_theme(light_theme)

    def _ensure_themes_registered(self) -> None:
        """Register Celune's built-in themes when the app is not fully mounted yet."""
        if colors.THEME.name not in self.available_themes:
            self.register_theme(colors.THEME)
        if colors.THEME_LIGHT.name not in self.available_themes:
            self.register_theme(colors.THEME_LIGHT)
        if colors.THEME_APRIL_FOOLS.name not in self.available_themes:
            self.register_theme(colors.THEME_APRIL_FOOLS)
        self._register_runtime_error_themes()

    def _apply_theme(self, theme_name: str) -> None:
        """Apply theme and repaint theme-sensitive widgets."""
        self._clear_border_pulses()
        self.active_theme_name = theme_name
        self.theme = self._runtime_theme_name()
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()

    def _has_celune(self) -> bool:
        """Is the app attached to this UI instance?"""
        return self.celune is not None

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

        def repaint(widget: Widget) -> None:
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
        except ScreenStackError:
            screen = None
        if screen is not None and hasattr(screen, "styles"):
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
        if self.progress_bar is not None and hasattr(self.progress_bar, "styles"):
            self.progress_bar.styles.color = None
            self.progress_bar.styles.background = None
            repaint(self.progress_bar)

    def _wrap_runtime_fatal_glow(self) -> None:
        """Mirror runtime fatal glow events into the UI fatal theme flag."""
        if self.celune is None or getattr(self.celune, "_ui_fatal_glow_wrapped", False):
            return

        glow = getattr(self.celune, "glow", None)
        if glow is None or not hasattr(glow, "fatal"):
            return
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            self._fatal_error_active = True
            self._run_on_ui_thread(self._refresh_theme_text)
            original_fatal()

        glow.fatal = wrapped_fatal
        self.celune._ui_fatal_glow_wrapped = True

    def _bind_runtime_callbacks(self) -> None:
        """Bind one attached Celune instance back into this UI."""
        if self.celune is None:
            return

        self.celune.log_callback = self.tts_log
        self.celune.status_callback = self.safe_status
        self.celune.error_callback = self.error
        self.celune.idle_callback = self.tts_idle
        self.celune.queue_avail_callback = self.tts_queue_avail
        self.celune.voice_changed_callback = self.tts_voice_changed
        self.celune.change_input_state_callback = self.change_input_state
        self.celune.change_voice_lock_state_callback = self.change_voice_lock_state
        self.celune.progress_callback = self.safe_progress

    def _is_ui_test_mode(self) -> bool:
        """Return whether the attached runtime is the interactive fake-backend UI test mode."""
        if self.celune is None:
            return False

        backend = getattr(self.celune, "backend", None)
        return bool(getattr(backend, "is_fake", False)) and "pytest" not in sys.modules

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
            return indent(self._status_text, spaces=2)

        loop = f"{self._status_text}{self._status_marquee_gap}"
        offset = self._status_marquee_offset % len(loop)
        window = (loop * 2)[offset : offset + width]
        return indent(window, spaces=2)

    def _update_status_label(self) -> None:
        """Push the current status text into the label."""
        if self.status is None:
            return
        self.status.update(self._render_status_text())
        self._refresh_status()

    def _advance_status_marquee(self) -> None:
        """Advance the marquee one character for long status messages."""
        if self.status is None:
            return
        playback_status = (
            current_playback_status(self.celune) if self.celune is not None else None
        )
        if playback_status is not None and playback_status != self._status_text:
            self._status_text = playback_status
            self.status_severity = "info"
            self._status_marquee_offset = 0
            self._refresh_theme_text()
        if len(self._status_text) <= self._status_view_width():
            self._update_status_label()
            return
        self._status_marquee_offset += 1
        self._update_status_label()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-render width-sensitive widgets after the window size changes.

        Args:
            _event: Textual resize event that triggered the redraw.
        """
        if self.status is not None:
            self._update_status_label()

    def _refresh_logs(self) -> None:
        """Repaint existing log entries using the active theme colors."""
        if self.logs is None:
            return

        scroll_offset = self.logs.scroll_offset
        auto_scroll = self.logs.auto_scroll
        self.logs.auto_scroll = False
        self.logs.clear()

        for message, severity in self.log_history:
            self.logs.write(
                Text(message, style=self._severity_color(severity)),
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

    def _persist_log_entry(self, msg: str, severity: str) -> None:
        """Append one UI log entry to the persisted main-window log file."""
        with contextlib.suppress(OSError):
            if not self._log_file_initialized:
                self._log_file_path.write_text("", encoding="utf-8")
                self._log_file_initialized = True

            timestamp = datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            )
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
        discard(output_stream)
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
        auto_pilot: Optional[AutopilotCallbackType] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Optional[ReturnType]:
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
            return super().run(
                headless=headless,
                inline=inline,
                inline_no_clear=inline_no_clear,
                mouse=mouse,
                size=size,
                auto_pilot=auto_pilot,
                loop=loop,
            )
        finally:
            self._shutdown_runtime()
            if output_stream is not None:
                sys.__stderr__ = original_stderr
                self._terminal_output_stream = None
                with contextlib.suppress(OSError, ValueError):
                    output_stream.close()

    def compose(self) -> ComposeResult:
        """Define the UI.

        Returns:
            ComposeResult: The root widget tree for the interface.
        """
        with Vertical(id="container"):
            with Horizontal(id="header-container"):
                yield Label("", classes="line")
                yield Label(APP_NAME, id="header")
                yield Label("", classes="line")
            yield RichLog(id="logs", wrap=True, markup=False)
            yield ProgressBar(
                id="progress", show_percentage=False, show_eta=False, total=1
            )
            with Horizontal(id="controls"):
                yield TextArea(id="input", placeholder=string("ui.wait_placeholder"))
                yield Button(string("ui.no_voice_set"), id="style", disabled=True)
                yield Button(string("ui.vc_mode_talk"), id="vc-mode", disabled=True)
                yield Button(
                    string("ui.vc_pitch_button", value="+0"),
                    id="vc-pitch",
                    disabled=True,
                )
            with Horizontal(id="bottom"):
                yield Label("", id="status")
                yield Label("", id="resources")

    def on_mount(self) -> None:
        """Prepare the UI runtime.

        Raises:
            RuntimeError: ``CeluneUI`` was run without an instance of ``Celune``.
        """
        if not self._has_celune():
            raise RuntimeError(
                f"{self.__class__.__name__} requires an instance of {APP_NAME} to be set"
            )

        colors.configure_theme()

        if os.name == "nt":
            self._install_windows_signal_handler()
        else:
            if SIGTSTP is not None:
                signal.signal(SIGTSTP, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self.set_interval(0.1, self._check_launcher_loss)

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
                    colors.configure_theme(
                        background,
                        accent,
                        faded_accent,
                    )

        self._ensure_themes_registered()
        self._bind_runtime_callbacks()
        self._wrap_runtime_fatal_glow()

        if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }:
            self.active_theme_name = "celune_april_fools"
        else:
            theme = os.getenv("CELUNE_THEME") or self.celune.config.get("theme", "dark")

            if theme == "dark":
                self.active_theme_name = "celune"
            elif theme == "light":
                self.active_theme_name = "celune_light"
            else:
                self.active_theme_name = "celune"
                self.safe_log(string("ui.invalid_theme_defaulting_dark"), "warning")

        self.theme = self.active_theme_name

        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.resources = self.query_one("#resources", Label)
        self.style_button = self.query_one("#style", Button)
        self.vc_mode_button = self.query_one("#vc-mode", Button)
        self.vc_pitch_button = self.query_one("#vc-pitch", Button)
        self.progress_bar = self.query_one("#progress", ProgressBar)
        self.header = self.query_one("#header", Label)
        self.header_lines = tuple(cast(Label, widget) for widget in self.query(".line"))
        self.set_focus(None)
        self._refresh_status()
        self.refresh_vc_controls()
        self._refresh_theme_text()
        self._refresh_logs()
        if not self.celune.backend.is_fake or "pytest" in sys.modules:
            self._enable_runtime_log_capture()
        ui_resources.prime_usage()
        self.set_interval(FOOTER_ROTATE_SECONDS, self.advance_resources)
        self._status_marquee_timer = self.set_interval(
            0.18, self._advance_status_marquee
        )

        self.call_after_refresh(self.start_background_init)
        self.safe_status(string("status.initializing"))
        self.update_resources()

    def _check_launcher_loss(self) -> None:
        """Run the normal UI shutdown path after the launcher disconnects."""
        if launcher_loss_requested() and self.cur_state != "exiting":
            self._graceful_exit()

    def update_resources(self) -> None:
        """Refresh the currently selected resource footer page."""
        if self.cur_state == "exiting" or self.resources is None:
            return

        def update() -> None:
            pages = ui_resources.resource_pages(self.celune, self.active_theme_name)
            text = pages[self._resource_page % len(pages)]

            if supports_ansi() and self.celune.cur_state == "error":
                self._write_terminal_escape(f"\x1b]2;{next(CRASH_LINES)}\x07")

            self.resources.update(indent(text, spaces=2, direction="right"))

        self._run_on_ui_thread(update)

    def _enable_runtime_log_capture(self) -> None:
        """Capture Celune runtime output after the Textual app has started cleanly."""
        if self._runtime_log_capture_enabled:
            return

        self._log_stdout = LogRedirect(
            write_callback=self.safe_log,
            default_severity="info",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )
        self._log_stderr = LogRedirect(
            write_callback=self.safe_log,
            default_severity="warning",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr
        self._redirect_dunder_stdio()
        self._install_runtime_log_redirects()
        self._install_low_level_stderr_capture()
        self._runtime_log_capture_enabled = True

    def _write_terminal_escape(self, escape: str) -> None:
        """Write one ANSI escape sequence to the real terminal when available."""
        if self._log_stdout is not None:
            self._log_stdout.ansi(escape)
            return

        if self._old_stdout is not None:
            self._old_stdout.write(escape)
            self._old_stdout.flush()

    def _install_runtime_log_redirects(self) -> None:
        """Route non-Celune Python logging output into Celune's UI log widget."""
        if self._runtime_redirect_handler is not None:
            return

        handler = UILogHandler(
            self.safe_log,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )
        original_call_handlers = logging.Logger.callHandlers

        def call_handlers(self: logging.Logger, record: logging.LogRecord) -> None:  # noqa
            if is_celune_log_record(record):
                original_call_handlers(self, record)
                return

            handler.handle(record)

        self._runtime_redirect_handler = handler
        self._runtime_redirect_original_call_handlers = original_call_handlers
        self._runtime_redirect_original_last_resort = logging.lastResort
        self._runtime_redirect_original_raise_exceptions = logging.raiseExceptions
        logging.Logger.callHandlers = call_handlers
        logging.lastResort = None
        logging.raiseExceptions = False

        logging.captureWarnings(True)
        self._warnings_capture_enabled = True

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
        if self._stderr_forward_thread is not None:
            return

        stderr_stream = self._old_stderr
        if stderr_stream is None or not hasattr(stderr_stream, "fileno"):
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

    @staticmethod
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
        stderr_stream = self._old_stderr
        original_fd_dup = self._stderr_original_fd_dup
        pipe_write_fd = self._stderr_pipe_write_fd
        pipe_read_fd = self._stderr_pipe_read_fd

        if (
            stderr_stream is not None
            and hasattr(stderr_stream, "fileno")
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
                logging.raiseExceptions = (
                    self._runtime_redirect_original_raise_exceptions
                )
            handler.close()

        if self._warnings_capture_enabled:
            logging.captureWarnings(False)
            self._warnings_capture_enabled = False

        self._runtime_redirect_handler = None
        self._runtime_redirect_original_call_handlers = None
        self._runtime_redirect_original_last_resort = None
        self._runtime_redirect_original_raise_exceptions = None

    def _disable_runtime_log_capture(self) -> None:
        """Restore global stdio once the UI is shutting down."""
        if self._log_stdout is not None:
            self._log_stdout.flush()
        if self._log_stderr is not None:
            self._log_stderr.flush()

        self._remove_low_level_stderr_capture()
        self._remove_runtime_log_redirects()
        self._restore_dunder_stdio()

        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._runtime_log_capture_enabled = False

    def advance_resources(self) -> None:
        """Advance the resource footer to the next page and refresh it."""
        if self.cur_state == "exiting" or self.resources is None:
            return

        self._resource_page = (self._resource_page + 1) % len(
            ui_resources.resource_pages(self.celune, self.active_theme_name)
        )
        self.update_resources()

    def _cancel_sleep_timer(self) -> None:
        """Cancel a pending automatic sleep transition."""
        if threading.current_thread() is not threading.main_thread():
            self.call_from_thread(self._cancel_sleep_timer)
            return

        if self._sleep_timer is not None:
            self._sleep_timer.stop()
            self._sleep_timer = None

    def _schedule_sleep_timer(self) -> None:
        """Schedule automatic sleep after the configured idle timeout."""
        if threading.current_thread() is not threading.main_thread():
            self.call_from_thread(self._schedule_sleep_timer)
            return

        self._cancel_sleep_timer()
        if (
            self.cur_state == "exiting"
            or not self.celune_ready
            or not hasattr(self.celune, "sleep_enabled")
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

    @work(exclusive=True)
    async def enter_sleep_mode(self) -> None:
        """Put the app to sleep without blocking the UI event loop."""
        if await self.celune.enter_sleep_mode_async():
            self.safe_log(
                string("ui.sleeping_log", app_name=APP_NAME),
                "sleeping",
            )
            self.safe_status(string("ui.sleeping_status"), "sleeping")
            self.change_voice_lock_state(locked=True)

    @work(exclusive=True)
    async def wake_from_sleep(self) -> None:
        """Wake the app after the user types into the sleeping UI."""
        try:
            if await self.celune.wake_from_sleep_async():
                self._schedule_sleep_timer()
        finally:
            if self.celune.sleeping:
                self.safe_status(string("ui.sleeping_status"), "sleeping")

    def start_background_init(self) -> None:
        """Run the initialization function."""
        self.load_tts()

    @work(thread=True, exclusive=True)
    def load_tts(self) -> None:
        """Load the app runtime."""
        try:
            if self.celune.load():
                self.celune_styles = self.celune.voices
                if not self.celune_styles:
                    if self._is_ui_test_mode():
                        if not self.celune.use_normalization:
                            self.safe_progress(1, 1)
                        self.change_input_state(locked=True)
                        self.change_voice_lock_state(locked=True)
                        self.safe_status(string("ui.test_mode_active"))
                        return

                    self.change_input_state(locked=True)
                    self.change_voice_lock_state(locked=True)
                    self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                    self.cur_state = "error"
                    return
                self.celune_voices = itertools.cycle(self.celune_styles)
                if self.celune.current_voice in self.celune_styles:
                    self.style_index = self.celune_styles.index(
                        self.celune.current_voice
                    )
                else:
                    self.style_index = 0
                self.celune_ready = True
                self.safe_status(string("ui.idle_status"))
                self.tts_voice_changed(
                    self.celune.current_voice or self.celune.voices[0]
                )
                if not self.celune.use_normalization:
                    self.safe_progress(1, 1)
                self.change_input_state(locked=False)
                self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
                self.safe_log(string("ui.tutorial_prompt", app_name=APP_NAME))
                self._schedule_sleep_timer()
                if supports_ansi(self._old_stdout):
                    self.call_from_thread(
                        lambda: self._write_terminal_escape(f"\x1b]2;{APP_NAME}\x07")
                    )
            else:
                self.cur_state = "error"
                self.change_input_state(locked=True)
                self.change_voice_lock_state(locked=True)
                self.error(string("ui.app_could_not_start", app_name=APP_NAME))
        except Exception as e:
            self.cur_state = "error"
            self.safe_log(
                string(
                    "ui.init_error",
                    error=format_error(e, self.celune.dev),
                ),
                "error",
            )
            self.celune.fatal()
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            self.error(string("ui.app_could_not_start", app_name=APP_NAME))

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Current progress, or ``None`` for an indeterminate bar.
            total: Total progress, or ``None`` for an indeterminate bar.
        """
        if self.cur_state == "exiting" or self.progress_bar is None:
            return

        def update() -> None:
            self.progress_bar.update(
                total=total,
                progress=0 if progress is None else progress,
            )

        self._run_on_ui_thread(update)

    @staticmethod
    def _with_brightness(color: Color, brightness: float) -> Color:
        """Return ``color`` blended toward the requested brightness."""
        brightness = max(0.0, min(1.0, brightness))
        current = color.brightness

        if abs(current - brightness) < 0.01:
            return color

        destination = Color(255, 255, 255) if current < brightness else Color(0, 0, 0)
        destination_brightness = destination.brightness
        factor = (brightness - current) / (destination_brightness - current)
        return color.blend(destination, max(0.0, min(1.0, factor)))

    @staticmethod
    def _with_darkened_brightness(color: Color) -> Color:
        """Return ``color`` with a visibly darker brightness."""
        target_brightness = max(0.0, color.brightness * 0.6)
        return CeluneUI.with_brightness(color, target_brightness)

    def pulse_border(self, target: Union[str, Widget]) -> None:
        """Softly pulse a widget border darker and back.

        Args:
            target: Widget or Textual selector for the target widget.
        """
        if threading.current_thread() is not threading.main_thread():
            self.call_from_thread(lambda: self.pulse_border(target))
            return

        duration = 2.06
        steps = 10

        widget = self.query_one(target) if isinstance(target, str) else target
        original_border: tuple[EdgeStyle, ...] = tuple(widget.styles.border)  # noqa

        if not any(edge_type for edge_type, _ in original_border):
            return

        widget_key = id(widget)
        token = self._border_pulse_tokens.get(widget_key, 0) + 1
        self._border_pulse_tokens[widget_key] = token
        self._border_pulse_widgets[widget_key] = widget

        target_border: tuple[EdgeStyle, ...] = tuple(  # noqa
            (
                edge_type,
                self._with_darkened_brightness(color) if edge_type else color,
            )
            for edge_type, color in original_border
        )
        steps = max(1, steps)
        duration = max(0.0, duration)
        hold_duration = min(0.2, duration / 3)
        transition_duration = max(0.0, duration - hold_duration)
        frame_delay = transition_duration / (steps * 2) if transition_duration else 0.0

        def set_border(border: tuple[EdgeStyle, ...]) -> None:  # noqa
            (
                widget.styles.border_top,
                widget.styles.border_right,
                widget.styles.border_bottom,
                widget.styles.border_left,
            ) = border
            widget.refresh(layout=False)

        def apply_blend(progress: float) -> None:
            if self._border_pulse_tokens.get(widget_key) != token:
                return

            eased = progress * progress * (3 - 2 * progress)
            set_border(
                tuple(
                    (
                        edge_type,
                        start_color.blend(end_color, eased)
                        if edge_type
                        else start_color,
                    )
                    for (edge_type, start_color), (_, end_color) in zip(
                        original_border, target_border
                    )
                )
            )

        def restore() -> None:
            if self._border_pulse_tokens.get(widget_key) != token:
                return
            widget.styles.border = None
            widget.refresh(layout=False)
            self._border_pulse_tokens.pop(widget_key, None)
            self._border_pulse_widgets.pop(widget_key, None)

        def schedule_frame(index: int, delay: float) -> None:
            if self._border_pulse_tokens.get(widget_key) != token:
                return

            if index >= steps * 2:
                restore()
                return

            def run_frame() -> None:
                if self._border_pulse_tokens.get(widget_key) != token:
                    return

                if index < steps:
                    apply_blend((index + 1) / steps)
                    next_delay = (
                        hold_duration + frame_delay
                        if index + 1 == steps
                        else frame_delay
                    )
                else:
                    apply_blend(1 - ((index - steps + 1) / steps))
                    next_delay = frame_delay

                schedule_frame(index + 1, next_delay)

            if delay <= 0:
                run_frame()
            else:
                self.set_timer(delay, run_frame)

        schedule_frame(0, frame_delay)

    def change_voice_lock_state(self, locked: bool) -> None:
        """Lock or unlock the ability to change Celune's voice.

        Args:
            locked: Whether voice changes should be disabled.
        """

        def update() -> None:
            self.style_button.disabled = locked
            self.update_resources()

        self._run_on_ui_thread(update)

    def _normal_input_placeholder(self) -> str:
        """Return the unlocked input placeholder without blocking the UI."""
        if self._is_voice_conversion_mode():
            return string("ui.voice_changer_placeholder")

        if (
            self._persona_loaded()
            and self._persona_available
            and persona_enabled(self.celune.config)
            and persona_talkback_enabled(self.celune.config)
        ):
            return string("ui.say_placeholder")

        return string("ui.input_placeholder")

    def _persona_loaded(self) -> bool:
        """Return whether the attached Celune instance currently has Persona."""
        return bool(getattr(self.celune, "vision", None))

    def _refresh_persona_availability(self) -> None:
        """Refresh Persona availability in the background for placeholder text."""
        if self._persona_probe_running:
            return

        self._persona_probe_running = True

        def probe() -> None:
            available = self._persona_loaded()

            def apply_result() -> None:
                self._persona_probe_running = False
                if self.cur_state == "exiting":
                    return

                changed = self._persona_available != available
                self._persona_available = available
                if changed and not self._input_locked:
                    self.input_box.placeholder = self._normal_input_placeholder()

            self._run_on_ui_thread(apply_result)

        threading.Thread(target=probe, daemon=True).start()

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: Whether user input should be disabled.
        """

        if not locked:
            self._schedule_sleep_timer()

        def update() -> None:
            self._input_locked = locked
            self.input_box.placeholder = (
                string("ui.wait_placeholder")
                if locked
                else self._normal_input_placeholder()
            )
            self.style_button.disabled = locked
            self.refresh_vc_controls()
            self.update_resources()

        self._run_on_ui_thread(update)
        if not locked:
            self._refresh_persona_availability()

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: The status text to display.
            severity: The status severity level.
        """
        if self.cur_state == "exiting" or self.status is None:
            return

        if severity not in colors.SEVERITY_COLORS["celune"]:
            self.safe_log(
                f"[WARNING] Unknown severity '{severity}', defaulting to info",
                "warning",
            )
            severity = "info"

        if self._fatal_error_active and severity != "error":
            return

        self.status_severity = severity

        def update() -> None:
            self._status_text = msg
            self._status_marquee_offset = 0
            self._refresh_theme_text()
            self._update_status_label()
            self.update_resources()

        self._run_on_ui_thread(update)

    def safe_log(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The log line to append.
            severity: The log severity level.
        """
        if self.cur_state == "exiting":
            return

        if severity not in colors.SEVERITY_COLORS["celune"]:
            severity = "info"

        self.log_history.append((msg, severity))
        self._persist_log_entry(msg, severity)
        if self.logs is None:
            return

        entry = Text(msg, style=self._severity_color(severity))

        if threading.current_thread() is threading.main_thread():
            self.logs.write(entry)
        else:
            self.post_message(UILogMessage(msg, severity))

    def on_uilog_message(self, message: UILogMessage) -> None:
        """Write a background log message on Textual's application thread.

        Args:
            message: Background log message to write to the UI.
        """
        if self.logs is not None:
            self.logs.write(
                Text(message.message, style=self._severity_color(message.severity))
            )

    def safe_log_dev(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The log line to append.
            severity: The log severity level.
        """
        if self.celune.dev:
            self.safe_log(msg, severity)

    def _is_voice_conversion_mode(self) -> bool:
        """Return whether the attached Celune instance is running in VC mode."""
        return bool(
            self.celune is not None
            and getattr(self.celune, "vc_backend", None) is not None
        )

    @staticmethod
    def _format_vc_pitch_shift(value: int) -> str:
        """Return one signed semitone label for the VC pitch control."""
        return f"{value:+d}"

    def _set_vc_controls_visibility(self, visible: bool) -> None:
        """Show or hide the VC-only controls in the bottom input row."""
        if self.vc_mode_button is not None:
            self.vc_mode_button.display = visible
        if self.vc_pitch_button is not None:
            self.vc_pitch_button.display = visible

    def refresh_vc_controls(self) -> None:
        """Refresh VC control labels and enabled state from the current engine state."""
        if (
            self.vc_mode_button is None
            or self.vc_pitch_button is None
            or self.celune is None
        ):
            return

        is_vc_mode = self._is_voice_conversion_mode()
        self._set_vc_controls_visibility(is_vc_mode)
        if not is_vc_mode:
            self._cancel_vc_recording(announce=False)
        f0_condition = bool(getattr(self.celune, "vc_f0_condition", False))
        pitch_shift = int(getattr(self.celune, "vc_pitch_shift", 0))
        self.vc_mode_button.label = string(
            "ui.vc_mode_sing" if f0_condition else "ui.vc_mode_talk"
        )
        self.vc_pitch_button.label = string(
            "ui.vc_pitch_button",
            value=self._format_vc_pitch_shift(pitch_shift),
        )
        self.vc_mode_button.disabled = (not is_vc_mode) or self._input_locked
        self.vc_pitch_button.disabled = (not is_vc_mode) or self._input_locked

    def set_vc_f0_condition(self, enabled: bool, announce: bool = True) -> None:
        """Update the active VC talk-vs-sing mode in the UI and backend state.

        Args:
            enabled: Whether to enable sing-mode F0 conditioning.
            announce: Whether to log the new mode to the user.
        """
        if self.celune is None:
            return

        self.celune.vc_f0_condition = enabled
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and hasattr(backend, "f0_condition"):
            backend.f0_condition = enabled
        self.refresh_vc_controls()

        if announce:
            self.safe_log(
                string(
                    "ui.vc_mode_changed",
                    mode=string("ui.vc_mode_sing" if enabled else "ui.vc_mode_talk"),
                )
            )

    def set_vc_pitch_shift(self, value: int, announce: bool = True) -> None:
        """Update the active VC pitch-shift value in the UI and backend state.

        Args:
            value: The requested pitch shift in semitones before clamping.
            announce: Whether to log the new pitch shift to the user.
        """
        if self.celune is None:
            return

        clamped = clamp_vc_pitch_shift(value)
        self.celune.vc_pitch_shift = clamped
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and hasattr(backend, "pitch_shift"):
            backend.pitch_shift = clamped
        self.refresh_vc_controls()

        if announce:
            self.safe_log(
                string(
                    "ui.vc_pitch_changed",
                    value=self._format_vc_pitch_shift(clamped),
                )
            )

    def _persona_recording_active(self) -> bool:
        """Return whether Persona microphone capture is active."""
        return self._persona_recording_stream is not None

    def _persona_speech_model_id(self) -> str:
        """Return the configured Hugging Face Whisper model ID."""
        configured = persona_config(self.celune.config).get("speech_model_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return DEFAULT_PERSONA_SPEECH_MODEL_ID

    def _persona_speech_language(self) -> Optional[str]:
        """Return a configured Whisper language, or ``None`` for auto-detection."""
        configured = persona_config(self.celune.config).get("speech_language")
        if not isinstance(configured, str) or configured.strip().lower() in {
            "",
            "auto",
        }:
            return None
        return configured.strip()

    def _persona_speech_end_delay_seconds(self) -> float:
        """Return the extra VAD silence delay before Persona submission."""
        configured = persona_config(self.celune.config).get("speech_end_delay_seconds")
        if (
            isinstance(configured, (int, float))
            and not isinstance(configured, bool)
            and configured >= 0
        ):
            return float(configured)
        return PERSONA_SPEECH_END_DELAY_SECONDS

    def _persona_recording_audio_locked(self) -> npt.NDArray[np.float32]:
        """Return captured Persona audio while holding its lock."""
        if not self._persona_recording_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._persona_recording_chunks, axis=0).astype(
            np.float32,
            copy=False,
        )

    def _queue_persona_recording_item_locked(self, final: bool) -> None:
        """Queue a partial or final Persona transcription snapshot."""
        recording_queue = self._persona_recording_queue
        if recording_queue is None:
            return

        audio = self._persona_recording_audio_locked().copy()
        if final:
            while True:
                try:
                    recording_queue.get_nowait()
                except queue_module.Empty:
                    break
            recording_queue.put_nowait((audio, True))
            return

        try:
            recording_queue.put_nowait((audio, False))
        except queue_module.Full:
            with contextlib.suppress(queue_module.Empty):
                recording_queue.get_nowait()
            with contextlib.suppress(queue_module.Full):
                recording_queue.put_nowait((audio, False))

    def _set_persona_recording_text(self, transcript: str) -> None:
        """Display a live Whisper transcript in the main input box."""
        prefix = self._persona_recording_text_prefix
        text = f"{prefix} {transcript}".strip() if prefix else transcript.strip()

        def update() -> None:
            if self.cur_state == "exiting" or self.input_box is None:
                return
            self._suppress_input_change = True
            try:
                self.input_box.load_text(text)
            finally:
                self._suppress_input_change = False

        self._run_on_ui_thread(update)

    def _complete_persona_transcription(
        self,
        transcript: str,
        prefix: str,
        error: Optional[Exception] = None,
        error_already_reported: bool = False,
    ) -> None:
        """Submit the final Persona transcript or report its transcription error."""
        if error is not None and not error_already_reported:
            self.safe_log(
                string(
                    "ui.persona_transcription_failed",
                    error=format_error(error, self.celune.dev),
                ),
                "error",
            )
        if error is not None or error_already_reported:
            self.safe_status(string("ui.idle_status"))
            if self.style_button is not None:
                self.style_button.disabled = self._input_locked
            self.update_resources()
            return

        text = f"{prefix} {transcript}".strip() if prefix else transcript.strip()
        if text:
            self._set_persona_recording_text(transcript)
            self._submit_text(text, process_commands=False)
        else:
            self.safe_log(string("ui.recording_empty"), "warning")
        self.safe_status(string("ui.idle_status"))
        if self.style_button is not None:
            self.style_button.disabled = self._input_locked
        self.update_resources()

    def _persona_transcription_worker(
        self,
        recording_queue: queue_module.Queue[tuple[AudioChunk, bool]],
        transcriber: WhisperTranscriber,
        sample_rate: int,
        prefix: str,
    ) -> None:
        """Transcribe Persona microphone snapshots off the UI thread."""
        partial_error_reported = False
        while True:
            audio, final = recording_queue.get()
            transcript = ""
            error: Optional[Exception] = None
            try:
                transcript = transcriber.transcribe(audio, sample_rate)
            except Exception as exc:
                error = exc

            if transcript:
                self._set_persona_recording_text(transcript)

            if (
                error is not None
                and (final or not partial_error_reported)
                and not final
            ):
                partial_error_reported = True
                self.safe_log(
                    string(
                        "ui.persona_transcription_failed",
                        error=format_error(error, self.celune.dev),
                    ),
                    "warning",
                )

            if not final:
                continue

            with self._persona_recording_lock:
                stream = self._persona_recording_stream
                self._persona_recording_stream = None
                self._persona_recording_queue = None
                self._persona_recording_worker = None
                self._persona_recording_transcriber = None
                self._persona_recording_chunks = []
                self._persona_recording_stop_requested = False
                self._persona_recording_speech_started = False
                self._persona_recording_silence_frames = 0

            self._shutdown_vc_stream(stream)
            self._run_on_ui_thread(
                lambda: self._complete_persona_transcription(
                    transcript,  # noqa: B023
                    prefix,  # noqa: B023
                    error,  # noqa: B023
                    error_already_reported=partial_error_reported and error is not None,  # noqa: B023
                )
            )
            return

    def _request_persona_recording_stop(self) -> bool:
        """Queue final Persona audio for transcription and automatic submission."""
        with self._persona_recording_lock:
            if self._persona_recording_stream is None:
                return False
            if self._persona_recording_stop_requested:
                return True
            self._persona_recording_stop_requested = True
            self._queue_persona_recording_item_locked(final=True)

        self.safe_status(string("ui.persona_transcribing"))
        return True

    def _start_persona_recording(self) -> bool:
        """Start push-to-talk microphone capture for the active Persona."""
        if (
            self.celune is None
            or self._is_voice_conversion_mode()
            or not self._persona_loaded()
            or not persona_talkback_enabled(self.celune.config)
        ):
            return False
        if self._persona_recording_active():
            return True

        input_config = getattr(self.celune, "config", None)
        input_device_key = (
            "input_recording_device"
            if isinstance(input_config, dict)
            and "input_recording_device" in input_config
            else "input_device"
        )
        try:
            input_device, direct_device_info = resolve_audio_device_with_info(
                input_config,
                input_device_key,
                "input",
            )
            device_info = (
                cast(dict[str, AudioDeviceScalar], dict(direct_device_info))
                if direct_device_info is not None
                else cast(
                    dict[str, AudioDeviceScalar],
                    sd.query_devices(device=input_device, kind="input"),
                )
            )
        except Exception as exc:
            self.safe_log(
                string(
                    "ui.recording_open_input_failed",
                    error=format_error(exc, self.celune.dev),
                ),
                "error",
            )
            return False

        channels = _device_scalar_int(device_info.get("max_input_channels"), 0)
        if channels <= 0:
            self.safe_log(string("pipeline.no_audio_device"), "warning")
            return False

        sample_rate = _device_scalar_int(
            device_info.get("default_samplerate"),
            48000,
        )
        channel_count = 2 if channels >= 2 else 1
        vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate) + int(
            sample_rate * self._persona_speech_end_delay_seconds()
        )
        ai_vad = create_live_voice_activity_detector(input_config)
        recording_queue: queue_module.Queue[tuple[AudioChunk, bool]] = (
            queue_module.Queue(maxsize=1)
        )
        transcriber = WhisperTranscriber(
            self._persona_speech_model_id(),
            language=self._persona_speech_language(),
        )
        prefix = self.input_box.text.strip() if self.input_box is not None else ""
        should_stop = False

        def callback(
            indata: npt.NDArray[np.float32],
            frames: int,
            time_info: Optional[tuple[float, float, float]],
            status: Optional[sd.CallbackFlags],
        ) -> None:
            discard(frames)
            discard(time_info)
            discard(status)
            nonlocal should_stop

            callback_audio = np.asarray(indata, dtype=np.float32).copy()
            if ai_vad is not None:
                try:
                    voice_detected = ai_vad.has_voice(callback_audio, sample_rate)
                except (RuntimeError, AssertionError, ValueError):
                    ai_vad.reset()
                    voice_detected = self._vc_input_has_voice(callback_audio)
            else:
                voice_detected = self._vc_input_has_voice(callback_audio)

            with self._persona_recording_lock:
                if (
                    self._persona_recording_stream is None
                    or self._persona_recording_stop_requested
                ):
                    return

                if voice_detected:
                    self._persona_recording_speech_started = True
                    self._persona_recording_silence_frames = 0
                elif self._persona_recording_speech_started:
                    self._persona_recording_silence_frames += len(callback_audio)

                if self._persona_recording_speech_started:
                    self._persona_recording_chunks.append(callback_audio)
                    if (
                        time.monotonic() - self._persona_recording_last_partial_at
                        >= 0.8
                    ):
                        self._queue_persona_recording_item_locked(final=False)
                        self._persona_recording_last_partial_at = time.monotonic()

                if (
                    self._persona_recording_speech_started
                    and self._persona_recording_silence_frames >= vad_hangover_frames
                ):
                    self._persona_recording_stop_requested = True
                    self._queue_persona_recording_item_locked(final=True)
                    should_stop = True

            if should_stop:
                self.safe_status(string("ui.persona_transcribing"))

        worker: Optional[threading.Thread] = None
        stream: Optional[sd.InputStream] = None
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channel_count,
                dtype="float32",
                callback=callback,
                device=input_device,
            )
            worker = threading.Thread(
                target=self._persona_transcription_worker,
                args=(recording_queue, transcriber, sample_rate, prefix),
                daemon=True,
            )
            with self._persona_recording_lock:
                self._persona_recording_stream = stream
                self._persona_recording_queue = recording_queue
                self._persona_recording_worker = worker
                self._persona_recording_transcriber = transcriber
                self._persona_recording_sample_rate = sample_rate
                self._persona_recording_chunks = []
                self._persona_recording_silence_frames = 0
                self._persona_recording_speech_started = False
                self._persona_recording_stop_requested = False
                self._persona_recording_text_prefix = prefix
                self._persona_recording_last_partial_at = time.monotonic()
            stream.start()
            worker.start()
        except Exception as exc:
            with self._persona_recording_lock:
                stream = self._persona_recording_stream
                self._persona_recording_stream = None
                self._persona_recording_queue = None
                self._persona_recording_worker = None
                self._persona_recording_transcriber = None
                self._persona_recording_chunks = []
                self._persona_recording_stop_requested = True
            self._shutdown_vc_stream(stream)
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)
            self.safe_log(
                string(
                    "ui.recording_start_failed",
                    label=string("ui.audio_input_label"),
                    error=format_error(exc, self.celune.dev),
                ),
                "error",
            )
            return False

        self.safe_log(string("ui.persona_recording_started"), "info")
        self.safe_status(string("ui.persona_recording_listening"))
        self.update_resources()
        return True

    def toggle_persona_recording(self) -> bool:
        """Toggle Persona microphone capture and final transcription.

        Returns:
            ``True`` when recording was stopped or started successfully.
        """
        if self._persona_recording_active():
            return self._request_persona_recording_stop()
        return self._start_persona_recording()

    def _shutdown_persona_recording(self) -> None:
        """Stop Persona microphone capture without submitting a final utterance."""
        with self._persona_recording_lock:
            stream = self._persona_recording_stream
            recording_queue = self._persona_recording_queue
            worker = self._persona_recording_worker
            self._persona_recording_stream = None
            self._persona_recording_queue = None
            self._persona_recording_worker = None
            self._persona_recording_transcriber = None
            self._persona_recording_chunks = []
            self._persona_recording_stop_requested = True
            if recording_queue is not None:
                while True:
                    try:
                        recording_queue.get_nowait()
                    except queue_module.Empty:
                        break
                with contextlib.suppress(queue_module.Full):
                    recording_queue.put_nowait((np.zeros(0, dtype=np.float32), True))
        self._shutdown_vc_stream(stream)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def _vc_recording_active(self) -> bool:
        """Return whether live VC recording is active in the TUI."""
        return self._vc_recording_stream is not None

    @staticmethod
    def _vc_input_rms(audio: npt.NDArray[np.float32]) -> float:
        """Return RMS energy for one microphone callback buffer."""
        return vc_input_rms(audio)

    def _request_vc_recording_feedback_stop(self) -> None:
        """Request a feedback-triggered recording stop on a dedicated thread."""
        stop_thread = threading.Thread(
            target=self._stop_vc_recording_for_feedback,
            daemon=True,
        )
        self._vc_recording_stop_thread = stop_thread
        stop_thread.start()

    def _flush_vc_recording_buffer_locked(self) -> Optional[AudioChunk]:
        """Return and clear the buffered microphone chunk accumulator."""
        if not self._vc_recording_chunks:
            return None
        audio = np.concatenate(self._vc_recording_chunks, axis=0)
        self._vc_recording_chunks = []
        self._vc_recording_buffered_frames = 0
        return audio

    def _flush_vc_recording_chunk_locked(
        self,
        keep_tail_frames: int = 0,
    ) -> Optional[AudioChunk]:
        """Return one buffered VC chunk while optionally retaining a tail overlap."""
        audio = self._flush_vc_recording_buffer_locked()
        if audio is None or keep_tail_frames <= 0:
            return audio

        if len(audio) <= keep_tail_frames:
            self._vc_recording_chunks = [audio]
            self._vc_recording_buffered_frames = len(audio)
            return None

        retained = np.asarray(audio[-keep_tail_frames:], dtype=np.float32).copy()
        flushed = np.asarray(audio[:-keep_tail_frames], dtype=np.float32).copy()
        self._vc_recording_chunks = [retained]
        self._vc_recording_buffered_frames = len(retained)
        return flushed

    @staticmethod
    def _vc_vad_hangover_frames(sample_rate: int) -> int:
        """Return how many trailing silent frames to tolerate before flushing."""
        return vc_vad_hangover_frames(sample_rate)

    @staticmethod
    def _vc_vad_preroll_frames(sample_rate: int) -> int:
        """Return how much recent pre-speech audio to keep before VAD triggers."""
        return vc_vad_preroll_frames(sample_rate)

    @staticmethod
    def _vc_live_chunk_frames(sample_rate: int) -> int:
        """Return how much active speech to collect before a live VC flush."""
        return vc_live_chunk_frames(sample_rate)

    @staticmethod
    def _vc_live_chunk_overlap_frames(sample_rate: int) -> int:
        """Return how much tail audio to retain between live VC chunks."""
        return vc_live_chunk_overlap_frames(sample_rate)

    def _append_vc_preroll_audio_locked(
        self,
        audio: npt.NDArray[np.float32],
        max_frames: int,
    ) -> None:
        """Retain only the newest pre-speech audio frames for VAD onset recovery."""
        copied = np.asarray(audio, dtype=np.float32).copy()
        self._vc_recording_preroll_chunks.append(copied)
        self._vc_recording_preroll_frames += len(copied)

        while (
            self._vc_recording_preroll_chunks
            and self._vc_recording_preroll_frames > max_frames
        ):
            overflow_frames = self._vc_recording_preroll_frames - max_frames
            oldest = self._vc_recording_preroll_chunks[0]
            if len(oldest) <= overflow_frames:
                self._vc_recording_preroll_chunks.pop(0)
                self._vc_recording_preroll_frames -= len(oldest)
                continue
            trimmed = np.asarray(oldest[overflow_frames:], dtype=np.float32).copy()
            self._vc_recording_preroll_chunks[0] = trimmed
            self._vc_recording_preroll_frames -= overflow_frames
            break

    def _prepend_vc_preroll_locked(self) -> None:
        """Move retained pre-speech audio into the active VC speech buffer."""
        if not self._vc_recording_preroll_chunks:
            return
        self._vc_recording_chunks = [
            *self._vc_recording_preroll_chunks,
            *self._vc_recording_chunks,
        ]
        self._vc_recording_buffered_frames += self._vc_recording_preroll_frames
        self._vc_recording_preroll_chunks = []
        self._vc_recording_preroll_frames = 0

    def _clear_vc_preroll_locked(self) -> None:
        """Discard any retained pre-speech VC audio."""
        self._vc_recording_preroll_chunks = []
        self._vc_recording_preroll_frames = 0

    @staticmethod
    def _vc_input_has_voice(audio: npt.NDArray[np.float32]) -> bool:
        """Return whether one microphone callback likely contains voice activity."""
        return vc_input_has_voice(audio)

    @staticmethod
    def _normalize_vc_overlap_audio(
        audio: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Normalize one VC overlap chunk into valid mono or stereo time-first audio."""
        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.ndim == 1:
            return normalized
        if normalized.ndim != 2:
            raise ValueError(
                f"expected 1D or 2D VC overlap audio, got {normalized.shape}"
            )
        if normalized.shape[1] == 1:
            return normalized[:, 0]
        if normalized.shape[1] == 2:
            return normalized
        raise ValueError(
            f"expected mono or stereo VC overlap audio, got {normalized.shape}"
        )

    def _crossfade_vc_overlap(
        self,
        previous_tail: npt.NDArray[np.float32],
        current_head: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Crossfade two same-rate VC overlap regions into one seamless bridge."""
        overlap_frames = min(len(previous_tail), len(current_head))
        if overlap_frames <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        previous = self._normalize_vc_overlap_audio(previous_tail[-overlap_frames:])
        current = self._normalize_vc_overlap_audio(current_head[:overlap_frames])

        if previous.ndim != current.ndim:
            if previous.ndim == 1:
                previous = np.column_stack((previous, previous))
            if current.ndim == 1:
                current = np.column_stack((current, current))

        fade = np.linspace(0.0, 1.0, overlap_frames, dtype=np.float32)
        if previous.ndim == 2:
            fade = fade[:, None]

        return np.asarray(
            (previous * (1.0 - fade)) + (current * fade),
            dtype=np.float32,
        )

    @staticmethod
    def _enqueue_vc_submission_chunk(
        submission_queue: queue_module.Queue[
            Optional[tuple[AudioChunk, int, str, bool]]
        ],
        item: tuple[AudioChunk, int, str, bool],
    ) -> None:
        """Queue one live VC chunk while dropping only the stalest backlog item."""
        try:
            submission_queue.put_nowait(item)
            return
        except queue_module.Full:
            pass

        with contextlib.suppress(queue_module.Empty):
            submission_queue.get_nowait()

        with contextlib.suppress(queue_module.Full):
            submission_queue.put_nowait(item)

    @staticmethod
    def _finish_vc_submission_queue(
        submission_queue: Optional[
            queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]  # noqa
        ],
        final_item: Optional[tuple[AudioChunk, int, str, bool]] = None,
    ) -> None:
        """Flush stale live VC chunks and end the submission worker."""
        if submission_queue is None:
            return

        with contextlib.suppress(queue_module.Empty):
            while True:
                submission_queue.get_nowait()

        if final_item is not None:
            with contextlib.suppress(queue_module.Full):
                submission_queue.put_nowait(final_item)

        with contextlib.suppress(queue_module.Full):
            submission_queue.put_nowait(None)

    def _clear_vc_recording_state(self) -> None:
        """Clear transient VC recording buffers after stop or cancel."""
        self._vc_recording_stream = None
        self._vc_recording_chunks = []
        self._vc_recording_buffered_frames = 0
        self._vc_recording_captured_frames = 0
        self._vc_recording_feedback_detected = False
        self._vc_recording_feedback_spike_count = 0
        self._vc_recording_sample_rate = 0
        self._vc_recording_label = string("ui.audio_input_label")
        self._vc_recording_preroll_chunks = []
        self._vc_recording_preroll_frames = 0
        self._vc_recording_previous_rms = 0.0
        self._vc_recording_silence_frames = 0
        self._vc_recording_submission_queue = None
        self._vc_recording_stop_thread = None
        self._vc_recording_worker = None

    def _stop_vc_recording_stream(
        self,
    ) -> tuple[
        Optional[sd.InputStream],
        Optional[AudioChunk],
        int,
        str,
        Optional[
            queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]  # noqa
        ],
        int,
        Optional[threading.Thread],
        Optional[threading.Thread],
    ]:
        """Stop the active VC recording stream and return any pending live-state data."""
        stream = self._vc_recording_stream
        buffered_audio = self._flush_vc_recording_buffer_locked()
        sample_rate = self._vc_recording_sample_rate
        label = self._vc_recording_label
        submission_queue = self._vc_recording_submission_queue
        captured_frames = self._vc_recording_captured_frames
        stop_thread = self._vc_recording_stop_thread
        worker = self._vc_recording_worker
        self._clear_vc_recording_state()
        return (
            stream,
            buffered_audio,
            sample_rate,
            label,
            submission_queue,
            captured_frames,
            stop_thread,
            worker,
        )

    @staticmethod
    def _shutdown_vc_stream(stream: Optional[sd.InputStream]) -> None:
        """Stop and close one VC input stream outside the recording lock."""
        if stream is None:
            return

        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    @staticmethod
    def _join_vc_recording_threads(
        stop_thread: Optional[threading.Thread],
        worker: Optional[threading.Thread],
        timeout: float = 2.0,
    ) -> None:
        """Wait briefly for live VC helper threads to finish."""
        for thread in (stop_thread, worker):
            if thread is None or thread is threading.current_thread():
                continue
            with contextlib.suppress(Exception):
                thread.join(timeout=timeout)

    def _cancel_vc_recording(self, announce: bool = True) -> bool:
        """Stop VC recording without submitting audio for conversion."""
        if not self._vc_recording_active():
            return False

        with self._vc_recording_lock:
            (
                stream,
                _audio,
                _sample_rate,
                label,
                submission_queue,
                _captured_frames,
                stop_thread,
                worker,
            ) = self._stop_vc_recording_stream()
            self._finish_vc_submission_queue(submission_queue)
        self._shutdown_vc_stream(stream)
        self._join_vc_recording_threads(stop_thread, worker)

        if announce:
            self.safe_log(string("ui.recording_stopped", label=label), "info")
        return True

    def _stop_vc_recording_for_feedback(self) -> None:
        """Stop live VC recording after detecting a sudden feedback-like RMS spike."""
        if not self._vc_recording_active():
            return

        with self._vc_recording_lock:
            (
                stream,
                buffered_audio,
                sample_rate,
                label,
                submission_queue,
                _captured_frames,
                stop_thread,
                worker,
            ) = self._stop_vc_recording_stream()
            self._finish_vc_submission_queue(
                submission_queue,
                (
                    np.asarray(buffered_audio, dtype=np.float32),
                    sample_rate,
                    label,
                    True,
                )
                if buffered_audio is not None
                else None,
            )
        self._shutdown_vc_stream(stream)
        self._join_vc_recording_threads(stop_thread, worker)

        self.safe_log(string("ui.recording_stopped_feedback", label=label), "warning")
        self.update_resources()

    def _start_vc_recording(self) -> bool:
        """Start recording from the active system input device for VC."""
        if self.celune is None or not self._is_voice_conversion_mode():
            return False
        if (
            getattr(self.celune, "sleeping", False)
            or getattr(
                self.celune,
                "cur_state",
                "",
            )
            == "waking"
        ):
            return False

        if self._vc_recording_active():
            return True

        input_config = getattr(self.celune, "config", None)
        input_device_key = (
            "input_recording_device"
            if isinstance(input_config, dict)
            and "input_recording_device" in input_config
            else "input_device"
        )
        try:
            input_device, direct_device_info = resolve_audio_device_with_info(
                input_config,
                input_device_key,
                "input",
            )
        except ValueError as error:
            self.safe_log(str(error), "warning")
            return False

        try:
            device_info = (
                cast(dict[str, AudioDeviceScalar], dict(direct_device_info))
                if direct_device_info is not None
                else cast(
                    dict[str, AudioDeviceScalar],
                    sd.query_devices(device=input_device, kind="input"),
                )
            )
        except Exception as e:
            self.safe_log(
                string(
                    "ui.recording_open_input_failed",
                    error=format_error(e, self.celune.dev),
                ),
                "error",
            )
            return False

        channels = _device_scalar_int(device_info.get("max_input_channels"), 0)
        if channels <= 0:
            self.safe_log(string("pipeline.no_audio_device"), "warning")
            return False

        sample_rate = _device_scalar_int(
            device_info.get("default_samplerate"),
            48000,
        )
        label = format_audio_device_name(device_info) or str(
            device_info.get("name", string("ui.audio_input_label"))
        )
        channel_count = 2 if channels >= 2 else 1
        vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate)
        vad_preroll_frames = self._vc_vad_preroll_frames(sample_rate)
        live_chunk_frames = self._vc_live_chunk_frames(sample_rate)
        live_chunk_overlap_frames = self._vc_live_chunk_overlap_frames(sample_rate)
        ai_vad = create_live_voice_activity_detector(input_config)
        submission_queue: queue_module.Queue[
            Optional[tuple[AudioChunk, int, str, bool]]
        ] = queue_module.Queue(maxsize=1)

        def submit_live_audio() -> None:
            live_source_id: Optional[int] = None
            live_playback_generation: Optional[int] = None

            def queue_playback_segment(
                audio_chunk: AudioChunk,
                sr: int,
                audio_label: str,
            ) -> None:
                nonlocal live_playback_generation, live_source_id
                if len(audio_chunk) <= 0 or self.celune is None:
                    return

                # noinspection PyBroadException
                try:
                    if live_source_id is None:
                        live_playback_generation = getattr(
                            self.celune,
                            "_playback_generation",
                            0,
                        )
                    live_source_id = queue_streaming_sfx_audio(
                        self.celune,
                        np.asarray(audio_chunk, dtype=np.float32),
                        sr,
                        audio_label,
                        source_id=live_source_id,
                        generation=live_playback_generation,
                        status_label_key="pipeline.revoicing_label",
                        reset_ready_announcement=live_source_id is None,
                    )
                    if live_source_id is None:
                        live_playback_generation = None
                except Exception:
                    self.safe_log(
                        string("ui.recording_stream_submit_failed"),
                        "warning",
                    )
                    return

            def finish_playback_segment() -> None:
                nonlocal live_playback_generation, live_source_id
                if self.celune is None:
                    return
                finish_streaming_sfx_audio(self.celune, live_source_id)
                live_source_id = None
                live_playback_generation = None

            while True:
                item = submission_queue.get()
                if self.celune is None or getattr(self.celune, "exit_requested", False):
                    return
                if item is None:
                    finish_playback_segment()
                    return

                audio, queued_sample_rate, queued_label, is_final_chunk = item
                try:
                    if self.celune is None:
                        continue
                    converted = self.celune.convert_audio(
                        audio,
                        queued_sample_rate,
                        label=queued_label,
                    )
                    if converted is None:
                        self.safe_log(
                            string("ui.recording_stream_submit_failed"),
                            "warning",
                        )
                        continue

                    converted_audio = np.asarray(converted.audio, dtype=np.float32)
                    playback_sample_rate = converted.sample_rate
                    playback_label = converted.label
                    if len(converted_audio) > 0:
                        queue_playback_segment(
                            converted_audio,
                            playback_sample_rate,
                            playback_label,
                        )

                    if is_final_chunk:
                        finish_playback_segment()
                except Exception as exc:
                    if self.celune is None:
                        continue
                    self.safe_log(
                        string(
                            "ui.recording_stream_chunk_failed",
                            label=queued_label,
                            error=format_error(exc, self.celune.dev),
                        ),
                        "warning",
                    )

        worker = threading.Thread(target=submit_live_audio, daemon=True)

        def callback(
            indata: npt.NDArray[np.float32],
            frames: int,
            time_info: Optional[tuple[float, float, float]],
            status: Optional[sd.CallbackFlags],
        ) -> None:
            discard(frames)
            discard(time_info)
            discard(status)

            callback_audio = np.asarray(indata, dtype=np.float32).copy()
            current_rms = self._vc_input_rms(callback_audio)
            if ai_vad is not None:
                try:
                    voice_detected = ai_vad.has_voice(callback_audio, sample_rate)
                except (RuntimeError, AssertionError, ValueError):
                    ai_vad.reset()
                    voice_detected = self._vc_input_has_voice(callback_audio)
            else:
                voice_detected = self._vc_input_has_voice(callback_audio)

            with self._vc_recording_lock:
                if self._vc_recording_stream is None:
                    return
                if self._vc_recording_feedback_detected:
                    return

                self._vc_recording_previous_rms = current_rms
                self._vc_recording_captured_frames += len(callback_audio)

                buffered_audio: Optional[AudioChunk] = None
                if voice_detected:
                    if self._vc_recording_buffered_frames <= 0:
                        self._prepend_vc_preroll_locked()
                    self._vc_recording_silence_frames = 0
                    self._vc_recording_chunks.append(callback_audio)
                    self._vc_recording_buffered_frames += len(callback_audio)
                    self._clear_vc_preroll_locked()
                    if self._vc_recording_buffered_frames >= live_chunk_frames:
                        buffered_audio = self._flush_vc_recording_chunk_locked(
                            keep_tail_frames=live_chunk_overlap_frames,
                        )
                elif self._vc_recording_buffered_frames > 0:
                    self._vc_recording_silence_frames += len(callback_audio)
                    if self._vc_recording_silence_frames <= vad_hangover_frames:
                        self._vc_recording_chunks.append(callback_audio)
                        self._vc_recording_buffered_frames += len(callback_audio)
                    else:
                        buffered_audio = self._flush_vc_recording_chunk_locked()
                        self._vc_recording_silence_frames = 0
                        self._append_vc_preroll_audio_locked(
                            callback_audio,
                            vad_preroll_frames,
                        )
                        if ai_vad is not None:
                            ai_vad.reset()
                else:
                    self._append_vc_preroll_audio_locked(
                        callback_audio,
                        vad_preroll_frames,
                    )

                if (
                    buffered_audio is not None
                    and self._vc_recording_submission_queue is not None
                ):
                    self._enqueue_vc_submission_chunk(
                        self._vc_recording_submission_queue,
                        (buffered_audio, sample_rate, label, False),
                    )

        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channel_count,
                dtype="float32",
                callback=callback,
                device=input_device,
            )
            stream.start()
        except Exception as e:
            self.safe_log(
                string(
                    "ui.recording_start_failed",
                    label=label,
                    error=format_error(e, self.celune.dev),
                ),
                "error",
            )
            return False

        with self._vc_recording_lock:
            self._vc_recording_stream = stream
            self._vc_recording_chunks = []
            self._vc_recording_buffered_frames = 0
            self._vc_recording_captured_frames = 0
            self._vc_recording_feedback_detected = False
            self._vc_recording_sample_rate = sample_rate
            self._vc_recording_label = label
            self._vc_recording_preroll_chunks = []
            self._vc_recording_preroll_frames = 0
            self._vc_recording_previous_rms = 0.0
            self._vc_recording_silence_frames = 0
            self._vc_recording_submission_queue = submission_queue
            self._vc_recording_stop_thread = None
            self._vc_recording_worker = worker

        worker.start()
        self.safe_log(string("ui.recording_started", label=label), "info")
        self.update_resources()
        return True

    def toggle_vc_recording(self) -> bool:
        """Toggle live VC recording for the current input device.

        Returns:
            bool: ``True`` when recording started or stopped successfully.
        """
        if self._vc_recording_active():
            with self._vc_recording_lock:
                (
                    stream,
                    buffered_audio,
                    sample_rate,
                    label,
                    submission_queue,
                    captured_frames,
                    stop_thread,
                    worker,
                ) = self._stop_vc_recording_stream()
                self._finish_vc_submission_queue(
                    submission_queue,
                    (
                        np.asarray(buffered_audio, dtype=np.float32),
                        sample_rate,
                        label,
                        True,
                    )
                    if buffered_audio is not None
                    else None,
                )
            self._shutdown_vc_stream(stream)
            self._join_vc_recording_threads(stop_thread, worker)

            self.safe_log(string("ui.recording_stopped", label=label), "info")
            self.update_resources()
            if captured_frames <= 0:
                self.safe_log(string("ui.recording_empty"), "warning")
                return False
            return True

        return self._start_vc_recording()

    def _shutdown_live_vc_recording(self) -> None:
        """Stop live VC recording immediately for application shutdown."""
        self._shutdown_persona_recording()
        if self.celune is not None:
            self.celune._exit_requested = True

        if not self._vc_recording_active():
            return

        with self._vc_recording_lock:
            (
                stream,
                _buffered_audio,
                _sample_rate,
                _label,
                submission_queue,
                _captured_frames,
                stop_thread,
                worker,
            ) = self._stop_vc_recording_stream()
            self._finish_vc_submission_queue(submission_queue)
        self._shutdown_vc_stream(stream)
        self._join_vc_recording_threads(stop_thread, worker)

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: The newly active voice name.
        """
        if self.cur_state == "exiting":
            return

        if name in self.celune_styles:
            self.style_index = self.celune_styles.index(name)

        label = name.capitalize()

        if threading.current_thread() is threading.main_thread():
            self.style_button.label = label
            self.refresh_vc_controls()
            self.update_resources()
        else:

            def update() -> None:
                self.style_button.label = label
                self.refresh_vc_controls()
                self.update_resources()

            self.call_from_thread(update)

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Handle log messages coming from Celune.

        Args:
            msg: The log message emitted by Celune.
            severity: The log severity level.
        """
        if self.cur_state == "exiting":
            return

        self.safe_log(msg, severity)

    def process_command(self, command: str, args: list[str]) -> None:
        """Process Celune control commands.

        Args:
            command: The control command to run.
            args: The command arguments to use.
        """
        process_ui_command(self, command, args)

    def consume_buffer(self, text_len: int) -> None:
        """Consume a sentence from live input and say it.

        Args:
            text_len: The number of characters to consume from the input buffer.
        """
        to_say = self.input_box.text[:text_len].strip()

        self._suppress_input_change = True
        try:
            self.input_box.load_text(self.input_box.text[text_len:])
        # yes, no except:
        # that is valid python
        finally:
            self._suppress_input_change = False

        if not to_say:
            return

        if all(char in ".!?;:, " for char in to_say):
            return

        if self.celune.config.get("ipa") is False:
            ipa_decoded, unmatched = replace_ipa(to_say, strict=True)
            if unmatched > 0:
                self.safe_log_dev(
                    f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                    "warning",
                )

            self.celune.say(ipa_decoded, display_text=to_say)
        else:
            self.celune.say(to_say)

    def _submit_text(self, text: str, process_commands: bool = True) -> bool:
        """Submit text through the same path as the input box."""
        text = text.strip()

        if not text:
            return False

        if self._is_ui_test_mode():
            self._suppress_input_change = True
            try:
                self.input_box.load_text("")
            finally:
                self._suppress_input_change = False
            self.safe_status(string("ui.test_mode_active"))
            return True

        if self.celune.cur_state == "waking":
            self._cancel_sleep_timer()
            self.safe_status(string("status.waking_up"))
            self.change_input_state(locked=True)
            return True

        if self.celune.sleeping:
            self._cancel_sleep_timer()
            self.safe_status(string("status.waking_up"))
            self._suppress_input_change = True
            try:
                self.input_box.load_text("")
            finally:
                self._suppress_input_change = False
            self.change_input_state(locked=True)
            self.wake_from_sleep()
            return True

        if process_commands and text.startswith("/"):
            try:
                parts = self.split_command_input(text[1:])
            except ValueError as e:
                self.safe_log(
                    string("ui.command_parsing_error", error=e),
                    "error",
                )
                return False

            if not parts:
                return False

            command = parts[0].lower()
            command_args = parts[1:]
            self.process_command(command, command_args)
            return True

        if persona_talkback_enabled(self.celune.config):
            handled = self.celune.think(text)
        else:
            if self.celune.config.get("ipa") is False:
                ipa_decoded, unmatched = replace_ipa(text, strict=True)
                if unmatched > 0:
                    self.safe_log_dev(
                        f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                        "warning",
                    )
                handled = self.celune.say(ipa_decoded, display_text=text)
            else:
                handled = self.celune.say(text)

        if not handled:
            return False

        self._cancel_sleep_timer()
        self.style_button.disabled = True
        self.input_box.placeholder = string("ui.wait_placeholder")
        self.input_box.load_text("")
        self.update_resources()
        return True

    def tutorial_after(self, delay: float, callback: Callable[[], None]) -> None:
        """Schedule a cancellable tutorial callback.

        Args:
            delay: Delay in seconds before running the callback.
            callback: Callback to run if the tutorial has not been canceled.
        """
        token = self._tutorial_token

        def run() -> None:
            if token != self._tutorial_token:
                return
            callback()

        if delay <= 0:
            self.call_later(run)
            return

        timer = self.set_timer(delay, run)
        self._tutorial_timers.append(timer)

    def begin_tutorial(self) -> None:
        """Start a new cancellable tutorial action sequence."""
        self.cancel_tutorial(stop_audio=True)
        self._tutorial_active = True
        self.change_input_state(locked=True)
        self.input_box.placeholder = "Currently in tutorial mode"
        self.celune.is_in_tutorial = True

    def finish_tutorial(self) -> None:
        """Mark the current tutorial sequence as complete."""
        self._tutorial_active = False
        self._tutorial_timers.clear()
        self.celune.is_in_tutorial = False
        self.change_input_state(locked=False)
        self.change_voice_lock_state(locked=len(self.celune.voices) < 2)

    def cancel_tutorial(self, stop_audio: bool = True) -> bool:
        """Cancel pending tutorial actions and any active tutorial typing.

        Args:
            stop_audio: Whether active tutorial playback should be interrupted.

        Returns:
            bool: ``True`` when tutorial work was canceled.
        """
        was_active = self._tutorial_active or bool(self._tutorial_timers)
        if not was_active:
            return False

        self._tutorial_token += 1
        self._tutorial_active = False
        self.celune.is_in_tutorial = False

        for timer in self._tutorial_timers:
            timer.stop()
        self._tutorial_timers.clear()

        if stop_audio and was_active and self.celune is not None:

            def stop_tutorial_audio() -> None:
                try:
                    asyncio.run(self.celune.force_stop_speech_async())
                except Exception as exc:
                    self.safe_log(
                        string(
                            "ui.tutorial_stop_failed",
                            error=format_error(exc, self.celune.dev),
                        ),
                        "error",
                    )

            threading.Thread(
                target=stop_tutorial_audio,
                daemon=True,
            ).start()

        self._suppress_input_change = True
        try:
            self.input_box.load_text("")
        finally:
            self._suppress_input_change = False
        self.change_input_state(locked=False)
        self.change_voice_lock_state(locked=len(self.celune.voices) < 2)

        return True

    def type_and_send(
        self,
        text: str,
        process_commands: bool = False,
        cancellable: bool = True,
    ) -> None:
        """Type text into the input box using Celune's typing animation and submit it.

        Args:
            text: The text to type into the input box.
            process_commands: Whether typed slash commands should be executed.
            cancellable: Whether tutorial cancellation should stop this typing.
        """
        token = self._tutorial_token

        def worker() -> None:
            typed = ""

            def replace_input(value: str) -> None:
                self._suppress_input_change = True
                try:
                    self.input_box.load_text(value)
                finally:
                    self._suppress_input_change = False

            self.call_from_thread(lambda: replace_input(""))

            for char in typing_animation(text):
                if cancellable and token != self._tutorial_token:
                    return
                if self.cur_state == "exiting":
                    return
                typed += char
                self.call_from_thread(lambda value=typed: replace_input(value))

            final_char = text[-1] if text else " "
            time.sleep(typing_delay(final_char))

            if self.cur_state != "exiting" and (
                not cancellable or token == self._tutorial_token
            ):
                self.call_from_thread(
                    lambda value=typed: self._submit_text(value, process_commands)
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_key(self, event: events.Key) -> None:
        """Accept input and send text to Celune.

        Args:
            event: The key event received from Textual.
        """
        with contextlib.suppress(EOFError):
            if self.cur_state == "exiting":
                return

            if event.key == "ctrl+q":
                event.prevent_default()
                event.stop()
                self._graceful_exit()
                return

            if event.key in {"ctrl+j", "ctrl+enter"} and self.cancel_tutorial():
                event.prevent_default()
                event.stop()
                return

            if event.key == "ctrl+t":
                if self.active_theme_name == "celune_april_fools":
                    event.prevent_default()
                    return

                next_theme = (
                    self.themes[1]
                    if self.active_theme_name == self.themes[0]
                    else self.themes[0]
                )
                self._apply_theme(next_theme)
                self.celune.config["theme"] = (
                    "dark" if self.theme == self.themes[0] else "light"
                )
                with open(config_path(create_parent=True), "w", encoding="utf-8") as f:
                    yaml.dump(self.celune.config, f)
                self.update_resources()

                event.prevent_default()
                return

            if event.key == "ctrl+r":
                if getattr(self.celune, "sleeping", False):
                    self._cancel_sleep_timer()
                    self.safe_status(string("status.waking_up"))
                    self.change_input_state(locked=True)
                    self.wake_from_sleep()
                    event.prevent_default()
                    event.stop()
                    return
                if getattr(self.celune, "cur_state", "") == "waking":
                    event.prevent_default()
                    event.stop()
                    return
                if self._is_voice_conversion_mode():
                    recording_toggled = self.toggle_vc_recording()
                else:
                    recording_toggled = self.toggle_persona_recording()
                if recording_toggled:
                    event.prevent_default()
                    event.stop()
                return

            if event.key == "ctrl+j" and self._submit_text(self.input_box.text):
                event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Change Celune's tone.

        Args:
            event: The button press event emitted by Textual.
        """
        if self.cur_state == "exiting":
            return

        if self.celune.is_in_tutorial:
            return

        if event.button == self.vc_mode_button:
            if self._is_voice_conversion_mode():
                self.set_vc_f0_condition(
                    not bool(getattr(self.celune, "vc_f0_condition", False))
                )
            return

        if event.button == self.vc_pitch_button:
            if self._is_voice_conversion_mode():
                current_value = int(getattr(self.celune, "vc_pitch_shift", 0))
                next_value = current_value + 1
                if next_value > VC_PITCH_SHIFT_MAX:
                    next_value = VC_PITCH_SHIFT_MIN
                self.set_vc_pitch_shift(next_value)
            return

        if event.button != self.style_button:
            return

        if len(self.celune.voices) == 0 or not self.celune_styles:
            self.safe_log(string("ui.no_voices_loaded"), "warning")
            self.change_voice_lock_state(locked=True)
            return

        if not self.celune_ready and not self.celune.backend.is_fake:
            self.safe_log(string("ui.core_engine_not_loaded"), "warning")
            self.change_voice_lock_state(locked=True)
            return

        self.style_index = (self.style_index + 1) % len(self.celune_styles)
        next_voice = self.celune_styles[self.style_index]
        threading.Thread(
            target=self.celune.set_voice,
            args=(next_voice,),
            daemon=True,
        ).start()

    def on_unmount(self) -> None:
        """Unload Celune."""
        self._shutdown_runtime()

    def _shutdown_runtime(self) -> None:
        """Release Celune and UI resources exactly once during any exit path."""
        with self._shutdown_lock:
            if self.cur_state == "exiting":
                return

            self.cur_state = "exiting"
            self._write_terminal_escape(
                f"\x1b]2;{string('osc.exiting', app_name=APP_NAME)}\x07"
            )
            self._shutdown_live_vc_recording()
            if self.celune is not None:
                self.celune.close()

            if self._runtime_log_capture_enabled:
                self._disable_runtime_log_capture()

            CeluneUI._instance = None

    def _shutdown_from_windows_signal(self) -> None:
        """Schedule graceful teardown after a Windows console-close notification."""
        try:
            self.call_from_thread(self._graceful_exit)
        except RuntimeError:
            self._shutdown_runtime()

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""
        if self.cur_state in {"exiting", "error"} or not self.celune_ready:
            if self.input_box is not None:
                self.input_box.placeholder = string("ui.wait_placeholder")
            if self.style_button is not None:
                self.style_button.disabled = True
            return
        if self.celune.cur_state in {"reloading", "waking"}:
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            if self.celune.cur_state == "waking":
                self.safe_status(string("status.waking_up"))
            return
        self.celune.locked = False
        if self.celune.sleeping:
            self.safe_status(string("status.sleeping"), "sleeping")
            return
        self.celune.cur_state = "idle"
        if self.celune.is_in_tutorial:
            self.input_box.placeholder = string("ui.tutorial_placeholder")
            self.style_button.disabled = True
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
        self.safe_status(string("status.idle"))
        self._schedule_sleep_timer()

    def tts_queue_avail(
        self,
    ) -> None:  # allow enqueuing new inputs while speaking but after generation
        """Unlock input queueing after Celune completes generation."""
        if self.cur_state in {"exiting", "error"} or not self.celune_ready:
            return
        self.celune.locked = False
        self._cancel_sleep_timer()
        self.safe_status(string("status.speaking"))
        if self.celune.is_in_tutorial:
            self.input_box.placeholder = string("ui.tutorial_placeholder")
            self.style_button.disabled = True
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(self.celune.voices) < 2)

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: The error text to display.
        """
        if self.cur_state == "exiting":
            return
        self.safe_status(error, "error")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Monitor text area changes and perform actions.

        Args:
            event: The Textual text-area change event.
        """
        if self.cur_state == "exiting":
            return

        if self._suppress_input_change:
            return

        if event.text_area.id != "input":
            return

        text = event.text_area.text
        if self.celune.sleeping and text.strip():
            self._submit_text(text, process_commands=False)
            return

        line_count = text.count("\n") + 1
        min_lines = 1
        max_lines = 8

        visible_lines = max(min_lines, min(line_count, max_lines))
        event.text_area.styles.height = visible_lines + 2

        if self.consume_on_boundary and text and text[-1] in ".!?":
            if text in ".!?":
                return
            self.consume_buffer(len(text))

    def _signal_handler(self, sig: int, frame: Optional[types.FrameType]) -> None:
        """Handle incoming signals."""
        discard(frame)

        if SIGTSTP is not None and sig == SIGTSTP:
            return

        self._graceful_exit()

    def _install_windows_signal_handler(self) -> None:
        """Install Windows console shutdown handler."""
        winfunctype = getattr(ctypes, "WINFUNCTYPE", None)
        windll = getattr(ctypes, "windll", None)

        if winfunctype is None or windll is None:
            return

        handler_type = winfunctype(ctypes.c_bool, ctypes.c_uint)
        self._windows_signal_handler = handler_type(self._signal_handler_windows)

        windll.kernel32.SetConsoleCtrlHandler(
            self._windows_signal_handler,
            True,
        )

    def _signal_handler_windows(self, sig: int) -> bool:
        """Handle incoming Windows signals."""
        if sig in (2, 5, 6):
            self._shutdown_from_windows_signal()
            return True
        return False

    def _graceful_exit(self) -> None:
        """Exit from Celune gracefully."""
        # while Python cleanup would tear down the core, we'd rather explicitly tell Celune to shut down
        # before we tell Textual to exit its main loop
        self._shutdown_runtime()
        self.exit()

    def graceful_exit(self) -> None:
        """Exit the UI through the same graceful shutdown path as internal callers."""
        self._graceful_exit()

    @property
    def tutorial_token(self) -> int:  # noqa
        """Return the active tutorial cancellation token.

        Returns:
            int: The tutorial token currently used to invalidate pending tutorial work.
        """
        return self._tutorial_token

    @property
    def tutorial_active(self) -> bool:  # noqa
        """Return whether a tutorial flow is currently active.

        Returns:
            bool: ``True`` when tutorial work is active, otherwise ``False``.
        """
        return self._tutorial_active

    @staticmethod
    def _split_command_input(text: str) -> list[str]:
        """Split one slash-command string into a command name and arguments."""
        posix = os.name != "nt"
        parts = shlex.split(text, posix=posix)
        if posix:
            return parts

        normalized: list[str] = []
        for part in parts:
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
                normalized.append(part[1:-1])
            else:
                normalized.append(part)
        return normalized

    @staticmethod
    def split_command_input(text: str) -> list[str]:
        """Split one slash-command string into a command name and arguments.

        Args:
            text: The command input to split.

        Returns:
            list[str]: The parsed command name followed by its arguments.
        """
        posix = os.name != "nt"
        parts = shlex.split(text, posix=posix)
        if posix:
            return parts

        normalized: list[str] = []
        for part in parts:
            if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
                normalized.append(part[1:-1])
            else:
                normalized.append(part)
        return normalized

    register_runtime_error_themes = _register_runtime_error_themes
    wrap_runtime_fatal_glow = _wrap_runtime_fatal_glow
    advance_status_marquee = _advance_status_marquee
    enable_runtime_log_capture = _enable_runtime_log_capture
    install_runtime_log_redirects = _install_runtime_log_redirects
    disable_runtime_log_capture = _disable_runtime_log_capture
    with_brightness = _with_brightness
    normal_input_placeholder = _normal_input_placeholder
    persona_loaded = _persona_loaded
