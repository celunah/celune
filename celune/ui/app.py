# SPDX-License-Identifier: MIT
"""Frontend layer."""

import os
import sys
import time
import types
import shlex
import signal
import ctypes
import logging
import datetime
import itertools
import threading
import contextlib
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Optional, Callable, Protocol, Union, TextIO, cast

import yaml
from rich.text import Text
from textual.color import Color
from textual.timer import Timer
from textual.theme import Theme
from textual import work, events
from textual.widget import Widget
from textual.css.types import EdgeStyle
from textual.app import ScreenStackError
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, RichLog, TextArea, Button, ProgressBar

from .. import colors
from ..celune import Celune
from ..cevoice import default_loader
from . import resources as ui_resources
from ..constants import APP_NAME, SIGTSTP
from .theme import CELUNE_CSS, severity_color
from .terminal import LogRedirect, UILogHandler
from ..paths import config_path, main_window_log_path
from .commands import process_command as process_ui_command
from ..persona.impl import (
    persona_talkback_enabled,
    persona_enabled,
)
from ..utils import (
    format_error,
    indent,
    replace_ipa,
    typing_animation,
    typing_delay,
    is_april_fools,
    supports_ansi,
    discard,
)


@dataclass
class CeluneUIWidgetState:
    """Resolved widget references owned by the UI."""

    logs: Optional[RichLog] = None
    input_box: Optional[TextArea] = None
    style_button: Optional[Button] = None
    status: Optional[Label] = None
    resources: Optional[Label] = None
    progress_bar: Optional[ProgressBar] = None
    header: Optional[Label] = None
    header_lines: tuple[Label, ...] = ()


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
    celune_styles: tuple[str, ...] = ()
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
    runtime_redirect_loggers: Optional[dict[str, logging.Logger]] = None
    runtime_redirect_handlers: Optional[dict[str, UILogHandler]] = None
    runtime_redirect_original_handlers: Optional[dict[str, list[logging.Handler]]] = (
        None
    )
    runtime_redirect_original_propagate: Optional[dict[str, bool]] = None
    warnings_capture_enabled: bool = False
    log_file_path: Path = field(default_factory=Path)
    log_file_initialized: bool = False


@dataclass
class CeluneUIInteractionState:
    """Transient UI effects, sleep scheduling, and tutorial state."""

    border_pulse_tokens: dict[int, int] = field(default_factory=dict)
    border_pulse_widgets: dict[int, Widget] = field(default_factory=dict)
    tutorial_timers: list[Timer] = field(default_factory=list)
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

        CeluneUI._instance = self

    logs = _forward_ui_property("_widgets", "logs")
    input_box = _forward_ui_property("_widgets", "input_box")
    style_button = _forward_ui_property("_widgets", "style_button")
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
    _runtime_redirect_loggers = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_loggers"
    )
    _runtime_redirect_handlers = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_handlers"
    )
    _runtime_redirect_original_handlers = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_handlers"
    )
    _runtime_redirect_original_propagate = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_propagate"
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

        class _RefreshableWidget(Protocol):
            def refresh(self, *args, **kwargs) -> object:
                """Refresh one widget in place."""

        def repaint(widget: _RefreshableWidget) -> None:
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
        setattr(self.celune, "_ui_fatal_glow_wrapped", True)

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
        try:
            if not self._log_file_initialized:
                self._log_file_path.write_text("", encoding="utf-8")
                self._log_file_initialized = True

            timestamp = datetime.datetime.now().isoformat(timespec="seconds")
            with self._log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] [{severity.upper()}] {msg}\n")
        except OSError:
            pass

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
                yield TextArea(id="input", placeholder="Please wait")
                yield Button("No Voice Set", id="style", disabled=True)
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
                self.safe_log("Invalid theme, defaulting to dark", "warning")

        self.theme = self.active_theme_name

        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.resources = self.query_one("#resources", Label)
        self.style_button = self.query_one("#style", Button)
        self.progress_bar = self.query_one("#progress", ProgressBar)
        self.header = self.query_one("#header", Label)
        self.header_lines = tuple(cast(Label, widget) for widget in self.query(".line"))
        self.set_focus(None)
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()
        self._install_runtime_log_redirects()
        ui_resources.prime_usage()
        self.set_interval(2.06, self.advance_resources)
        self._status_marquee_timer = self.set_interval(
            0.18, self._advance_status_marquee
        )

        self.call_after_refresh(self.start_background_init)
        self.safe_status("Initializing")
        self.update_resources()

    def update_resources(self) -> None:
        """Refresh the currently selected resource footer page."""
        if self.cur_state == "exiting" or self.resources is None:
            return

        def update() -> None:
            pages = ui_resources.resource_pages(self.celune, self.active_theme_name)
            text = pages[self._resource_page % len(pages)]

            if supports_ansi() and self.celune.cur_state == "error":
                if self._resource_page % 2:
                    self._old_stdout.write(f"\x1b]2;{APP_NAME}\x07")
                else:
                    self._old_stdout.write("\x1b]2;has crashed\x07")

            self.resources.update(indent(text, spaces=2, direction="right"))

        self._run_on_ui_thread(update)

    def _enable_runtime_log_capture(self) -> None:
        """Capture Celune runtime output after the Textual app has started cleanly."""
        if self._runtime_log_capture_enabled:
            return

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = LogRedirect(
            write_callback=self.safe_log,
            default_severity="info",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages={"`torch_dtype` is deprecated! Use `dtype` instead!"},
        )
        self._log_stderr = LogRedirect(
            write_callback=self.safe_log,
            default_severity="warning",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages={"`torch_dtype` is deprecated! Use `dtype` instead!"},
        )

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr
        self._install_runtime_log_redirects()
        self._runtime_log_capture_enabled = True

    def _install_runtime_log_redirects(self) -> None:
        """Route known runtime logger output into Celune's UI log widget."""
        if self._runtime_redirect_loggers is not None:
            return

        self._runtime_redirect_loggers = {}
        self._runtime_redirect_handlers = {}
        self._runtime_redirect_original_handlers = {}
        self._runtime_redirect_original_propagate = {}

        for logger_name in (
            "torch.utils.flop_counter",
            "py.warnings",
            "huggingface_hub",
            "transformers",
        ):
            logger = logging.getLogger(logger_name)
            handler = UILogHandler(self.safe_log)
            self._runtime_redirect_loggers[logger_name] = logger
            self._runtime_redirect_handlers[logger_name] = handler
            self._runtime_redirect_original_handlers[logger_name] = list(
                logger.handlers
            )
            self._runtime_redirect_original_propagate[logger_name] = logger.propagate
            logger.handlers = [handler]
            logger.propagate = False

        logging.captureWarnings(True)
        self._warnings_capture_enabled = True

    def _remove_runtime_log_redirects(self) -> None:
        """Restore Python logger output handlers replaced by the UI."""
        loggers = self._runtime_redirect_loggers
        handlers = self._runtime_redirect_handlers
        original_handlers = self._runtime_redirect_original_handlers
        original_propagate = self._runtime_redirect_original_propagate
        if (
            loggers is not None
            and handlers is not None
            and original_handlers is not None
            and original_propagate is not None
        ):
            for logger_name, logger in loggers.items():
                logger.handlers = original_handlers[logger_name]
                logger.propagate = original_propagate[logger_name]
                handlers[logger_name].close()

        if self._warnings_capture_enabled:
            logging.captureWarnings(False)
            self._warnings_capture_enabled = False

        self._runtime_redirect_loggers = None
        self._runtime_redirect_handlers = None
        self._runtime_redirect_original_handlers = None
        self._runtime_redirect_original_propagate = None

    def _disable_runtime_log_capture(self) -> None:
        """Restore global stdio once the UI is shutting down."""
        if self._log_stdout is not None:
            self._log_stdout.flush()
        if self._log_stderr is not None:
            self._log_stderr.flush()

        self._remove_runtime_log_redirects()

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

        if self.celune.enter_sleep_mode():
            self.safe_log(
                f"{APP_NAME} is currently sleeping. Type anything to wake up.",
                "sleeping",
            )
            self.safe_status("Sleeping", "sleeping")
            self.change_voice_lock_state(locked=True)

    @work(thread=True, exclusive=True)
    def wake_from_sleep(self) -> None:
        """Wake the app after the user types into the sleeping UI."""
        try:
            if self.celune.wake_from_sleep():
                self._schedule_sleep_timer()
        finally:
            if self.celune.sleeping:
                self.safe_status("Sleeping", "sleeping")

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
                    self.change_input_state(locked=True)
                    self.change_voice_lock_state(locked=True)
                    self.error(f"{APP_NAME} could not start")
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
                self.safe_status("Idle")
                self.tts_voice_changed(
                    self.celune.current_voice or self.celune.voices[0]
                )
                if not self.celune.use_normalization:
                    self.safe_progress(1, 1)
                self.change_input_state(locked=False)
                self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
                self.call_from_thread(self._enable_runtime_log_capture)
                self.safe_log(
                    f"New to {APP_NAME}? Type /tutorial to begin the tutorial."
                )
                self._schedule_sleep_timer()
            else:
                self.cur_state = "error"
                self.change_input_state(locked=True)
                self.change_voice_lock_state(locked=True)
                self.error(f"{APP_NAME} could not start")
        except Exception as e:
            self.cur_state = "error"
            self.safe_log(f"[INIT ERROR] {format_error(e, self.celune.dev)}", "error")
            self.celune.glow.fatal()
            if not self.celune.try_play_signal("error"):
                self.safe_log_dev("Could not play the error signal.", "warning")
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            self.error(f"{APP_NAME} could not start")

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
            self.call_from_thread(self.pulse_border, target)
            return

        duration = 2.06
        steps = 10

        widget = self.query_one(target) if isinstance(target, str) else target
        original_border: tuple[EdgeStyle, ...] = tuple(widget.styles.border)

        if not any(edge_type for edge_type, _ in original_border):
            return

        widget_key = id(widget)
        token = self._border_pulse_tokens.get(widget_key, 0) + 1
        self._border_pulse_tokens[widget_key] = token
        self._border_pulse_widgets[widget_key] = widget

        target_border: tuple[EdgeStyle, ...] = tuple(
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

        def set_border(border: tuple[EdgeStyle, ...]) -> None:
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
        if (
            self._persona_loaded()
            and self._persona_available
            and persona_enabled(self.celune.config)
            and persona_talkback_enabled(self.celune.config)
        ):
            return "Say something..."

        return "Enter text to speak here"

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
                "Please wait" if locked else self._normal_input_placeholder()
            )
            self.style_button.disabled = locked
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
            self.call_from_thread(self.logs.write, entry)

    def safe_log_dev(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The log line to append.
            severity: The log severity level.
        """
        if self.celune.dev:
            self.safe_log(msg, severity)

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
            self.update_resources()
        else:

            def update() -> None:
                self.style_button.label = label
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

        if self.celune.cur_state == "waking":
            self._cancel_sleep_timer()
            self.safe_status("Waking up")
            self.change_input_state(locked=True)
            return True

        if self.celune.sleeping:
            self._cancel_sleep_timer()
            self.safe_status("Waking up")
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
                self.safe_log(f"Command parsing error: {e}", "error")
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
        self.input_box.placeholder = "Please wait"
        self.input_box.load_text("")
        self.update_resources()
        return True

    def tutorial_after(self, delay: float, callback: Callable[[], None]) -> None:
        """Schedule a cancellable tutorial callback.

        Args:
            delay: Delay in seconds before running the callback.
            callback: Callback to run if the tutorial has not been cancelled.
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
            bool: ``True`` when tutorial work was cancelled.
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
            self.celune.force_stop_speech()

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

            self.call_from_thread(replace_input, "")

            for char in typing_animation(text):
                if cancellable and token != self._tutorial_token:
                    return
                if self.cur_state == "exiting":
                    return
                typed += char
                self.call_from_thread(replace_input, typed)

            final_char = text[-1] if text else " "
            time.sleep(typing_delay(final_char))

            if self.cur_state != "exiting" and (
                not cancellable or token == self._tutorial_token
            ):
                self.call_from_thread(self._submit_text, typed, process_commands)

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

            if event.key in {"ctrl+j", "ctrl+enter"}:
                if self.cancel_tutorial():
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

            if event.key == "ctrl+j":
                if self._submit_text(self.input_box.text):
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

        if event.button != self.style_button:
            return

        if len(self.celune.voices) == 0 or not self.celune_styles:
            self.safe_log("No voices are loaded.", "warning")
            self.change_voice_lock_state(locked=True)
            return

        if not self.celune_ready:
            self.safe_log("Core engine is not loaded.", "warning")
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
        if self.celune is not None:
            self.celune.close()

        self.cur_state = "exiting"
        if self._runtime_log_capture_enabled:
            self._disable_runtime_log_capture()

        CeluneUI._instance = None

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""
        if self.cur_state in {"exiting", "error"} or not self.celune_ready:
            if self.input_box is not None:
                self.input_box.placeholder = "Please wait"
            if self.style_button is not None:
                self.style_button.disabled = True
            return
        if self.celune.cur_state in {"reloading", "waking"}:
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            if self.celune.cur_state == "waking":
                self.safe_status("Waking up")
            return
        self.celune.locked = False
        if self.celune.sleeping:
            self.safe_status("Sleeping", "sleeping")
            return
        self.celune.cur_state = "idle"
        if self.celune.is_in_tutorial:
            self.input_box.placeholder = "Currently in tutorial mode"
            self.style_button.disabled = True
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
        self.safe_status("Idle")
        self._schedule_sleep_timer()

    def tts_queue_avail(
        self,
    ) -> None:  # allow enqueuing new inputs while speaking but after generation
        """Unlock input queueing after Celune completes generation."""
        if self.cur_state in {"exiting", "error"} or not self.celune_ready:
            return
        self.celune.locked = False
        self._cancel_sleep_timer()
        self.safe_status("Speaking")
        if self.celune.is_in_tutorial:
            self.input_box.placeholder = "Currently in tutorial mode"
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

        if self.consume_on_boundary:
            if text and text[-1] in ".!?":
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
            self._graceful_exit()
            return True
        return False

    def _graceful_exit(self) -> None:
        """Exit from Celune gracefully."""
        # while Python cleanup would tear down the core, we'd rather explicitly tell Celune to shut down
        # before we tell Textual to exit its main loop
        self.celune.close()
        self.exit()

    def graceful_exit(self) -> None:
        """Exit the UI through the same graceful shutdown path as internal callers."""
        self._graceful_exit()

    @property
    def tutorial_token(self) -> int:
        """Return the active tutorial cancellation token.

        Returns:
            int: The tutorial token currently used to invalidate pending tutorial work.
        """
        return self._tutorial_token

    @property
    def tutorial_active(self) -> bool:
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
