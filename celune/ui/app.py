# SPDX-License-Identifier: MIT
"""Celune's frontend layer."""

import os
import sys
import time
import shlex
import signal
import threading
import itertools
import contextlib
from types import FrameType
from collections.abc import Iterator
from typing import cast, Optional, Callable, Union

import yaml
from textual.timer import Timer
from textual.color import Color
from textual import work, events
from textual.widget import Widget
from textual.css.types import EdgeStyle
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, RichLog, TextArea, Button, ProgressBar
from rich.text import Text

from ..celune import Celune
from ..utils import (
    format_error,
    indent,
    replace_ipa,
    typing_animation,
    typing_delay,
    is_april_fools,
)
from ..constants import SIGTSTP
from ..cevoice import default_loader
from .. import colors
from .commands import process_command as process_ui_command
from . import resources as ui_resources
from .terminal import LogRedirect
from .theme import CELUNE_CSS, severity_color


class CeluneUI(App):
    """Celune's user interface."""

    ENABLE_COMMAND_PALETTE = False
    CSS = CELUNE_CSS
    _instance: Optional["CeluneUI"] = None

    def __init__(self) -> None:
        super().__init__()

        if CeluneUI._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")

        self.logs = cast(RichLog, None)
        self.input_box = cast(TextArea, None)
        self.style_button = cast(Button, None)
        self.status = cast(Label, None)
        self.resources = cast(Label, None)
        self.progress_bar = cast(ProgressBar, None)

        if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }:
            self.themes = ("celune_april_fools", "celune_april_fools")
            self.active_theme_name = "celune_april_fools"
        else:
            self.themes = ("celune", "celune_light")
            self.active_theme_name = "celune"
        self.log_history: list[tuple[str, str]] = []
        self.status_severity = "info"

        self.celune = cast(Celune, None)
        self.celune_ready = False
        self.celune_styles: tuple[str, ...] = ()
        self.celune_voices: Iterator[str] = itertools.cycle(self.celune_styles)

        self.style_index = 0

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        self._log_stdout = cast(LogRedirect, None)
        self._log_stderr = cast(LogRedirect, None)

        self.cur_state = "active"

        self.consume_on_boundary = False
        self._suppress_input_change = False
        self._resource_page = 0
        self._border_pulse_tokens: dict[int, int] = {}
        self._border_pulse_widgets: dict[int, Widget] = {}
        self._tutorial_timers: list[Timer] = []
        self._tutorial_token = 0
        self._tutorial_active = False

        CeluneUI._instance = self

    def _run_on_ui_thread(self, callback: Callable[[], None]) -> None:
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            self.call_from_thread(callback)

    def _severity_color(self, severity: str = "info") -> str:
        """Return the current theme color for a log severity.

        Args:
            severity: The severity label to map to a theme color.

        Returns:
            str: The configured color string for the requested severity.
        """
        return severity_color(self.active_theme_name, severity)

    def _apply_theme(self, theme_name: str) -> None:
        """Apply theme and repaint theme-sensitive widgets.

        Args:
            theme_name: The theme name to activate.

        Returns:
            None: This method updates theme state and redraws status and logs.
        """
        self._clear_border_pulses()
        self.active_theme_name = theme_name
        self.theme = theme_name
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()

    def _has_celune(self) -> bool:
        """Is Celune attached to this UI instance?"""
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
        """Refresh widgets that use the active theme's normal text color.

        Returns:
            None: This method reapplies the active theme color to widgets.
        """
        color = self._severity_color("info")
        if self.logs is not None:
            self.logs.styles.color = color
        if self.resources is not None:
            self.resources.styles.color = color

    def _refresh_status(self) -> None:
        """Refresh the status color for the active theme.

        Returns:
            None: This method reapplies the active severity color to the status
                widget.
        """
        if self.status is None:
            return
        self.status.styles.color = self._severity_color(self.status_severity)

    def _refresh_logs(self) -> None:
        """Repaint existing log entries using the active theme colors.

        Returns:
            None: This method redraws the log widget while preserving scroll
                position.
        """
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

    def compose(self) -> ComposeResult:
        """Define the UI.

        Returns:
            ComposeResult: The root widget tree for the Celune interface.
        """
        with Vertical(id="container"):
            with Horizontal(id="header-container"):
                yield Label("", classes="line")
                yield Label("Celune", id="header")
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
        """Prepare Celune.

        Returns:
            None: This method sets up themes, widgets, output redirection, and
                background initialization.
        """
        if not self._has_celune():
            raise RuntimeError(
                f"{self.__class__.__name__} requires an instance of Celune to be set"
            )

        loader = default_loader()
        if loader is not None:
            theme = loader.bundle.metadata.get("theme")
            if isinstance(theme, dict):
                background = theme.get("background")
                accent = theme.get("accent")
                if isinstance(background, str) and isinstance(accent, str):
                    colors.configure_theme(background, accent)

        self.register_theme(colors.THEME)
        self.register_theme(colors.THEME_LIGHT)
        self.register_theme(colors.THEME_APRIL_FOOLS)

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
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()
        ui_resources.prime_usage()
        self.set_interval(2.06, self.advance_resources)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = LogRedirect(
            write_callback=self.safe_log,
            default_severity="info",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
        )
        self._log_stderr = LogRedirect(
            write_callback=self.safe_log,
            default_severity="warning",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
        )

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr

        self.call_after_refresh(self.start_background_init)
        signal.signal(signal.SIGINT, self.signal_handler)
        if SIGTSTP is not None:
            signal.signal(SIGTSTP, self.signal_handler)
        self.safe_status("Initializing")
        self.update_resources()

    def update_resources(self) -> None:
        """Refresh the currently selected resource footer page.

        Returns:
            None: This method updates the footer widget when available.
        """
        if self.cur_state == "exiting" or self.resources is None:
            return

        def update() -> None:
            """Update the resource widget on the UI thread.

            Returns:
                None: This callback writes the selected resource page.
            """
            pages = ui_resources.resource_pages(self.celune, self.active_theme_name)
            text = pages[self._resource_page % len(pages)]
            self.resources.update(indent(text, spaces=2, direction="right"))

        self._run_on_ui_thread(update)

    def advance_resources(self) -> None:
        """Advance the resource footer to the next page and refresh it.

        Returns:
            None: This method rotates the footer page index.
        """
        if self.cur_state == "exiting" or self.resources is None:
            return

        self._resource_page = (self._resource_page + 1) % len(
            ui_resources.resource_pages(self.celune, self.active_theme_name)
        )
        self.update_resources()

    def start_background_init(self) -> None:
        """Run Celune's initialization function.

        Returns:
            None: This method triggers background TTS loading.
        """
        self.load_tts()

    @work(thread=True, exclusive=True)
    def load_tts(self) -> None:
        """Load Celune.

        Returns:
            None: This worker initializes the engine and updates UI state.
        """
        try:
            if self.celune.load():
                self.celune_styles = self.celune.voices
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
                    self.celune.current_voice or self.celune_styles[0]
                )
                if not self.celune.use_normalization:
                    self.safe_progress(1, 1)
                self.change_input_state(locked=False)
                self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
                self.safe_log("New to Celune? Type /tutorial to begin the tutorial.")

        except Exception as e:
            self.safe_log(f"[INIT ERROR] {format_error(e, self.celune.dev)}", "error")
            self.error("Celune could not start")
            self.cur_state = "error"

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Current progress, or ``None`` for an indeterminate bar.
            total: Total progress, or ``None`` for an indeterminate bar.

        Returns:
            None: This method safely updates progress from any thread.
        """
        if self.cur_state == "exiting" or self.progress_bar is None:
            return

        def update() -> None:
            """Apply a progress update on the UI thread."""
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
        return CeluneUI._with_brightness(color, target_brightness)

    def pulse_border(self, target: Union[str, Widget]) -> None:
        """Softly pulse a widget border darker and back.

        Args:
            target: Widget or Textual selector for the target widget.

        Returns:
            None: This method schedules a border color pulse.
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
            """Set each border edge directly and refresh the widget."""
            (
                widget.styles.border_top,
                widget.styles.border_right,
                widget.styles.border_bottom,
                widget.styles.border_left,
            ) = border
            widget.refresh(layout=False)

        def apply_blend(progress: float) -> None:
            """Apply one transition frame if this pulse is still current."""
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
            """Return border styling to CSS/theme ownership."""
            if self._border_pulse_tokens.get(widget_key) != token:
                return
            widget.styles.border = None
            widget.refresh(layout=False)
            self._border_pulse_tokens.pop(widget_key, None)
            self._border_pulse_widgets.pop(widget_key, None)

        def schedule_frame(index: int, delay: float) -> None:
            """Schedule the next pulse frame without filling the timer queue."""
            if self._border_pulse_tokens.get(widget_key) != token:
                return

            if index >= steps * 2:
                restore()
                return

            def run_frame() -> None:
                """Apply one frame and then queue the next one."""
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

        Returns:
            None: This method locks or unlocks the voice change button.
        """

        def update() -> None:
            """Apply the voice change lock state on the UI thread.

            Returns:
                None: This callback updates input widgets and resources
            """
            self.style_button.disabled = locked
            self.update_resources()

        self._run_on_ui_thread(update)

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: Whether user input should be disabled.

        Returns:
            None: This method updates the input placeholder and style button
                state.
        """

        def update() -> None:
            """Apply the input lock state on the UI thread.

            Returns:
                None: This callback updates input widgets and resources.
            """
            self.input_box.placeholder = (
                "Please wait" if locked else "Enter text to speak here"
            )
            self.style_button.disabled = locked
            self.update_resources()

        self._run_on_ui_thread(update)

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: The status text to display.
            severity: The status severity level.

        Returns:
            None: This method safely updates the status widget from any thread.
        """
        if self.cur_state == "exiting" or self.status is None:
            return

        if severity not in colors.SEVERITY_COLORS["celune"]:
            self.safe_log(
                f"[WARNING] Unknown severity '{severity}', defaulting to info",
                "warning",
            )
            severity = "info"

        self.status_severity = severity

        def update() -> None:
            """Apply a status update on the UI thread.

            Returns:
                None: This callback updates status text and color.
            """
            self.status.update(indent(msg, spaces=2))
            self._refresh_status()
            self.update_resources()

        self._run_on_ui_thread(update)

    def safe_log(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The log line to append.
            severity: The log severity level.

        Returns:
            None: This method safely updates the log widget from any thread.
        """
        if self.cur_state == "exiting":
            return

        if severity not in colors.SEVERITY_COLORS["celune"]:
            severity = "info"

        self.log_history.append((msg, severity))
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

        Returns:
            None: This method safely updates the log widget from any thread.
        """
        if self.celune.dev:
            self.safe_log(msg, severity)

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: The newly active voice name.

        Returns:
            None: This method synchronizes the style button label with Celune.
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
                """Apply a voice label update on the UI thread.

                Returns:
                    None: This callback updates the style button label.
                """
                self.style_button.label = label
                self.update_resources()

            self.call_from_thread(update)

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Handle log messages coming from Celune.

        Args:
            msg: The log message emitted by Celune.
            severity: The log severity level.

        Returns:
            None: This method forwards engine logs to the UI log panel.
        """
        if self.cur_state == "exiting":
            return

        self.safe_log(msg, severity)

    def process_command(self, command: str, args: list[str]) -> None:
        """Process Celune control commands.

        Args:
            command: The control command to run.
            args: The command arguments to use.

        Returns:
            None: This method asks Celune to process a control command.
        """
        process_ui_command(self, command, args)

    def consume_buffer(self, text_len: int) -> None:
        """Consume a sentence from live input and say it.

        Args:
            text_len: The number of characters to consume from the input buffer.

        Returns:
            None: This method removes the consumed text and queues it for speech.
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

        ipa_decoded, unmatched = replace_ipa(to_say, strict=True)
        if unmatched > 0:
            self.safe_log_dev(
                f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                "warning",
            )

        self.celune.say(ipa_decoded, display_text=to_say)

    def _submit_text(self, text: str, process_commands: bool = True) -> bool:
        """Submit text through the same path as the input box.

        Args:
            text: The text to submit.
            process_commands: Whether slash commands should be executed.

        Returns:
            bool: ``True`` when the text was handled, otherwise ``False``.
        """
        text = text.strip()

        if not text:
            return False

        if process_commands and text.startswith("/"):
            try:
                parts = shlex.split(text[1:])
            except ValueError as e:
                self.safe_log(f"Command parsing error: {e}", "error")
                return False

            if not parts:
                return False

            command = parts[0].lower()
            command_args = parts[1:]
            self.process_command(command, command_args)
            return True

        ipa_decoded, unmatched = replace_ipa(text, strict=True)
        if unmatched > 0:
            self.safe_log_dev(
                f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                "warning",
            )

        if not self.celune.say(ipa_decoded, display_text=text):
            return False

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

        Returns:
            None: This method stores a cancellable Textual timer.
        """
        token = self._tutorial_token

        def run() -> None:
            """Run a scheduled tutorial callback if still current."""
            if token != self._tutorial_token:
                return
            callback()

        if delay <= 0:
            self.call_later(run)
            return

        timer = self.set_timer(delay, run)
        self._tutorial_timers.append(timer)

    def begin_tutorial(self) -> None:
        """Start a new cancellable tutorial action sequence.

        Returns:
            None: This method sets Celune's tutorial flags and begins tutorial actions.
        """
        self.cancel_tutorial(stop_audio=True)
        self._tutorial_active = True
        self.change_input_state(locked=True)
        self.input_box.placeholder = "Currently in tutorial mode"
        self.celune.is_in_tutorial = True

    def finish_tutorial(self) -> None:
        """Mark the current tutorial sequence as complete.

        Returns:
            None: This method unsets Celune's tutorial flags after a completed or cancelled tutorial.
        """
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

        Returns:
            None: This method starts a background typing helper.
        """
        token = self._tutorial_token

        def worker() -> None:
            """Animate the text input without blocking Textual."""
            typed = ""

            def replace_input(value: str) -> None:
                """Replace input text without triggering live consumption."""
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

        Returns:
            None: This handler processes shortcuts, commands, and speech requests.
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
                with open("config.yaml", "w", encoding="utf-8") as f:
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

        Returns:
            None: This handler cycles to the next available voice.
        """
        if self.cur_state == "exiting":
            return

        if self.celune.is_in_tutorial:
            return

        if event.button != self.style_button:
            return

        self.style_index = (self.style_index + 1) % len(self.celune_styles)
        next_voice = self.celune_styles[self.style_index]
        threading.Thread(
            target=self.celune.set_voice,
            args=(next_voice,),
            daemon=True,
        ).start()

    def on_unmount(self) -> None:
        """Unload Celune.

        Returns:
            None: This handler shuts down Celune and restores redirected output.
        """
        if self.celune is not None:
            self.celune.close()

        if self._log_stdout is not None:
            self._log_stdout.flush()
        if self._log_stderr is not None:
            self._log_stderr.flush()

        self.cur_state = "exiting"

        if hasattr(self, "_old_stdout"):
            sys.stdout = self._old_stdout
        if hasattr(self, "_old_stderr"):
            sys.stderr = self._old_stderr

        CeluneUI._instance = None

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking.

        Returns:
            None: This method restores idle UI state when playback finishes.
        """
        if self.cur_state == "exiting":
            return
        self.celune.locked = False
        self.celune.cur_state = "idle"
        if self.celune.is_in_tutorial:
            self.input_box.placeholder = "Currently in tutorial mode"
            self.style_button.disabled = True
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
        self.safe_status("Idle")

    def tts_queue_avail(
        self,
    ) -> None:  # allow enqueuing new inputs while speaking but after generation
        """Unlock input queueing after Celune completes generation.

        Returns:
            None: This method re-enables input while Celune is still speaking.
        """
        if self.cur_state == "exiting":
            return
        self.celune.locked = False
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

        Returns:
            None: This method shows the message with error severity.
        """
        if self.cur_state == "exiting":
            return
        self.safe_status(error, "error")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Monitor text area changes and perform actions.

        Args:
            event: The Textual text-area change event.

        Returns:
            None: This handler resizes the input and optionally consumes
                sentence-boundary text.
        """
        if self.cur_state == "exiting":
            return

        if self._suppress_input_change:
            return

        if event.text_area.id != "input":
            return

        text = event.text_area.text
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

    def _graceful_exit(self) -> None:
        """Exit from Celune gracefully.

        Returns:
            None: This method requests a normal Textual application exit.
        """
        self.exit()

    def graceful_exit(self) -> None:
        """Public interface for CeluneUI._graceful_exit().

        Returns:
            None: This method returns the same value as CeluneUI._graceful_exit().
        """
        self._graceful_exit()

    def signal_handler(self, sig: int, _frame: Optional[FrameType]) -> None:
        """Trap CTRL+C and exit Celune if pressed, while ignoring CTRL+Z.

        Args:
            sig: The received signal number.
            _frame: The current stack frame from the signal handler.

        Returns:
            None: This handler schedules a graceful application shutdown.
        """
        if SIGTSTP is not None and sig == SIGTSTP:
            return

        self._run_on_ui_thread(self._graceful_exit)

    @property
    def tutorial_token(self) -> int:
        """Property for accessing the tutorial token held by Celune.

        Returns:
            int: The tutorial token currently in use by Celune.
        """
        return self._tutorial_token

    @property
    def tutorial_active(self) -> bool:
        """Property for accessing whether the tutorial is active or not.

        Returns:
            bool: Celune's current tutorial flag.
        """
        return self._tutorial_active
