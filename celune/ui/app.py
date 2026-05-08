"""Celune's frontend layer."""

import datetime
import os
import sys
import shlex
import signal
import threading
import itertools
import contextlib
from types import FrameType
from typing import cast, Optional
from collections.abc import Iterator

import yaml
from textual import work, events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, RichLog, TextArea, Button
from rich.text import Text

from ..celune import Celune
from ..utils import format_error, indent, replace_ipa
from ..constants import SIGTSTP
from ..colors import THEME, THEME_LIGHT, THEME_APRIL_FOOLS, SEVERITY_COLORS
from .commands import process_command as process_ui_command
from . import resources as ui_resources
from .terminal import LogRedirect
from .theme import CELUNE_CSS, severity_color


class CeluneUI(App):
    """Celune's user interface."""

    ENABLE_COMMAND_PALETTE = False

    CSS = CELUNE_CSS

    def __init__(self) -> None:
        """Initialize UI state.

        Returns:
            None: This constructor prepares widget references and runtime flags.
        """
        super().__init__()

        self.logs = cast(RichLog, None)
        self.input_box = cast(TextArea, None)
        self.style_button = cast(Button, None)
        self.status = cast(Label, None)
        self.resources = cast(Label, None)

        if self._is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
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
        self.celune_styles: tuple[str, ...] = ("balanced", "calm", "bold", "upbeat")
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

    @staticmethod
    def _is_april_fools() -> bool:
        """Return whether the UI should use the April Fools theme."""
        now = datetime.datetime.now()
        return now.month == 4 and now.day == 1

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
        self.active_theme_name = theme_name
        self.theme = theme_name
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()

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
            with Horizontal(id="controls"):
                yield TextArea(id="input", placeholder="Please wait")
                yield Button(
                    self.celune_styles[0].capitalize(), id="style", disabled=True
                )
            with Horizontal(id="bottom"):
                yield Label("", id="status")
                yield Label("", id="resources")

    def on_mount(self) -> None:
        """Prepare Celune.

        Returns:
            None: This method sets up themes, widgets, output redirection, and
                background initialization.
        """
        self.register_theme(THEME)
        self.register_theme(THEME_LIGHT)
        self.register_theme(THEME_APRIL_FOOLS)

        if self._is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
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
        self._refresh_status()
        self._refresh_theme_text()
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
            pages = ui_resources.resource_pages(self.celune)
            text = pages[self._resource_page % len(pages)]
            self.resources.update(indent(text, spaces=2, direction="right"))

        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.call_from_thread(update)

    def advance_resources(self) -> None:
        """Advance the resource footer to the next page and refresh it.

        Returns:
            None: This method rotates the footer page index.
        """
        if self.cur_state == "exiting" or self.resources is None:
            return

        self._resource_page = (self._resource_page + 1) % len(
            ui_resources.resource_pages(self.celune)
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
            tts_voices: tuple[str, ...] = tuple(self.celune.backend.voices)

            self.celune.set_voices(tts_voices)
            self.celune_styles = tts_voices or ("balanced", "calm", "bold", "upbeat")
            self.celune_voices = itertools.cycle(tts_voices)
            if self.celune.current_voice in self.celune_styles:
                self.style_index = self.celune_styles.index(self.celune.current_voice)
            else:
                self.style_index = 0

            if self.celune.load():
                self.celune_ready = True
                self.safe_status("Idle")
                self.tts_voice_changed(
                    self.celune.current_voice or self.celune_styles[0]
                )
                self.change_input_state(locked=False)

        except Exception as e:
            self.safe_log(f"[INIT ERROR] {format_error(e, self.celune.dev)}", "error")
            self.error("Celune could not start")
            self.cur_state = "error"

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

        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.call_from_thread(update)

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

        if severity not in SEVERITY_COLORS["celune"]:
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

        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.call_from_thread(update)

    def safe_log(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The log line to append.
            severity: The log severity level.

        Returns:
            None: This method safely updates the log widget from any thread.
        """
        if self.cur_state == "exiting" or self.logs is None:
            return

        if severity not in SEVERITY_COLORS["celune"]:
            severity = "info"

        self.log_history.append((msg, severity))
        entry = Text(msg, style=self._severity_color(severity))

        if threading.current_thread() is threading.main_thread():
            self.logs.write(entry)
        else:
            self.call_from_thread(self.logs.write, entry)

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
        """Process Celune control commands."""
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
        if self.celune.dev:
            self.safe_log(
                f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                "warning",
            )

        self.celune.say(ipa_decoded, display_text=to_say)

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
                text = self.input_box.text.strip()

                if not text:
                    return

                if text.startswith("/"):
                    try:
                        parts = shlex.split(text[1:])
                    except ValueError as e:
                        self.safe_log(f"Command parsing error: {e}", "error")
                        return

                    if not parts:
                        return

                    command = parts[0].lower()
                    command_args = parts[1:]

                    self.process_command(command, command_args)
                    return

                ipa_decoded, unmatched = replace_ipa(text, strict=True)
                if self.celune.dev:
                    self.safe_log(
                        f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                        "warning",
                    )

                if self.celune.say(ipa_decoded, display_text=text):
                    self.style_button.disabled = True
                    self.input_box.placeholder = "Please wait"
                    self.input_box.load_text("")
                    self.update_resources()
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

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking.

        Returns:
            None: This method restores idle UI state when playback finishes.
        """
        if self.cur_state == "exiting":
            return
        self.celune.locked = False
        self.celune.cur_state = "idle"
        self.change_input_state(locked=False)
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
        self.change_input_state(locked=False)

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

        if threading.current_thread() is threading.main_thread():
            self.call_later(self._graceful_exit)
        else:
            self.call_from_thread(self._graceful_exit)
