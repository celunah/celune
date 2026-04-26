# pylint: disable=C0114, C0302, R0912, W0718, R0911, R0902, R0904, R0915
"""Celune's frontend layer."""

import os
import sys
import time
import shlex
import signal
import shutil
import datetime
import threading
import itertools
import subprocess
from typing import Optional, Callable

import yaml
import torch
import psutil
import readchar
from textual import work, events
from textual.app import App, ComposeResult
from textual.theme import Theme
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, RichLog, TextArea, Button
from rich.text import Text

from .celune import Celune
from .config import config_bool
from .utils import format_number, celune_day_status, lunar_phase, lunar_info
from .exceptions import InvalidExtensionError

SEVERITY_COLORS = {
    "celune": {
        "info": "#cebaff",
        "warning": "#f0e68c",
        "error": "#f07178",
    },
    "celune_light": {
        "info": "#33293f",
        "warning": "#6b5e00",
        "error": "#7a1f24",
    },
}

# Celune theme
THEME = Theme(
    name="celune",
    primary="#cebaff",  # Celune primary
    secondary="#a595cc",  # Celune secondary
    accent="#7c7099",  # Celune tertiary
    foreground="#e2ceff",  # Celune highlight
    background="#1d1826",  # Celune background
    surface="#1d1826",  # same as background
    warning="#f0e68c",  # Celune warning
    error="#f07178",  # Celune error
    dark=True,
)

THEME_LIGHT = Theme(
    name="celune_light",
    primary="#33293f",  # Celune light primary
    secondary="#281732",  # Celune light secondary
    accent="#1e1126",  # Celune light tertiary
    foreground="#473d53",  # Celune highlight
    background="#ece8ff",  # Celune light background
    surface="#ece8ff",  # same as background
    warning="#f0e68c",  # Celune warning
    error="#f07178",  # Celune error
    dark=False,
)


class CeluneUI(App):
    """Celune's user interface."""

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #logs {
        height: 1fr;
        border: round $primary;
        color: $primary;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1;
    }

    /* give scrollbar colors only to the elements that will have a scrollbar */
    #logs, #input {
        scrollbar-color: $accent;
        scrollbar-color-hover: $secondary;
        scrollbar-color-active: $primary;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        background: $background;
    }

    #logs:focus {
        border: round $primary;
        background: transparent;
    }

    #input {
        min-height: 3;
        height: 3;
        width: 1fr;
        border: round $primary;
    }

    #style {
        width: 14;
        height: 3;
        border: round $primary;
        margin-right: 1;
        text-align: center;
        background: $background;
    }

    #input:focus {
        border: round $foreground;
        background: $background;
        background-tint: transparent;
    }

    #input .text-area--cursor-line {
        background: transparent;
    }

    #style:focus, #style:hover, #style.-active, #input:hover {
        border: round $foreground;
        background: $background;
        background-tint: transparent;
        tint: transparent;
    }

    #logs, #input {
        margin-left: 1;
        margin-right: 1;
    }

    #bottom {
        height: 1;
        background: $background;
        margin-left: 1;
        margin-right: 1;
        margin-bottom: 1;
        color: $primary;
    }

    #status {
        height: 1;
        width: 1fr;
        text-align: left;
        color: $primary;
    }

    #resources {
        height: 1;
        width: 1fr;
        text-align: right;
        color: $primary;
    }

    #header-container {
        height: 1;
        width: 1fr;
        layout: horizontal;
        align: center middle;
        margin-bottom: 1;
        margin-top: 1;
    }

    #header {
        width: auto;
        content-align: center middle;
        color: $primary;
        text-style: bold;
        padding: 0 2;
    }

    .line {
        width: 1fr;
        height: 1;
        border-top: solid $primary;
        margin: 0 2;  /* when zero two works, arno would be proud */
    }

    #controls {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()

        self.logs = None
        self.input_box = None
        self.style_button = None
        self.status = None
        self.resources = None
        self.themes = ("celune", "celune_light")
        self.active_theme_name = "celune"
        self.log_history: list[tuple[str, str]] = []
        self.status_severity = "info"

        self.celune: Optional[Celune] = None
        self.celune_ready = False
        self.celune_styles = ["balanced"]
        self.celune_voices = None

        self.style_index = 0

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        self._log_stdout = None
        self._log_stderr = None

        self.cur_state = "active"

        self.consume_on_boundary = False
        self._suppress_input_change = False
        self._resource_page = 0
        self.indent = 2

    def _severity_color(self, severity: str = "info") -> str:
        """Return the current theme color for a log severity.

        Args:
            severity: The severity label to map to a theme color.

        Returns:
            str: The configured color string for the requested severity.
        """
        palette = SEVERITY_COLORS.get(self.active_theme_name, SEVERITY_COLORS["celune"])
        return palette.get(severity, palette["info"])

    def _apply_theme(self, theme_name: str) -> None:
        """Apply theme and repaint theme-sensitive widgets.

        Args:
            theme_name: The theme name to activate.

        Returns:
            None: This method updates theme state and redraws status and logs.
        """
        self.active_theme_name = theme_name
        self.theme = theme_name  # pylint: disable=W0201
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs()

    def _refresh_theme_text(self) -> None:
        """Refresh widgets that use the active theme's normal text color."""
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

        theme = os.getenv("CELUNE_THEME") or self.celune.config["theme"]

        if theme == "dark":
            self.active_theme_name = "celune"
        elif theme == "light":
            self.active_theme_name = "celune_light"
        else:
            self.active_theme_name = "celune"
            self.safe_log("Invalid theme, defaulting to dark", "warning")
        self.theme = self.active_theme_name  # pylint: disable=W0201

        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.resources = self.query_one("#resources", Label)
        self.style_button = self.query_one("#style", Button)
        self._refresh_status()
        self._refresh_theme_text()
        psutil.cpu_percent(interval=None)
        self.set_interval(10 / len(self._resource_pages()), self.advance_resources)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = LogRedirect(self.safe_log, "info")
        self._log_stderr = LogRedirect(self.safe_log, "warning")

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr

        self.call_after_refresh(self.start_background_init)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.safe_status("Initializing")
        self.update_resources()

    @staticmethod
    def _format_vram() -> str:
        """Return available CUDA memory in a compact display format."""
        if not torch.cuda.is_available():
            return "VRAM: nothing to fetch"

        try:
            device = torch.cuda.current_device()
            avail, total = torch.cuda.mem_get_info(device)
        except Exception:
            return "VRAM: cannot fetch"

        return f"VRAM: {avail / 1024**3:.2f}/{total / 1024**3:.2f} GB available"

    _NVIDIA_SMI: Optional[str] = shutil.which("nvidia-smi")

    @classmethod
    def _gpu_usage(cls) -> Optional[int]:
        """Read GPU utilization from nvidia-smi when it is available."""
        if not cls._NVIDIA_SMI:
            return None

        try:
            result = subprocess.run(
                [
                    cls._NVIDIA_SMI,
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=1,
            )
        except Exception:
            return None

        first_line = result.stdout.strip().splitlines()[0:1]
        if not first_line:
            return None

        try:
            return int(first_line[0].strip())
        except ValueError:
            return None

    def _format_usage(self) -> str:
        """Return CPU/GPU utilization in a compact display format."""
        cpu = psutil.cpu_percent(interval=None)
        gpu = self._gpu_usage()
        gpu_text = f"{gpu}%" if gpu is not None else "N/A"
        return f"CPU: {cpu:.0f}% \u2022 GPU: {gpu_text}"

    def _format_seed(self) -> str:
        """Return the current backend seed when Celune exposes one."""
        seed = None
        if self.celune is not None:
            seed = getattr(self.celune.backend, "current_seed", None)

        return f"Seed: {seed}" if seed is not None else "Seed: N/A"

    def _resource_pages(self) -> tuple[Callable[[], str], ...]:
        """Return resource footer pages in their display order."""

        # system information & volatile state
        pages: list[Callable[[], str]] = [self._format_vram, self._format_usage]
        if (
            self.celune is not None
            and getattr(self.celune.backend, "current_seed", None) is not None
        ):
            pages.append(self._format_seed)

        # Celune status report
        now = datetime.datetime.now()
        lunar = lunar_info(now)
        suffix = "s" if int(lunar[2]) != 1 else ""

        pages.append(lambda: now.strftime("%A, %B %d, %Y"))
        pages.append(lambda: celune_day_status(now))
        pages.append(lambda: lunar_phase(lunar[0]).title())
        pages.append(lambda: f"{int(lunar[2])} day{suffix} until full moon")

        # usage help
        pages.append(lambda: "/help commands")
        pages.append(
            lambda: (
                f"CTRL+C/CTRL+Q exit \u2022 CTRL+T {self.celune.config['theme']} \u2022 CTRL+ENTER say"
            )
        )

        return tuple(pages)

    def update_resources(self) -> None:
        """Refresh the currently selected resource footer page."""
        if self.cur_state == "exiting" or self.resources is None:
            return

        def update() -> None:
            pages = self._resource_pages()
            text = pages[self._resource_page % len(pages)]()
            self.resources.update(text + " " * self.indent)

        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.call_from_thread(update)

    def advance_resources(self) -> None:
        """Advance the resource footer to the next page and refresh it."""
        if self.cur_state == "exiting" or self.resources is None:
            return

        self._resource_page = (self._resource_page + 1) % len(self._resource_pages())
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
            tts_voices = list(self.celune.backend.voices)

            self.celune.set_voices(tts_voices)
            self.celune_styles = tts_voices or ["balanced"]
            self.celune_voices = itertools.cycle(tts_voices)
            if self.celune.current_voice in self.celune_styles:
                self.style_index = self.celune_styles.index(self.celune.current_voice)
            else:
                self.style_index = 0

            if self.celune.load():
                self.celune_ready = True
                self.safe_status("Idle")
                self.style_button.disabled = False
                self.tts_voice_changed(
                    self.celune.current_voice or self.celune_styles[0]
                )
                self.input_box.placeholder = (
                    "Enter text to speak here or run /help for commands"
                )

                self.safe_log("Ready to speak.")

        except Exception as e:
            self.safe_log(
                f"[INIT ERROR] {self.celune.format_error(e, self.celune.dev)}", "error"
            )
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
            self.input_box.placeholder = (
                "Please wait"
                if locked
                else "Enter text to speak here or run /help for commands"
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
            self.status.update(" " * self.indent + msg)
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
            command: The slash-command name without the leading slash.
            args: The parsed command arguments.

        Returns:
            None: This method mutates UI or engine state based on the command.
        """

        self.input_box.load_text("")
        if command == "help":
            self.safe_log("Celune help topics")
            self.safe_log("Available commands:")
            self.safe_log(
                "Arguments marked in <> are required, those marked in [] are optional."
            )
            self.safe_log(
                "/consumebuffer <true/false> - Make Celune consume text from the live buffer without "
                "pressing CTRL+SHIFT+ENTER."
            )
            self.safe_log(
                "Caution: This feature may interfere with typing '...'.", "warning"
            )
            self.safe_log(
                "/invoke <extension> [args] - Invoke a Celune extension by its name."
            )
            self.safe_log("/extensions - List currently available Celune extensions.")
            self.safe_log(
                "/voiceprompt <prompt> - Change Celune's voice prompt. This will allow you to steer her voice."
            )
            self.safe_log(
                "Caution: Some prompts may cause adverse effects. Choose prompts that enhance personality, "
                "rather than replace it.",
                "warning",
            )
            self.safe_log("/speed <speed> - Change speaking speed.")
            self.safe_log("/reverb <strength> - Change reverb strength.")
            self.safe_log(
                "/play <file> - Play a sound effect by path. Only WAV files are supported."
            )
            self.safe_log("/stop - Terminate ongoing speech.")
            self.safe_log("/exit - Exit Celune.")
            self.safe_log("/help - Display this help message.")
            return
        if command == "consumebuffer":
            if not args:
                self.safe_log("Usage: /consumebuffer <true/false>", "warning")
                return

            if args[0].lower() in ["true", "false"]:
                boolean = args[0].lower() == "true"
                self.consume_on_boundary = boolean

                if boolean:
                    self.safe_log("Now consuming from live input")
                else:
                    self.safe_log("No longer consuming from live input")
                return
            self.safe_log(
                f"Invalid argument for '{command}', must be true/false.", "warning"
            )
            return
        if command == "invoke":
            if not args:
                self.safe_log("Usage: /invoke <extension> [args]")
                return

            if not self.celune or not self.celune.extension_manager:
                self.safe_log("Extension system not initialized.", "warning")
                return

            name = args[0]
            invoke_args = args[1:]

            try:
                self.celune.extension_manager.invoke(name, *invoke_args)
            except InvalidExtensionError:
                self.safe_log(f"Extension not found: {name}", "warning")
            except Exception as e:
                self.safe_log(f"[EXT ERROR] {e}", "error")

            return
        if command == "extensions":
            if not self.celune or not self.celune.extension_manager:
                self.safe_log("Extension system not initialized.", "warning")
                return

            names = self.celune.extension_manager.list_extensions()
            if not names:
                self.safe_log("No extensions loaded.", "warning")
            else:
                self.safe_log("Extensions: " + ", ".join(names))
            return
        if command == "voiceprompt":
            if not self.celune:
                self.safe_log("Celune is not initialized.", "warning")
                return

            if not args:
                self.safe_log("Usage: /voiceprompt <prompt>", "warning")
                return

            new_prompt = " ".join(args).strip()
            self.celune.voice_prompt = new_prompt

            if not new_prompt or new_prompt.lower() == "clear":
                self.celune.voice_prompt = None
                self.safe_log("Voice prompt cleared.")
                return

            self.safe_log(f"Voice prompt set to '{new_prompt}'.")
            return
        if command == "speed":
            if not self.celune:
                self.safe_log("Celune is not initialized.", "warning")
                return

            if not self.celune.can_use_rubberband:
                self.safe_log("Celune cannot currently use Rubber Band.", "warning")
                return

            if not args:
                self.safe_log("Usage: /speed <speed>", "warning")
                return

            try:
                speed = float(args[0])
                if not 0.8 <= speed <= 1.2:
                    self.safe_log("Value out of range. Expected 0.8-1.2.", "warning")
                    return
                self.celune.speed = speed
            except ValueError:
                self.safe_log(f"Invalid argument: {args[0]}", "warning")
            else:
                self.safe_log(f"Speaking speed set to x{args[0]}.")
            return
        if command == "reverb":
            if not self.celune:
                self.safe_log("Celune is not initialized.", "warning")
                return

            if not args:
                self.safe_log("Usage: /reverb <strength>", "warning")
                return

            try:
                strength = float(args[0])
                if not 0.0 <= strength <= 1.0:
                    self.safe_log("Value out of range. Expected 0.0-1.0.", "warning")
                    return
                self.celune.reverb.strength = strength
            except ValueError:
                self.safe_log(f"Invalid argument: {args[0]}", "warning")
            else:
                self.safe_log(
                    f"Reverb strength set to {format_number(strength * 100)}%."
                )
            return
        if command == "play":
            if not self.celune:
                self.safe_log("Celune is not initialized.", "warning")
                return

            if not args:
                self.safe_log("Usage: /play <path>", "warning")
                return

            try:
                self.safe_log(f"Playing {args[0]}")
                self.celune.play(args[0])
            except Exception as e:
                self.safe_log(
                    f"Cannot play this file: {self.celune.format_error(e, self.celune.dev)}",
                    "error",
                )
                return
            return
        if command == "stop":
            if not self.celune:
                self.safe_log("Celune is not initialized.", "warning")
                return

            if not self.celune.force_stop_speech():
                self.safe_log("Nothing to stop.")
                return

            return
        if command == "exit":
            self._graceful_exit()
            return

        self.safe_log(
            f"Unknown command: {command}. Run /help for a list of commands.", "warning"
        )

    def consume_buffer(self, tlen: int) -> None:
        """Consume a sentence from live input and say it.

        Args:
            tlen: The number of characters to consume from the input buffer.

        Returns:
            None: This method removes the consumed text and queues it for speech.
        """
        to_say = self.input_box.text[:tlen].strip()

        self._suppress_input_change = True
        try:
            self.input_box.load_text(self.input_box.text[tlen:])
        # yes, no except:
        # that is valid python
        finally:
            self._suppress_input_change = False

        if not to_say:
            return

        if all(char in ".!?;:, " for char in to_say):
            return

        self.celune.say(to_say)

    def on_key(self, event: events.Key) -> None:
        """Accept input and send text to Celune.

        Args:
            event: The key event received from Textual.

        Returns:
            None: This handler processes shortcuts, commands, and speech requests.
        """
        if self.cur_state == "exiting":
            return

        if event.key == "ctrl+q":
            event.prevent_default()
            event.stop()
            self._graceful_exit()
            return

        if event.key == "ctrl+t":
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
            if not self.celune:
                return

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

            if self.celune.say(text):
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
        self.cur_state = "exiting"

        if self.celune is not None:
            self.celune.close()

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
        self.input_box.placeholder = "Enter text to speak here"
        self.safe_status("Idle")
        self.update_resources()

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
        self.input_box.placeholder = (
            "Enter text to speak here or run /help for commands"
        )
        self.style_button.disabled = False
        self.update_resources()

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

        text = event.text_area.text

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

    def signal_handler(self, _sig, _frame) -> None:
        """Trap CTRL+C and exit Celune if pressed.

        Args:
            _sig: The received signal number.
            _frame: The current stack frame from the signal handler.

        Returns:
            None: This handler schedules a graceful application shutdown.
        """
        # cannot be wrapped in try-except, or it won't be effective
        if threading.current_thread() is threading.main_thread():
            self.call_later(self._graceful_exit)
        else:
            self.call_from_thread(self._graceful_exit)


class CeluneHeadlessUI:
    """Celune headless interface methods."""

    def __init__(self):
        # not using Celune palette for compatibility purposes
        self.colors = {
            "black": "\x1b[0;30m",
            "red": "\x1b[0;31m",
            "green": "\x1b[0;32m",
            "yellow": "\x1b[0;33m",
            "blue": "\x1b[0;34m",
            "magenta": "\x1b[0;35m",
            "cyan": "\x1b[0;36m",
            "white": "\x1b[0;37m",
        }
        self.celune: Optional[Celune] = None

        # for Celune terminals not supporting colored text
        self.no_color = (
            config_bool(
                self.celune.config, "CELUNE_HEADLESS_NOCOLOR", "headless_nocolor"
            )
            or not sys.stdout.isatty()
        )
        self.reset: str = "\x1b[0m" if not self.no_color else ""

    def severity_color(self, severity: str) -> str:
        """Get color from the VGA text mode palette.

        Args:
            severity: The severity label to map to a terminal color.

        Returns:
            str: The ANSI color sequence for the requested severity.
        """
        if self.no_color:
            return ""
        if severity == "warning":
            return self.colors["yellow"]
        if severity == "error":
            return self.colors["red"]
        return self.colors["white"]

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: The log message to print.
            severity: The log severity level.

        Returns:
            None: This method prints a formatted line to stdout.
        """
        prefix = ""
        if severity == "warning":
            prefix = "[WARN] "
        elif severity == "error":
            prefix = "[ERROR] "
        print(f"{prefix}{self.severity_color(severity)}{msg}{self.reset}", flush=True)

    def headless_error(self, error: str) -> None:
        """Log an error to the headless interface.

        Args:
            error: The error message to print.

        Returns:
            None: This method forwards the message as an error log.
        """
        self.headless_log(error, "error")

    def run(self) -> None:
        """Start the headless interface.

        Returns:
            None: This method installs the signal handler and keeps the process
                alive.
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        while True:
            time.sleep(1)

    def signal_handler(self, _sig, _frame) -> None:
        """Exit Celune in headless mode on CTRL+C.

        Args:
            _sig: The received signal number.
            _frame: The current stack frame from the signal handler.

        Returns:
            None: This handler closes Celune and exits the process.
        """
        if self.celune is not None:
            self.celune.close()
        sys.exit(0)


class SelectMenu:
    """A selection menu.

    Args:
        choices: Human-readable choice names.
        raw_choices: Internal choice values.
        prompt: Selection prompt to override the default one.
    """

    def __init__(
        self,
        choices: list[str],
        raw_choices: list[str],
        prompt: str = "Select an option",
    ):
        self.choices = choices
        self.raw_choices = raw_choices
        self.prompt = prompt
        self.idx = 0

    def render(self) -> None:
        """Render available selections.

        Returns:
            None: This function does not return a value.
        """
        sys.stdout.write("\r")
        for n, choice in enumerate(self.choices):
            if n == self.idx:
                sys.stdout.write(f"\033[7m -> {choice} \033[0m\n")
            else:
                sys.stdout.write(f"    {choice} \n")

        sys.stdout.write(f"\033[{len(self.choices)}A")
        sys.stdout.flush()

    def start(self) -> str:
        """Start the selection menu.

        Returns:
            str: The chosen option.
        """
        sys.stdout.write("\033[?25l")
        sys.stdout.write(self.prompt)
        sys.stdout.write("\n\n")

        try:
            while True:
                self.render()
                key = readchar.readkey()
                if key == readchar.key.UP:
                    self.idx = (self.idx - 1) % len(self.choices)
                elif key == readchar.key.DOWN:
                    self.idx = (self.idx + 1) % len(self.choices)
                elif key == readchar.key.ENTER:
                    return self.raw_choices[self.idx]
        finally:
            sys.stdout.write(f"\033[{len(self.choices)}B")
            sys.stdout.write("\033[?25h\n")


class LogRedirect:
    """Redirect logs to the logger."""

    def __init__(
        self,
        write_callback: Callable[[str, str], None],
        default_severity: str = "info",
    ) -> None:
        self.write_callback = write_callback
        self.default_severity = default_severity
        self._buffer = ""

    def write(self, text: str) -> None:
        """Write text to the logger.

        Args:
            text: The raw text chunk captured from redirected output.

        Returns:
            None: This method buffers partial lines and forwards complete ones.
        """
        if not text:
            return

        if "is deprecated" in text:
            return

        self._buffer += text

        while "\n" in self._buffer or "\r" in self._buffer:
            newline_pos = self._buffer.find("\n") if "\n" in self._buffer else 10**9
            cr_pos = self._buffer.find("\r") if "\r" in self._buffer else 10**9
            pos = min(newline_pos, cr_pos)

            chunk = self._buffer[:pos].strip()
            self._buffer = self._buffer[pos + 1 :]

            if chunk:
                self.write_callback(chunk, self.default_severity)

    def flush(self) -> None:
        """Flush the buffers.

        Returns:
            None: This method emits any buffered text and clears the buffer.
        """
        if self._buffer.strip():
            self.write_callback(self._buffer.strip(), self.default_severity)
        self._buffer = ""
