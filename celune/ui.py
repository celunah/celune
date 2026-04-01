# pylint: disable=C0114, R0912, W0718, R0911, R0902
"""Celune's frontend layer."""

import os
import sys
import shlex
import hashlib
import threading
import itertools
from typing import Callable

from textual import work, events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, RichLog, TextArea, Button
from rich.text import Text

from .celune import Celune

SEVERITY_COLORS = {
    "info": "#ceaaff",  # lunar.css accent 100 - Celune accent
    "warning": "#fcf283",  # Celune warning
    "error": "#ff6b6b",  # Celune error
}


class CeluneUI(App):
    """Celune's user interface."""

    CSS = """
    Screen {
        layout: vertical;
        background: #1d1824;
    }

    #logs {
        height: 1fr;
        border: round #ceaaff;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1;
    }

    /* give scrollbar colors only to the elements that will have a scrollbar */
    #logs, #input {
        scrollbar-color: #9a7fbf; /* lunar.css accent 900 */
        scrollbar-color-hover: #af90d8; /* lunar.css accent 500 */
        scrollbar-color-active: #ceaaff; /* lunar.css accent 100 - Celune accent */
        scrollbar-background: #1d1824; /* lunah.site --toggle-accent (50%) - Celune background */
        scrollbar-background-hover: #1d1824;
        scrollbar-background-active: #1d1824;
        background: #1d1824;
    }

    #logs:focus {
        border: round #ceaaff;
        background: transparent;
    }

    #input {
        min-height: 3;
        height: 3;
        width: 1fr;
        border: round #ceaaff;
    }

    #style {
        width: 14;
        height: 3;
        border: round #ceaaff;
        margin-right: 1;
        text-align: center;
        background: #1d1824;
    }

    #input:focus {
        border: round #ceaaff;
        background-tint: #ceaaff 10%;
    }

    #logs, #input {
        margin-left: 1;
        margin-right: 1;
    }

    #status {
        height: 1;
        background: #1d1824;
        width: 1fr;
        margin-left: 2;
        margin-bottom: 1;
        color: #ceaaff;
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
        color: #ceaaff;
        text-style: bold;
        padding: 0 2;
    }
    
    .line {
        width: 1fr;
        height: 1;
        border-top: solid #ceaaff;
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

        self.celune: Celune | None = None
        self.celune_ready = False
        self.celune_styles = ["balanced", "calm", "enthusiastic", "upbeat"]
        self.celune_voices = None

        self.style_index = 0

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        self._log_stdout = None
        self._log_stderr = None

        self.cur_state = "active"

        self.consume_on_boundary = False
        self._suppress_input_change = False

    def compose(self) -> ComposeResult:
        """Define the UI."""
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
            yield Label("Initializing", id="status")

    def on_mount(self) -> None:
        """Prepare Celune."""
        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.style_button = self.query_one("#style", Button)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = LogRedirect(self.safe_log, "info")
        self._log_stderr = LogRedirect(self.safe_log, "warning")

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr

        self.call_after_refresh(self.start_background_init)

    def start_background_init(self) -> None:
        """Run Celune's initialization function."""
        self.load_tts()

    @work(thread=True, exclusive=True)
    def load_tts(self) -> None:
        """Load Celune."""
        try:
            tts_voices = {
                "balanced": (
                    "refs/balanced.wav",
                    "My name is Celune, pronounced Celune. It is a pleasure to meet you.",
                    4243102495,
                ),
                "calm": (
                    "refs/calm.wav",
                    "My name is... Celune... It is so... quiet.",
                    418977738,
                ),
                "enthusiastic": (
                    "refs/enthusiastic.wav",
                    "My name is Celune! Let's do this, we have to get it done!",
                    590298652,
                ),
                "upbeat": (
                    "refs/upbeat.wav",
                    "Hehehe... Hi, I'm Celune. Look, I have something to tell... might as well make it fun. Shall we?",
                    3771593946
                )
            }

            self.celune.set_voices(tts_voices)
            self.celune_voices = itertools.cycle(tts_voices.values())
            tts_hashes = {
                "balanced": "",
                "calm": "",
                "enthusiastic": "",
                "upbeat": "",
            }

            for voice_name, (
                voice_path,
                _,
                _,
            ) in tts_voices.items():  # ignore seed and ref text
                if not os.path.exists(voice_path):
                    self.safe_log(f"Reference voice '{voice_name}' not found.", "error")
                    self.safe_status(f"Missing reference voice '{voice_name}'", "error")
                    return

                checksum_path = f"{os.path.splitext(voice_path)[0]}.sha256"

                if os.path.exists(checksum_path):
                    with open(checksum_path, "r", encoding="utf-8") as f:
                        # the type checker doesn't like this one
                        tts_hashes[voice_name] = f.read().strip()

                    with open(voice_path, "rb") as f:
                        voice_hash = hashlib.file_digest(f, "sha256").hexdigest()

                    if voice_hash != tts_hashes[voice_name]:
                        self.safe_log(
                            f"Voice file mismatch, voice '{voice_name}' may be affected."
                        )
                else:
                    self.safe_log(
                        f"Reference voice '{voice_name}' has no checksum.", "warning"
                    )

            if self.celune.load():
                self.celune_ready = True
                self.safe_status("Idle")
                self.style_button.disabled = False
                self.input_box.placeholder = (
                    "Enter text to speak here or run /help for commands"
                )

                if self.celune.extension_manager is not None:
                    self.safe_log("[EXT] Running extension autostart")
                    self.celune.extension_manager.autostart_all()

                self.safe_log("Ready to speak.")

        except Exception as e:
            self.safe_log(
                f"[INIT ERROR] {self.celune.format_error(e, self.celune.dev)}", "error"
            )
            self.error("Celune could not start")
            self.cur_state = "error"

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status."""
        if self.cur_state == "exiting" or self.status is None:
            return

        if severity not in SEVERITY_COLORS:
            self.safe_log(
                f"[WARNING] Unknown severity '{severity}', defaulting to info",
                "warning",
            )

        color = SEVERITY_COLORS.get(severity, "#ceaaff")

        def update() -> None:
            self.status.update(msg)
            self.status.styles.color = color

        if threading.current_thread() is threading.main_thread():
            update()
        else:
            self.call_from_thread(update)

    def safe_log(self, msg: str, severity: str = "info") -> None:
        """Log a message."""
        if self.cur_state == "exiting" or self.logs is None:
            return

        if threading.current_thread() is threading.main_thread():
            self.logs.write(Text(msg, style=SEVERITY_COLORS.get(severity, "#ceaaff")))
        else:
            self.call_from_thread(
                self.logs.write,
                Text(msg, style=SEVERITY_COLORS.get(severity, "#ceaaff")),
            )

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice."""
        if self.cur_state == "exiting":
            return

        if name in self.celune_styles:
            self.style_index = self.celune_styles.index(name)

        label = name.capitalize()

        if threading.current_thread() is threading.main_thread():
            self.style_button.label = label
        else:
            self.call_from_thread(lambda: setattr(self.style_button, "label", label))

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Set status from TTS log."""
        if self.cur_state == "exiting":
            return

        self.safe_log(msg, severity)

    def process_command(self, command: str, args: list[str]) -> None:
        """Process Celune control commands."""

        self.input_box.load_text("")
        if command == "help":
            self.safe_log("Available commands:")
            self.safe_log(
                "/consumebuf <true/false> - Make Celune consume text from the live buffer without "
                "pressing CTRL+ENTER."
            )
            self.safe_log(
                "Caution: This feature may interfere with typing '...'.", "warning"
            )
            self.safe_log(
                "/invoke <extension> <args> - Invoke a Celune extension by its name."
            )
            self.safe_log("/extensions - List currently available Celune extensions.")
            self.safe_log("/exit - Exit Celune.")
            self.safe_log("/help - Display this help message.")
            return
        if command == "consumebuf":
            if not args:
                self.safe_log("Usage: /consumebuf <true/false>", "warning")
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
                self.safe_log("Usage: /invoke <extension_name>")
                return

            if not self.celune or not self.celune.extension_manager:
                self.safe_log("Extension system not initialized.", "warning")
                return

            name = args[0]
            invoke_args = args[1:]

            try:
                self.celune.extension_manager.invoke(name, *invoke_args)
            except KeyError:
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
        if command == "exit":
            self.safe_log("Exiting Celune...")
            self.celune.close()
            self.exit()
            return

        self.safe_log(
            f"Unknown command: {command}. Run /help for a list of commands.", "warning"
        )

    def consume_buffer(self, tlen: int) -> None:
        """Consume a sentence from live input and say it."""
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
        """Accept input and send text to Celune."""
        if self.cur_state == "exiting":
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
                event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Change Celune's tone."""
        if self.cur_state == "exiting":
            return

        if event.button != self.style_button:
            return

        self.style_index = (self.style_index + 1) % len(self.celune_styles)
        next_voice = self.celune_styles[self.style_index]
        self.celune.set_voice(next_voice)

    def on_unmount(self) -> None:
        """Unload Celune."""
        self.cur_state = "exiting"

        if self.celune is not None:
            self.celune.close()

        if hasattr(self, "_old_stdout"):
            sys.stdout = self._old_stdout
        if hasattr(self, "_old_stderr"):
            sys.stderr = self._old_stderr

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""
        if self.cur_state == "exiting":
            return
        self.celune.locked = False
        self.celune.cur_state = "idle"
        self.input_box.placeholder = (
            "Enter text to speak here or run /help for commands"
        )
        self.safe_status("Idle")

    def tts_queue_avail(
        self,
    ) -> None:  # allow enqueuing new inputs while speaking but after generation
        """Unlock input queueing after Celune completes the generation."""
        if self.cur_state == "exiting":
            return
        self.celune.locked = False
        self.safe_status("Speaking")
        self.input_box.placeholder = (
            "Enter text to speak here or run /help for commands"
        )
        self.style_button.disabled = False

    def error(self, error: str) -> None:
        """Set the UI status to the error message."""
        if self.cur_state == "exiting":
            return
        self.safe_status(error, "error")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Monitor text area changes and perform actions."""
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
        """Write text to the logger."""
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
        """Flush the buffers."""
        if self._buffer.strip():
            self.write_callback(self._buffer.strip(), self.default_severity)
        self._buffer = ""
