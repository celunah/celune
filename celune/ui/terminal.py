# SPDX-License-Identifier: MIT
"""Terminal UI helpers."""

import logging
import re
import sys
from collections.abc import Collection
from typing import Callable, Optional

import readchar

from ..constants import JSONSerializable


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
        raw_choices: list[JSONSerializable],
        prompt: str = "Select an option",
    ) -> None:
        if not choices:
            raise ValueError("choices must not be empty")
        if len(choices) != len(raw_choices):
            raise ValueError("choices and raw_choices must have same length")

        self.choices = choices
        self.raw_choices = raw_choices
        self.prompt = prompt
        self.idx = 0

    def render(self) -> None:
        """Render available selections."""
        sys.stdout.write("\r")
        for n, choice in enumerate(self.choices):
            if n == self.idx:
                sys.stdout.write(f"\033[7m -> {choice} \033[0m\n")
            else:
                sys.stdout.write(f"    {choice} \n")

        sys.stdout.write(f"\033[{len(self.choices)}A")
        sys.stdout.flush()

    def start(self) -> JSONSerializable:
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
        stdout,
        stderr,
        write_callback: Callable[[str, str], None],
        default_severity: str = "info",
        filter_messages: Optional[Collection[str]] = None,
    ) -> None:
        self.write_callback = write_callback
        self.default_severity = default_severity
        self._buffer = ""
        self.underlying_stdout = stdout
        self.underlying_stderr = stderr
        self.filter_messages = (
            filter_messages  # these messages will be filtered out by the logger
        )

    def _is_filtered_message(self, message: str) -> bool:
        """Return whether one redirected message should be suppressed."""
        if self.filter_messages is None:
            return False

        return any(
            filtered_message in message for filtered_message in self.filter_messages
        )

    @staticmethod
    def _severity_for_message(message: str, default_severity: str) -> str:
        """Infer log severity from one redirected text line."""
        lowered = message.casefold()

        if "[error]" in lowered:
            return "error"
        if "[warning]" in lowered:
            return "warning"
        if "traceback (most recent call last):" in lowered:
            return "error"
        if re.search(
            r"\b(?:error|exception|fatal(?: error)?)\b",
            lowered,
        ):
            return "error"
        if re.search(
            (
                r"\b(?:warning|futurewarning|deprecationwarning|"
                r"pendingdeprecationwarning|runtimewarning|resourcewarning|"
                r"userwarning|syntaxwarning|importwarning|unicodewarning|"
                r"byteswarning)\b"
            ),
            lowered,
        ):
            return "warning"
        return default_severity

    def write(self, text: str) -> None:
        """Write text to the logger.

        Args:
            text: The raw text chunk captured from redirected output.
        """
        if not text:
            return

        # strip any incoming ANSI, but keep TTY specific input
        ansi_regex = re.compile(
            r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
        )
        text = re.sub(ansi_regex, "", text)

        self._buffer += text

        while "\n" in self._buffer or "\r" in self._buffer:
            newline_pos = self._buffer.find("\n") if "\n" in self._buffer else 10**9
            cr_pos = self._buffer.find("\r") if "\r" in self._buffer else 10**9
            pos = min(newline_pos, cr_pos)

            chunk = self._buffer[:pos].strip()
            self._buffer = self._buffer[pos + 1 :]

            if chunk and not self._is_filtered_message(chunk):
                self.write_callback(
                    chunk,
                    self._severity_for_message(chunk, self.default_severity),
                )

    def ansi(self, escape: str) -> None:
        """Write ANSI escape code(s) to the terminal directly.

        Args:
            escape: The ANSI escape code(s) to process.
        """
        ansi_regex = re.compile(
            r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
        )
        any_ansi = re.findall(ansi_regex, escape)
        if any_ansi:
            escapes = "".join(any_ansi)
            self.underlying_stdout.write(escapes)
            self.underlying_stdout.flush()

    def flush(self) -> None:
        """Flush the buffers."""
        if self._buffer.strip():
            chunk = self._buffer.strip()
            if not self._is_filtered_message(chunk):
                self.write_callback(
                    chunk,
                    self._severity_for_message(chunk, self.default_severity),
                )
        self._buffer = ""

    def isatty(self) -> bool:
        """Return if the underlying terminal is a TTY.

        Returns:
            bool: Whether the underlying terminal is a TTY.
        """
        return self.underlying_stdout.isatty()


class UILogHandler(logging.Handler):
    """Route Python logging records into Celune's UI log callback."""

    def __init__(
        self,
        write_callback: Callable[[str, str], None],
        filter_messages: Optional[Collection[str]] = None,
    ) -> None:
        super().__init__()
        self.write_callback = write_callback
        self.filter_messages = filter_messages

    def _is_filtered_message(self, message: str) -> bool:
        """Return whether one logging message should be suppressed."""
        if self.filter_messages is None:
            return False

        return any(
            filtered_message in message for filtered_message in self.filter_messages
        )

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one Python logging record into the UI log stream.

        Args:
            record: The logging record to be emitted.
        """
        try:
            message = record.getMessage().strip()
        except Exception:
            self.handleError(record)
            return

        if not message:
            return

        if (
            "triton not found; flop counting will not work for triton kernels"
            in message
        ):
            message = "triton not found; flop counting will not work for triton kernels"

        if self._is_filtered_message(message):
            return

        if record.levelno >= logging.ERROR:
            severity = "error"
            prefix = "Internal runtime error:"
        elif record.levelno >= logging.WARNING:
            severity = "warning"
            prefix = "Internal runtime warning:"
        else:
            severity = "info"
            prefix = "Internal runtime notice:"

        self.write_callback(" ".join([prefix, message]), severity)


def is_celune_log_record(record: logging.LogRecord) -> bool:
    """Return whether a logging record belongs to Celune itself.

    Args:
        record: The logging record to classify.

    Returns:
        bool: ``True`` when the record originated from Celune loggers.
    """
    return record.name == "celune" or record.name.startswith("celune.")
