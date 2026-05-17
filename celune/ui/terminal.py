# SPDX-License-Identifier: MIT
"""Terminal UI helpers."""

import re
import sys
from typing import Callable, Generic

import readchar

from ..constants import T


class SelectMenu(Generic[T]):
    """A selection menu.

    Args:
        choices: Human-readable choice names.
        raw_choices: Internal choice values.
        prompt: Selection prompt to override the default one.
    """

    def __init__(
        self,
        choices: list[str],
        raw_choices: list[T],
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

    def start(self) -> T:
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
    ) -> None:
        self.write_callback = write_callback
        self.default_severity = default_severity
        self._buffer = ""
        self.underlying_stdout = stdout
        self.underlying_stderr = stderr
        self.filter_messages = []  # these messages will be filtered out by the logger

    def write(self, text: str) -> None:
        """Write text to the logger.

        Args:
            text: The raw text chunk captured from redirected output.

        Returns:
            None: This method buffers partial lines and forwards complete ones.
        """
        if not text:
            return

        if text in self.filter_messages:
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

            if chunk:
                self.write_callback(chunk, self.default_severity)

    def ansi(self, escape: str) -> None:
        """Write ANSI escape code(s) to the terminal directly.

        Args:
            escape: The ANSI escape code(s) to process.

        Returns:
            None: This function writes ANSI escape codes to the underlying terminal.
        """
        ansi_regex = re.compile(
            r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
        )
        any_ansi = re.findall(ansi_regex, escape)
        if any_ansi:
            escapes = "".join(any_ansi)
            self.underlying_stdout.write(escapes)

    def flush(self) -> None:
        """Flush the buffers.

        Returns:
            None: This method emits any buffered text and clears the buffer.
        """
        if self._buffer.strip():
            self.write_callback(self._buffer.strip(), self.default_severity)
        self._buffer = ""

    def isatty(self) -> bool:
        """Return if the underlying terminal is a TTY.

        Returns:
            bool: Whether the underlying terminal is a TTY.
        """
        return self.underlying_stdout.isatty()
