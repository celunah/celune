# SPDX-License-Identifier: MIT
"""Headless Celune UI."""

import signal
import sys
import time
from types import FrameType
from typing import Any, Optional, cast

from ..celune import Celune
from ..config import config_bool
from ..constants import ExitCodes, SIGTSTP
from ..utils import supports_ansi


class CeluneHeadlessUI:
    """Celune headless interface methods."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize headless UI state.

        Args:
            config: Loaded configuration dictionary used for terminal color
                settings.

        Returns:
            None: This constructor prepares terminal color handling.
        """
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
        self.celune = cast(Celune, None)

        # for Celune terminals not supporting colored text
        self.no_color = (
            config_bool(config, "CELUNE_HEADLESS_NOCOLOR", "headless_nocolor")
            or not supports_ansi()
        )
        self.reset = "\x1b[0m" if not self.no_color else ""

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
        if SIGTSTP is not None:
            signal.signal(SIGTSTP, self.signal_handler)
        while True:
            time.sleep(1)

    def signal_handler(self, sig: int, _frame: Optional[FrameType]) -> None:
        """Exit Celune in headless mode on CTRL+C and handle CTRL+Z.

        Args:
            sig: The received signal number.
            _frame: The current stack frame from the signal handler.

        Returns:
            None: This handler closes Celune and exits the process.
        """
        if SIGTSTP is not None and sig == SIGTSTP:
            return

        if self.celune is not None:
            self.celune.close()
        sys.exit(ExitCodes.EXIT_SUCCESS.value)
