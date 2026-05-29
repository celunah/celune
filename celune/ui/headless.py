# SPDX-License-Identifier: MIT
"""Headless Celune UI."""

import signal
import sys
import time
from types import FrameType
from typing import Optional, cast
import warnings

from ..celune import Celune
from ..utils import discard
from ..config import Config, config_bool
from ..constants import ExitCodes, SIGTSTP


class CeluneHeadlessUI:
    """Celune headless interface methods."""

    _instance: Optional["CeluneHeadlessUI"] = None

    def __init__(self, config: Optional[Config] = None) -> None:
        if CeluneHeadlessUI._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")

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
        self.no_color = config_bool(
            config, "CELUNE_HEADLESS_NOCOLOR", "headless_nocolor"
        )
        self.reset = "\x1b[0m" if not self.no_color else ""

        CeluneHeadlessUI._instance = self

    def _has_celune(self) -> bool:
        """Is Celune attached to this UI instance?"""
        return self.celune is not None

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
        if severity == "sleeping":
            return self.colors["blue"]
        # sleeping severity does not have a match in the VGA palette
        return self.colors["magenta"]

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: The log message to print.
            severity: The log severity level.
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
        """
        self.headless_log(error, "error")

    def run(self) -> None:
        """Start the headless interface."""
        if not self._has_celune():
            warnings.warn(
                f"{self.__class__.__name__} has no attached Celune instance: this will do nothing",
                RuntimeWarning,
            )

        signal.signal(signal.SIGINT, self.signal_handler)
        if SIGTSTP is not None:
            signal.signal(SIGTSTP, self.signal_handler)
        while True:
            time.sleep(1)

    def close(self) -> None:
        """Exit from Celune's headless interface."""

        if self.celune is not None:
            self.celune.close()

        CeluneHeadlessUI._instance = None

    def signal_handler(self, sig: int, frame: Optional[FrameType]) -> None:
        """Exit Celune in headless mode on CTRL+C and handle CTRL+Z.

        Args:
            sig: The received signal number.
            frame: The current stack frame from the signal handler.
        """
        if SIGTSTP is not None and sig == SIGTSTP:
            return

        discard(frame)
        self.close()
        sys.exit(ExitCodes.EXIT_SUCCESS.value)
