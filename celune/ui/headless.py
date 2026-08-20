# SPDX-License-Identifier: MIT
"""Headless UI."""

import os
import time
import ctypes
import signal
import warnings
from types import FrameType
from typing import Optional, cast
from collections.abc import Callable

from ..i18n import string
from ..utils import discard
from ..celune import Celune
from ..typing.aliases import LogLevel
from ..config import Config, config_bool
from ..constants import SIGTSTP, APP_NAME
from ..watchdog import launcher_loss_requested


class CeluneHeadlessUI:
    """Headless interface methods."""

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

        # for app terminals not supporting colored text
        self.no_color = config_bool(
            config, "CELUNE_HEADLESS_NOCOLOR", "headless_nocolor"
        )
        self.reset = "\x1b[0m" if not self.no_color else ""
        self._exit = False
        self._windows_signal_handler: Optional[Callable[[int], bool]] = None

        CeluneHeadlessUI._instance = self

    def _has_celune(self) -> bool:
        """Is the app attached to this UI instance?"""
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

    def headless_log(
        self,
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Log to the headless interface.

        Args:
            msg: The log message to print.
            severity: The log severity level.
            loglevel: The minimum configured log level required to display the message.
        """
        levels = {"info": 0, "verbose": 1, "debug": 2}
        active_log_level = getattr(self.celune, "log_level", "info")
        if levels.get(active_log_level, 0) < levels.get(loglevel, 0):
            return

        prefix = ""
        if severity == "warning":
            prefix = string("headless.warn_prefix")
        elif severity == "error":
            prefix = string("headless.error_prefix")
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
                string(
                    "headless.no_attached_instance",
                    class_name=self.__class__.__name__,
                    app_name=APP_NAME,
                ),
                RuntimeWarning,
            )

        if os.name == "nt":
            self._install_windows_signal_handler()
        else:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            if SIGTSTP is not None:
                signal.signal(SIGTSTP, self._signal_handler)

        while not self._exit:
            if launcher_loss_requested():
                self.close()
                self._exit = True
                continue
            time.sleep(0.1)

    def close(self) -> None:
        """Exit from Celune's headless interface."""

        if self.celune is not None:
            self.celune.close()

        CeluneHeadlessUI._instance = None

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
            self.close()
            self._exit = True
            return True
        return False

    def _signal_handler(self, sig: int, frame: Optional[FrameType]) -> None:
        """Handle incoming signals."""
        discard(frame)

        if SIGTSTP is not None and sig == SIGTSTP:
            return

        self.close()
        self._exit = True
