# SPDX-License-Identifier: Apache-2.0
"""Engine callback and status dispatch."""

from __future__ import annotations

from ..typing.aliases import LogLevel


def status(self, msg: str, severity: str = "info") -> None:
    """Update Celune's status display.

    Args:
        msg: The status message to show.
        severity: The status severity level.
    """
    self.status_callback(msg, severity)


def log(
    self,
    msg: str,
    severity: str = "info",
    *,
    loglevel: LogLevel = "info",
) -> None:
    """Log a message.

    Args:
        msg: The message to emit.
        severity: The message severity level.
        loglevel: The minimum configured log level required to emit the message.
    """
    levels = {"info": 0, "verbose": 1, "debug": 2}
    if levels.get(self.log_level, 0) < levels.get(loglevel, 0):
        return
    if loglevel == "info":
        self.log_callback(msg, severity)
        return
    try:
        self.log_callback(msg, severity, loglevel=loglevel)
    except TypeError as error:
        if "loglevel" not in str(error):
            raise
        self.log_callback(msg, severity)


def _buffer_startup_log(
    self,
    msg: str,
    severity: str = "info",
    *,
    loglevel: LogLevel = "info",
) -> None:
    """Buffer startup diagnostics until the runtime banner is emitted."""
    levels = {"info": 0, "verbose": 1, "debug": 2}
    if levels.get(self.log_level, 0) < levels.get(loglevel, 0):
        return
    if not self._startup_banner_emitted:
        self._startup_log_buffer.append((msg, severity, loglevel))
        return
    _deliver_startup_log(self, msg, severity, loglevel)


def _emit_runtime_banner_line(
    self,
    msg: str,
    severity: str = "info",
) -> None:
    """Emit one runtime-banner line without entering the startup buffer."""
    _deliver_startup_log(self, msg, severity, "info")


def _deliver_startup_log(
    self,
    msg: str,
    severity: str,
    loglevel: LogLevel,
) -> None:
    """Forward one startup log while retaining legacy callback compatibility."""
    try:
        self._startup_log_sink(msg, severity, loglevel=loglevel)
    except TypeError as error:
        if "loglevel" not in str(error):
            raise
        self._startup_log_sink(msg, severity)


def _flush_startup_logs(self) -> None:
    """Flush buffered constructor diagnostics after the runtime banner."""
    buffered_logs = self._startup_log_buffer
    self._startup_log_buffer = []
    for msg, severity, loglevel in buffered_logs:
        _deliver_startup_log(self, msg, severity, loglevel)


def log_dev(self, msg: str, severity: str = "info") -> None:
    """Log a legacy developer message at verbose level.

    Args:
        msg: The message to emit.
        severity: The message severity level.
    """
    self.log(msg, severity, loglevel="verbose")
