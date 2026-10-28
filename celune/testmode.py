# SPDX-License-Identifier: Apache-2.0
"""Small runtime state used by the interactive UI test mode."""

from typing import Optional
from collections.abc import Callable

from .i18n import string
from .typing.common import Config, JSON
from .typing.modes import BackendMode, OperationState


class _UITestBackend:
    """Identify the UI test runtime without importing a TTS backend."""

    is_fake = True
    singing = False


class UITestRuntime:
    """Provide the state and lifecycle callbacks needed to display the UI."""

    backend_mode: BackendMode = "ui_test"
    backend = _UITestBackend()
    voices: tuple[str, ...] = ()
    current_voice: Optional[str] = None
    use_normalization = False

    def __init__(
        self,
        config: Config,
        backend_mode: BackendMode,
        log_level: str,
        startup_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        if backend_mode != "ui_test":
            raise ValueError(
                f"unknown backend mode for UI test runtime: {backend_mode}"
            )

        self.config = config
        self.log_level = log_level
        self._startup_callback = startup_callback
        self.cur_state = "idle"
        self.operation_state = OperationState.NORMAL
        self.test_finished = False
        self.test_result: Optional[JSON] = None
        self.locked = False
        self._closed = False

    def load(self) -> bool:
        """Complete lightweight startup without loading models or audio devices."""
        if self._startup_callback is not None:
            self._startup_callback(string("ui.startup_initializing_core"))
        return not self._closed

    def finish_test_mode(
        self,
        mode: str,
        success: bool,
        *,
        detail: Optional[str] = None,
    ) -> JSON:
        """Record the UI test result and stop accepting runtime actions."""
        if self.test_result is not None:
            return self.test_result

        result: JSON = {
            "mode": mode,
            "success": success,
            "engine_state": "stopped",
            "task_state": None,
            "detail": detail,
        }
        self.test_result = result
        self.test_finished = True
        self.operation_state = OperationState.STOPPED
        self.cur_state = "stopped"
        self.locked = True
        return result

    def fatal(self) -> None:
        """Mark lightweight startup as failed."""
        self.operation_state = OperationState.FATAL
        self.cur_state = "error"

    def close(self) -> None:
        """Release the lightweight test runtime."""
        self._closed = True
        self.cur_state = "stopped"
