# SPDX-License-Identifier: Apache-2.0
"""Command host adapting shared UI commands to browser actions."""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Callable

from .runtime import append_webui_log, set_webui_status, publish_active_task_log

from ..utils import available
from ..typing.aliases import LogLevel

if TYPE_CHECKING:
    from ..celune import Celune


class _WebUiInputProxy:
    """Minimal input surface required by the shared slash-command handler."""

    @staticmethod
    def load_text(_value: str) -> None:
        """Discard command input after the browser has submitted it."""


class WebUiCommandHost:
    """Core-backed command host used when the Textual UI is not mounted."""

    def __init__(
        self,
        celune: Celune,
        submit_command: Callable[[str], bool],
    ) -> None:
        self.celune = celune
        self.input_box = _WebUiInputProxy()
        self.consume_on_boundary = False
        self.tutorial_token = 0
        self._submit_command = submit_command

    @property
    def tutorial_active(self) -> bool:
        """Return whether the core is currently in tutorial mode."""
        return bool(getattr(self.celune, "is_in_tutorial", False))

    def safe_log(
        self,
        message: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Forward command output to browser state and active task logs."""
        publish_active_task_log(message, severity)
        append_webui_log(message, severity)
        _ = loglevel

    def safe_status(self, message: str, severity: str = "info") -> None:
        """Forward a command status to the browser footer."""
        set_webui_status(message, severity, source="callback")

    @staticmethod
    def call_from_thread(
        callback: Callable[..., None], *args: object, **kwargs: object
    ) -> None:
        """Run a command callback immediately in the API worker context."""
        callback(*args, **kwargs)

    def refresh_vc_controls(self) -> None:
        """Refresh browser controls on the next snapshot."""

    def set_vc_f0_condition(self, enabled: bool) -> None:
        """Set VC talk or sing conditioning through the core state."""
        self.celune.vc_f0_condition = enabled
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and available("f0_condition", obj=backend):
            backend.f0_condition = enabled

    def open_settings_menu(self) -> None:
        """Report that configuration editing belongs to the Textual UI."""
        from ..i18n import string

        self.safe_log(string("commands.settings_unavailable"), "warning")

    def begin_tutorial(self) -> None:
        """Leave tutorial lifecycle ownership to the core runtime."""

    def finish_tutorial(self) -> None:
        """Leave tutorial lifecycle ownership to the core runtime."""

    def cancel_tutorial(self, _restore_input: bool = False) -> bool:
        """Return whether the core was already outside tutorial mode."""
        return not self.tutorial_active

    def tutorial_after(self, _delay: float, callback: Callable[[], None]) -> None:
        """Run a command tutorial callback without a second timer source."""
        callback()

    def type_and_send(self, text: str, process_commands: bool = True) -> None:
        """Submit a tutorial string through the browser command path."""
        if process_commands and text.startswith("/"):
            self._submit_command(text)
        else:
            self.celune.say(text)

    @staticmethod
    def pulse_border(_selector: str) -> None:
        """Ignore a Textual-only tutorial animation in the browser host."""

    def graceful_exit(self) -> None:
        """Close the bound runtime when the browser receives ``/exit``."""
        self.celune.close()
