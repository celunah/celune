# SPDX-License-Identifier: Apache-2.0
"""Lightweight loading overlay owned by the main UI."""

from __future__ import annotations

from typing import Optional

from textual.widget import Widget
from textual.widgets import Static
from textual.app import ComposeResult
from textual.timer import Timer
from textual.css.query import NoMatches
from textual.containers import Center, Vertical, Horizontal

from ..i18n import string
from ..constants import APP_NAME


class CeluneLoadingScreen(Widget):
    """Display startup progress as an in-place overlay widget."""

    def __init__(self, *, widget_id: Optional[str] = None) -> None:
        super().__init__(id=widget_id)
        self._spinner_frames = (
            "⠋",
            "⠙",
            "⠹",
            "⠸",
            "⠼",
            "⠴",
            "⠦",
            "⠧",
            "⠇",
            "⠏",
        )
        self._spinner_index = 0
        self._startup_messages: list[str] = []
        self._status_message = string("status.initializing")
        self._latest_log_message = string("ui.loading_waiting_for_log")
        self._wait_message = string("ui.loading_wait")
        self._footer_message = string("ui.loading_starting", app_name=APP_NAME)
        self._spinner_timer: Optional[Timer] = None
        self._failed = False

    @property
    def opacity(self) -> float:
        """Return the overlay opacity used by the Textual animator."""
        return self.styles.opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set the overlay opacity through its CSS styles."""
        self.styles.opacity = value

    def compose(self) -> ComposeResult:
        """Compose the startup overlay widgets.

        Returns:
            ComposeResult: The loading overlay widget tree.
        """
        with Center(id="loading-center"), Vertical(id="loading-content"):
            yield Static(APP_NAME, id="loading-brand", markup=False)
            yield Static(
                self._status_message,
                id="loading-state-label",
                markup=False,
            )
            with Vertical(id="loading-log"):
                yield Static(
                    "",
                    id="loading-diagnostics",
                    markup=False,
                )
                yield Static(
                    self._latest_log_message,
                    id="loading-log-message",
                    markup=False,
                )
            yield Static(self._spinner_frames[0], id="loading-spinner", markup=False)
            yield Static(
                self._wait_message,
                id="loading-wait",
                markup=False,
            )
        with Horizontal(id="loading-footer"):
            yield Static(
                self._footer_message,
                id="loading-footer-starting",
                markup=False,
            )
            yield Static(
                string("ui.loading_quit"),
                id="loading-footer-quit",
                markup=False,
            )

    def on_mount(self) -> None:
        """Start the loading spinner after the overlay is mounted."""
        if self._failed:
            self.query_one("#loading-spinner", Static).display = False
            return
        self._spinner_timer = self.set_interval(0.26, self._advance_spinner)

    def _advance_spinner(self) -> None:
        """Advance the loading spinner by one frame."""
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        try:
            self.query_one("#loading-spinner", Static).update(
                self._spinner_frames[self._spinner_index]
            )
        except NoMatches:
            pass

    def set_latest_log_message(self, message: str) -> None:
        """Show the latest useful startup log message.

        Args:
            message: Log message to display below the loading state.
        """
        if self._failed:
            return
        self._latest_log_message = message
        try:
            self.query_one("#loading-log-message", Static).update(message)
        except NoMatches:
            pass

    def set_startup_messages(self, messages: list[str]) -> None:
        """Show diagnostic startup messages that occurred before Textual mounted.

        Args:
            messages: Startup messages to keep visible while the engine loads.
        """
        self._startup_messages = list(messages)
        self._update_startup_messages_widget()

    def append_startup_message(self, message: str) -> None:
        """Append one diagnostic startup message to the loading screen.

        Args:
            message: Startup message emitted after the loading screen mounted.
        """
        self._startup_messages.append(message)
        self._update_startup_messages_widget()

    def _update_startup_messages_widget(self) -> None:
        """Refresh the diagnostic startup message block when it is mounted."""
        try:
            diagnostics = self.query_one("#loading-diagnostics", Static)
        except NoMatches:
            return

        diagnostics.update("\n".join(self._startup_messages))
        diagnostics.display = bool(self._startup_messages)

    def set_status_message(self, message: str) -> None:
        """Show the current startup status.

        Args:
            message: Current status text.
        """
        if self._failed:
            return
        self._status_message = message
        try:
            self.query_one("#loading-state-label", Static).update(message)
        except NoMatches:
            pass

    def show_error(
        self,
        message: str,
        *,
        status_message: Optional[str] = None,
        footer_message: Optional[str] = None,
    ) -> None:
        """Switch the overlay from startup progress to its failure state.

        Args:
            message: Initialization failure to show to the user.
            status_message: Optional replacement for the failure heading.
            footer_message: Optional status to show in the lower-left footer.
        """
        self._failed = True
        self._status_message = status_message or string("status.failed_to_start")
        self._latest_log_message = message
        self._wait_message = string(
            "ui.loading_cannot_continue",
            app_name=APP_NAME,
        )
        self._footer_message = footer_message or string(
            "ui.app_could_not_start",
            app_name=APP_NAME,
        )
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
        try:
            self.query_one("#loading-state-label", Static).update(self._status_message)
            self.query_one("#loading-log-message", Static).update(message)
            self.query_one("#loading-wait", Static).update(self._wait_message)
            self.query_one("#loading-spinner", Static).display = False
            self.query_one("#loading-footer-starting", Static).update(
                self._footer_message
            )
        except NoMatches:
            pass
