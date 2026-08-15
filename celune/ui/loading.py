# SPDX-License-Identifier: MIT
"""Lightweight loading overlay owned by the main UI."""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Static

from ..constants import APP_NAME
from ..i18n import string


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
        self._error_message = ""

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
                string("ui.loading_wait"),
                id="loading-wait",
                markup=False,
            )
            yield Static(
                self._error_message,
                id="loading-error",
                markup=False,
            )
        with Horizontal(id="loading-footer"):
            yield Static(
                string("ui.loading_starting", app_name=APP_NAME),
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
        self._update_error_widget()
        self.set_interval(0.26, self._advance_spinner)

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
        self._status_message = message
        try:
            self.query_one("#loading-state-label", Static).update(message)
        except NoMatches:
            pass

    def show_error(self, message: str) -> None:
        """Show an initialization error without hiding the overlay.

        Args:
            message: Initialization failure to show to the user.
        """
        self._error_message = message
        self._update_error_widget()

    def _update_error_widget(self) -> None:
        """Refresh the optional initialization error widget."""
        try:
            error = self.query_one("#loading-error", Static)
        except NoMatches:
            return

        error.update(self._error_message)
        error.display = bool(self._error_message)
