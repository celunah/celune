# SPDX-License-Identifier: MIT
"""UI callback protocols."""

from typing import Protocol, Optional

from ..celune import Celune


class CeluneBaseUI(Protocol):
    """Celune base UI protocols."""

    celune: Celune

    def run(self) -> None:
        """Run the UI's main loop."""


class CeluneTextualUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's interactive Textual UI callbacks."""

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Handle log messages coming from Celune.

        Args:
            msg: The message to be logged.
            severity: The severity to log the message as.
        """

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: The message to be logged.
            severity: The severity to log the message as.
        """

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: How many steps were processed.
            total: How many total steps are to be processed.
        """

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: The error message to log.
        """

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""

    def tts_queue_avail(self) -> None:
        """Unlock input queueing after Celune completes generation."""

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: The loaded voice name.
        """

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: The new UI lock state.
        """

    def change_voice_lock_state(self, locked: bool) -> None:
        """Lock or unlock Celune's voice change button.

        Args:
            locked: The new voice change lock state.
        """


class CeluneHeadlessBaseUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's headless UI callbacks."""

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: The message to be logged.
            severity: The severity to log the message as.
        """

    def headless_error(self, error: str) -> None:
        """Log an error to the headless interface.

        Args:
            error: The error message to log.
        """
