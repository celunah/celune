# SPDX-License-Identifier: MIT
"""UI callback protocols."""

from typing import Protocol, Optional

from ..celune import Celune


class CeluneBaseUI(Protocol):
    """Celune base UI protocols."""

    celune: Celune

    def run(self) -> None:
        """Run the UI's main loop.

        Returns:
            None: Implementations block until the UI exits.
        """


class CeluneTextualUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's interactive Textual UI callbacks."""

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Handle log messages coming from Celune.

        Args:
            msg: The log message emitted by Celune.
            severity: The log severity level.

        Returns:
            None: Implementations forward the log message.
        """

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: The status text to display.
            severity: The status severity level.

        Returns:
            None: Implementations update their status display.
        """

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Current progress, or ``None`` for an indeterminate bar.
            total: Total progress, or ``None`` for an indeterminate bar.

        Returns:
            None: Implementations update their progress display.
        """

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: The error text to display.

        Returns:
            None: Implementations expose the error to the user.
        """

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking.

        Returns:
            None: Implementations restore idle state.
        """

    def tts_queue_avail(self) -> None:
        """Unlock input queueing after Celune completes generation.

        Returns:
            None: Implementations re-enable queueing while playback continues.
        """

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: The newly active voice name.

        Returns:
            None: Implementations synchronize visible voice state.
        """

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: Whether input should be disabled.

        Returns:
            None: Implementations update input availability.
        """

    def change_voice_lock_state(self, locked: bool) -> None:
        """Lock or unlock Celune's voice change button.

        Args:
            locked: Whether voice changes should be disabled.

        Returns:
            None: Implementations update voice change availability.
        """


class CeluneHeadlessBaseUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's headless UI callbacks."""

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: The log message to print.
            severity: The log severity level.

        Returns:
            None: Implementations emit the log line.
        """

    def headless_error(self, error: str) -> None:
        """Log an error to the headless interface.

        Args:
            error: The error message to print.

        Returns:
            None: Implementations emit the error line.
        """
