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
            msg: Value for `msg`.
            severity: Value for `severity`.
        """

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: Value for `msg`.
            severity: Value for `severity`.
        """

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Value for `progress`.
            total: Value for `total`.
        """

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: Value for `error`.
        """

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""

    def tts_queue_avail(self) -> None:
        """Unlock input queueing after Celune completes generation."""

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: Value for `name`.
        """

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: Value for `locked`.
        """

    def change_voice_lock_state(self, locked: bool) -> None:
        """Lock or unlock Celune's voice change button.

        Args:
            locked: Value for `locked`.
        """


class CeluneHeadlessBaseUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's headless UI callbacks."""

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: Value for `msg`.
            severity: Value for `severity`.
        """

    def headless_error(self, error: str) -> None:
        """Log an error to the headless interface.

        Args:
            error: Value for `error`.
        """
