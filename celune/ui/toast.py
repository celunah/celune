# SPDX-License-Identifier: Apache-2.0
"""Transient, reusable toast messages for the Textual interface."""

from __future__ import annotations

from typing import Optional

from textual.events import Resize
from textual.timer import Timer
from textual.widgets import Static


class CeluneToast(Static):
    """Show one centered message near the bottom of the terminal."""

    DEFAULT_DURATION = 3.0
    FADE_DURATION = 0.25

    def __init__(self) -> None:
        super().__init__("", id="celune-toast", markup=False)
        self._hide_timer: Optional[Timer] = None
        self._dismiss_timer: Optional[Timer] = None
        self._generation = 0

    def on_mount(self) -> None:
        """Keep the toast hidden until a message is requested."""
        self.display = False
        self.styles.opacity = 0.0

    def on_unmount(self) -> None:
        """Stop pending timers when the toast leaves the interface."""
        self._cancel_timers()

    def on_resize(self, _event: Resize) -> None:
        """Keep the toast centered when the terminal changes size."""
        if self.display:
            self.call_after_refresh(self._center_toast)

    def show(self, message: str, duration: float = DEFAULT_DURATION) -> None:
        """Display a message, then fade it out after ``duration`` seconds."""
        self._generation += 1
        generation = self._generation
        self._cancel_timers()
        self.update(" ".join(message.split()))
        self.display = True
        self.styles.opacity = 0.0
        self.call_after_refresh(self._fade_in, generation, max(0.0, duration))

    def _fade_in(self, generation: int, duration: float) -> None:
        """Center and reveal the current message after its layout is ready."""
        if generation != self._generation:
            return

        self._center_toast()
        self.styles.opacity = 1.0
        self._hide_timer = self.set_timer(duration, lambda: self._fade_out(generation))

    def _center_toast(self) -> None:
        """Place the toast at the horizontal center and terminal bottom."""
        center_x = max(0, (self.screen.size.width - self.size.width) // 2)
        self.styles.offset = (center_x, "100%")

    def _fade_out(self, generation: int) -> None:
        """Fade the current message before removing its widget."""
        if generation != self._generation:
            return

        self._hide_timer = None
        self.styles.opacity = 0.0
        self._dismiss_timer = self.set_timer(
            self.FADE_DURATION, lambda: self._dismiss(generation)
        )

    def _dismiss(self, generation: int) -> None:
        """Hide the toast after its fade-out transition finishes."""
        if generation != self._generation:
            return

        self._dismiss_timer = None
        self.update("")
        self.display = False

    def _cancel_timers(self) -> None:
        """Cancel timers left by an earlier message."""
        if self._hide_timer is not None:
            self._hide_timer.stop()
            self._hide_timer = None
        if self._dismiss_timer is not None:
            self._dismiss_timer.stop()
            self._dismiss_timer = None
