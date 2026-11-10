# SPDX-License-Identifier: Apache-2.0
"""Transient, reusable toast messages for the Textual interface."""

from __future__ import annotations

from collections.abc import Callable
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
        self._generation = 0

    @property
    def opacity(self) -> float:
        """Return the toast opacity used by Textual's animator."""
        return self.styles.opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set the toast opacity through its CSS styles."""
        self.styles.opacity = value

    def on_mount(self) -> None:
        """Keep the toast hidden until a message is requested."""
        self.display = False
        self.styles.opacity = 0.0

    def on_unmount(self) -> None:
        """Stop pending timers when the toast leaves the interface."""
        self._cancel_timers()

    def on_resize(self, _event: Resize) -> None:
        """Recenter against the visible content area after a resize."""
        if self.display:
            self.call_after_refresh(self._center_toast)

    def show(
        self,
        message: str,
        duration: float = DEFAULT_DURATION,
        *,
        severity: str = "info",
    ) -> None:
        """Display a message, then fade it out after ``duration`` seconds."""
        self._generation += 1
        generation = self._generation
        self._cancel_timers()
        self.set_class(severity == "warning", "severity-warning")
        self.set_class(severity == "error", "severity-error")
        self.update(" ".join(message.split()))
        self.display = True
        self.call_after_refresh(self._fade_in, generation, max(0.0, duration))

    def _fade_in(self, generation: int, duration: float) -> None:
        """Center and reveal the current message after its layout is ready."""
        if generation != self._generation:
            return

        self._center_toast()
        self._animate_opacity(1.0)
        self._hide_timer = self.set_timer(duration, lambda: self._fade_out(generation))

    def _center_toast(self) -> None:
        """Center the toast in the screen's scrollable content region."""
        screen = self.screen
        content_region = screen.scrollable_content_region
        current_offset = self.styles.offset.resolve(screen.size, screen.size)
        horizontal_offset = round(
            content_region.center[0] - screen.region.x - self.region.width / 2
        )
        self.styles.offset = (
            horizontal_offset,
            current_offset.y,
        )

    def _fade_out(self, generation: int) -> None:
        """Fade the current message before removing its widget."""
        if generation != self._generation:
            return

        self._hide_timer = None
        self._animate_opacity(0.0, lambda: self._dismiss(generation))

    def _dismiss(self, generation: int) -> None:
        """Hide the toast after its fade-out transition finishes."""
        if generation != self._generation:
            return

        self.update("")
        self.display = False

    def _cancel_timers(self) -> None:
        """Cancel timers left by an earlier message."""
        if self._hide_timer is not None:
            self._hide_timer.stop()
            self._hide_timer = None

    def _animate_opacity(
        self,
        opacity: float,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        """Animate opacity when mounted, or apply it directly before mounting."""
        if self.is_attached:
            self.animate(
                "opacity",
                opacity,
                duration=self.FADE_DURATION,
                on_complete=on_complete,
            )
            return
        self.styles.opacity = opacity
        if on_complete is not None:
            on_complete()
