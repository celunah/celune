# SPDX-License-Identifier: Apache-2.0
"""Tests for transient Textual toast notifications."""

import pytest
from textual.app import App, ComposeResult

from celune.ui.theme import CELUNE_CSS
from celune.ui.toast import CeluneToast


class _ToastHarness(App[None]):
    """Minimal Textual host with Celune's toast styles."""

    CSS = CELUNE_CSS + "\nScreen { scrollbar-gutter: stable; }"

    def compose(self) -> ComposeResult:
        """Mount one toast for layout and animation checks."""
        yield CeluneToast()


@pytest.mark.anyio
async def test_toast_is_bottom_centered_and_fades_in_and_out() -> None:
    """Place toast at the bottom center and animate both opacity transitions."""
    app = _ToastHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        toast = app.query_one(CeluneToast)
        toast.show("Toast test message", duration=0.5)

        await pilot.pause(0.1)
        assert toast.display
        assert (
            abs(toast.region.center[0] - app.screen.scrollable_content_region.center[0])
            <= 0.5
        )
        assert toast.region.bottom == app.screen.size.height - 1
        assert 0.0 < toast.opacity < 1.0

        await pilot.pause(0.2)
        assert toast.opacity == 1.0

        await pilot.pause(0.3)
        assert 0.0 < toast.opacity < 1.0

        await pilot.pause(0.3)
        assert not toast.display
        assert toast.opacity == 0.0


@pytest.mark.anyio
async def test_toast_colors_warning_and_error_severities() -> None:
    """Use separate theme colors for warning and error toasts."""
    app = _ToastHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        toast = app.query_one(CeluneToast)
        toast.show("Warning", severity="warning")
        await pilot.pause(0.05)

        warning_color = toast.styles.color
        assert toast.has_class("severity-warning")

        toast.show("Error", severity="error")
        await pilot.pause(0.05)

        assert toast.has_class("severity-error")
        assert toast.styles.color != warning_color
