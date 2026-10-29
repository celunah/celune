# SPDX-License-Identifier: Apache-2.0
"""Tests for the terminal license acknowledgement screen."""

import asyncio

from celune.ui.app import CeluneUI, ModelConsentScreen

from .fixtures import reset_ui_singletons


def test_model_consent_modal_keeps_the_app_screen_visible() -> None:
    """Verify the consent modal leaves the app screen visible behind it."""
    ui = CeluneUI()

    async def run() -> None:
        async with ui.run_test(size=(80, 24)) as pilot:
            main_container = ui.query_one("#container")
            main_container.display = True
            ui.query_one("#loading-overlay").display = False
            underlying_screen = ui.screen
            modal = ModelConsentScreen(
                prompt="License terms",
                instruction="Type the acknowledgement to continue.",
                details=(),
                entry_hint="Acknowledgement",
                consent_phrase="I consent",
                accept_label="Accept",
                decline_label="Decline",
            )

            ui.push_screen(modal)
            await pilot.pause()

            assert modal.styles.background.a == 0
            assert ui._background_screens == [underlying_screen]

    try:
        asyncio.run(run())
    finally:
        reset_ui_singletons()
