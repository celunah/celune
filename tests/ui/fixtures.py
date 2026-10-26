# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for independently collected Textual UI test classes."""

from celune.ui.app import CeluneUI
from celune.ui.headless import CeluneHeadlessUI


def reset_ui_singletons() -> None:
    """Reset UI application singletons between frontend test cases."""
    CeluneUI._instance = None
    CeluneHeadlessUI._instance = None
