# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for the Textual UI runtime."""

from __future__ import annotations

import sys
import asyncio

from textual import timer as textual_timer


async def _asyncio_timer_sleep(secs: float) -> None:
    """Sleep through the event loop without occupying its default executor."""
    await asyncio.sleep(secs)


def install_textual_timer_sleep() -> None:
    """Use cancellable event-loop sleeps for Textual timers on Windows."""
    if sys.platform == "win32":
        textual_timer.sleep = _asyncio_timer_sleep
