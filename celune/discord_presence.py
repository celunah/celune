# SPDX-License-Identifier: Apache-2.0
"""Publish Celune's coarse runtime state through Discord Rich Presence."""

from __future__ import annotations

import time
import threading
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from .i18n import string
from .constants import APP_NAME

if TYPE_CHECKING:
    from pypresence import Presence

DISCORD_APPLICATION_ID = "1131161920804765768"
POLL_INTERVAL_SECONDS = 1.0
PRESENCE_REFRESH_SECONDS = 30.0
RECONNECT_INTERVAL_SECONDS = 15.0
_STATE_STATUS_KEYS = {
    "generating": "status.generating",
    "idle": "status.idle",
    "reloading": "status.reloading",
    "sleeping": "status.sleeping",
    "speaking": "status.speaking",
    "thinking": "status.thinking",
    "waking": "status.waking_up",
}


class DiscordPresence:
    """Run a reconnecting, presence-only Discord IPC client.

    Args:
        state_provider: Return Celune's current internal runtime state.

    The worker publishes only a localized activity label. It does not invoke
    Discord authorization, messaging, guild, friend, or voice APIs.
    """

    def __init__(self, state_provider: Callable[[], str]) -> None:
        self._state_provider = state_provider
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background presence worker once."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="celune-discord-presence",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Clear the published activity and stop the worker promptly."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=4)

    def _run(self) -> None:
        """Connect to Discord locally and publish state transitions."""
        try:
            from pypresence import Presence
            from pypresence.types import ActivityType, StatusDisplayType
        except ImportError:
            return

        client: Optional[Presence] = None
        last_status_key: Optional[str] = None
        last_update_at = 0.0
        retry_after = 0.0

        try:
            while not self._stop_event.is_set():
                if client is None and time.monotonic() >= retry_after:
                    candidate: Optional[Presence] = None
                    try:
                        candidate = Presence(
                            DISCORD_APPLICATION_ID,
                            connection_timeout=2,
                            response_timeout=2,
                        )
                        candidate.connect()
                    except Exception:
                        if candidate is not None:
                            with contextlib.suppress(Exception):
                                candidate.close()
                        retry_after = time.monotonic() + RECONNECT_INTERVAL_SECONDS
                    else:
                        client = candidate
                        last_status_key = None
                        last_update_at = 0.0

                if client is not None:
                    try:
                        state = self._state_provider()
                    except Exception:
                        state = ""
                    status_key = _STATE_STATUS_KEYS.get(state)

                    refresh_due = (
                        status_key is not None
                        and time.monotonic() - last_update_at
                        >= PRESENCE_REFRESH_SECONDS
                    )
                    if status_key != last_status_key or refresh_due:
                        try:
                            if status_key is None:
                                client.clear()
                            else:
                                client.update(
                                    name=APP_NAME,
                                    details=string(status_key),
                                    activity_type=ActivityType.LISTENING,
                                    status_display_type=StatusDisplayType.DETAILS,
                                )
                        except Exception:
                            with contextlib.suppress(Exception):
                                client.clear()
                            with contextlib.suppress(Exception):
                                client.close()
                            client = None
                            retry_after = time.monotonic() + RECONNECT_INTERVAL_SECONDS
                            last_status_key = None
                            last_update_at = 0.0
                        else:
                            last_status_key = status_key
                            last_update_at = time.monotonic()

                if self._stop_event.wait(POLL_INTERVAL_SECONDS):
                    break
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.clear()
                with contextlib.suppress(Exception):
                    client.close()
