# SPDX-License-Identifier: Apache-2.0
"""Publish Celune's runtime state and optional caption chunks to Discord."""

from __future__ import annotations

import math
import time
import threading
import contextlib
from typing import TYPE_CHECKING, Optional
from collections import deque
from collections.abc import Callable

from .i18n import string
from .constants import APP_NAME

if TYPE_CHECKING:
    from pypresence import Presence

DISCORD_APPLICATION_ID = "1131161920804765768"
DISCORD_ARTWORK_URL = (
    "https://raw.githubusercontent.com/celunah/celune/main/Celune.AppDir/celune.png"
)
POLL_INTERVAL_SECONDS = 1.0
PRESENCE_REFRESH_SECONDS = 30.0
RECONNECT_INTERVAL_SECONDS = 15.0
PRESENCE_UPDATE_LIMIT = 5
PRESENCE_UPDATE_WINDOW_SECONDS = 20.0
CAPTION_INTRO_SECONDS = 1.0
CAPTION_UPDATE_INTERVAL_SECONDS = 6.0
DISCORD_DETAILS_MAX_LENGTH = 128
MAX_PENDING_CAPTION_WORDS = 512
_STATE_STATUS_KEYS = {
    "error": "discord.presence.error",
    "generating": "discord.presence.generating",
    "idle": "discord.presence.idle",
    "reloading": "discord.presence.reloading",
    "sleeping": "discord.presence.sleeping",
    "speaking": "discord.presence.speaking",
    "thinking": "discord.presence.thinking",
    "waking": "discord.presence.waking",
}


class DiscordPresence:
    """Run a reconnecting, presence-only Discord IPC client.

    Args:
        state_provider: Return Celune's current internal runtime state.
        show_utterances: Publish available speech captions as activity details.

    The worker publishes only localized activity labels and, when enabled,
    speech captions. It does not invoke Discord authorization, messaging,
    guild, friend, or voice APIs.
    """

    def __init__(
        self,
        state_provider: Callable[[], str],
        show_utterances: bool = False,
    ) -> None:
        self._state_provider = state_provider
        self._show_utterances = show_utterances
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._caption_lock = threading.Lock()
        self._caption_source_id: Optional[int] = None
        self._caption_generation = 0
        self._caption_active = False
        self._caption_finished = False
        self._caption_words: list[str] = []
        self._caption_current_words: list[str] = []
        self._caption_visible_words = 0
        self._caption_started_at: Optional[float] = None
        self._caption_next_update_at: Optional[float] = None
        self._caption_displayed_details: Optional[str] = None

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
        """Clear the published activity when allowed and stop the worker."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=4)

    def update_caption(
        self,
        caption: Optional[str],
        source_id: Optional[int],
        include_text: bool = False,
    ) -> None:
        """Queue one caption chunk or clear the active caption session."""
        if not self._show_utterances:
            return

        text = caption.strip() if caption is not None else ""
        if not text:
            with self._caption_lock:
                if self._caption_words:
                    self._caption_current_words.clear()
                    self._caption_visible_words = 0
                    self._caption_finished = True
                else:
                    self._reset_caption_locked()
            return

        with self._caption_lock:
            if not self._caption_active or source_id != self._caption_source_id:
                self._reset_caption_locked()
                self._caption_active = True
                self._caption_source_id = source_id
            self._caption_finished = False

            self._caption_current_words = text.split()
            self._caption_visible_words = 0
            if include_text:
                self._append_pending_words_locked(self._caption_current_words)
                self._caption_visible_words = len(self._caption_current_words)

    def update_caption_progress(
        self,
        progress: Optional[float],
        total: Optional[float],
        visible_words: Optional[int],
    ) -> None:
        """Queue only newly visible caption words from playback progress."""
        if not self._show_utterances:
            return

        with self._caption_lock:
            if not self._caption_active or not self._caption_current_words:
                return

            if visible_words is None:
                if (
                    progress is None
                    or total is None
                    or not math.isfinite(progress)
                    or not math.isfinite(total)
                    or total <= 0
                ):
                    return
                visible_word_count = math.ceil(
                    progress / total * len(self._caption_current_words)
                )
            else:
                visible_word_count = visible_words
            visible_word_count = min(
                len(self._caption_current_words),
                max(self._caption_visible_words, visible_word_count),
            )
            newly_visible_words = self._caption_current_words[
                self._caption_visible_words : visible_word_count
            ]
            self._append_pending_words_locked(newly_visible_words)
            self._caption_visible_words = visible_word_count

    def _append_pending_words_locked(self, words: list[str]) -> None:
        """Append words while keeping the pending caption queue bounded."""
        self._caption_words.extend(words)
        if len(self._caption_words) > MAX_PENDING_CAPTION_WORDS:
            self._caption_words = self._caption_words[-MAX_PENDING_CAPTION_WORDS:]

    def _reset_caption_locked(self) -> None:
        """Clear caption state while the caption lock is held."""
        self._caption_generation += 1
        self._caption_source_id = None
        self._caption_active = False
        self._caption_finished = False
        self._caption_words.clear()
        self._caption_current_words.clear()
        self._caption_visible_words = 0
        self._caption_started_at = None
        self._caption_next_update_at = None
        self._caption_displayed_details = None

    def _finish_caption(self) -> None:
        """Flush one final caption batch after speech playback ends."""
        with self._caption_lock:
            if self._caption_words:
                self._caption_finished = True
                self._caption_current_words.clear()
                self._caption_visible_words = 0
            else:
                self._reset_caption_locked()

    def _reset_caption(self) -> None:
        """Discard caption state when speech stops for a non-idle reason."""
        with self._caption_lock:
            self._reset_caption_locked()

    def _caption_activity(
        self,
        state: str,
        now: float,
    ) -> tuple[Optional[str], Optional[tuple[str, int, int, Optional[str]]]]:
        """Return caption details and a commit token for a due update."""
        with self._caption_lock:
            if not self._caption_active:
                return None, None

            generation = self._caption_generation
            if self._caption_finished:
                if state not in {"speaking", "idle"}:
                    self._reset_caption_locked()
                    return None, None
                if not self._caption_words:
                    self._reset_caption_locked()
                    return None, None
                details, word_count, remainder = _take_caption_batch(
                    self._caption_words
                )
                if details is None:
                    self._reset_caption_locked()
                    return None, None
                return details, ("final", generation, word_count, remainder)

            if state != "speaking":
                return None, None

            if self._caption_started_at is None:
                return (
                    string("discord.presence.says", app_name=APP_NAME),
                    ("intro", generation, 0, None),
                )

            if now < self._caption_started_at + CAPTION_INTRO_SECONDS:
                return string("discord.presence.says", app_name=APP_NAME), None

            if (
                self._caption_words
                and self._caption_next_update_at is not None
                and now >= self._caption_next_update_at
            ):
                details, word_count, remainder = _take_caption_batch(
                    self._caption_words
                )
                if details is None:
                    return None, None
                return details, ("caption", generation, word_count, remainder)

            if self._caption_displayed_details == string(
                "discord.presence.says",
                app_name=APP_NAME,
            ):
                return None, None

            return self._caption_displayed_details, None

    def _commit_caption_activity(
        self,
        action: tuple[str, int, int, Optional[str]],
        details: str,
        now: float,
    ) -> None:
        """Commit a caption update only if its source session is still current."""
        action_kind, generation, word_count, remainder = action
        with self._caption_lock:
            if generation != self._caption_generation or not self._caption_active:
                return

            if action_kind == "intro":
                if self._caption_started_at is None:
                    self._caption_started_at = now
                    self._caption_next_update_at = now + CAPTION_INTRO_SECONDS
                self._caption_displayed_details = details
                return

            del self._caption_words[:word_count]
            if remainder:
                self._caption_words.insert(0, remainder)
            if action_kind == "final":
                self._reset_caption_locked()
                return
            self._caption_displayed_details = details
            self._caption_next_update_at = now + CAPTION_UPDATE_INTERVAL_SECONDS

    def _run(self) -> None:
        """Connect to Discord locally and publish activity transitions."""
        try:
            from pypresence import Presence
            from pypresence.types import ActivityType, StatusDisplayType
        except ImportError:
            return

        client: Optional[Presence] = None
        last_status_key: Optional[str] = None
        last_details: Optional[str] = None
        last_update_at = 0.0
        retry_after = 0.0
        previous_state: Optional[str] = None
        update_times: deque[float] = deque()

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
                        last_details = None
                        last_update_at = 0.0

                if client is not None:
                    try:
                        state = self._state_provider()
                    except Exception:
                        state = ""

                    if state == "idle" and previous_state != "idle":
                        self._finish_caption()
                    elif previous_state == "speaking" and state != "speaking":
                        self._reset_caption()
                    previous_state = state
                    status_key = _STATE_STATUS_KEYS.get(state)
                    details = (
                        string(status_key, app_name=APP_NAME)
                        if status_key is not None
                        else None
                    )
                    caption_action: Optional[tuple[str, int, int, Optional[str]]] = None
                    if self._show_utterances:
                        caption_details, caption_action = self._caption_activity(
                            state, time.monotonic()
                        )
                        if caption_details is not None:
                            details = caption_details

                    refresh_due = (
                        details is not None
                        and time.monotonic() - last_update_at
                        >= PRESENCE_REFRESH_SECONDS
                    )
                    if (
                        status_key != last_status_key
                        or details != last_details
                        or refresh_due
                        or caption_action is not None
                    ):
                        now = time.monotonic()
                        if _reserve_presence_update(update_times, now):
                            try:
                                if details is None:
                                    client.clear()
                                else:
                                    client.update(
                                        name=APP_NAME,
                                        details=details,
                                        large_image=DISCORD_ARTWORK_URL,
                                        activity_type=ActivityType.LISTENING,
                                        status_display_type=StatusDisplayType.DETAILS,
                                    )
                            except Exception:
                                with contextlib.suppress(Exception):
                                    client.close()
                                client = None
                                retry_after = (
                                    time.monotonic() + RECONNECT_INTERVAL_SECONDS
                                )
                                last_status_key = None
                                last_details = None
                                last_update_at = 0.0
                            else:
                                last_status_key = status_key
                                last_details = details
                                last_update_at = time.monotonic()
                                if details is not None and caption_action is not None:
                                    self._commit_caption_activity(
                                        caption_action,
                                        details,
                                        last_update_at,
                                    )

                if self._stop_event.wait(POLL_INTERVAL_SECONDS):
                    break
        finally:
            if client is not None:
                if _reserve_presence_update(update_times, time.monotonic()):
                    with contextlib.suppress(Exception):
                        client.clear()
                with contextlib.suppress(Exception):
                    client.close()


def _reserve_presence_update(update_times: deque[float], now: float) -> bool:
    """Reserve one activity update within Discord's rolling rate limit."""
    while update_times and now - update_times[0] >= PRESENCE_UPDATE_WINDOW_SECONDS:
        update_times.popleft()
    if len(update_times) >= PRESENCE_UPDATE_LIMIT:
        return False

    update_times.append(now)
    return True


def _take_caption_batch(words: list[str]) -> tuple[Optional[str], int, Optional[str]]:
    """Build one Discord-sized caption detail without discarding long text."""
    selected: list[str] = []
    selected_length = 0
    consumed = 0
    remainder: Optional[str] = None

    for word in words:
        separator_length = 1 if selected else 0
        available = DISCORD_DETAILS_MAX_LENGTH - selected_length - separator_length
        if len(word) <= available:
            selected.append(word)
            selected_length += separator_length + len(word)
            consumed += 1
            continue

        if not selected and available > 0:
            selected.append(word[:available])
            consumed = 1
            remainder = word[available:]
        break

    details = " ".join(selected)
    if len(details) < 2:
        return None, 0, None

    return details, consumed, remainder
