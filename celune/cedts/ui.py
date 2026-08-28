# SPDX-License-Identifier: Apache-2.0
"""CEDTS-framed timed updates shared by Celune frontends."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from collections.abc import Callable
from typing import cast

from .protocol import build_packet, decode_message, encode_message
from ..typing.worker import WorkerValue

UiTimedUpdateCallback = Callable[["UiTimedUpdate"], None]


@dataclass(frozen=True, slots=True)
class UiTimedUpdate:
    """One authoritative timed UI update emitted by the TUI runtime."""

    runtime_id: str
    sequence: int
    emitted_at: float
    resource_page: int
    theme_name: str
    status_text: str
    status_severity: str
    status_marquee_offset: int

    def as_data(self) -> dict[str, object]:
        """Return the update data carried by its CEDTS event packet."""
        return {
            "runtime_id": self.runtime_id,
            "sequence": self.sequence,
            "emitted_at": self.emitted_at,
            "resource_page": self.resource_page,
            "theme_name": self.theme_name,
            "status_text": self.status_text,
            "status_severity": self.status_severity,
            "status_marquee_offset": self.status_marquee_offset,
        }


class UiTimedUpdateChannel:
    """Deliver CEDTS-framed timed updates to in-process frontend subscribers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: list[UiTimedUpdateCallback] = []

    def subscribe(self, callback: UiTimedUpdateCallback) -> Callable[[], None]:
        """Subscribe to updates and return an idempotent unsubscribe callback."""
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, update: UiTimedUpdate) -> None:
        """Frame and deliver one timed update to all current subscribers."""
        packet = build_packet(
            "event",
            "ui_timed_update",
            cast(dict[str, WorkerValue], update.as_data()),
        )
        control, payloads = encode_message(packet)
        decoded = decode_message(
            control,
            {payload.descriptor["id"]: payload for payload in payloads},
        )
        data = cast(dict[str, object], decoded["data"])
        framed_update = UiTimedUpdate(
            runtime_id=cast(str, data["runtime_id"]),
            sequence=cast(int, data["sequence"]),
            emitted_at=cast(float, data["emitted_at"]),
            resource_page=cast(int, data["resource_page"]),
            theme_name=cast(str, data["theme_name"]),
            status_text=cast(str, data["status_text"]),
            status_severity=cast(str, data["status_severity"]),
            status_marquee_offset=cast(int, data["status_marquee_offset"]),
        )
        with self._lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(framed_update)


ui_timed_update_channel = UiTimedUpdateChannel()
