# SPDX-License-Identifier: Apache-2.0
"""Shared UI update data records."""

from dataclasses import dataclass


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
