# SPDX-License-Identifier: Apache-2.0
"""Playback contention tracking for the shared audio pipeline."""

from __future__ import annotations

import os
import threading
import contextlib
from typing import TYPE_CHECKING

import psutil

from .pipelinecore import (
    _PLAYBACK_BUFFER_MAX_SECONDS,
    _PLAYBACK_BUFFER_MIN_SECONDS,
    _PLAYBACK_CONTENTION_CPU_START,
    _PLAYBACK_CONTENTION_CPU_CRITICAL,
    _PLAYBACK_CONTENTION_STABLE_DECAY,
    _PLAYBACK_CONTENTION_REBUFFER_LEVEL,
    _PLAYBACK_CONTENTION_SAMPLE_SECONDS,
    _PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS,
    _PLAYBACK_CONTENTION_LAG_START_SECONDS,
    _PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
)

if TYPE_CHECKING:
    from .celune import Celune


class _PlaybackContentionMonitor:
    """Estimate playback contention from CPU pressure and output timing."""

    def __init__(self, engine: Celune) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
        self._last_sample_at = 0.0
        self._level = 0.0
        self._underflows = 0
        with contextlib.suppress(psutil.Error, OSError):
            self._process.cpu_percent(interval=None)

    @staticmethod
    def _pressure(value: float, start: float, critical: float) -> float:
        """Normalize one observed pressure value to the inclusive 0..1 range."""
        if value <= start:
            return 0.0
        if value >= critical:
            return 1.0
        return (value - start) / (critical - start)

    def _publish_locked(self) -> None:
        """Publish lightweight diagnostics for logs and status views."""
        self._engine.playback_contention_level = self._level
        self._engine.playback_underflows = self._underflows

    def sample_cpu(self, now: float) -> None:
        """Sample system and process CPU without blocking the playback loop."""
        with self._lock:
            if now - self._last_sample_at < _PLAYBACK_CONTENTION_SAMPLE_SECONDS:
                return
            self._last_sample_at = now

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
        except (psutil.Error, OSError, TypeError, ValueError):
            cpu_percent = 0.0

        try:
            process_percent = self._process.cpu_percent(interval=None)
        except (psutil.Error, OSError, TypeError, ValueError):
            process_percent = 0.0

        logical_cpus = max(1, psutil.cpu_count(logical=True) or 1)
        normalized_process_percent = process_percent / logical_cpus
        pressure = max(
            self._pressure(
                cpu_percent,
                _PLAYBACK_CONTENTION_CPU_START,
                _PLAYBACK_CONTENTION_CPU_CRITICAL,
            ),
            self._pressure(
                normalized_process_percent,
                _PLAYBACK_CONTENTION_CPU_START,
                _PLAYBACK_CONTENTION_CPU_CRITICAL,
            ),
        )

        with self._lock:
            if pressure > 0.0:
                self._level = max(self._level * 0.9, pressure)
            else:
                self._level *= _PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def observe_scheduler_lag(self, delay_seconds: float) -> None:
        """Record a delayed playback-loop wakeup as contention evidence."""
        self._observe_pressure(
            self._pressure(
                delay_seconds,
                _PLAYBACK_CONTENTION_LAG_START_SECONDS,
                _PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
            )
        )

    def _observe_pressure(self, pressure: float) -> None:
        """Update the smoothed contention level from one pressure sample."""
        pressure = max(0.0, min(1.0, pressure))
        with self._lock:
            if pressure > 0.0:
                self._level = max(self._level * 0.9, pressure)
            else:
                self._level *= _PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def observe_write(
        self,
        elapsed_seconds: float,
        block_seconds: float,
        underflowed: bool,
    ) -> None:
        """Record one output write and any PortAudio-reported underflow."""
        with self._lock:
            if underflowed:
                self._underflows += 1
                self._level = 1.0
            else:
                write_pressure = self._pressure(
                    max(0.0, elapsed_seconds - block_seconds),
                    _PLAYBACK_CONTENTION_LAG_START_SECONDS,
                    _PLAYBACK_CONTENTION_LAG_CRITICAL_SECONDS,
                )
                if write_pressure > 0.0:
                    self._level = max(self._level * 0.9, write_pressure)
                else:
                    self._level *= _PLAYBACK_CONTENTION_STABLE_DECAY
            self._publish_locked()

    def target_seconds(self) -> float:
        """Return the current reserve target, rising as contention increases."""
        with self._lock:
            max_seconds = _PLAYBACK_BUFFER_MAX_SECONDS + (
                (_PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS - _PLAYBACK_BUFFER_MAX_SECONDS)
                * self._level
            )
            return _PLAYBACK_BUFFER_MIN_SECONDS + (
                (max_seconds - _PLAYBACK_BUFFER_MIN_SECONDS) * self._level
            )

    def capacity_seconds(self) -> float:
        """Return the maximum reserve allowed at the current contention level."""
        with self._lock:
            return _PLAYBACK_BUFFER_MAX_SECONDS + (
                (_PLAYBACK_BUFFER_CRITICAL_MAX_SECONDS - _PLAYBACK_BUFFER_MAX_SECONDS)
                * self._level
            )

    def requires_rebuffer(self) -> bool:
        """Return whether contention is high enough to pause for more reserve."""
        with self._lock:
            return self._level >= _PLAYBACK_CONTENTION_REBUFFER_LEVEL
