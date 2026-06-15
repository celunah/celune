# SPDX-License-Identifier: MIT
"""Celune Razer Chroma and OpenRGB-compatible RGB glow effect."""

from __future__ import annotations

import os
import time
import datetime
import threading
import contextlib
from collections import deque
from typing import Union, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor

from .dsp import _split
from .colors import RGB, ERROR
from .constants import BASE_SR
from .utils import to_rgb, lunar_info, range_interpolated, is_celune_day

if TYPE_CHECKING:
    from .celune import Celune


class AudioRGBGlow:
    """OpenRGB-compatible speaking-aware glow effect."""

    def __init__(
        self,
        celune: Optional["Celune"],
        color: str,
        host: str = "127.0.0.1",
        port: int = 6742,
    ) -> None:
        self.base_color = np.array(
            self._fix_color_rendering(to_rgb(color)), dtype=np.float32
        )
        self.fatal_color = np.array(
            self._fix_color_rendering(to_rgb(ERROR)), dtype=np.float32
        )
        self._current_color = self.base_color.copy()
        self._target_color = self.base_color.copy()

        self.celune = celune

        self.host = host
        self.port = port
        self.connect_failed = False
        self.finished = threading.Event()
        self.client = None
        self.devices = []

        self._scheduled_chunks = deque()
        self.fps = 60

        self.transition_rate = 0.02
        self.color_transition_rate = 0.08
        self.glow_multiplier = 1.0

        # force Celune Day flag
        self.max_glow_forced = os.getenv("CELUNE_FORCE_CELUNE_DAY") in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }

        # Celune glows much brighter on Celune Day, else she'll glow according to the lunar phase
        # this effect is muted down to 25% of current brightness while Celune is sleeping
        current_date = datetime.datetime.now()
        if is_celune_day():
            self.glow_multiplier *= 3.0
        else:
            _, illumination, _ = lunar_info(current_date)
            self.glow_multiplier *= range_interpolated(illumination, 1.0, 2.0)

        if not self.max_glow_forced:
            self.idle_brightness = 0.05 * self.glow_multiplier
        else:
            self.idle_brightness = 0.05 * 2.0 * 3.0  # max glow level

        self.max_brightness = 1.0

        self.input_gain = 4.0
        self.gamma = 1.8
        self.smoothing_factor = 0.8
        self.fast = True

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None

        self._current_brightness = 0.0
        self._target_brightness = self.idle_brightness
        self._sleep_restore_brightness = self.idle_brightness
        self._smoothed_level = 0.0

        self._state = "none"

    def connect(self) -> bool:
        """Connect to the OpenRGB backend and initialize devices.

        Returns:
            bool: ``True`` when the client is connected and devices are ready, otherwise ``False``.
        """
        if self.client is not None:
            return True

        if self.connect_failed:
            return False

        try:
            self.client = OpenRGBClient(address=self.host, port=self.port)
            self.devices = list(self.client.ee_devices)
            for device in self.devices:
                with contextlib.suppress(Exception):
                    device.set_custom_mode()
            return True
        except (TimeoutError, OSError):
            self.client = None
            self.connect_failed = True
            self.devices = []
            return False

    def start(self) -> bool:
        """Start the glow effect worker thread.

        Returns:
            bool: ``True`` when the worker is running, otherwise ``False``.
        """
        if self._worker is not None and self._worker.is_alive():
            return True

        if not self.connect():
            return False

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return True

    def stop(self, reset: bool = True, wait: bool = False) -> None:
        """Hard-stop the glow effect.

        Args:
            reset: Whether to turn all registered devices off after stopping.
            wait: Whether to block until the worker thread finishes.
        """
        self._stop_event.set()
        worker = self._worker
        if wait and worker is not None:
            worker.join()
        self._worker = None
        if reset:
            self._set_all_devices((0, 0, 0))

    def enter(self) -> None:
        """Fade in from black to idle presence."""
        if not self.start():
            return

        with self._lock:
            self._state = "entering"
            self._current_brightness = 0.0
            self._target_brightness = self.idle_brightness
            self._target_color = self.base_color.copy()

    def fatal(self) -> None:
        """Fade the glow into Celune's fixed fatal-error color."""
        if not self.start():
            return

        with self._lock:
            self._state = "fatal"
            self._target_brightness = max(self.idle_brightness, 0.2)
            self._target_color = self.fatal_color.copy()

    def sleep(self) -> None:
        """Fade the glow down to Celune's sleeping brightness."""
        if not self.start():
            return

        with self._lock:
            if self._state != "sleeping":
                self._sleep_restore_brightness = max(
                    self._target_brightness,
                    self._current_brightness,
                    self.idle_brightness,
                )
            self._state = "sleeping"
            self._target_brightness = self.idle_brightness * 0.25

    def wake(self) -> None:
        """Restore the glow brightness that was active before sleep."""
        if not self.start():
            return

        with self._lock:
            self._state = "waking"
            self._target_brightness = max(
                self._sleep_restore_brightness,
                self.idle_brightness,
            )

    def leave(self) -> None:
        """Fade out from current brightness to black and stop."""
        if self._worker is None or not self._worker.is_alive():
            return

        with self._lock:
            self._state = "leaving"
            self._target_brightness = 0.0
            self._target_color = self._target_color.copy()
            self.finished.clear()

    def schedule(self, audio: npt.NDArray[np.float32]) -> None:
        """Chop up audio into chunks and schedule glow activation.

        Args:
            audio: The audio to glow to.
        """
        if not self.start():
            return

        chunk_seconds = 1.0 / float(self.fps)
        chunks = _split(audio, BASE_SR, chunk_seconds)
        now = time.monotonic()
        offset = 0.0

        with self._lock:
            for chunk in chunks:
                duration = chunk.shape[0] / float(BASE_SR)
                self._scheduled_chunks.append((now + offset, chunk))
                offset += duration

    def glow(self, audio: npt.NDArray[np.float32]) -> None:
        """Update brightness target based on incoming audio chunk.

        Args:
            audio: The latest audio chunk used to estimate speaking intensity.
        """
        if not self.start():
            return

        self._process_glow_chunk(audio, time.monotonic())

    def reset_audio_reactivity(self) -> None:
        """Clear queued audio and fade the glow back to its idle brightness."""
        if self._worker is None or not self._worker.is_alive():
            return

        with self._lock:
            self._scheduled_chunks.clear()
            self._smoothed_level = 0.0
            if self._state not in {"fatal", "sleeping", "waking", "leaving", "none"}:
                self._state = "normal"
                self._target_brightness = self.idle_brightness

    def _process_glow_chunk(self, audio: npt.NDArray[np.float32], now: float) -> None:
        """Process one audio chunk and update audio-reactive brightness."""
        del now
        level = self._speech_level(audio)
        smoothing = float(np.clip(self.smoothing_factor, 0.0, 0.98))
        self._smoothed_level = (self._smoothed_level * smoothing) + (
            level * (1.0 - smoothing)
        )
        smoothed_level = float(np.clip(self._smoothed_level, 0.0, 1.0))

        with self._lock:
            self._state = "normal"
            self._target_brightness = float(
                np.clip(
                    self.idle_brightness
                    + (self.max_brightness - self.idle_brightness) * smoothed_level,
                    self.idle_brightness,
                    self.max_brightness,
                )
            )

    @staticmethod
    def _to_mono(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Convert stereo or multi-channel audio to mono."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    @staticmethod
    def _fix_color_rendering(rgb: RGB) -> RGB:
        """Compensate for LED green dominance and prevent channel clipping."""
        r, g, b = map(float, rgb)
        g *= 0.65
        r *= 1.12
        g *= 1.12
        b *= 1.12

        peak = max(r, g, b)
        if peak > 0xFF:
            scale = 0xFF / peak
            r *= scale
            g *= scale
            b *= scale

        return int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))

    def _speech_level(self, audio: npt.NDArray[np.float32]) -> float:
        """Calculate normalized audio activity level from RMS energy."""
        audio = self._to_mono(audio)
        if audio.size == 0:
            return 0.0

        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        level = np.clip(rms * self.input_gain, 0.0, 1.0)
        level = float(np.log1p(6.0 * level) / np.log1p(6.0))
        level = level ** (1.0 / self.gamma)
        return float(np.clip(level, 0.0, 1.0))

    def _set_all_devices(self, rgb: Union[RGB, npt.NDArray[np.floating]]) -> None:
        """Apply color to all registered OpenRGB devices."""
        rgb = np.clip(rgb, 0, 255).astype(int)
        color = RGBColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        for device in self.devices:
            with contextlib.suppress(Exception):
                device.set_color(color, fast=self.fast)

    def _run(self) -> None:
        """Interpolate brightness and push to hardware."""
        frame_sleep = 1.0 / self.fps

        while not self._stop_event.is_set():
            now = time.monotonic()

            scheduled_chunks = []

            with self._lock:
                while self._scheduled_chunks and self._scheduled_chunks[0][0] <= now:
                    _, chunk = self._scheduled_chunks.popleft()
                    scheduled_chunks.append(chunk)

            for chunk in scheduled_chunks:
                self._process_glow_chunk(chunk, now)

            with self._lock:
                state = self._state
                target = self._target_brightness

            if state == "entering":
                target = self.idle_brightness
                alpha = self.transition_rate
                self._current_brightness += (target - self._current_brightness) * alpha

                if self._current_brightness >= self.idle_brightness - 0.001:
                    self._current_brightness = self.idle_brightness
                    with self._lock:
                        if self._state == "entering":
                            self._state = "normal"

            elif state == "leaving":
                target = 0.0
                alpha = self.transition_rate
                self._current_brightness += (target - self._current_brightness) * alpha

                if self._current_brightness <= 0.001:
                    self._current_brightness = 0.0
                    self._set_all_devices((0, 0, 0))
                    self._stop_event.set()
                    self.finished.clear()
                    break

            elif state == "none":
                self._current_brightness = 0.0

            elif state == "fatal":
                target = max(target, self.idle_brightness, 0.2)
                alpha = self.transition_rate
                self._current_brightness += (target - self._current_brightness) * alpha
                self._current_brightness = float(
                    np.clip(self._current_brightness, 0.0, target)
                )

            elif state == "sleeping":
                target = self.idle_brightness * 0.25
                alpha = self.transition_rate
                self._current_brightness += (target - self._current_brightness) * alpha

            elif state == "waking":
                target = max(target, self.idle_brightness)
                alpha = self.transition_rate
                self._current_brightness += (target - self._current_brightness) * alpha

                if abs(self._current_brightness - target) <= 0.001:
                    self._current_brightness = target
                    with self._lock:
                        if self._state == "waking":
                            self._state = "normal"

            else:
                target = max(target, self.idle_brightness)
                alpha = max(self.transition_rate, min(0.25, frame_sleep * 6.0))
                self._current_brightness += (target - self._current_brightness) * alpha
                self._current_brightness = float(
                    np.clip(
                        self._current_brightness,
                        self.idle_brightness,
                        self.max_brightness,
                    )
                )

            self._current_color += (
                self._target_color - self._current_color
            ) * self.color_transition_rate
            current_rgb = self._current_color * self._current_brightness
            self._set_all_devices(current_rgb)
            time.sleep(frame_sleep)

        self._set_all_devices((0, 0, 0))
