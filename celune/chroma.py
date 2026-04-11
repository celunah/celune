# pylint: disable=R0902
"""Celune Razer Chroma and OpenRGB-compatible RGB glow effect."""

import time
import threading
import contextlib

import numpy as np
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor

from .utils import to_rgb


class AudioRGBGlow:
    """OpenRGB-compatible speaking-aware glow effect."""

    def __init__(self, color, host="127.0.0.1", port=6742):
        self.color = np.array(
            self._fix_color_rendering(to_rgb(color)), dtype=np.float32
        )

        self.host = host
        self.port = port
        self.connect_failed = False
        self.finished = threading.Event()
        self.client = None
        self.devices = []

        self.speech_threshold = 0.06
        self._level_history = np.zeros(3, dtype=np.float32)

        self.hold_duration = 1.25
        self.fade_in_rate = 0.03
        self.fade_out_rate = 0.02
        self.fps = 60

        self.transition_rate = 0.02

        self.idle_brightness = 0.05
        self.max_brightness = 1.0

        self.input_gain = 4.0
        self.gamma = 1.4
        self.fast = True

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None

        self._current_brightness = 0.0
        self._target_brightness = self.idle_brightness
        self._last_speech_time = 0.0

        self._state = "none"

    def connect(self) -> bool:
        """Connect to the OpenRGB backend and initialize devices."""
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
        except TimeoutError:
            self.client = None
            self.connect_failed = True
            self.devices = []
            return False

    def start(self) -> bool:
        """Start the glow effect worker thread."""
        if self._worker is not None and self._worker.is_alive():
            return True

        if not self.connect():
            return False

        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return True

    def stop(self, reset=True, wait=False) -> None:
        """Hard-stop the glow effect."""
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

    def leave(self) -> None:
        """Fade out from current brightness to black and stop."""
        if self._worker is None or not self._worker.is_alive():
            return

        with self._lock:
            self._state = "leaving"
            self._target_brightness = 0.0
            self.finished.clear()

    def glow(self, audio) -> None:
        """Update brightness target based on incoming audio chunk."""
        if not self.start():
            return

        level = self._speech_level(audio)
        now = time.monotonic()

        self._level_history[:-1] = self._level_history[1:]
        self._level_history[-1] = level
        smoothed_level = float(np.mean(self._level_history))

        with self._lock:
            if smoothed_level > self.speech_threshold:
                self._state = "normal"
                self._target_brightness = self.max_brightness
                self._last_speech_time = now

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert stereo/multi-channel audio to mono."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    @staticmethod
    def _fix_color_rendering(rgb: tuple) -> tuple[int, int, int]:
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

    def _speech_level(self, audio: np.ndarray) -> float:
        """Calculate normalized speech activity level."""
        audio = self._to_mono(audio)
        if audio.size == 0:
            return 0.0

        amp = float(np.mean(np.abs(audio), dtype=np.float64))
        level = np.clip(amp * self.input_gain, 0.0, 1.0)
        level = level ** (1.0 / self.gamma)
        return float(np.clip(level, 0.0, 1.0))

    def _set_all_devices(self, rgb) -> None:
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

            with self._lock:
                state = self._state
                target = self._target_brightness
                last_speech = self._last_speech_time

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
                self._set_all_devices((0, 0, 0))

            else:
                if now - last_speech > self.hold_duration:
                    target = self.idle_brightness

                alpha = (
                    self.fade_in_rate
                    if target > self._current_brightness
                    else self.fade_out_rate
                )
                self._current_brightness += (target - self._current_brightness) * alpha
                self._current_brightness = float(
                    np.clip(
                        self._current_brightness,
                        self.idle_brightness,
                        self.max_brightness,
                    )
                )

            current_rgb = self.color * self._current_brightness
            self._set_all_devices(current_rgb)
            time.sleep(frame_sleep)

        self._set_all_devices((0, 0, 0))
