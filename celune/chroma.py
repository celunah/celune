# SPDX-License-Identifier: Apache-2.0
"""Celune Razer Chroma and OpenRGB-compatible RGB glow effect."""

from __future__ import annotations

import os
import math
import time
import random
import datetime
import threading
import contextlib
from collections import deque
from typing import TYPE_CHECKING, Union, Optional

import numpy as np
import numpy.typing as npt
from openrgb import OpenRGBClient
from openrgb.utils import RGBColor

from .constants import BASE_SR
from .typing.common import RGB
from .theme.colors import ERROR
from .utils import to_rgb, discard, lunar_info, is_celune_day, range_interpolated

if TYPE_CHECKING:
    from .celune import Celune
    from .typing.aliases import AudioChunkBroad


class AudioRGBGlow:
    """OpenRGB-compatible speaking-aware glow effect."""

    _celune_fatal_wrapped: bool = False
    SPECIAL_EFFECTS: tuple[tuple[str, frozenset[tuple[int, int]]], ...] = (
        ("june", frozenset({(6, 2)})),
        ("halloween", frozenset({(10, 31)})),
        ("christmas", frozenset({(12, 24), (12, 25), (12, 26)})),
    )
    CHRISTMAS_COLORS = ("#baceff", "#ffbace", "#ceffba")
    JUNE_CYCLE_SECONDS = 12.0
    HALLOWEEN_PULSE_SECONDS = 4.12
    HALLOWEEN_FADE_SECONDS = 1.0

    def __init__(
        self,
        celune: Optional[Celune],
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
        self._last_device_rgb: Optional[RGB] = None

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
        current_date = datetime.datetime.now(datetime.UTC)
        if is_celune_day():
            self.glow_multiplier *= 3.0
        else:
            _, illumination, _ = lunar_info(current_date)
            self.glow_multiplier *= range_interpolated(illumination, 1.0, 2.0)

        self._base_idle_brightness = 0.05
        if not self.max_glow_forced:
            self._base_idle_brightness *= self.glow_multiplier
        else:
            self._base_idle_brightness *= 2.0 * 3.0  # max glow level

        self.idle_brightness = self._base_idle_brightness

        self.max_brightness = 1.0

        self.input_gain = 4.0
        self.gamma = 1.8
        self.smoothing_factor = 0.8
        self.fast = True

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = None
        self.finished.set()

        self._current_brightness = 0.0
        self._target_brightness = self.idle_brightness
        self._sleep_restore_brightness = self.idle_brightness
        self._smoothed_level = 0.0

        self._state = "none"
        self._speaking = False
        self._audio_reactivity_active = False
        self._manual_special_effect = False
        self._suppressed_effect_date: Optional[datetime.date] = None
        self._custom_color_override = False
        self._custom_color_transition_active = False
        self._custom_color_transition_start = self._current_color.copy()
        self._custom_color_transition_target = self._current_color.copy()
        self._custom_color_transition_elapsed = 0.0
        self._custom_color_transition_seconds = 0.0
        self._effect_date = datetime.date.today()
        self._special_effect = self._select_special_effect(self._effect_date)
        self._brightness_effect_elapsed = 0.0
        self._last_christmas_color: Optional[str] = None
        self._christmas_color_start = self._current_color.copy()
        self._christmas_color_target = self._current_color.copy()
        self._christmas_transition_elapsed = 0.0
        self._christmas_transition_seconds = 0.0
        self._christmas_hold_seconds = 0.0
        self._apply_special_effect_color()
        self._refresh_idle_brightness()
        self._target_brightness = self.idle_brightness
        self._sleep_restore_brightness = self.idle_brightness

    @classmethod
    def _select_special_effect(cls, current_date: datetime.date) -> Optional[str]:
        """Select the calendar effect for a local date, if one is active."""
        month_day = (current_date.month, current_date.day)
        return next(
            (name for name, dates in cls.SPECIAL_EFFECTS if month_day in dates),
            None,
        )

    def _default_brightness_multiplier(self) -> float:
        """Return the idle baseline used when speech starts."""
        if self._special_effect == "june":
            return 0.75
        if self._special_effect == "halloween":
            return 0.5
        return 1.0

    def _brightness_multiplier(self) -> float:
        """Return the current idle brightness multiplier for the active effect."""
        if self._special_effect == "june":
            phase = math.sin(
                2.0
                * math.pi
                * self._brightness_effect_elapsed
                / self.JUNE_CYCLE_SECONDS
            )
            return 0.75 + (0.25 * phase)

        if self._special_effect == "halloween":
            elapsed = self._brightness_effect_elapsed
            if elapsed < self.HALLOWEEN_PULSE_SECONDS:
                return 0.5
            pulse_age = elapsed % self.HALLOWEEN_PULSE_SECONDS
            if pulse_age < self.HALLOWEEN_FADE_SECONDS:
                return 1.0 - (0.5 * pulse_age / self.HALLOWEEN_FADE_SECONDS)
            return 0.5

        return 1.0

    def _refresh_idle_brightness(self) -> None:
        """Apply the active calendar multiplier to lunar-adjusted brightness."""
        self.idle_brightness = (
            self._base_idle_brightness * self._brightness_multiplier()
        )

    def _apply_special_effect_color(self) -> None:
        """Set the active seasonal color target without replacing the pack color."""
        if self._state == "fatal" or self._custom_color_override:
            return

        effect_color = None
        if self._special_effect == "june":
            effect_color = "#cebaff"
        elif self._special_effect == "halloween":
            effect_color = "#9d1017"
        if effect_color is not None:
            self._target_color = np.array(
                self._fix_color_rendering(to_rgb(effect_color)), dtype=np.float32
            )
            return

        if self._special_effect == "christmas":
            self._choose_christmas_color()
            return

        self._target_color = self.base_color.copy()
        self._last_christmas_color = None

    def _choose_christmas_color(self) -> None:
        """Start a randomized fade to a different Christmas color."""
        choices = [
            color
            for color in self.CHRISTMAS_COLORS
            if color != self._last_christmas_color
        ]
        color = random.choice(choices)
        self._last_christmas_color = color
        self._christmas_color_start = self._current_color.copy()
        self._christmas_color_target = np.array(
            self._fix_color_rendering(to_rgb(color)), dtype=np.float32
        )
        self._target_color = self._christmas_color_target.copy()
        self._christmas_transition_elapsed = 0.0
        self._christmas_transition_seconds = random.uniform(0.5, 2.0)
        self._christmas_hold_seconds = 0.0

    def _advance_christmas_color(self, elapsed: float) -> None:
        """Advance the Christmas color fade and hold timer."""
        if self._christmas_transition_seconds > 0.0:
            self._christmas_transition_elapsed = min(
                self._christmas_transition_seconds,
                self._christmas_transition_elapsed + elapsed,
            )
            amount = (
                self._christmas_transition_elapsed / self._christmas_transition_seconds
            )
            self._current_color = (
                self._christmas_color_start
                + (self._christmas_color_target - self._christmas_color_start) * amount
            )
            if self._christmas_transition_elapsed >= self._christmas_transition_seconds:
                self._current_color = self._christmas_color_target.copy()
                self._christmas_transition_seconds = 0.0
                self._christmas_hold_seconds = random.uniform(1.0, 3.0)
            return

        self._christmas_hold_seconds -= elapsed
        if self._christmas_hold_seconds <= 0.0:
            self._choose_christmas_color()

    def _advance_custom_color(self, elapsed: float) -> None:
        """Advance a user-requested color fade to its exact duration."""
        if not self._custom_color_transition_active:
            return

        self._custom_color_transition_elapsed = min(
            self._custom_color_transition_seconds,
            self._custom_color_transition_elapsed + elapsed,
        )
        amount = (
            self._custom_color_transition_elapsed
            / self._custom_color_transition_seconds
        )
        self._current_color = (
            self._custom_color_transition_start
            + (
                self._custom_color_transition_target
                - self._custom_color_transition_start
            )
            * amount
        )
        if (
            self._custom_color_transition_elapsed
            >= self._custom_color_transition_seconds
        ):
            self._current_color = self._custom_color_transition_target.copy()
            self._custom_color_transition_active = False

    def set_color(self, color: str, transition_seconds: float) -> None:
        """Fade to a custom color over a positive duration in seconds.

        Args:
            color: Hex color to use until the custom color is cleared.
            transition_seconds: Positive duration for the fade, in seconds.

        Raises:
            ValueError: If the color or transition duration is invalid.
        """
        if not math.isfinite(transition_seconds) or transition_seconds <= 0.0:
            raise ValueError("transition duration must be positive and finite")

        target_color = np.array(
            self._fix_color_rendering(to_rgb(color)), dtype=np.float32
        )
        with self._lock:
            if self._state == "fatal":
                return
            self._custom_color_override = True
            self._custom_color_transition_active = True
            self._custom_color_transition_start = self._current_color.copy()
            self._custom_color_transition_target = target_color.copy()
            self._custom_color_transition_elapsed = 0.0
            self._custom_color_transition_seconds = transition_seconds
            self._target_color = target_color.copy()

    def activate_special_effect(self, name: str) -> None:
        """Activate one built-in seasonal effect until it is reset.

        Args:
            name: Built-in effect name: ``halloween``, ``christmas``, or ``june``.

        Raises:
            ValueError: If ``name`` is not a built-in effect name.
        """
        if name not in {effect_name for effect_name, _ in self.SPECIAL_EFFECTS}:
            raise ValueError("unknown glow effect")

        with self._lock:
            if self._state == "fatal":
                return
            self._manual_special_effect = True
            self._suppressed_effect_date = None
            self._special_effect = name
            self._brightness_effect_elapsed = 0.0
            self._custom_color_override = False
            self._custom_color_transition_active = False
            self._apply_special_effect_color()
            self._refresh_idle_brightness()
            if not self._audio_reactivity_active:
                self._target_brightness = self.idle_brightness

    def reset_special_effects(self) -> None:
        """Cancel seasonal effects and restore the pack's default glow color."""
        current_date = datetime.date.today()
        with self._lock:
            self._effect_date = current_date
            self._manual_special_effect = False
            self._suppressed_effect_date = current_date
            self._special_effect = None
            self._brightness_effect_elapsed = 0.0
            self._custom_color_override = False
            self._custom_color_transition_active = False
            self._last_christmas_color = None
            if self._state != "fatal":
                self._target_color = self.base_color.copy()
            self._refresh_idle_brightness()
            self._target_brightness = self.idle_brightness

    def _sync_special_effect(self, current_date: datetime.date) -> None:
        """Refresh the selected calendar effect when the local date changes."""
        with self._lock:
            if current_date == self._effect_date:
                return

            self._effect_date = current_date
            if self._manual_special_effect:
                return
            if self._suppressed_effect_date == current_date:
                self._special_effect = None
            else:
                self._suppressed_effect_date = None
                self._special_effect = self._select_special_effect(current_date)
            self._brightness_effect_elapsed = 0.0
            self._apply_special_effect_color()
            self._refresh_idle_brightness()
            if not self._speaking:
                self._target_brightness = self.idle_brightness

    def _advance_idle_brightness(self, elapsed: float, state: str) -> None:
        """Advance calendar brightness only while idle audio is inactive."""
        if state not in {"normal", "entering", "waking"}:
            return
        if self._speaking or self._audio_reactivity_active:
            return

        self._brightness_effect_elapsed += elapsed
        self._refresh_idle_brightness()
        if state == "normal":
            self._target_brightness = self.idle_brightness

    def _set_color(self, color: str) -> None:
        """Update the selected pack's glow color and active target."""
        base_color = np.array(
            self._fix_color_rendering(to_rgb(color)), dtype=np.float32
        )
        with self._lock:
            self.base_color = base_color
            if (
                self._state != "fatal"
                and self._special_effect is None
                and not self._custom_color_override
            ):
                self._target_color = base_color.copy()

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
        self.finished.clear()
        self._worker = threading.Thread(
            target=self._run_with_completion,
            daemon=True,
        )
        self._worker.start()
        return True

    def _run_with_completion(self) -> None:
        """Run the glow worker and signal that it has stopped."""
        try:
            self._run()
        finally:
            self.finished.set()

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
            self._apply_special_effect_color()

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

        frames = max(1, round(BASE_SR / max(self.fps, 1)))
        now = time.monotonic()
        offset = 0.0

        with self._lock:
            for start in range(0, len(audio), frames):
                chunk = audio[start : start + frames]
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
            self._speaking = False
            self._audio_reactivity_active = False
            if self._state not in {"fatal", "sleeping", "waking", "leaving", "none"}:
                self._state = "normal"
                self._target_brightness = self.idle_brightness

    def begin_speaking(self) -> None:
        """Pause idle brightness effects and reset to their speech baseline."""
        if not self.start():
            return

        with self._lock:
            if self._state in {"fatal", "sleeping", "leaving", "none"}:
                return
            if self._speaking:
                return

            self._scheduled_chunks.clear()
            self._smoothed_level = 0.0
            self._speaking = True
            self._audio_reactivity_active = True
            baseline = (
                self._base_idle_brightness * self._default_brightness_multiplier()
            )
            self._current_brightness = baseline
            self._target_brightness = baseline
            self._state = "normal"

    def _process_glow_chunk(self, audio: npt.NDArray[np.float32], now: float) -> None:
        """Process one audio chunk and update audio-reactive brightness."""
        discard(now)
        level = self._speech_level(audio)
        smoothing = float(np.clip(self.smoothing_factor, 0.0, 0.98))
        self._smoothed_level = (self._smoothed_level * smoothing) + (
            level * (1.0 - smoothing)
        )
        smoothed_level = float(np.clip(self._smoothed_level, 0.0, 1.0))

        with self._lock:
            if self._state in {"fatal", "sleeping", "waking", "leaving", "none"}:
                return
            self._state = "normal"
            self._audio_reactivity_active = True
            baseline = (
                self._base_idle_brightness * self._default_brightness_multiplier()
                if self._speaking
                else self.idle_brightness
            )
            self._target_brightness = float(
                np.clip(
                    baseline + (self.max_brightness - baseline) * smoothed_level,
                    baseline,
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

        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))
        level = np.clip(rms * self.input_gain, 0.0, 1.0)
        level = float(np.log1p(6.0 * level) / np.log1p(6.0))
        level = level ** (1.0 / self.gamma)
        return float(np.clip(level, 0.0, 1.0))

    def _set_all_devices(self, rgb: Union[RGB, AudioChunkBroad]) -> None:
        """Apply color to all registered OpenRGB devices."""
        rgb = np.clip(rgb, 0, 255).astype(int)
        current_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        if current_rgb == self._last_device_rgb:
            return
        self._last_device_rgb = current_rgb
        color = RGBColor(*current_rgb)
        for device in self.devices:
            with contextlib.suppress(Exception):
                device.set_color(color, fast=self.fast)

    def _run(self) -> None:
        """Interpolate brightness and push to hardware."""
        frame_sleep = 1.0 / self.fps

        while not self._stop_event.is_set():
            now = time.monotonic()
            self._sync_special_effect(datetime.date.today())

            scheduled_chunks = []

            with self._lock:
                while self._scheduled_chunks and self._scheduled_chunks[0][0] <= now:
                    _, chunk = self._scheduled_chunks.popleft()
                    scheduled_chunks.append(chunk)

            for chunk in scheduled_chunks:
                self._process_glow_chunk(chunk, now)

            with self._lock:
                state = self._state
                self._advance_idle_brightness(frame_sleep, state)
                custom_color_active = self._custom_color_override and state not in {
                    "fatal",
                    "leaving",
                    "none",
                }
                if custom_color_active:
                    self._advance_custom_color(frame_sleep)
                elif self._special_effect == "christmas" and state not in {
                    "fatal",
                    "leaving",
                    "none",
                }:
                    self._advance_christmas_color(frame_sleep)
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
                brightness_floor = (
                    self._base_idle_brightness * self._default_brightness_multiplier()
                    if self._speaking
                    else self.idle_brightness
                )
                target = max(target, brightness_floor)
                alpha = max(self.transition_rate, min(0.25, frame_sleep * 6.0))
                self._current_brightness += (target - self._current_brightness) * alpha
                self._current_brightness = float(
                    np.clip(
                        self._current_brightness,
                        brightness_floor,
                        self.max_brightness,
                    )
                )

            if not custom_color_active and (
                self._special_effect != "christmas"
                or state in {"fatal", "leaving", "none"}
            ):
                self._current_color += (
                    self._target_color - self._current_color
                ) * self.color_transition_rate
            current_rgb = self._current_color * self._current_brightness
            self._set_all_devices(current_rgb)
            time.sleep(frame_sleep)

        self._set_all_devices((0, 0, 0))

    process_glow_chunk = _process_glow_chunk
    to_mono = _to_mono
    fix_color_rendering = _fix_color_rendering
    speech_level = _speech_level
    run = _run
