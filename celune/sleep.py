# SPDX-License-Identifier: Apache-2.0
"""Celune core sleep mode handler methods."""

from __future__ import annotations

import contextlib
import gc
import threading

import torch

from . import celune as _core
from .config import config_value
from .constants import APP_NAME
from .exceptions import NotAvailableError
from .i18n import string, tagged_string
from .persona.impl import persona_enabled
from .utils import format_error_message
from .binding import install_class_functions

__all__ = (
    "_run_wake_background_jobs",
    "_sleep_config",
    "_start_wake_background_jobs",
    "enter_sleep_mode",
    "sleep_enabled",
    "sleep_timeout_seconds",
    "unload_normalizer_state",
    "wake_from_sleep",
)


def unload_normalizer_state(self, release_cuda_cache: bool = True) -> None:
    """Unload only CeluneNorm components and release unused memory.

    Args:
        release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
    """
    self._unload_normalizer_components()
    gc.collect()

    if release_cuda_cache and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def _sleep_config(self) -> tuple[bool, int, dict[str, bool]]:
    """Return sleep enablement, timeout, and unload settings."""
    sleep_config = config_value(self.config, "sleep", {})
    if isinstance(sleep_config, bool):
        return (
            sleep_config,
            10,
            {
                "persona": True,
                "normalizer": True,
                "tts": False,
                "vc": False,
            },
        )

    if not isinstance(sleep_config, dict):
        sleep_config = {}

    unload_config = sleep_config.get("unload", {})
    if not isinstance(unload_config, dict):
        unload_config = {}

    try:
        timeout = _core._config_int(sleep_config.get("timeout", 10), 10)
    except (TypeError, ValueError):
        timeout = 10

    unload_tts = bool(unload_config.get("tts", False))

    return (
        bool(sleep_config.get("enabled", False)),
        max(1, timeout),
        {
            "persona": bool(unload_config.get("persona", True)),
            "normalizer": bool(unload_config.get("normalizer", True)),
            "tts": unload_tts,
            "vc": bool(unload_config.get("vc", unload_tts)),
        },
    )


def sleep_enabled(self) -> bool:
    """Return whether automatic sleep mode is enabled.

    Returns:
        bool: Whether automatic sleep mode is enabled.
    """
    enabled, _, _ = self._sleep_config()
    return enabled


def sleep_timeout_seconds(self) -> float:
    """Return the configured idle timeout in seconds.

    Returns:
        float: The configured idle timeout in seconds.
    """
    _, timeout_minutes, _ = self._sleep_config()
    return timeout_minutes * 60.0


def enter_sleep_mode(self) -> bool:
    """Put Celune to sleep and unload models according to configuration.

    Returns:
        bool: Whether Celune was put to sleep.
    """
    enabled, _, unload = self._sleep_config()
    self.log(
        f"[SLEEP] enter requested enabled={enabled} sleeping={self.sleeping} "
        f"unload={unload}",
        loglevel="debug",
    )
    if not enabled or self.sleeping:
        return False

    with self.say_lock:
        if (
            self.locked
            or self.cur_state in {"generating", "speaking", "reloading"}
            or self._any_playback_active()
        ):
            return False
        self.sleeping = True
        self.loaded = False
        self.cur_state = "sleeping"
        self.glow.sleep()

    if not self._try_play_signal("sleeping"):
        self.log(
            "Could not play the sleeping signal.",
            "warning",
            loglevel="verbose",
        )

    self._ready_announced = False
    self.model_ready.clear()
    self.progress_callback(0, 1)

    with (
        self._wake_background_lock,
        self._model_lock,
    ):
        if unload["persona"]:
            self._unload_persona_state()

        if unload["tts"]:
            self.unload_runtime_state(
                include_normalizer=unload["normalizer"],
                include_vc=unload["vc"],
                close_backends=True,
                release_cuda_cache=False,
            )
            self.model_name = ""
        elif unload["normalizer"]:
            self.unload_normalizer_state(release_cuda_cache=False)

        if unload["vc"] and not unload["tts"] and self.vc_backend is not None:
            _core._dispose_backend(self.vc_backend, release_cuda_cache=False)

    self.model_ready.set()
    self.log("[SLEEP] enter complete model_ready=True", loglevel="debug")
    return True


def wake_from_sleep(self) -> bool:
    """Wake Celune and reload anything unloaded by sleep mode.

    Returns:
        bool: Whether Celune was woken up from sleep.

    Raises:
        NotAvailableError: Celune has no valid model ID to reload after waking up.
        WarmupError: Celune cannot warm up after waking up.
    """
    with self._wake_lock:
        self.log(
            f"[SLEEP] wake requested sleeping={self.sleeping}",
            loglevel="debug",
        )
        if not self.sleeping:
            return True

        _, _, unload = self._sleep_config()
        self.model_ready.clear()
        self.status_callback(string("status.waking_up"))
        self.progress_callback(None, None)
        self.cur_state = "waking"

        try:
            with self._model_lock:
                if self._is_voice_conversion_mode():
                    if self.vc_backend is None:
                        raise NotAvailableError(
                            "cannot wake without a configured voice conversion backend"
                        )
                    if unload["vc"] and self._recreate_vc_backend():
                        self.log("[SLEEP] Recreated VC backend", loglevel="verbose")
                    self.vc_backend.preload_models()
                else:
                    active_voice = self.current_voice or (
                        self.voices[0] if self.voices else None
                    )
                    if active_voice is None:
                        raise NotAvailableError("cannot wake without an active voice")

                    if unload["tts"] or self.model is None:
                        if unload["tts"] and self._recreate_tts_backend():
                            self.log(
                                "[SLEEP] Recreated TTS backend", loglevel="verbose"
                            )
                        model_id = self.backend.model_id_for_voice(active_voice)
                        self.log(
                            f"[SLEEP] Loading model: {model_id}",
                            loglevel="verbose",
                        )
                        self.model = self.backend.load_model(model_id)
                        self.model_name = model_id
                        if not self._warmup():
                            self._raise_warmup_error("warmup failed after sleep")

                is_voice_conversion = self._is_voice_conversion_mode()
                if (
                    not is_voice_conversion
                    and unload["persona"]
                    and persona_enabled(self.config)
                ):
                    self.vision = self._persona_conn()
                    self.persona_ready = False

                self.loaded = True
                self.sleeping = False
                self.cur_state = "idle"
                self.glow.wake()

            self.progress_callback(1, 1)
            self.status_callback(string("status.idle"))
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)
            self.log("[SLEEP] wake complete state=idle", loglevel="debug")
            if not is_voice_conversion:
                self._start_wake_background_jobs(unload)
            return True
        except Exception as error:
            self.fatal()
            self.log(
                format_error_message(
                    tagged_string("celune.wake_error", "WAKE ERROR"),
                    error,
                    self.log_level,
                ),
                "error",
            )
            self.status_callback(
                string("status.could_not_wake", app_name=APP_NAME), "error"
            )
            self.error_callback(string("status.could_not_wake", app_name=APP_NAME))
            self.progress_callback(0, 1)
            return False
        finally:
            self.model_ready.set()


def _start_wake_background_jobs(self, unload: dict[str, bool]) -> None:
    """Start optional wake-up restoration after TTS becomes ready."""
    existing = self._wake_background_thread
    if existing is not None and existing.is_alive():
        return

    thread = threading.Thread(
        target=self._run_wake_background_jobs,
        args=(unload,),
        daemon=True,
    )
    self._wake_background_thread = thread
    thread.start()


def _run_wake_background_jobs(self, unload: dict[str, bool]) -> None:
    """Restore optional wake-up resources without delaying TTS readiness."""
    try:
        with self._wake_background_lock:
            if self.exit_requested or self.sleeping:
                return

            if (
                unload["vc"]
                and self.vc_backend is not None
                and not self._is_voice_conversion_mode()
            ):
                with self._model_lock:
                    if self.exit_requested or self.sleeping:
                        return
                    if self._recreate_vc_backend():
                        self.log("[SLEEP] Recreated VC backend", loglevel="verbose")
                    self.vc_backend.preload_models()

            if unload["normalizer"] and self.use_normalization:
                self.load_normalizer()

            if not unload["persona"] or not persona_enabled(self.config):
                return

            with self._model_lock:
                vision = self.vision
                if vision is None or self.persona_ready or self.persona_loading:
                    return
                self.persona_loading = True
            self._load_persona_background(vision)
    except Exception as e:
        self.log(
            format_error_message(
                string("celune.wake_error"),
                e,
                self.log_level,
            ),
            "error",
        )
    finally:
        if self._wake_background_thread is threading.current_thread():
            self._wake_background_thread = None


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(target, {name: globals()[name] for name in __all__})
