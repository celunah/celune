# SPDX-License-Identifier: MIT
"""Celune's backend layer."""

import os
import gc
import time
import queue
import threading
import contextlib
from pathlib import Path
from typing import Optional, Callable, Union, cast

import torch
import numpy as np
import numpy.typing as npt
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.logging import disable_progress_bar
from transformers.utils import logging as hf_logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .dataclasses.celune import (
    CELUNE_CONSTANT_PROPERTIES,
    CELUNE_FORWARDED_PROPERTIES,
    CeluneAudioState,
    CeluneBackendState,
    CeluneCallbackState,
    CeluneModelState,
    CelunePipelineState,
    CeluneRuntimeState,
    CeluneVoiceState,
)
from .dataclasses.properties import (
    bind_constant_properties,
    bind_forwarded_properties,
)
from .dataclasses.events import (
    CharacterChangedEvent,
    CharacterLoadedEvent,
    CharacterUnloadedEvent,
    ReadyEvent,
    ShutdownEvent,
    StateChangedEvent,
    VoiceChangedEvent,
)
from .chroma import AudioRGBGlow
from .backends.qwen3 import Qwen3
from .extensions.base import CeluneContext
from .extensions.events import EventDispatcher
from .extensions.manager import CeluneExtensionManager
from .config import Config, config_bool, config_value
from .paths import project_root
from .runtime import log_runtime_banner, validate_runtime
from .backends import CeluneBackend, resolve_backend
from .exceptions import NotAvailableError, WarmupError, BackendError
from .modeling import normalizer_device, load_normalizer_components
from .constants import APP_NAME, JSONSerializable, NORMALIZER_MODEL_ID
from .typing.celune import (
    CeluneStateAccessors,
    Generative,
    InputStateCallback,
    MessageCallback,
    NormalizerTokenizer,
    ProgressCallback,
    ReleasableObject,
    VoiceLockStateCallback,
)
from .typing.events import EventName, EventPayload
from .utils import format_number, format_error, discard, is_port_usable, custom_assert
from .vram import (
    QWEN3_0_6B_MODEL,
    VramPreset,
    backend_allowed,
    resolve_backend_name,
    resolve_vram_preset,
    validate_vram_preset,
)
from .persona.impl import (
    PersonaClient,
    create_persona_client,
    persona_enabled,
    persona_is_available,
    persona_model_id,
    persona_quantization,
)
from .cevoice import (
    announce_default_bundle,
    bundle_character_name,
    default_bundle_path,
    default_loader,
    persona_metadata_from_manifest,
    select_voice_bundle,
)
from .pipeline import (
    acquire_pipeline,
    clear_queue,
    close as close_pipeline,
    close_stream,
    force_stop_speech as force_stop_pipeline,
    generation_worker,
    queue_sfx_audio,
    playback_worker,
    play as play_pipeline,
    queue_speech,
    release_pipeline,
    say as say_pipeline,
    think as think_pipeline,
    split_text,
    play_signal,
)
from .typing.pipeline import SpeechStreamQueue


def _config_str(value: JSONSerializable) -> Optional[str]:
    """Return a config value only when it is a string."""
    return value if isinstance(value, str) else None


def _config_int(value: JSONSerializable, default: int) -> int:
    """Return a config value as an integer when it is scalar-like."""
    if isinstance(value, bool):
        raise TypeError("boolean is not an integer config value")
    if isinstance(value, (int, float, str)):
        return int(value)
    if value is None:
        return default
    raise TypeError("config value cannot be converted to int")


def _release_loaded_object(value: ReleasableObject) -> None:
    """Best-effort release hook for one loaded runtime object."""
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()
        return

    unload = getattr(value, "unload", None)
    if callable(unload):
        with contextlib.suppress(Exception):
            unload()


class Celune(CeluneStateAccessors):
    """The character engine for Celune."""

    _instance: Optional["Celune"] = None

    def __init__(
        self,
        config: Config,
        tts_backend: Optional[Union[str, CeluneBackend, type[CeluneBackend]]] = None,
        chunk_size: int = 0,  # defaulted to 0 because not all backends use this
        target_chunk_length: float = 0.64,
        language: str = "Auto",  # Qwen3 backend accepts a language, others may not
        log_callback: Optional[MessageCallback] = None,
        status_callback: Optional[MessageCallback] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
        queue_avail_callback: Optional[Callable[[], None]] = None,
        voice_changed_callback: Optional[Callable[[str], None]] = None,
        change_input_state_callback: Optional[InputStateCallback] = None,
        change_voice_lock_state_callback: Optional[VoiceLockStateCallback] = None,
        progress_callback: Optional[ProgressCallback] = None,
        dev: bool = False,
    ) -> None:
        if Celune._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")

        self._callbacks = CeluneCallbackState(
            log_callback=log_callback or self._noop_message,
            status_callback=status_callback or self._noop_message,
            error_callback=error_callback or (lambda error: None),
            idle_callback=idle_callback or (lambda: None),
            queue_avail_callback=queue_avail_callback or (lambda: None),
            voice_changed_callback=voice_changed_callback or (lambda name: None),
            change_input_state_callback=(
                change_input_state_callback or self._noop_input_state
            ),
            change_voice_lock_state_callback=(
                change_voice_lock_state_callback or self._noop_voice_lock_state
            ),
            progress_callback=(progress_callback or self._noop_progress),
        )
        self._event_dispatcher = EventDispatcher(log_warning=self.log, dev=dev)

        self._backend_state = CeluneBackendState(config=config)
        self._model_state = CeluneModelState()
        self._voice_state = CeluneVoiceState()
        self._pipeline_state = CelunePipelineState()
        self._audio_state = CeluneAudioState()
        self._runtime_state = CeluneRuntimeState()
        self._pipeline_state.model_ready.set()
        self._pipeline_state.playback_done.set()

        self.config = config
        select_voice_bundle(_config_str(config_value(config, "voice_bundle")))
        preset = resolve_vram_preset(config)

        if tts_backend is None:
            tts_backend = preset.default_backend

        backend_kwargs: dict[str, Optional[Union[bool, str]]] = {}
        if isinstance(tts_backend, CeluneBackend):
            if not backend_allowed(config, tts_backend.name):
                raise BackendError(
                    f"backend '{tts_backend.name}' is not available for VRAM tier '{preset.tier}'"
                )
        elif isinstance(tts_backend, type) and issubclass(tts_backend, CeluneBackend):
            backend_type_name = getattr(tts_backend, "name", "").strip().lower()
            if not backend_allowed(config, backend_type_name):
                raise BackendError(
                    f"backend '{backend_type_name}' is not available for VRAM tier '{preset.tier}'"
                )

        if isinstance(tts_backend, str):
            requested_backend = tts_backend
            tts_backend = resolve_backend_name(config, tts_backend)
            if tts_backend != requested_backend.strip().lower():
                self.log(
                    (
                        f"Backend '{requested_backend}' is not available for VRAM tier "
                        f"'{preset.tier}', using '{tts_backend}' instead."
                    ),
                    "warning",
                )

        if not isinstance(tts_backend, CeluneBackend) and (
            (isinstance(tts_backend, str) and tts_backend.strip().lower() == "qwen3")
            or (isinstance(tts_backend, type) and issubclass(tts_backend, Qwen3))
        ):
            backend_kwargs["x_vector_only"] = config_bool(
                config,
                "CELUNE_QWEN3_X_VECTOR_ONLY",
                "qwen3_x_vector_only",
            )
            backend_kwargs["clone_model_id"] = preset.qwen3_clone_model_id

        try:
            if not isinstance(tts_backend, CeluneBackend):
                self._backend_spec = tts_backend
                self._backend_kwargs = dict(backend_kwargs)
            self.backend = resolve_backend(
                tts_backend,
                log=self.log_callback,
                **backend_kwargs,
            )
            self._validate_backend_against_preset(self.backend, preset)
            self.tts_backend = self.backend.name
        except ValueError as e:
            raise BackendError(str(e)) from e
        except TypeError as e:
            raise BackendError(f"invalid backend specification: '{tts_backend}'") from e
        except ModuleNotFoundError as e:
            raise BackendError(
                f"backend '{tts_backend}' has unmet dependencies: '{e.name}'"
            ) from e
        except Exception as e:
            raise BackendError(f"internal backend error: {format_error(e, dev)}") from e

        if chunk_size:
            self.chunk_size = chunk_size
        else:
            # chunk length must be evenly divisible by target backend's base chunk size
            # e.g. if chunk rate = 12.5, then chunk length must be evenly divisible by 0.08s
            #
            # examples:
            # Qwen3 = length must be divisible by 0.08s (12.5 Hz)
            # VoxCPM2 = length must be divisible by 0.16s (6.25 Hz)
            multiple = target_chunk_length * self.backend.chunk_rate
            nearest = round(multiple)

            if abs(multiple - nearest) > 1e-6:
                raise BackendError(
                    f"invalid chunk length: {target_chunk_length}s is not divisible by {1 / self.backend.chunk_rate}s"
                )

            self.chunk_size = max(
                1, round(target_chunk_length / (1 / self.backend.chunk_rate))
            )

        self.language = language
        self.dev = dev
        self.use_normalization = config_bool(
            config, "CELUNE_NORMALIZE", "use_normalizer"
        )

        glow_color = "#cebaff"
        loader = default_loader()
        if loader is not None:
            theme = loader.bundle.metadata.get("theme")
            if isinstance(theme, dict):
                accent = theme.get("accent")
                if isinstance(accent, str):
                    glow_color = accent

                configured_glow = theme.get("glow_color")
                if isinstance(configured_glow, str):
                    glow_color = configured_glow

        self.glow = AudioRGBGlow(celune=self, color=glow_color)
        self._wrap_fatal_glow()
        self.glow.start()

        self.vision = self._persona_conn()

        Celune._instance = self

    bind_forwarded_properties(locals(), CELUNE_FORWARDED_PROPERTIES)
    bind_constant_properties(locals(), CELUNE_CONSTANT_PROPERTIES)

    @property
    def cur_state(self) -> str:
        """Return Celune's current runtime state.

        Returns:
            str: The current runtime-state label.
        """
        return self._runtime_state.cur_state

    @cur_state.setter
    def cur_state(self, value: str) -> None:
        """Store Celune's runtime state and emit transition events on change.

        Args:
            value: The new runtime-state label to store.
        """
        old_state = self._runtime_state.cur_state
        self._runtime_state.cur_state = value
        if old_state == value:
            return
        self._emit_event(
            "state_changed",
            StateChangedEvent(
                celune=self,
                old_state=old_state,
                new_state=value,
            ),
        )

    @staticmethod
    def _noop_message(msg: str, severity: str = "info") -> None:
        """Discard a message callback."""

    @staticmethod
    def _noop_input_state(locked: bool) -> None:
        """Discard an input lock-state callback."""

    @staticmethod
    def _noop_voice_lock_state(locked: bool) -> None:
        """Discard a voice lock-state callback."""

    @staticmethod
    def _noop_progress(progress: Optional[float], total: Optional[float]) -> None:
        """Discard a progress callback."""

    def _enter_fatal_error_state(self) -> None:
        """Mark the runtime as unrecoverably failed before fatal handlers run."""
        self.cur_state = "error"
        self.loaded = False
        self.locked = True
        self._ready_announced = False

    def _wrap_fatal_glow(self) -> None:
        """Ensure all fatal glow paths also stamp the runtime state as failed."""
        if getattr(self.glow, "_celune_fatal_wrapped", False):
            return

        original_fatal = self.glow.fatal

        def wrapped_fatal() -> None:
            self._enter_fatal_error_state()
            original_fatal()

        self.glow.fatal = wrapped_fatal
        setattr(self.glow, "_celune_fatal_wrapped", True)

    def _emit_event(self, event_name: EventName, event: EventPayload) -> None:
        """Dispatch one typed event through Celune's internal event bus."""
        self._event_dispatcher.emit(event_name, event)

    @staticmethod
    def _bundle_path_string(bundle: object) -> Optional[str]:
        """Return one bundle path as a string when it is available."""
        path = getattr(bundle, "path", None)
        if path is None:
            return None
        return str(path)

    def _emit_character_event_transition(
        self,
        old_character: Optional[str],
        old_bundle_path: Optional[str],
        new_character: Optional[str],
        new_bundle_path: Optional[str],
        new_is_default: bool,
    ) -> None:
        """Emit the appropriate character lifecycle event for one bundle transition."""
        if old_character and new_character:
            if old_character == new_character and old_bundle_path == new_bundle_path:
                return
            self._emit_event(
                "character_changed",
                CharacterChangedEvent(
                    celune=self,
                    old_character=old_character,
                    new_character=new_character,
                    old_bundle_path=old_bundle_path,
                    new_bundle_path=new_bundle_path,
                    new_is_default=new_is_default,
                ),
            )
            return

        if new_character:
            self._emit_event(
                "character_loaded",
                CharacterLoadedEvent(
                    celune=self,
                    character_name=new_character,
                    bundle_path=new_bundle_path,
                    is_default=new_is_default,
                ),
            )
            return

        if old_character:
            self._emit_event(
                "character_unloaded",
                CharacterUnloadedEvent(
                    celune=self,
                    character_name=old_character,
                    bundle_path=old_bundle_path,
                ),
            )

    @staticmethod
    def _validate_backend_against_preset(
        backend: CeluneBackend,
        preset: VramPreset,
    ) -> None:
        """Reject backend instances that bypass preset-specific runtime limits."""
        if (
            isinstance(backend, Qwen3)
            and backend.clone_model_id != preset.qwen3_clone_model_id
        ):
            raise BackendError(
                f"backend '{backend.name}' is not available with model "
                f"'{backend.clone_model_id}' for VRAM tier '{preset.tier}'"
            )

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        """Drain all pending items from a queue."""
        clear_queue(q)

    def _persona_conn(self) -> Optional[PersonaClient]:
        """Return a connection to the Persona runtime, if available."""

        if not persona_enabled(self.config):
            return None

        if not persona_is_available():
            self.log("Persona could not be initialized.", "warning")
            return None

        return create_persona_client(self.config, log_dev=self.log_dev)

    def _close_stream(self, abort: bool = False) -> None:
        """Close the current audio stream if one exists."""
        close_stream(self, abort=abort)

    def _unload_persona_state(self) -> None:
        """Release Persona runtime state and clear the active client."""
        vision = self.vision
        self.vision = None
        if vision is not None:
            with contextlib.suppress(Exception):
                vision.close()

    def _unload_normalizer_components(self) -> None:
        """Release normalizer model references and invalidate pending loads."""
        self._normalizer_load_epoch += 1
        llm = self.llm
        tokenizer = self.tokenizer
        self.llm = None
        self.tokenizer = None
        if llm is not None:
            _release_loaded_object(llm)
        if tokenizer is not None:
            _release_loaded_object(tokenizer)

    def unload_runtime_state(self, include_normalizer: bool = False) -> None:
        """Unload unused models to regain memory.

        Args:
            include_normalizer: Whether to also unload the normalization model and tokenizer.
        """
        discard(self, "model")

        self.backend.unload_model()

        if include_normalizer:
            self._unload_normalizer_components()

        gc.collect()

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def _recreate_tts_backend(self) -> bool:
        """Rebuild the TTS backend from its original constructor recipe."""
        if self._backend_spec is None:
            return False

        self.backend = resolve_backend(
            self._backend_spec,
            log=self.log_callback,
            **self._backend_kwargs,
        )
        self.tts_backend = self.backend.name
        return True

    def _raise_warmup_error(self, message: str) -> None:
        """Raise a Celune warmup error while preserving any original cause."""
        if self._last_warmup_error is not None:
            raise WarmupError(message) from self._last_warmup_error
        raise WarmupError(message)

    raise_warmup_error = _raise_warmup_error

    def unload_normalizer_state(self) -> None:
        """Unload only CeluneNorm components and release unused memory."""
        self._unload_normalizer_components()
        gc.collect()

        if torch.cuda.is_available():
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
                },
            )

        if not isinstance(sleep_config, dict):
            sleep_config = {}

        unload_config = sleep_config.get("unload", {})
        if not isinstance(unload_config, dict):
            unload_config = {}

        try:
            timeout = _config_int(sleep_config.get("timeout", 10), 10)
        except (TypeError, ValueError):
            timeout = 10

        return (
            bool(sleep_config.get("enabled", False)),
            max(1, timeout),
            {
                "persona": bool(unload_config.get("persona", True)),
                "normalizer": bool(unload_config.get("normalizer", True)),
                "tts": bool(unload_config.get("tts", False)),
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
        if not enabled or self.sleeping:
            return False

        with self.say_lock:
            if self.locked or self.cur_state in {"generating", "speaking", "reloading"}:
                return False
            self.sleeping = True
            self.loaded = False
            self.cur_state = "sleeping"
            self.glow.sleep()

        if not self._try_play_signal("sleeping"):
            self.log_dev("Could not play the sleeping signal.", "warning")

        self._ready_announced = False
        self.model_ready.clear()
        self.progress_callback(0, 1)

        with self._model_lock:
            if unload["persona"]:
                self._unload_persona_state()

            if unload["tts"]:
                self.unload_runtime_state(include_normalizer=unload["normalizer"])
                self.model_name = ""
            elif unload["normalizer"]:
                self.unload_normalizer_state()

        self.model_ready.set()
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
            if not self.sleeping:
                return True

            _, _, unload = self._sleep_config()
            self.model_ready.clear()
            self.status_callback("Waking up")
            self.progress_callback(None, None)
            self.cur_state = "waking"

            try:
                with self._model_lock:
                    active_voice = self.current_voice or (
                        self.voices[0] if self.voices else None
                    )
                    if active_voice is None:
                        raise NotAvailableError("cannot wake without an active voice")

                    if unload["tts"] or self.model is None:
                        if unload["tts"] and self._recreate_tts_backend():
                            self.log_dev("[SLEEP] Recreated TTS backend")
                        model_id = self.backend.model_id_for_voice(active_voice)
                        self.log_dev(f"[SLEEP] Loading model: {model_id}")
                        self.model = self.backend.load_model(model_id)
                        self.model_name = model_id
                        if not self._warmup():
                            self._raise_warmup_error("warmup failed after sleep")

                    if unload["normalizer"] and self.use_normalization:
                        self.load_normalizer()

                    if unload["persona"] and persona_enabled(self.config):
                        self.vision = self._persona_conn()
                        if self.vision is not None:
                            try:
                                self.vision.load(
                                    persona_model_id(self.config),
                                    persona_quantization(self.config),
                                )
                            except Exception as e:
                                self.log("Persona not initialized.", "warning")
                                self.log("Continuing in speech-only mode.", "warning")
                                self.log(format_error(e, self.dev), "warning")
                                self.vision.close()
                                self.vision = None

                    self.loaded = True
                    self.sleeping = False
                    self.cur_state = "idle"
                    self.glow.wake()

                self.progress_callback(1, 1)
                self.status_callback("Idle")
                self.change_input_state_callback(locked=False)
                self.change_voice_lock_state_callback(locked=len(self.voices) < 2)
                return True
            except Exception as e:
                self.cur_state = "error"
                self.loaded = False
                self.log(f"[WAKE ERROR] {format_error(e, self.dev)}", "error")
                self.glow.fatal()
                if not self._try_play_signal("error"):
                    self.log_dev("Could not play the error signal.", "warning")
                self.cur_state = "error"
                self.status_callback(f"{APP_NAME} could not wake", "error")
                self.progress_callback(0, 1)
                self.error_callback(f"{APP_NAME} could not wake")
                return False
            finally:
                self.model_ready.set()

    def set_voices(self, voices: tuple[str, ...]) -> None:
        """Configure Celune's voice information.

        Args:
            voices: The list of available voice names.
        """
        self.voices = voices

    def load_voice_bundle(self, bundle: Optional[Union[str, Path]] = None) -> bool:
        """Select and load a CEVOICE bundle into Celune's active voice set.

        Args:
            bundle: A built-in bundle name, explicit bundle path, or ``None`` to use Celune's default bundle.

        Returns:
            bool: ``True`` when a CEVOICE bundle was loaded, otherwise ``False``.
        """
        previous_loader = default_loader()
        previous_bundle_path = (
            self._bundle_path_string(previous_loader.bundle)
            if previous_loader is not None
            else None
        )
        previous_character = self.current_character
        select_voice_bundle(bundle)
        loader = default_loader()
        if loader is None:
            self.current_character_persona = None
            self.current_character = None
            self.voice_bundle_is_default = True
            voices = tuple(self.backend.voices)
            self.voices = voices
            self.current_voice = (
                self.backend.default_voice
                if self.backend.default_voice in voices
                else voices[0]
                if voices
                else None
            )
            self._emit_character_event_transition(
                previous_character,
                previous_bundle_path,
                None,
                None,
                True,
            )
            return bool(voices)

        new_bundle_path = self._bundle_path_string(loader.bundle)
        self.voice_bundle_is_default = loader.bundle.path == default_bundle_path()
        self.current_character_persona = persona_metadata_from_manifest(
            loader.bundle.metadata
        )
        self.current_character = bundle_character_name(loader.bundle)
        voices = loader.bundle.voice_order
        configured_default = loader.bundle.metadata.get("default_voice")
        preferred_voice = (
            configured_default
            if isinstance(configured_default, str)
            else self.backend.default_voice
        )
        self.voices = voices
        self.current_voice = (
            preferred_voice
            if preferred_voice in voices
            else voices[0]
            if voices
            else None
        )
        self._emit_character_event_transition(
            previous_character,
            previous_bundle_path,
            self.current_character,
            new_bundle_path,
            self.voice_bundle_is_default,
        )
        return bool(voices)

    def load_available_voices(self) -> bool:
        """Load the active voice set appropriate for the selected backend.

        Returns:
            bool: ``True`` when at least one voice is available.
        """
        if self.backend.uses_voice_bundles:
            return self.load_voice_bundle(
                _config_str(config_value(self.config, "voice_bundle"))
            )

        voices = tuple(self.backend.voices)
        self.voices = voices
        self.current_voice = (
            self.backend.default_voice
            if self.backend.default_voice in voices
            else voices[0]
            if voices
            else None
        )
        return bool(voices)

    def set_voice(self, name: str) -> bool:
        """Extension method for changing Celune's voice.

        Args:
            name: The voice name to load.

        Returns:
            bool: ``True`` when the reload thread was started, otherwise ``False``.
        """
        if name not in self.voices:
            # this voice was not found in the current CEVOICE/CECHAR pack
            self.log(f"Unknown voice: {name}", "warning")
            return False

        self.change_input_state_callback(locked=True)

        if not self._model_ready.is_set():
            self.log("Waiting for models to load...")
            self._model_ready.wait(timeout=5)

        self._model_ready.clear()
        self.loaded = False

        threading.Thread(
            target=self.change_voice,
            args=(name,),
            daemon=True,
        ).start()
        return True

    def set_voice_and_wait(self, name: str, timeout: float = 30.0) -> bool:
        """Change Celune's voice and wait until the reload finishes.

        Args:
            name: The voice name to load.
            timeout: How long to wait before considering the reload a failure.

        Returns:
            bool: ``True`` when the requested voice finished loading, otherwise ``False``.
        """
        if not self.set_voice(name):
            return False

        if not self._model_ready.wait(timeout=timeout):
            self.log("Timed out while processing a voice change.", "warning")
            return False
        return self.loaded and self.current_voice == name

    def _wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Wait until the model and playback pipeline are ready."""
        # don't wait a timeout while Celune is downloading a model
        ok = self._model_ready.wait(timeout=timeout)
        if not ok:
            self.log("Timed out while waiting to become ready.", "warning")
            self.log(
                "A possible reason for this may be a model download or high GPU activity.",
                "warning",
            )
            self.log(
                "This is not a fatal error, the utterance may be retried.", "warning"
            )
            return False

        if not self.loaded:
            self.log("Model was unloaded while waiting to become ready.", "warning")
            return False

        ok = self._playback_done.wait(timeout=timeout)
        if not ok:
            self.log(
                "Timed out while waiting for playback pipeline to become idle.",
                "warning",
            )
            return False

        with self._say_lock:
            return (not self.locked) and self.loaded

    wait_until_idle = _wait_until_idle

    def setup_extensions(self) -> None:
        """Configure Celune's extension manager."""
        ctx = CeluneContext(
            log=self.log,
            say=self.say,
            think=self.think,
            play=self.play,
            status=self.status_callback,
            set_voice=self.set_voice,
            get_state=lambda: self.cur_state,
            wait_until_ready=self._wait_until_idle,
            name=APP_NAME,
            version=__version__,
            dev=self.dev,
            log_dev=self.log_dev,
        )
        self.extension_manager = CeluneExtensionManager(ctx, self._event_dispatcher)
        self.extension_manager.autoload(str(project_root() / "extensions"))

        self.log_dev(
            f"[Core] Loaded extensions: {', '.join(self.extension_manager.list_extensions())}"
        )

    def log(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The message to emit.
            severity: The message severity level.
        """
        self.log_callback(msg, severity)

    def log_dev(self, msg: str, severity: str = "info") -> None:
        """Log a developer message.

        Args:
            msg: The message to emit.
            severity: The message severity level.
        """
        if self.dev:
            self.log_callback(msg, severity)

    def try_play_signal(self, signal_type: str) -> bool:
        """Public interface for Celune._try_play_signal.

        Args:
            signal_type: The signal type to play.

        Returns:
            bool: ``True`` when the requested signal was played, otherwise ``False``.

        Raises:
            ValueError: An invalid signal was requested.
        """
        return self._try_play_signal(signal_type)

    def _try_play_signal(self, signal_type: str) -> bool:
        """Play a runtime signal only when the playback pipeline can currently accept it."""
        playback_thread = self.playback_thread
        if playback_thread is None or not playback_thread.is_alive():
            return False

        if self.locked and self._playback_done.is_set():
            self._release_pipeline()

        return play_signal(self, signal_type)

    def voice_prompt_supported(self) -> bool:
        """Return whether the active TTS configuration supports voice prompts.

        Returns:
            bool: Whether the currently loaded TTS model supports voice prompting.
        """
        backend = self.backend
        return not (
            isinstance(backend, Qwen3) and backend.clone_model_id == QWEN3_0_6B_MODEL
        )

    def effective_voice_prompt(self) -> Optional[str]:
        """Return the active voice prompt only when the current model supports it.

        Returns:
            Optional[str]: The current voice prompt if voice prompts are supported, else ``None``.
        """
        if not self.voice_prompt_supported():
            return None
        return self.voice_prompt

    def change_voice(self, voice: str) -> None:
        """Change Celune's voice parameters.

        Args:
            voice: The voice name to load and warm up.

        Raises:
            WarmupError: The newly loaded voice fails warmup.
        """

        self.log(f"{APP_NAME} is reloading, please stand by...")
        self._ready_announced = False
        self.status_callback("Reloading")
        self.progress_callback(None, None)
        self.cur_state = "reloading"
        active_voice = self.current_voice or voice

        try:
            with self._model_lock:
                new_model_name = self.backend.model_id_for_voice(voice)

                # VoxCPM2 uses the same model for all voices, so we don't have to reload every time
                if new_model_name != self.model_name:
                    if not self._try_play_signal("working"):
                        self.log_dev("Could not play the working signal.", "warning")
                    self.log_dev(f"[RELOAD] Unloading model: {self.model_name}")
                    self.unload_runtime_state(include_normalizer=False)
                    self.log_dev(f"[RELOAD] Loading model: {new_model_name}")
                    self.model = self.backend.load_model(new_model_name)

                    self.log("Rewarming up...")
                    if not self._warmup():
                        self._raise_warmup_error("warmup failed after reload")

                    if not self._try_play_signal("readiness"):
                        self.log_dev("Could not play the readiness signal.", "warning")

                self.log_dev(
                    "[RELOAD] The target model is the same as the model currently in use."
                )

                self.current_voice = voice
                self.loaded = True

            self.voice_changed_callback(voice)
            if active_voice != voice:
                self._emit_event(
                    "voice_changed",
                    VoiceChangedEvent(
                        celune=self,
                        old_voice=active_voice,
                        new_voice=voice,
                    ),
                )
            self.log(f"Voice {voice} loaded.")
            self.progress_callback(1, 1)
            self.cur_state = "idle"
            self.status_callback("Idle")
        except Exception as e:
            self.cur_state = "error"
            self.loaded = False
            self.log(f"[RELOAD ERROR] {format_error(e, self.dev)}", "error")
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log_dev("Could not play the error signal.", "warning")
            self.status_callback(f"{APP_NAME} could not reload", "error")
            self.progress_callback(0, 1)
            self.error_callback(f"{APP_NAME} could not reload")
        finally:
            self._model_ready.set()
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def force_stop_speech(self) -> bool:
        """Forcefully stop Celune from speaking.

        Returns:
            bool: ``True`` when an active utterance was interrupted, otherwise ``False``.
        """
        return force_stop_pipeline(self)

    def load(self) -> bool:
        """Load and initialize Celune.

        Returns:
            bool: ``True`` when initialization completed successfully, otherwise ``False``.
        """
        disable_progress_bar()
        disable_progress_bars()
        hf_logging.set_verbosity_error()

        log_runtime_banner(self.log, self.backend.name)
        if not self.load_available_voices():
            self.cur_state = "error"
            self.log("No voices were loaded.", "error")
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log_dev("Could not play the error signal.", "warning")
            self.progress_callback(0, 1)
            self.error_callback("No voices loaded")
            return False

        if self.backend.uses_voice_bundles:
            announced_character = announce_default_bundle(self.log)
            character = self.current_character or announced_character
            self.current_character = character

            # the default pack's SHA256 hash is:
            # 22ff70762e7f6f3e734cc62c81c286f7482de6155b2394e7a6ddec1a892f63e0
            # please check it later, or else non-default packs named Celune will show the "default" tag
            if character == "Celune":
                self.log(f"Current character: {character} (default)")
            else:
                self.log(f"Current character: {character}")

        self.setup_extensions()

        vram_message = validate_vram_preset(self.config)
        if vram_message:
            self.log(vram_message, "warning")

        self.log(
            f"Current VRAM preset: {str(self.config.get('vram', 'unknown')).title()}"
        )

        self.progress_callback(None, None)
        self.backend.preload_models()

        self.log("All voices are available.")
        try:
            self.model = self.backend.load_default_model()
            active_voice = self.current_voice or self.voices[0]
            self.model_name = self.backend.model_id_for_voice(active_voice)
        except Exception as e:
            self.cur_state = "error"
            self.log(f"{APP_NAME} could not load the default model.", "error")
            self.log(format_error(e, self.dev), "error")
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log_dev("Could not play the error signal.", "warning")
            self.progress_callback(0, 1)
            self.error_callback("Default model failed to load")
            return False

        if self.vision is not None:
            self.log("Initializing Persona...")
            try:
                self.vision.load(
                    persona_model_id(self.config),
                    persona_quantization(self.config),
                )
            except Exception as e:
                self.log("Persona not initialized.", "warning")
                self.log("Continuing in speech-only mode.", "warning")
                self.log(format_error(e, self.dev), "warning")
                self.vision.close()
                self.vision = None
            else:
                self.log("Persona initialized.")

        generation_thread = threading.Thread(
            target=self._generation_worker, daemon=True
        )
        playback_thread = threading.Thread(target=self._playback_worker, daemon=True)

        self._generation_thread = generation_thread
        self._playback_thread = playback_thread

        generation_thread.start()
        playback_thread.start()

        if not validate_runtime(
            log=self.log,
            error=self.error_callback,
            set_state=lambda state: setattr(self, "cur_state", state),
            glow_connect_failed=self.glow.connect_failed,
            format_error=format_error,
            dev=self.dev,
            backend_name=self.backend.name,
        ):
            self.cur_state = "error"
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log_dev("Could not play the error signal.", "warning")
            return False

        if self._warmup():
            self.loaded = True
            self._model_ready.set()
            self._release_pipeline()
            self.glow.enter()  # Celune has entered your PC
        else:
            self.cur_state = "error"
            self.log("[WARMUP] Warmup failed.", "error")
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log("Could not play the error signal.", "warning")
            return False

        if self.use_normalization:
            self.load_normalizer()

        self._start_configured_api()

        if persona_enabled(self.config) and not persona_is_available():
            self.log(
                f"Personas are unavailable. {APP_NAME} is operating in speech-only mode.",
                "warning",
            )

        # notify readiness
        if not self._try_play_signal("readiness"):
            self.log_dev("Could not play the readiness signal.", "warning")

        self._emit_event("ready", ReadyEvent(celune=self))

        return True

    def _api_settings(self) -> tuple[bool, str, int, Optional[str], int]:
        """Resolve API settings from Celune's configuration."""
        api_config = config_value(self.config, "api", {})

        if isinstance(api_config, bool):
            return api_config, "0.0.0.0", 2060, None, 60

        if api_config is None:
            return False, "0.0.0.0", 2060, None, 60

        if not isinstance(api_config, dict):
            return bool(api_config), "0.0.0.0", 2060, None, 60

        enabled = bool(api_config.get("enabled", True))
        host = str(api_config.get("host", "0.0.0.0"))
        token_value = api_config.get("token")
        token = str(token_value).strip() if token_value is not None else None
        if not token:
            self.log(
                f"No API token set. {APP_NAME} API will bind only to the local network.",
                "warning",
            )
            token = None
            host = "127.0.0.1"
        try:
            port = _config_int(api_config.get("port", 2060), 2060)
        except (TypeError, ValueError):
            invalid_port = api_config.get("port", 2060)
            self.log(
                f"{APP_NAME} API port ({invalid_port}) is invalid, will use 2060 instead.",
                "warning",
            )
            port = 2060

        if not 1 <= port <= 65535:
            self.log(
                f"{APP_NAME} API port ({port}) is out of range, will use 2060 instead.",
                "warning",
            )
            port = 2060

        try:
            requests_per_minute = _config_int(
                api_config.get("rate_limit_per_minute", 60),
                60,
            )
        except (TypeError, ValueError):
            invalid_ratelimit = api_config.get("rate_limit_per_minute", 60)
            self.log(
                f"{APP_NAME} API rate limit ({invalid_ratelimit}) is invalid, using 60/min.",
                "warning",
            )
            requests_per_minute = 60

        return enabled, host, port, token, max(0, requests_per_minute)

    api_settings = _api_settings

    def _start_configured_api(self) -> None:
        """Start the API from config without blocking Celune startup."""
        enabled, host, port, token, requests_per_minute = self._api_settings()
        if not enabled or self._api_thread is not None:
            return

        if not is_port_usable(port):
            self.log(f"Port {port} is unavailable.", "warning")
            self.log(f"{APP_NAME} API will not be available.", "warning")
            return

        try:
            from .api import start_api
        except ModuleNotFoundError as package:
            self.log(
                f"A required package ({package.name}) isn't installed.",
                "warning",
            )
            self.log(f"{APP_NAME} API will not be available.", "warning")
            return
        except Exception as e:
            self.log(f"Package import failed: {format_error(e, self.dev)}", "warning")
            self.log(f"{APP_NAME} API will not be available.", "warning")
            return

        try:
            self._api_thread = start_api(
                self,
                host=host,
                port=port,
                token=token,
                requests_per_minute=requests_per_minute,
            )
        except Exception as e:
            self.log(
                f"An internal error occurred: {format_error(e, self.dev)}", "warning"
            )
            self.log(f"{APP_NAME} API will not be available.", "warning")
            return

    def load_normalizer(self) -> None:
        """Load the normalizer LLM."""
        load_epoch = self._normalizer_load_epoch + 1
        self._normalizer_load_epoch = load_epoch

        def _worker():
            loaded_tokenizer: Optional[PreTrainedTokenizerBase] = None
            loaded_llm: Optional[PreTrainedModel] = None

            discard(loaded_tokenizer)
            discard(loaded_llm)
            try:
                loaded_tokenizer, loaded_llm = load_normalizer_components(
                    self.log, self.backend, self.config
                )
                with self._model_lock:
                    if (
                        self._normalizer_load_epoch != load_epoch
                        or self.sleeping
                        or self.exit_requested
                    ):
                        discard(loaded_llm)
                        discard(loaded_tokenizer)
                        gc.collect()
                        if torch.cuda.is_available():
                            with contextlib.suppress(Exception):
                                torch.cuda.synchronize()
                            with contextlib.suppress(Exception):
                                torch.cuda.empty_cache()
                        self.log_dev("[NORMALIZER] Discarded stale normalizer load.")
                        return

                    self.tokenizer = loaded_tokenizer
                    self.llm = loaded_llm
                self.log("Normalizer loaded.")
                self.progress_callback(1, 1)
            except Exception as e:
                self.log(f"[NORMALIZER ERROR] {format_error(e, self.dev)}", "error")
                self.log("Normalizer failed to load.", "warning")
                self.log("Normalization will not be available.", "warning")
                self.progress_callback(0, 1)

        if self.vision is not None:
            # we don't need to normalize out of the VLM
            return

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self.progress_callback(None, None)
        self.log(
            f"Loading normalizer {NORMALIZER_MODEL_ID} on "
            f"{normalizer_device(self.config)}..."
        )

    def _warmup(self) -> bool:
        """Warm up Celune's speech capabilities."""
        self.log("[WARMUP] Warming up...")
        self.status_callback("Warming up")
        self.progress_callback(None, None)
        warmup_text = "A"
        self._last_warmup_error = None

        forced_error = os.getenv("CELUNE_FORCE_ERROR") in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }

        if forced_error:
            raise WarmupError("forced warmup failure")

        try:
            warmup_start = time.perf_counter()

            with self._model_lock:
                if self.model is None:
                    raise WarmupError("cannot warm up a null model")

                for _, _, _ in self.backend.generate_stream(
                    self.model,
                    text=warmup_text,
                    language=self.language,
                    chunk_size=self.chunk_size,
                    instruct=self.effective_voice_prompt(),
                    voice=self.current_voice,
                ):
                    pass

            warmup_end = time.perf_counter()
            warmup_took = warmup_end - warmup_start
            self.log_dev(f"[WARMUP] done, took {format_number(warmup_took, 2)} seconds")

            self.progress_callback(1, 1)
            return True
        except Exception as e:
            self.cur_state = "error"
            self._last_warmup_error = e
            self.log(f"[WARMUP ERROR] {format_error(e, self.dev)}", "error")
            self.cur_state = "error"
            self.glow.fatal()
            if not self._try_play_signal("error"):
                self.log_dev("Could not play the error signal.", "warning")
            self.progress_callback(0, 1)
            self.error_callback(f"{APP_NAME} could not warm up")
            return False

    # as of CeluneNorm 2.0, normalization ACTUALLY works with long inputs
    def normalize(self, text: str) -> Optional[str]:
        """Normalize input text using CeluneNorm.

        Args:
            text: The raw text to normalize before speech generation.

        Returns:
            Optional[str]: The normalized text, the original text for blank input, or ``None`` when normalization is
            unavailable or has failed.
        """

        if not self.use_normalization:
            return None

        if not text or not text.strip():
            return text

        if self.llm is None or self.tokenizer is None:
            return None

        llm = cast(Generative, self.llm)
        tokenizer = cast(NormalizerTokenizer, self.tokenizer)

        def _run_inference() -> Optional[str]:
            inf_start = time.perf_counter()
            try:
                bad_text = text.strip()
                norm_token = "<NORM>"

                # Are we using CeluneNorm?
                norm_token_id = tokenizer.convert_tokens_to_ids(norm_token)
                custom_assert(
                    norm_token_id is not None, ValueError("not a CeluneNorm normalizer")
                )
                assert norm_token_id is not None

                custom_assert(
                    norm_token_id != tokenizer.unk_token_id,
                    ValueError("not a CeluneNorm normalizer"),
                )
                assert norm_token_id != tokenizer.unk_token_id

                prompt = f"{bad_text}{norm_token}"

                tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                device = next(llm.parameters()).device
                inputs = tokens.to(device)
                token_ids = cast(torch.Tensor, tokens["input_ids"])
                len_tokens = token_ids.shape[1]

                self.log(f"Tokens to normalize: {len_tokens}")
                if len_tokens > 512:
                    self.log("Input is too long to normalize.", "warning")
                    return None

                with torch.inference_mode():
                    output_ids = llm.generate(
                        **inputs,
                        # CeluneNorm will likely return less, unless you use up your whole context allowance
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_ids = cast(torch.Tensor, inputs["input_ids"])
                prompt_len = input_ids.shape[1]
                new_ids = output_ids[0][prompt_len:]

                # CeluneNorm shouldn't do this, but if it does happen, stop Celune from saying nothing
                if new_ids.numel() == 0:
                    self.log("Normalizer returned no tokens.", "warning")
                    return None

                out = tokenizer.decode(new_ids, skip_special_tokens=True)

                # fix type checker
                if isinstance(out, list):
                    out = out[0] if out else ""

                # too many <NORM>'s can break splitting
                if "<NORM>" in out:
                    out = out.split("<NORM>", 1)[0].strip()

                # are we absolutely sure CeluneNorm did produce something before Celune gets to say it?
                if not out:
                    self.log("Normalizer did not produce normal output.", "warning")
                    return None

                inf_total = time.perf_counter() - inf_start
                self.log(f"Normalized text: {out}")
                self.log(f"Normalization took {format_number(inf_total, 2)} seconds.")

                return out

            except Exception as e:
                self.log(
                    f"[NORMALIZATION ERROR] {format_error(e, self.dev)}",
                    "error",
                )
                return None

        return _run_inference()  # blocks the generation thread, but Celune doesn't mind it since the main thread is up

    def _acquire_pipeline(self, action: str) -> bool:
        """Atomically claim Celune's shared playback pipeline."""
        return acquire_pipeline(self, action)

    def _release_pipeline(self) -> None:
        """Release Celune's shared playback pipeline."""
        release_pipeline(self)

    def think(
        self,
        text: str,
    ) -> bool:
        """Let Celune reply to an input request.

        Args:
            text: The request that will be sent to Persona for processing.

        Returns:
            bool: ``True`` if Celune processed this smart request, otherwise ``False``.
        """
        if self.is_in_tutorial:
            self.log("Speech input is disabled during the tutorial.", "warning")
            return False

        if self.sleeping:
            self.log(f"Cannot think while {APP_NAME} is sleeping.", "warning")
            self.error_callback(f"{APP_NAME} is currently sleeping")
            return False

        with self.say_lock:
            if self.locked or self.cur_state in {"generating", "speaking"}:
                self.log(f"Tried to think while {APP_NAME} was busy.", "warning")
                self.error_callback(f"{APP_NAME} is currently busy")
                return False

        self.status_callback("Thinking")
        self.cur_state = "thinking"
        self.progress_callback(None, None)
        self._ready_announced = False
        thread = threading.Thread(
            target=self._think_worker,
            args=(text,),
            daemon=True,
        )
        self._persona_thread = thread
        thread.start()
        return True

    def _think_worker(self, text: str) -> None:
        """Fetch a Persona response without blocking Celune's UI thread."""

        if not self.vision:
            self.vision = self._persona_conn()
            if not self.vision:
                self.say(text)
                return

        if not think_pipeline(self, text):
            self.log("Will say the input instead.", "warning")
            self.say(text)

    def say(
        self,
        text: str,
        save: bool = True,
        display_text: Optional[str] = None,
    ) -> bool:
        """Queue text for Celune to say.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.
            display_text: Optional text to show in logs instead of the synthesis text.

        Returns:
            bool: ``True`` when the text was queued successfully, otherwise ``False``.
        """
        return say_pipeline(self, text, save=save, display_text=display_text)

    def say_stream(self, text: str, save: bool = True) -> Optional[SpeechStreamQueue]:
        """Queue text for playback and mirror generated chunks to a queue.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            Optional[queue.Queue]: Queue receiving 48 kHz stereo float32 chunks, or ``None`` when the request could not
            be queued.
        """
        stream_queue: SpeechStreamQueue = queue.Queue(maxsize=2)
        if not queue_speech(self, text, save=save, stream_queue=stream_queue):
            return None
        return stream_queue

    def play(self, sound_path: str, keep: bool = False, volume: float = 1.0) -> bool:
        """Play a sound via Celune's pipeline.

        Args:
            sound_path: The path to the audio file to play.
            keep: Whether to prepend this SFX to the next saved utterance.
            volume: How loud should the SFX be played at.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise ``False``.
        """
        return play_pipeline(self, sound_path, keep=keep, volume=volume)

    def play_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        label: str = "uploaded SFX",
        keep: bool = False,
    ) -> bool:
        """Play decoded audio via Celune's pipeline.

        Args:
            audio: Decoded mono or stereo audio.
            sample_rate: Source sample rate for the decoded audio.
            label: Human-readable label for logs and status.
            keep: Whether to prepend this SFX to the next saved utterance.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise ``False``.
        """
        return queue_sfx_audio(self, audio, sample_rate, label, keep=keep)

    def close(self) -> None:
        """Shut off Celune and release loaded runtime state."""
        self._emit_event("shutdown", ShutdownEvent(celune=self))
        try:
            close_pipeline(self)
            self._unload_persona_state()
            with self._model_lock:
                self.unload_runtime_state(include_normalizer=True)
        finally:
            Celune._instance = None

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        return split_text(self, text)

    def _generation_worker(self) -> None:
        """Generate audio tokens and send them to the audio pipeline."""
        generation_worker(self)

    def _playback_worker(self) -> None:
        """Receive audio chunks and play them."""
        playback_worker(self)
