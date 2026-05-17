# SPDX-License-Identifier: MIT
"""Celune's backend layer."""

import gc
import time
import queue
import threading
import contextlib
from pathlib import Path
from typing import Any, Optional, Callable, Protocol, Union

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging
from transformers.utils.logging import disable_progress_bar
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .backends import CeluneBackend, resolve_backend
from .backends.qwen3 import Qwen3
from .cevoice import announce_default_bundle, default_loader, select_voice_bundle
from .config import config_bool, config_value
from .constants import NORMALIZER_MODEL_ID, PipelineStates
from .dsp import StreamingPedalboardReverb
from .utils import format_number, format_error, discard, is_port_usable, custom_assert
from .chroma import AudioRGBGlow
from .exceptions import NotAvailableError, WarmupError, BackendError
from .extensions.base import CeluneContext
from .extensions.manager import CeluneExtensionManager
from .modeling import load_normalizer_components
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
    split_text,
    play_readiness_signal,
)
from .runtime import log_runtime_banner, validate_runtime


class MessageCallback(Protocol):
    """Callback accepting a message and optional severity."""

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Handle a message emitted by Celune.

        Args:
            msg: Message text to handle.
            severity: Severity label associated with the message.

        Returns:
            None: Implementations consume or forward the message.

        Raises:
            NotImplementedError: The protocol placeholder is called directly.
        """
        raise NotImplementedError


class InputStateCallback(Protocol):
    """Callback accepting either positional or named lock state."""

    def __call__(self, locked: bool) -> None:
        """Handle input lock-state changes.

        Args:
            locked: Whether input should be treated as locked.

        Returns:
            None: Implementations update their input state.

        Raises:
            NotImplementedError: The protocol placeholder is called directly.
        """
        raise NotImplementedError


class VoiceLockStateCallback(Protocol):
    """Callback accepting either positional or named lock state."""

    def __call__(self, locked: bool) -> None:
        """Handle voice lock-state changes.

        Args:
            locked: Whether voice changes should be treated as locked.

        Returns:
            None: Implementations update their voice lock state.

        Raises:
            NotImplementedError: The protocol placeholder is called directly.
        """
        raise NotImplementedError


class ProgressCallback(Protocol):
    """Callback accepting progress and total values."""

    def __call__(self, progress: Optional[float], total: Optional[float]) -> None:
        """Handle a progress update emitted by Celune.

        Args:
            progress: Current progress, or ``None`` for an indeterminate bar.
            total: Total progress, or ``None`` for an indeterminate bar.

        Returns:
            None: Implementations update their progress display.

        Raises:
            NotImplementedError: The protocol placeholder is called directly.
        """
        raise NotImplementedError


class Celune:
    """The character engine for Celune."""

    _instance: Optional["Celune"] = None

    def __init__(
        self,
        config: dict[str, Any],
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

        Celune._instance = self

        if tts_backend is None:
            raise BackendError("no backend set")

        self.log_callback: MessageCallback = log_callback or self._noop_message
        self.status_callback: MessageCallback = status_callback or self._noop_message
        self.error_callback = error_callback or (lambda error: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.voice_changed_callback = voice_changed_callback or (lambda name: None)
        self.change_input_state_callback: InputStateCallback = (
            change_input_state_callback or self._noop_input_state
        )
        self.change_voice_lock_state_callback: VoiceLockStateCallback = (
            change_voice_lock_state_callback or self._noop_voice_lock_state
        )
        self.progress_callback: ProgressCallback = (
            progress_callback or self._noop_progress
        )

        self.config = config
        select_voice_bundle(config_value(config, "voice_bundle"))

        backend_kwargs = {}

        if not isinstance(tts_backend, CeluneBackend) and (
            (isinstance(tts_backend, str) and tts_backend.strip().lower() == "qwen3")
            or (isinstance(tts_backend, type) and issubclass(tts_backend, Qwen3))
        ):
            backend_kwargs["mode"] = config_value(config, "qwen3_mode", "clone")
            backend_kwargs["x_vector_only"] = config_bool(
                config,
                "CELUNE_QWEN3_X_VECTOR_ONLY",
                "qwen3_x_vector_only",
            )

        try:
            self.backend = resolve_backend(
                tts_backend,
                log=self.log_callback,
                **backend_kwargs,
            )
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

        self.model: Optional[PreTrainedModel] = None
        self.model_name = ""
        self.llm: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        self.current_voice: Optional[str] = None
        self.current_character: Optional[str] = None
        self.voices: tuple[str, ...] = ()
        self.voice_prompt: Optional[str] = None

        self.text_queue: queue.Queue[Any] = queue.Queue()
        self.audio_queue: queue.Queue[Any] = queue.Queue()

        self._playback_thread: Optional[threading.Thread] = None
        self._generation_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None

        self._queue_lock = threading.Lock()
        self._utterance_force_stop = threading.Event()
        self.regenerate = False

        self._stream: Optional[sd.OutputStream] = None
        self._current_sr: Optional[int] = None
        self._audio_unavailable = False
        self.can_use_rubberband = True
        self.speed: float = 1.0
        self.reverb = StreamingPedalboardReverb()

        self.locked = True
        self.loaded = False
        self.recently_saved: Optional[str] = None
        self.kept_sfx_audio: Optional[npt.NDArray[np.float32]] = None

        self._last_flavor: Optional[str] = None
        self._model_ready = threading.Event()
        self._model_ready.set()
        self._exit_requested = False
        self._playback_done = threading.Event()
        self._playback_done.set()
        self._say_lock = threading.Lock()
        self._model_lock = threading.RLock()

        self.cur_state = "init"
        self.is_in_tutorial = False

        self.dev = dev
        self.use_normalization = config_bool(
            config, "CELUNE_NORMALIZE", "use_normalizer"
        )

        self.extension_manager: Optional[CeluneExtensionManager] = None

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
        self.glow = AudioRGBGlow(color=glow_color)
        self.glow.start()

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

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        """Drain all pending items from a queue.

        Args:
            q: The queue to empty.

        Returns:
            None: This helper removes all currently pending items.
        """
        clear_queue(q)

    def _close_stream(self, abort: bool = False) -> None:
        """Close the current audio stream if one exists.

        Args:
            abort: Whether to abort immediately instead of stopping gracefully.

        Returns:
            None: This helper closes Celune's active output stream.
        """
        close_stream(self, abort=abort)

    def unload_runtime_state(self, include_normalizer: bool = False) -> None:
        """Unload unused models to regain memory.

        Args:
            include_normalizer: Whether to also unload the normalization model and
                tokenizer.

        Returns:
            None: This method clears model references and frees CUDA memory when
                possible.
        """
        discard(self, "model")

        self.backend.unload_model()

        if include_normalizer:
            discard(self, "llm")
            discard(self, "tokenizer")

        gc.collect()

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def set_voices(self, voices: tuple[str, ...]) -> None:
        """Configure Celune's voice information.

        Args:
            voices: The list of available voice names.

        Returns:
            None: This method replaces Celune's current voice list.
        """
        self.voices = voices

    def load_voice_bundle(self, bundle: Optional[Union[str, Path]] = None) -> bool:
        """Select and load a CEVOICE bundle into Celune's active voice set.

        Args:
            bundle: A built-in bundle name, explicit bundle path, or ``None`` to
                use Celune's default bundle.

        Returns:
            bool: ``True`` when a CEVOICE bundle was loaded, otherwise ``False``.
        """
        select_voice_bundle(bundle)
        loader = default_loader()
        if loader is None:
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
        return bool(voices)

    def load_available_voices(self) -> bool:
        """Load the active voice set appropriate for the selected backend.

        Returns:
            bool: ``True`` when at least one voice is available.
        """
        if self.backend.uses_voice_bundles:
            return self.load_voice_bundle(config_value(self.config, "voice_bundle"))

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
            self.log(f"Unknown voice: {name}")
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
            bool: ``True`` when the requested voice finished loading,
                otherwise ``False``.
        """
        if not self.set_voice(name):
            return False

        if not self._model_ready.wait(timeout=timeout):
            self.log("Timed out while processing a voice change.", "warning")
            return False
        return self.loaded and self.current_voice == name

    def _wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Wait until the model and playback pipeline are ready.

        Args:
            timeout: Maximum seconds to wait for each readiness step.

        Returns:
            bool: ``True`` when Celune is loaded and idle, otherwise ``False``.
        """
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

    def setup_extensions(self) -> None:
        """Configure Celune's extension manager.

        Returns:
            None: This method builds the extension context and autoloads user
                extensions.
        """
        ctx = CeluneContext(
            log=self.log,
            say=self.say,
            play=self.play,
            status=self.status_callback,
            set_voice=self.set_voice,
            get_state=lambda: self.cur_state,
            wait_until_ready=self._wait_until_idle,
            name="Celune",
            version=__version__,
            dev=self.dev,
            log_dev=self.log_dev,
        )
        self.extension_manager = CeluneExtensionManager(ctx)
        self.extension_manager.autoload("extensions")

        self.log_dev(
            f"[Core] Loaded extensions: {', '.join(self.extension_manager.list_extensions())}"
        )

    def log(self, msg: str, severity: str = "info") -> None:
        """Log a message.

        Args:
            msg: The message to emit.
            severity: The message severity level.

        Returns:
            None: This method forwards the log event to the configured callback.
        """
        self.log_callback(msg, severity)

    def log_dev(self, msg: str, severity: str = "info") -> None:
        """Log a developer message.

        Args:
            msg: The message to emit.
            severity: The message severity level.

        Returns:
            None: This method forwards the log event to the configured callback.
        """
        if self.dev:
            self.log_callback(msg, severity)

    def change_voice(self, voice: str) -> None:
        """Change Celune's voice parameters.

        Args:
            voice: The voice name to load and warm up.

        Returns:
            None: This method reloads the backend model for the requested voice.

        Raises:
            WarmupError: The newly loaded voice fails warmup.
        """

        self.log("Celune is reloading, please stand by...")
        self.status_callback("Reloading")
        self.progress_callback(None, None)
        self.cur_state = "reloading"

        try:
            with self._model_lock:
                new_model_name = self.backend.model_id_for_voice(voice)

                # VoxCPM2 uses the same model for all voices, so we don't have to reload every time
                if new_model_name != self.model_name:
                    self.log_dev(f"[RELOAD] Unloading model: {self.model_name}")
                    self.unload_runtime_state(include_normalizer=False)
                    self.log_dev(f"[RELOAD] Loading model: {new_model_name}")
                    self.model = self.backend.load_model(new_model_name)

                    self.log("Rewarming up...")
                    if not self._warmup():
                        raise WarmupError("warmup failed after reload")

                self.log_dev(
                    "[RELOAD] The target model is the same as the model currently in use."
                )

                self.current_voice = voice
                self.loaded = True

            # play readiness signal upon voice change success
            # may not always be possible in cases where many voice changes occurred
            # while Celune did not have to reload completely because of the target model
            # remaining the same
            if not play_readiness_signal(self):
                self.log_dev("Could not play the readiness signal.", "warning")

            self.voice_changed_callback(voice)
            self.log(f"Voice {voice} loaded.")
            self.progress_callback(1, 1)
            self.status_callback("Idle")
        except Exception as e:
            self.loaded = False
            self.log(f"[RELOAD ERROR] {format_error(e, self.dev)}", "error")
            self.status_callback("Celune could not reload", "error")
            self.progress_callback(0, 1)
            self.error_callback("Celune could not reload")
        finally:
            self._model_ready.set()
            self.change_input_state_callback(locked=False)
            self.change_voice_lock_state_callback(locked=len(self.voices) < 2)

    def force_stop_speech(self) -> bool:
        """Forcefully stop Celune from speaking.

        Returns:
            bool: ``True`` when an active utterance was interrupted, otherwise
                ``False``.
        """
        return force_stop_pipeline(self)

    def load(self) -> bool:
        """Load and initialize Celune.

        Returns:
            bool: ``True`` when initialization completed successfully, otherwise
                ``False``.
        """
        disable_progress_bar()
        disable_progress_bars()
        hf_logging.set_verbosity_error()

        log_runtime_banner(self.log, self.backend.name)
        if not self.load_available_voices():
            self.log("No voices were loaded.", "error")
            self.progress_callback(0, 1)
            self.error_callback("No voices loaded")
            return False

        if self.backend.uses_voice_bundles:
            character = announce_default_bundle(self.log)
            self.current_character = character
            if character == "Celune":
                self.log(f"Current character: {character} (default)")
            else:
                self.log(f"Current character: {character}")

        self.setup_extensions()
        self.progress_callback(None, None)
        self.backend.preload_models()

        self.log("All voices are available.")
        try:
            self.model = self.backend.load_default_model()
            active_voice = self.current_voice or self.voices[0]
            self.model_name = self.backend.model_id_for_voice(active_voice)
        except Exception as e:
            self.log("Celune could not load the default model.", "error")
            self.log(format_error(e, self.dev), "error")
            self.progress_callback(0, 1)
            self.error_callback("Default model failed to load")
            return False

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
        ):
            return False

        if self._warmup():
            self.loaded = True
            self._model_ready.set()
            self._release_pipeline()
            self.glow.enter()  # Celune has entered your PC
        else:
            self.log("[WARMUP] Warmup failed.", "error")
            return False

        if self.use_normalization:
            self.load_normalizer()

        if self.extension_manager is not None:
            self.extension_manager.autostart_all()

        self._start_configured_api()

        # notify readiness
        if not play_readiness_signal(self):
            self.log_dev("Could not play the readiness signal.", "warning")

        return True

    def _api_settings(self) -> tuple[bool, str, int, Optional[str], int]:
        """Resolve API settings from Celune's configuration.

        Returns:
            tuple[bool, str, int, Optional[str], int]: Whether to start the API,
                bind host, port, auth token, and per-minute request limit.
        """
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
                "No API token set. Celune API will bind only to the local network.",
                "warning",
            )
            token = None
            host = "127.0.0.1"
        try:
            port = int(api_config.get("port", 2060))
        except (TypeError, ValueError):
            invalid_port = api_config.get("port", 2060)
            self.log(
                f"Celune API port ({invalid_port}) is invalid, will use 2060 instead.",
                "warning",
            )
            port = 2060

        try:
            requests_per_minute = int(api_config.get("rate_limit_per_minute", 60))
        except (TypeError, ValueError):
            invalid_ratelimit = api_config.get("rate_limit_per_minute", 60)
            self.log(
                f"Celune API rate limit ({invalid_ratelimit}) is invalid, using 60/min.",
                "warning",
            )
            requests_per_minute = 60

        return enabled, host, port, token, max(0, requests_per_minute)

    def _start_configured_api(self) -> None:
        """Start the API from config without blocking Celune startup.

        Returns:
            None: This method tries to start the API detached from Celune.
        """
        enabled, host, port, token, requests_per_minute = self._api_settings()
        if not enabled or self._api_thread is not None:
            return

        if not is_port_usable(port):
            self.log(f"Port {port} is unavailable.", "warning")
            self.log("Celune API will not be available.", "warning")
            return

        try:
            from .api import start_api
        except ModuleNotFoundError as package:
            self.log(
                f"A required package ({package.name}) isn't installed.",
                "warning",
            )
            self.log("Celune API will not be available.", "warning")
            return
        except Exception as e:
            self.log(f"Package import failed: {format_error(e, self.dev)}", "warning")
            self.log("Celune API will not be available.", "warning")
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
            self.log("Celune API will not be available.", "warning")
            return

    def load_normalizer(self) -> None:
        """Load the normalizer LLM.

        Returns:
            None: This method starts a background thread to load normalization
                components.
        """

        def _worker():
            """Load normalizer components on a background thread.

            Returns:
                None: This worker updates normalizer fields or logs failures.
            """
            try:
                self.tokenizer, self.llm = load_normalizer_components(
                    self.log, self.backend
                )
                self.log("Normalizer loaded.")
                self.progress_callback(1, 1)
            except Exception as e:
                self.log(f"[NORMALIZER ERROR] {format_error(e, self.dev)}", "error")
                self.log("Normalizer failed to load.", "warning")
                self.log("Normalization will not be available.", "warning")
                self.progress_callback(0, 1)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self.progress_callback(None, None)
        self.log(f"Loading normalizer {NORMALIZER_MODEL_ID}...")

    def _warmup(self) -> bool:
        """Warm up Celune's speech capabilities.

        Returns:
            bool: ``True`` when warmup generation succeeds, otherwise ``False``.
        """
        self.log("[WARMUP] Warming up...")
        self.status_callback("Warming up")
        self.progress_callback(None, None)
        warmup_text = "A"

        try:
            warmup_start = time.perf_counter()

            with self._model_lock:
                if self.model is None:
                    raise NotAvailableError("cannot warm up without a model reference")

                for _, _, _ in self.backend.generate_stream(
                    self.model,
                    text=warmup_text,
                    language=self.language,
                    chunk_size=self.chunk_size,
                    instruct=self.voice_prompt,
                    voice=self.current_voice,
                ):
                    pass

            warmup_end = time.perf_counter()
            warmup_took = warmup_end - warmup_start
            self.log_dev(f"[WARMUP] done, took {format_number(warmup_took, 2)} seconds")
            self.progress_callback(1, 1)
            return True
        except Exception as e:
            self.log(f"[WARMUP ERROR] {format_error(e, self.dev)}", "error")
            self.cur_state = "error"
            self.progress_callback(0, 1)
            self.error_callback("Celune could not warm up")
            return False

    # NOTE: do NOT normalize long inputs, CeluneNorm doesn't support inputs of above 512 tokens, and WILL choke!
    def normalize(self, text: str) -> Optional[str]:
        """Normalize input text using CeluneNorm.

        Args:
            text: The raw text to normalize before speech generation.

        Returns:
            Optional[str]: The normalized text, the original text for blank input,
                or ``None`` when normalization is unavailable or fails.
        """

        if not self.use_normalization:
            return None

        if not text or not text.strip():
            return text

        if self.llm is None or self.tokenizer is None:
            return None

        llm: Any = self.llm
        tokenizer: Any = self.tokenizer

        def _run_inference() -> Optional[str]:
            """Run a blocking normalization request.

            Returns:
                Optional[str]: Normalized text, or ``None`` when normalization
                fails or is unsuitable.
            """
            inf_start = time.perf_counter()
            try:
                bad_text = text.strip()
                norm_token = "<NORM>"

                # Are we using CeluneNorm?
                norm_token_id = tokenizer.convert_tokens_to_ids(norm_token)
                custom_assert(
                    norm_token_id is not None, ValueError("not a CeluneNorm normalizer")
                )
                custom_assert(
                    norm_token_id != tokenizer.unk_token_id,
                    ValueError("not a CeluneNorm normalizer"),
                )

                prompt = f"{bad_text}{norm_token}"

                tokens = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                inputs = tokens.to(llm.device)
                len_tokens = tokens["input_ids"].shape[1]

                self.log(f"Tokens to normalize: {len_tokens}")
                if len_tokens > 128:
                    self.log("Input is too long to normalize.", "warning")
                    return None

                with torch.inference_mode():
                    output_ids = llm.generate(  # type: ignore[operator]
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                new_ids = output_ids[0][prompt_len:]

                # CeluneNorm shouldn't do this, but if it does happen, stop Celune from saying nothing
                if new_ids.numel() == 0:
                    self.log("Normalizer returned no tokens.", "warning")
                    return None

                out = tokenizer.decode(new_ids, skip_special_tokens=True)

                # fix type checker
                if isinstance(out, list):
                    out = out[0] if out else ""

                # too many <NORM>s can break splitting
                if "<NORM>" in out:
                    out = out.split("<NORM>", 1)[0].strip()

                # are we absolutely sure CeluneNorm did produce something before Celune gets to say it?
                if not out:
                    self.log("Normalizer did not produce normal output.", "warning")
                    return None

                inf_total = time.perf_counter() - inf_start
                self.log(f"Normalization took {format_number(inf_total, 2)} seconds.")

                return out

            except Exception as e:
                self.log(
                    f"[NORMALIZATION ERROR] {format_error(e, self.dev)}",
                    "error",
                )
                return None

        return _run_inference()  # blocks for the duration of normalization, but CeluneNorm should be fast enough

    def _acquire_pipeline(self, action: str) -> bool:
        """Atomically claim Celune's shared playback pipeline.

        Args:
            action: A short label describing the action requesting the lock.

        Returns:
            bool: ``True`` when the pipeline lock was acquired, otherwise
                ``False``.
        """
        return acquire_pipeline(self, action)

    def _release_pipeline(self) -> None:
        """Release Celune's shared playback pipeline.

        Returns:
            None: This helper clears Celune's busy state.
        """
        release_pipeline(self)

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
            bool: ``True`` when the text was queued successfully, otherwise
                ``False``.
        """
        return say_pipeline(self, text, save=save, display_text=display_text)

    def say_stream(self, text: str, save: bool = True) -> Optional[queue.Queue]:
        """Queue text for playback and mirror generated chunks to a queue.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            Optional[queue.Queue]: Queue receiving 48 kHz stereo float32 chunks,
                or ``None`` when the request could not be queued.
        """
        stream_queue: queue.Queue = queue.Queue(maxsize=2)
        if not queue_speech(self, text, save=save, stream_queue=stream_queue):
            return None
        return stream_queue

    def play(self, sound_path: str, keep: bool = False) -> bool:
        """Play a sound via Celune's pipeline.

        Args:
            sound_path: The path to the audio file to play.
            keep: Whether to prepend this SFX to the next saved utterance.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise
                ``False``.
        """
        return play_pipeline(self, sound_path, keep=keep)

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
            bool: ``True`` when playback was queued successfully, otherwise
                ``False``.
        """
        return queue_sfx_audio(self, audio, sample_rate, label, keep=keep)

    def close(self) -> None:
        """Shut off Celune and exit.

        Returns:
            None: This method shuts down workers and unloads runtime state.
        """
        close_pipeline(self)
        with self._model_lock:
            self.unload_runtime_state(include_normalizer=True)

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The input text to split.

        Returns:
            list[str]: The generated text chunks.
        """
        return split_text(self, text)

    def _generation_worker(self) -> None:
        """Generate audio tokens and send them to the audio pipeline.

        Returns:
            None: This method runs Celune's generation worker loop.
        """
        generation_worker(self)

    def _playback_worker(self) -> None:
        """Receive audio chunks and play them.

        Returns:
            None: This method runs Celune's playback worker loop.
        """
        playback_worker(self)

    @property
    def stream(self) -> Optional[sd.OutputStream]:
        """Get the current audio output stream.

        Returns:
            Optional[sounddevice.OutputStream]: The active audio stream, if any.
        """
        return self._stream

    @stream.setter
    def stream(self, value: Optional[sd.OutputStream]) -> None:
        """Set the current audio output stream.

        Args:
            value: The new output stream object.

        Returns:
            None: This setter updates Celune's active stream reference.
        """
        self._stream = value

    @property
    def say_lock(self):
        """Get the speech pipeline lock.

        Returns:
            threading.Lock: The lock guarding speech and playback state changes.
        """
        return self._say_lock

    @property
    def utterance_force_stop(self):
        """Get the force-stop event for the current utterance.

        Returns:
            threading.Event: The event used to interrupt active speech.
        """
        return self._utterance_force_stop

    @property
    def queue_lock(self):
        """Get the queue coordination lock.

        Returns:
            threading.Lock: The lock guarding queue mutations.
        """
        return self._queue_lock

    @property
    def force_stop_marker(self):
        """Get the queue marker used to stop playback immediately.

        Returns:
            object: The sentinel object inserted into the audio queue.
        """
        return PipelineStates.UTTERANCE_FORCE_END.value

    @property
    def playback_done(self):
        """Get the playback completion event.

        Returns:
            threading.Event: The event set when playback is idle.
        """
        return self._playback_done

    @property
    def model_ready(self):
        """Get the model readiness event.

        Returns:
            threading.Event: The event set when the speech model is ready to use.
        """
        return self._model_ready

    @property
    def utterance_done(self):
        """Get the marker that signals utterance completion.

        Returns:
            object: The sentinel object inserted when generation finishes.
        """
        return PipelineStates.UTTERANCE_END.value

    @property
    def sentinel(self):
        """Get the global shutdown sentinel.

        Returns:
            object: The sentinel object used to stop worker threads.
        """
        return PipelineStates.TERMINATE.value

    @property
    def generation_thread(self) -> Optional[threading.Thread]:
        """Get the generation worker thread.

        Returns:
            Optional[threading.Thread]: The active generation thread, if started.
        """
        return self._generation_thread

    @property
    def playback_thread(self) -> Optional[threading.Thread]:
        """Get the playback worker thread.

        Returns:
            Optional[threading.Thread]: The active playback thread, if started.
        """
        return self._playback_thread

    @property
    def exit_requested(self):
        """Get the exit flag.

        Returns:
            bool: ``True`` when Celune is shutting down, otherwise ``False``.
        """
        return self._exit_requested

    @property
    def model_lock(self):
        """Get the model access lock.

        Returns:
            threading.RLock: The lock guarding model access and reloads.
        """
        return self._model_lock

    @property
    def audio_unavailable(self):
        """Get the audio availability flag.

        Returns:
            bool: ``True`` when audio output initialization has failed.
        """
        return self._audio_unavailable

    @property
    def current_sr(self) -> Optional[int]:
        """Get the active stream sample rate.

        Returns:
            Optional[int]: The current playback sample rate, if a stream exists.
        """
        return self._current_sr

    @current_sr.setter
    def current_sr(self, value: Optional[int]) -> None:
        """Set the active stream sample rate.

        Args:
            value: The new playback sample rate.

        Returns:
            None: This setter updates Celune's current sample-rate tracking.
        """
        self._current_sr = value
