# pylint: disable=R0902, R0904, R0911, R0912, R0913, R0914, R0915, R0917, W0718
"""Celune's backend layer."""

import gc
import time
import queue
import threading
import traceback
import contextlib
from typing import Optional, Callable, Union

import torch
import sounddevice as sd
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging
from transformers.utils.logging import disable_progress_bar
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .backends import CeluneBackend, resolve_backend
from .backends.qwen3 import Qwen3
from .config import config_bool, config_value
from .constants import NORMALIZER_MODEL_ID
from .dsp import StreamingPedalboardReverb
from .utils import format_number
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
    playback_worker,
    play as play_pipeline,
    release_pipeline,
    say as say_pipeline,
    split_text,
)
from .runtime import log_runtime_banner, validate_runtime


class Celune:
    """The character engine for Celune."""

    def __init__(
        self,
        tts_backend: Optional[Union[str, CeluneBackend, type[CeluneBackend]]] = None,
        chunk_size: int = 24,  # only used in Qwen3 backend; ~1.92s
        language: str = "Auto",  # Qwen3 backend accepts a language, others may not
        log_callback: Optional[Callable[[str, str], None]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
        queue_avail_callback: Optional[Callable[[], None]] = None,
        voice_changed_callback: Optional[Callable[[str], None]] = None,
        change_input_state_callback: Optional[Callable[[bool], None]] = None,
        dev: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        if tts_backend is None:
            raise BackendError("no backend set")

        self.log_callback = log_callback or (lambda msg, severity="info": None)
        self.status_callback = status_callback or (lambda msg, severity="info": None)
        self.error_callback = error_callback or (lambda error: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.voice_changed_callback = voice_changed_callback or (lambda name: None)
        self.change_input_state_callback = change_input_state_callback or (
            lambda locked: None
        )

        self.config = config

        backend_kwargs = {}

        if (
            isinstance(tts_backend, str) and tts_backend.strip().lower() == "qwen3"
        ) or isinstance(tts_backend, Qwen3):
            backend_kwargs["mode"] = config_value(config, "qwen3_mode", "native")

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
            raise BackendError(
                f"internal backend error: {self.format_error(e, dev)}"
            ) from e

        self.language = language

        self.model = None
        self.llm: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        self.current_voice = self.backend.default_voice
        self.voices: list[str] = self.backend.voices
        self.voice_prompt: Optional[str] = None

        self.chunk_size = chunk_size
        self.prebuffer_chunks = 5 if self.backend.name == "qwen3" else 10

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self._playback_thread = None
        self._generation_thread = None

        self._queue_lock = threading.Lock()
        self._sentinel = object()  # dispatched upon exit
        self._utterance_done = object()  # dispatched upon generation end
        self._force_stop_marker = (
            object()
        )  # dispatched when speech should stop immediately
        self._utterance_force_stop = threading.Event()

        self._stream: Optional[sd.OutputStream] = None
        self._current_sr: Optional[int] = None
        self._audio_unavailable = False
        self.can_use_rubberband = True
        self.speed: float = 1.0
        self.reverb = StreamingPedalboardReverb()

        self.locked = True
        self.loaded = False
        self.recently_saved = None
        self._model_ready = threading.Event()
        self._model_ready.set()
        self._exit_requested = False
        self._playback_done = threading.Event()
        self._playback_done.set()
        self._say_lock = threading.Lock()
        self._model_lock = threading.RLock()

        self.cur_state = "init"

        self.dev = dev
        self.use_normalization = config_bool(
            config, "CELUNE_NORMALIZE", "use_normalizer"
        )

        self.extension_manager: Optional[CeluneExtensionManager] = None

        self.glow = AudioRGBGlow(color="#cebaff")
        self.glow.start()

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
        _ = self.model
        self.model = None
        del _

        self.backend.unload_model()

        if include_normalizer:
            _ = self.llm
            self.llm = None
            del _

            _ = self.tokenizer
            self.tokenizer = None
            del _

        gc.collect()

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def set_voices(self, voices: list) -> None:
        """Configure Celune's voice information.

        Args:
            voices: The list of available voice names.

        Returns:
            None: This method replaces Celune's current voice list.
        """
        self.voices = voices

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

    def _wait_until_idle(self, timeout: float = 30.0) -> bool:
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
        if self.dev:
            self.log("[EXT] Setting up extension manager")

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
        )
        self.extension_manager = CeluneExtensionManager(ctx)
        self.extension_manager.autoload("extensions")

        if self.dev:
            self.log(
                f"[EXT] Loaded extensions: {self.extension_manager.list_extensions()}"
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

    def change_voice(self, voice: str) -> None:
        """Change Celune's voice parameters.

        Args:
            voice: The voice name to load and warm up.

        Returns:
            None: This method reloads the backend model for the requested voice.
        """

        self.log("Celune is reloading, please stand by...")
        self.status_callback("Reloading")
        self.cur_state = "reloading"

        try:
            with self._model_lock:
                self.unload_runtime_state(include_normalizer=False)

                self.model = self.backend.load_model(
                    self.backend.model_id_for_voice(voice)
                )

                self.log("Rewarming up...")
                if not self._warmup():
                    raise WarmupError("Warmup failed after reload")

                self.current_voice = voice
                self.loaded = True

            self.voice_changed_callback(voice)
            self.log(f"Voice {voice} loaded.")
            self.status_callback("Idle")
        except Exception as e:
            self.loaded = False
            self.log(f"[RELOAD ERROR] {self.format_error(e, self.dev)}", "error")
            self.status_callback("Celune could not reload", "error")
            self.error_callback("Celune could not reload")

        finally:
            self._model_ready.set()
            self._release_pipeline()
            self.change_input_state_callback(locked=False)

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
        self.backend.preload_models()

        self.log("All Celune voices are available.")
        try:
            self.model = self.backend.load_default_model()
        except Exception as e:
            self.log("Celune could not load the default model.", "error")
            self.log(self.format_error(e, self.dev), "error")
            self.error_callback("Default model failed to load")
            return False

        self._generation_thread = threading.Thread(
            target=self._generation_worker, daemon=True
        )
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )

        self._generation_thread.start()
        self._playback_thread.start()

        if not validate_runtime(
            log=self.log,
            error=self.error_callback,
            set_state=lambda state: setattr(self, "cur_state", state),
            glow_connect_failed=self.glow.connect_failed,
            format_error=self.format_error,
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

        return True

    def load_normalizer(self) -> None:
        """Load the normalizer LLM.

        Returns:
            None: This method starts a background thread to load normalization
                components.
        """

        def _worker():
            try:
                self.tokenizer, self.llm = load_normalizer_components(
                    self.log, self.backend
                )
                self.log("Normalizer loaded.")
            except Exception as e:
                self.log(
                    f"[NORMALIZER ERROR] {self.format_error(e, self.dev)}", "error"
                )
                self.log("Normalizer failed to load.", "warning")
                self.log("Normalization will not be available.", "warning")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        self.log(f"Loading normalizer {NORMALIZER_MODEL_ID}...")

    def _warmup(self) -> bool:
        """Warm up Celune's speech capabilities.

        Returns:
            bool: ``True`` when warmup generation succeeds, otherwise ``False``.
        """
        self.log("[WARMUP] Warming up...")
        self.status_callback("Warming up")
        warmup_text = "A"

        try:
            warmup_start = time.perf_counter()

            with self._model_lock:
                if self.model is None:
                    raise NotAvailableError("Model is not available during warmup")

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
            self.log(f"[WARMUP] done, took {format_number(warmup_took, 2)} seconds")
            return True

        except Exception as e:
            self.log(f"[WARMUP ERROR] {self.format_error(e, self.dev)}", "error")
            self.cur_state = "error"
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

        def _run_inference() -> Optional[str]:
            inf_start = time.perf_counter()
            try:
                bad_text = text.strip()
                norm_token = "<NORM>"

                # Are we using CeluneNorm?
                norm_token_id = self.tokenizer.convert_tokens_to_ids(norm_token)
                assert norm_token_id is not None, "not a CeluneNorm normalizer"
                assert norm_token_id != self.tokenizer.unk_token_id, (
                    "not a CeluneNorm normalizer"
                )

                prompt = f"{bad_text}{norm_token}"

                tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                inputs = tokens.to(self.llm.device)
                len_tokens = tokens["input_ids"].shape[1]

                self.log(f"Tokens to normalize: {len_tokens}")
                if len_tokens > 256:
                    self.log("Input is too long to normalize.", "warning")
                    return None

                with torch.inference_mode():
                    output_ids = self.llm.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                        or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                new_ids = output_ids[0][prompt_len:]

                # CeluneNorm shouldn't do this, but if it does happen, stop Celune from saying nothing
                if new_ids.numel() == 0:
                    self.log("Normalizer returned no tokens.", "warning")
                    return None

                out = self.tokenizer.decode(new_ids, skip_special_tokens=True)

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
                    f"[NORMALIZATION ERROR] {self.format_error(e, self.dev)}",
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

    def say(self, text: str, save: bool = True) -> bool:
        """Queue text for Celune to say.

        Args:
            text: The text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            bool: ``True`` when the text was queued successfully, otherwise
                ``False``.
        """
        return say_pipeline(self, text, save=save)

    def play(self, sound_path: str) -> bool:
        """Play a sound via Celune's pipeline.

        Args:
            sound_path: The path to the audio file to play.

        Returns:
            bool: ``True`` when playback was queued successfully, otherwise
                ``False``.
        """
        return play_pipeline(self, sound_path)

    def close(self) -> None:
        """Shut off Celune and exit.

        Returns:
            None: This method shuts down workers and unloads runtime state.
        """
        close_pipeline(self)
        with self._model_lock:
            self.unload_runtime_state(include_normalizer=True)

    @staticmethod
    def format_error(e: Exception, dev: bool) -> str:
        """Format an error message.

        Args:
            e: The exception to format.
            dev: Whether developer mode is enabled.

        Returns:
            str: Either the full traceback or the exception text.
        """
        if dev:
            trace = traceback.format_exc()
            with open("celune_traceback.txt", "w", encoding="utf-8") as f:
                f.write(trace)

        details = str(e) or "no error description"
        return traceback.format_exc() if dev else details

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
    def stream(self) -> sd.OutputStream:
        """Get the current audio output stream.

        Returns:
            Optional[sounddevice.OutputStream]: The active audio stream, if any.
        """
        return self._stream

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
        return self._force_stop_marker

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
        return self._utterance_done

    @property
    def sentinel(self):
        """Get the global shutdown sentinel.

        Returns:
            object: The sentinel object used to stop worker threads.
        """
        return self._sentinel

    @property
    def generation_thread(self) -> threading.Thread:
        """Get the generation worker thread.

        Returns:
            Optional[threading.Thread]: The active generation thread, if started.
        """
        return self._generation_thread

    @property
    def playback_thread(self) -> threading.Thread:
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
    def current_sr(self):
        """Get the active stream sample rate.

        Returns:
            Optional[int]: The current playback sample rate, if a stream exists.
        """
        return self._current_sr

    @stream.setter
    def stream(self, value):
        """Set the current audio output stream.

        Args:
            value: The new output stream object.

        Returns:
            None: This setter updates Celune's active stream reference.
        """
        self._stream = value

    @current_sr.setter
    def current_sr(self, value):
        """Set the active stream sample rate.

        Args:
            value: The new playback sample rate.

        Returns:
            None: This setter updates Celune's current sample-rate tracking.
        """
        self._current_sr = value
