# pylint: disable=R0902, R0911, R0912, R0913, R0914, R0915, R0917, W0718
"""Celune's backend layer."""

import gc
import os
import time
import queue
import threading
import traceback
from typing import Optional, Callable

import torch
import sounddevice as sd
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging
from transformers.utils.logging import disable_progress_bar
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .constants import (
    ALL_VOICE_MODEL_IDS,
    DEFAULT_MODEL_ID,
    DEFAULT_VOICE,
    NORMALIZER_MODEL_ID,
    VOICE_MODELS,
)
from .dsp import StreamingPedalboardReverb
from .utils import format_number
from .chroma import AudioRGBGlow
from .exceptions import NotAvailableError, WarmupError
from .extensions.base import CeluneContext
from .extensions.manager import CeluneExtensionManager
from .modeling import load_normalizer_components, load_tts_model, preload_models
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
        model_name: str,
        chunk_size: int = 24,  # approx. 1.92s per chunk
        sentences_per_chunk: int = 4,  # prevents Celune from running out of context
        language: str = "Auto",  # if you don't care about multilinguality, set it to "English"
        log_callback: Optional[Callable[[str, str], None]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
        queue_avail_callback: Optional[Callable[[], None]] = None,
        voice_changed_callback: Optional[Callable[[str], None]] = None,
        change_input_state_callback: Optional[Callable[[bool], None]] = None,
        dev: bool = False,
    ) -> None:
        self.model_name = model_name
        self.language = language

        self.model = None
        self.llm: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        self.current_voice = DEFAULT_VOICE
        self.voices: list[str] = list(VOICE_MODELS)
        self.voice_prompt: Optional[str] = None

        self.chunk_size = chunk_size
        self.prebuffer_chunks = min(
            max(self.chunk_size // 8, 1), 4
        )  # between 1-4 prebuffer chunks
        self.sentences_per_chunk = sentences_per_chunk

        self.log_callback = log_callback or (lambda msg, severity="info": None)
        self.status_callback = status_callback or (lambda msg, severity="info": None)
        self.error_callback = error_callback or (lambda error: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.voice_changed_callback = voice_changed_callback or (lambda name: None)
        self.change_input_state_callback = change_input_state_callback or (
            lambda locked: None
        )

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
        self._model_ready = threading.Event()
        self._model_ready.set()
        self._exit_requested = False
        self._playback_done = threading.Event()
        self._playback_done.set()
        self._say_lock = threading.Lock()
        self._model_lock = threading.RLock()

        self.cur_state = "init"

        self.dev = dev
        self.use_normalization = os.getenv("CELUNE_NORMALIZE") in {
            "1",
            "true",
            "on",
        }

        self.extension_manager: Optional[CeluneExtensionManager] = None

        self.glow = AudioRGBGlow(color="#cebaff")
        self.glow.start()

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        """Drain all pending items from a queue."""
        clear_queue(q)

    def _close_stream(self, abort: bool = False) -> None:
        """Close the current audio stream if one exists."""
        close_stream(self, abort=abort)

    def set_voices(self, voices: list) -> None:
        """Configure Celune's voice information."""
        self.voices = voices

    def set_voice(self, name: str) -> bool:
        """Extension method for changing Celune's voice."""
        if name not in self.voices:
            self.log(f"Unknown voice: {name}")
            return False

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
                "If this was not expected, Celune may have been downloading models.",
                "warning",
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
        """Configure Celune's extension manager."""
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
        """Log a message."""
        self.log_callback(msg, severity)

    def change_voice(self, voice: str) -> None:
        """Change Celune's voice parameters."""
        self.log("Celune is reloading, please stand by...")
        self.status_callback("Reloading")
        self.change_input_state_callback(locked=True)
        self.cur_state = "reloading"

        try:
            torch.cuda.synchronize()

            with self._model_lock:
                # noinspection PyUnusedLocal
                old_model = self.model
                self.model = None
                del old_model

                gc.collect()
                torch.cuda.empty_cache()

                self.model = load_tts_model(VOICE_MODELS[voice], self.log)

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
        """Forcefully stop Celune from speaking."""
        return force_stop_pipeline(self)

    def load(self) -> bool:
        """Load and initialize Celune."""
        disable_progress_bar()
        disable_progress_bars()
        hf_logging.set_verbosity_error()

        log_runtime_banner(self.log)
        preload_models(ALL_VOICE_MODEL_IDS, self.log)

        self.log("All Celune voices are available.")
        try:
            self.model = load_tts_model(DEFAULT_MODEL_ID, self.log)
        except Exception:
            self.log("Celune could not load the default model.", "error")
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
            self.extension_manager.autostart_all()  # why did this come to exactly line 314? pi confirmed?
        else:
            self.log("[WARMUP] Warmup failed.", "error")
            return False

        if self.use_normalization:
            self._load_normalizer()
        return True

    def _load_normalizer(self) -> None:
        """Load normalizer LLM."""

        def _worker():
            try:
                self.tokenizer, self.llm = load_normalizer_components(self.log)
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
        """Warm up Celune's speech capabilities."""
        self.log("[WARMUP] Warming up...")
        self.status_callback("Warming up")
        warmup_text = "A"

        try:
            warmup_start = time.perf_counter()

            with self._model_lock:
                if self.model is None:
                    raise NotAvailableError("Model is not available during warmup")

                for _, _, _ in self.model.generate_custom_voice_streaming(
                    text=warmup_text,
                    language=self.language,
                    speaker="celune",
                    chunk_size=self.chunk_size,
                    instruct=self.voice_prompt,
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

    def normalize(self, text: str) -> Optional[str]:
        """Normalize input text using CeluneNorm."""

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

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.llm.device)

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
        """Atomically claim Celune's shared playback pipeline."""
        return acquire_pipeline(self, action)

    def _release_pipeline(self) -> None:
        """Release Celune's shared playback pipeline."""
        release_pipeline(self)

    def say(self, text: str) -> bool:
        """Queue text for Celune to say."""
        return say_pipeline(self, text)

    def play(self, sound_path: str) -> bool:
        """Play a sound via Celune's pipeline."""
        return play_pipeline(self, sound_path)

    def close(self) -> None:
        """Shut off Celune and exit."""
        close_pipeline(self)

    @staticmethod
    def format_error(e: Exception, dev: bool) -> str:
        """Format an error message."""
        if dev:
            trace = traceback.format_exc()
            with open("celune_traceback.txt", "w", encoding="utf-8") as f:
                f.write(trace)

        return traceback.format_exc() if dev else str(e)

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        return split_text(self, text)

    def _generation_worker(self) -> None:
        """Generate audio tokens and send them to the audio pipeline."""
        generation_worker(self)

    def _playback_worker(self) -> None:
        """Receive audio chunks and play them."""
        playback_worker(self)
