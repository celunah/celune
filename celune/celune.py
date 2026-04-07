# pylint: disable=R0902, R0912, R0913, R0914, R0915, R0917, W0718
"""Celune's backend layer."""

import gc
import os
import re
import sys
import time
import glob
import queue
import platform
import threading
import traceback
from typing import Optional, Callable, Union

import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyrubberband as rb
from faster_qwen3_tts import FasterQwen3TTS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
from transformers.utils.logging import disable_progress_bar
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .dsp import _soften_onset, _to_48khz, StreamingPedalboardReverb
from .utils import format_number
from .exceptions import NotAvailableError, WarmupError
from .extensions.base import CeluneContext
from .extensions.manager import CeluneExtensionManager


class Celune:
    """The character engine for Celune."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int = 24,  # approx. 1.92s per chunk
        sentences_per_chunk: int = 4,  # prevents Celune from running out of context
        language: str = "Auto",  # if you don't care about multilinguality, set it to "English"
        log_callback: Optional[
            Union[Callable[[str], None], Callable[[str, str], None]]
        ] = None,
        status_callback: Optional[
            Union[Callable[[str], None], Callable[[str, str], None]]
        ] = None,
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
        self.llm = None
        self.tokenizer = None

        self.current_voice = "balanced"
        self.voices: list[str] = []
        self.voice_prompt: Optional[str] = None

        self.chunk_size = chunk_size
        self.prebuffer_chunks = min(
            max(self.chunk_size // 8, 1), 4
        )  # between 1-4 prebuffer chunks
        self.sentences_per_chunk = sentences_per_chunk

        self.log_callback = log_callback or (lambda msg, severity: None)
        self.status_callback = status_callback or (lambda msg, severity: None)
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

        self.extension_manager: CeluneExtensionManager | None = None

    @staticmethod
    def _model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if the TTS model is already available."""
        base = HF_HUB_CACHE
        model_dir = os.path.join(base, f"models--{model.replace('/', '--')}")

        refs_main = os.path.join(model_dir, "refs", "main")
        snapshots_dir = os.path.join(model_dir, "snapshots")

        expected_files = [
            "config.json",
            "generation_config.json",
            "model*.safetensors",
            "tokenizer_config.json",
        ]

        if not os.path.exists(refs_main):
            return False, None

        with open(refs_main, encoding="utf-8") as f:
            commit = f.read().strip()

        snapshot_path = os.path.join(snapshots_dir, commit)

        if not os.path.isdir(snapshot_path):
            return False, None

        if all(glob.glob(os.path.join(snapshot_path, f)) for f in expected_files):
            return True, snapshot_path

        return False, None

    @staticmethod
    def _clear_queue(q: queue.Queue) -> None:
        """Drain all pending items from a queue."""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _close_stream(self, abort: bool = False) -> None:
        """Close the current audio stream if one exists."""
        if self._stream is None:
            return

        try:
            if abort:
                self._stream.abort()
            else:
                self._stream.stop()
        except Exception:
            pass

        try:
            self._stream.close()
        except Exception:
            pass

        self._stream = None
        self._current_sr = None

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
        ok = self._model_ready.wait(timeout=timeout)

        if not ok:
            self.log("Timed out while waiting to become ready.", "warning")
            return False

        if not self.loaded:
            self.log("Model was unloaded while waiting to become ready.", "warning")
            return False

        return True

    def setup_extensions(self) -> None:
        """Configure Celune's extension manager."""
        self.log("[EXT] Setting up extension manager")

        ctx = CeluneContext(
            log=self.log,
            say=self.say,
            status=self.status_callback,
            set_voice=self.set_voice,
            get_state=lambda: self.cur_state,
            wait_until_ready=self._wait_until_idle,
            name="Celune",
            version=__version__,
        )
        self.extension_manager = CeluneExtensionManager(ctx)
        self.extension_manager.autoload("extensions")

        self.log(f"[EXT] Loaded extensions: {self.extension_manager.list_extensions()}")

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

                voice_map = {
                    "balanced": "lunahr/Celune-1.7B-Neutral",
                    "calm": "lunahr/Celune-1.7B-Calm",
                    "enthusiastic": "lunahr/Celune-1.7B-Energetic",
                    "upbeat": "lunahr/Celune-1.7B-Upbeat",
                }

                available, path = self._model_is_available_locally(voice_map[voice])

                if available:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    self.model = FasterQwen3TTS.from_pretrained(path)
                else:
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    self.log("Downloading TTS model...")
                    self.model = FasterQwen3TTS.from_pretrained(voice_map[voice])

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
            self.cur_state = "idle"
            self.change_input_state_callback(locked=False)

    def load(self) -> bool:
        """Load and initialize Celune."""
        disable_progress_bar()
        disable_progress_bars()
        hf_logging.set_verbosity_error()

        available, path = self._model_is_available_locally("lunahr/Celune-1.7B-Neutral")
        if available:
            self.log("TTS model is already available in cache")
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = FasterQwen3TTS.from_pretrained(path)
        else:
            self.log("Downloading TTS model...")
            os.environ["HF_HUB_OFFLINE"] = "0"
            self.model = FasterQwen3TTS.from_pretrained(self.model_name)

        self._generation_thread = threading.Thread(
            target=self._generation_worker, daemon=True
        )
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )

        self._generation_thread.start()
        self._playback_thread.start()

        self.log("Environment test...")
        self.log(
            f"Celune {__version__}, "
            f"Python {platform.python_version()}, "
            f"PyTorch {torch.__version__}, "
            f"CUDA {torch.version.cuda}"
        )

        if sys.version_info < (3, 12):
            self.log("Celune does not work on this version of Python", "error")
            self.cur_state = "error"
            self.error_callback("Incompatible Python version")
            return False

        cuda_avail = torch.cuda.is_available()
        self.log(f"CUDA available: {cuda_avail}")

        if not cuda_avail:
            self.log("No GPUs found.", "error")
            self.cur_state = "error"
            self.error_callback("CUDA is not available")
            return False

        current_gpu = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        self.log(f"GPU: {current_gpu} (Capability: {major}.{minor})")

        self.log("Compute test...")
        x = torch.rand(256, 256, device="cuda")
        y = x @ x
        self.log(f"[OK] Compute test succeeded on {y.device}")

        if self._warmup():
            self.locked = False
            self.loaded = True
            self._model_ready.set()
            self.cur_state = "idle"
        else:
            self.log("[WARMUP] Warmup failed.", "error")
            return False

        if self.use_normalization:
            self._load_normalizer()
        return True

    def _load_normalizer(self) -> None:
        """Load normalizer LLM."""

        def _worker():
            # CeluneNorm takes care of the input here.
            model_id = "lunahr/CeluneNorm-0.6B-v1.1"  # please use CeluneNorm or Celune receives bad input

            try:
                available, path = self._model_is_available_locally(model_id)
                if available:
                    self.log("Normalizer is already available in cache")
                    self.tokenizer = AutoTokenizer.from_pretrained(path)
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        path, torch_dtype=torch.bfloat16, device_map="cuda"
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
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
        self.log("Loading normalizer...")

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

    def say(self, text: str) -> bool:
        """Queue text for Celune to say."""
        if not self.loaded:
            self.log("Tried to speak before model was loaded.", "warning")
            self.error_callback("Celune is not currently ready")
            return False

        if not self._model_ready.is_set():
            self.status_callback("Waiting for model")
            self.log("Speak request is waiting for model reload to finish.", "info")

        self._model_ready.wait()

        if not self.loaded:
            self.log("Model became unavailable before speaking.", "warning")
            self.error_callback("Celune is not currently ready")
            return False

        # noinspection PyUnusedLocal
        normalized = None
        if self.use_normalization:
            self.status_callback("Normalizing")
            normalized = self.normalize(text)

        with self._say_lock:
            if not self.loaded:
                self.log("Model became unavailable before queueing speech.", "warning")
                self.error_callback("Celune is not currently ready")
                return False

            self.locked = True
            self._playback_done.clear()

            if normalized is None:
                self.text_queue.put(text)
            else:
                self.text_queue.put(normalized)

        return True

    def close(self) -> None:
        """Shut off Celune and exit."""
        self.log("Exiting...")
        self._exit_requested = True

        with self._queue_lock:
            self._clear_queue(self.text_queue)
            self._clear_queue(self.audio_queue)

        self.text_queue.put(self._sentinel)
        self.audio_queue.put(self._sentinel)

        if self._generation_thread is not None:
            self._generation_thread.join(timeout=2)

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2)

        self._close_stream(abort=True)

    @staticmethod
    def format_error(e: Exception, dev: bool) -> str:
        """Format an error message."""
        return traceback.format_exc() if dev else str(e)

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []

        current_sentences = []
        current_length = 0

        for sentence in sentences:
            if not sentence:
                continue

            new_length = current_length + len(sentence) + 1

            if len(current_sentences) < self.sentences_per_chunk and new_length <= (
                self.sentences_per_chunk * 150
            ):
                current_sentences.append(sentence)
                current_length = new_length
            else:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))

                current_sentences = [sentence]
                current_length = len(sentence)

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        self.log(f"Chunks: {len(chunks)}")
        return chunks

    def _generation_worker(self) -> None:
        """Generate audio tokens and send them to the audio pipeline."""
        while True:
            self.status_callback("Generating")
            text = self.text_queue.get()

            if text is self._sentinel:
                self.audio_queue.put(self._sentinel)
                break

            if self._exit_requested:
                self.locked = False
                continue

            try:
                # Wait here if a reload is in progress
                self._model_ready.wait()

                if not self.loaded:
                    self.log(
                        "Skipping generation because model is not ready.", "warning"
                    )
                    self.locked = False
                    continue

                start_time = time.perf_counter()
                self.log(f"[GEN] {text}")
                speech_len = 0.0

                chunks = self._split_text(text)
                buffer = []

                with self._model_lock:
                    if self.model is None:
                        raise NotAvailableError("self.model is None")

                    for chunk_index, chunk_text in enumerate(chunks):
                        if self._exit_requested:
                            break

                        is_first_chunk = chunk_index == 0

                        for (
                            audio_chunk,
                            sr,
                            timing,
                        ) in self.model.generate_custom_voice_streaming(
                            text=chunk_text,
                            language=self.language,
                            speaker="celune",
                            chunk_size=self.chunk_size,
                            instruct=self.voice_prompt,
                        ):
                            if self._exit_requested:
                                break

                            if hasattr(audio_chunk, "cpu"):
                                audio_chunk = audio_chunk.cpu().numpy()

                            audio_chunk = _to_48khz(audio_chunk, sr)
                            if self.speed != 1.0 and self.can_use_rubberband:
                                try:
                                    audio_chunk = rb.time_stretch(audio_chunk, 48000, self.speed)
                                except RuntimeError:
                                    self.log("Rubber Band is unavailable, speed controls disabled.", "warning")
                                    self.can_use_rubberband = False
                            if self.reverb.strength > 0.0:
                                audio_chunk = self.reverb.process(audio_chunk, 48000)

                            if is_first_chunk:
                                audio_chunk = _soften_onset(audio_chunk, 48000)
                                is_first_chunk = False

                            if self._exit_requested:
                                break

                            buffer.append(audio_chunk)
                            self.audio_queue.put((audio_chunk, 48000, timing))

                            chunk_dur = len(audio_chunk) / 48000
                            speech_len += chunk_dur

                if buffer:
                    wav = np.concatenate(buffer)
                    timestamp = time.time_ns() // 1000000
                    if not os.path.exists("outputs"):
                        self.log("Outputs path not found, creating...", "warning")
                        try:
                            os.mkdir("outputs")
                        except OSError as e:
                            self.log(
                                "Cannot create outputs directory, not saving WAV file: "
                                f"{self.format_error(e, self.dev)}",
                                "warning",
                            )

                    if os.path.exists("outputs"):
                        sf.write(f"outputs/celune_speech_{timestamp}.wav", wav, 48000)

                end_time = time.perf_counter()
                generation_time = end_time - start_time

                if self._exit_requested:
                    self.cur_state = "idle"
                    self.locked = False
                    continue

                self.log(
                    f"[GEN] {format_number(speech_len, 2)} seconds, took {format_number(generation_time, 2)} seconds, "
                    f"RTF: {format_number(speech_len / generation_time, 2)}"
                )
                self.status_callback("Speaking")
                self.cur_state = "speaking"
                self.log("[GEN] done")
                self.queue_avail_callback()

                if not self._exit_requested:
                    tail = self.reverb.flush(tail_seconds=1.5)
                    if len(tail) > 0:
                        self.audio_queue.put((tail, 48000, None))
                    self.reverb.reset()

                    self.audio_queue.put(self._utterance_done)

            except Exception as e:
                if self._exit_requested:
                    self.cur_state = "idle"
                    self.locked = False
                    continue

                self.log(f"[GEN ERROR] {self.format_error(e, self.dev)}", "error")
                self.cur_state = "error"
                self.locked = False
                self.error_callback("Celune could not generate the input")

    def _playback_worker(self) -> None:
        """Receive audio chunks and play them."""
        started = False

        while True:
            if self._exit_requested:
                with self._queue_lock:
                    self._clear_queue(self.audio_queue)

                self._close_stream(abort=True)

                self.cur_state = "idle"
                self.locked = False
                self.idle_callback()
                return

            if not started:
                while self.audio_queue.qsize() < self.prebuffer_chunks:
                    if self._exit_requested:
                        break
                    time.sleep(0.01)

                if self._exit_requested:
                    continue

            item = self.audio_queue.get()

            if item is self._sentinel:
                break

            if self._exit_requested:
                continue

            if item is self._utterance_done:
                self._playback_done.set()

                more_pending = (not self.audio_queue.empty()) or (
                    not self.text_queue.empty()
                )

                if more_pending:
                    silence = np.zeros((48000, 2), dtype=np.float32)
                    if self._stream is not None and not self._exit_requested:
                        self._stream.write(silence)
                else:
                    self.log("[STATE] idle")
                    self.cur_state = "idle"
                    self.locked = False
                    self.idle_callback()
                    self.log("Ready to speak.")

                    avail, total = tuple(
                        v / 1024**3 for v in torch.cuda.mem_get_info(0)
                    )
                    if avail <= 2:
                        self.log(
                            "Celune is running out of VRAM "
                            f"({format_number(avail, 2)}/{format_number(total, 2)} GB available).",
                            "warning",
                        )
                        self.log(
                            "Please close any memory-resident applications to improve performance.",
                            "warning",
                        )
                    else:
                        self.log(f"Available VRAM: {format_number(avail, 2)}/{format_number(total, 2)} GB")
                continue

            audio_chunk, sr, _ = item

            if self._stream is None:
                try:
                    self._current_sr = sr
                    self._stream = sd.OutputStream(
                        samplerate=sr,
                        channels=2,
                        dtype="float32",
                        blocksize=0,
                    )
                    self._stream.start()
                    started = True
                    self.log(f"[PLAY] started stream at {sr} Hz")
                except sd.PortAudioError:
                    if not self._audio_unavailable:
                        self.log(
                            "Celune could not initialize the audio stream.", "error"
                        )
                        self.log("No suitable audio device is available.", "error")
                        self.error_callback("No suitable audio devices")
                    self._audio_unavailable = True

            if self._exit_requested:
                continue

            try:
                self._stream.write(audio_chunk)
            except Exception as e:
                self.log(f"[PLAY ERROR] {self.format_error(e, self.dev)}", "error")
                self.error_callback("Playback error")
                self._close_stream(abort=True)
                self._stream = None
                self._current_sr = None
                continue
