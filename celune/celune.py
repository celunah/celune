# pylint: disable=R0902, R0912, R0913, R0914, R0915, R0917, W0718
"""Celune's backend layer."""

import os
import re
import sys
import math
import time
import glob
import queue
import random
import platform
import threading
import traceback
from typing import Optional, Callable, Union

import torch
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_qwen3_tts import FasterQwen3TTS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
from transformers.utils.logging import disable_progress_bar
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import disable_progress_bars

from . import __version__
from .extensions.base import CeluneContext
from .extensions.manager import CeluneExtensionManager


class Celune:
    """The character engine for Celune."""

    def __init__(
        self,
        model_name: str,
        ref_audio: str,
        ref_text: str,

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

        dev: bool = False,
    ) -> None:
        self.model_name = model_name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language

        self.model = None
        self.llm = None
        self.tokenizer = None

        self.current_voice = "balanced"
        self.voices: dict[str, tuple[str, str, int]] = {}

        self.chunk_size = chunk_size
        self.prebuffer_chunks = min(max(self.chunk_size // 8, 1), 3)  # between 1-4 prebuffer chunks
        self.sentences_per_chunk = sentences_per_chunk

        self.log_callback = log_callback or (lambda msg, severity: None)
        self.status_callback = status_callback or (lambda msg, severity: None)
        self.error_callback = error_callback or (lambda error: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.voice_changed_callback = voice_changed_callback or (lambda name: None)

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

        self.locked = True
        self.loaded = False
        self._exit_requested = False

        self.cur_state = "init"

        self.dev = dev
        self.use_normalization = os.getenv("CELUNE_NORMALIZE") in {
            "1",
            "true",
            "on",
        }
        self.normalization_timeout = 0.5

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

    def set_voices(self, voices: dict[str, tuple[str, str, int]]) -> None:
        """Configure Celune's voice information."""
        self.voices = voices

    def set_voice(self, name: str) -> bool:
        """Extension method for changing Celune's voice."""
        if name not in self.voices:
            self.log(f"Unknown voice: {name}")
            return False

        self.change_voice(self.voices[name])
        self.voice_changed_callback(name)
        return True

    def setup_extensions(self) -> None:
        """Configure Celune's extension manager."""
        self.log("[EXT] Setting up extension manager")

        ctx = CeluneContext(
            log=self.log,
            say=self.say,
            status=self.status_callback,
            set_voice=self.set_voice,
            name="Celune",
            version=__version__,
        )
        self.extension_manager = CeluneExtensionManager(ctx)
        self.extension_manager.autoload("extensions")

        self.log(f"[EXT] Loaded extensions: {self.extension_manager.list_extensions()}")

    def log(self, msg: str, severity: str = "info") -> None:
        """Log a message."""
        self.log_callback(msg, severity)

    def change_voice(self, voice: tuple[str, str, int]) -> None:
        """Change Celune's voice parameters."""
        self.log("Currently selected voice data:")
        self.log(", ".join([voice[0], voice[1]]))
        self.log(f"Seed changed to {voice[2]}")

        self.ref_audio = voice[0]
        self.ref_text = voice[1]

        random.seed(voice[2])
        np.random.seed(voice[2])
        torch.cuda.manual_seed_all(voice[2])

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self) -> bool:
        """Load and initialize Celune."""
        disable_progress_bar()
        disable_progress_bars()
        hf_logging.set_verbosity_error()

        available, path = self._model_is_available_locally(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        if available:
            self.log("TTS model is already available in cache")
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = FasterQwen3TTS.from_pretrained(path)
        else:
            self.log("Downloading TTS model...")
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
            for _, _, _ in self.model.generate_voice_clone_streaming(
                text=warmup_text,
                language=self.language,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                chunk_size=self.chunk_size,
            ):
                pass
            warmup_end = time.perf_counter()
            warmup_took = warmup_end - warmup_start
            self.log(f"[WARMUP] done, took {warmup_took:.2f} seconds")
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
                assert norm_token_id != self.tokenizer.unk_token_id, "not a CeluneNorm normalizer"

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
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
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
                self.log(f"Normalization took {inf_total:.2f} seconds.")

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

        # if the GPU is experiencing higher activity, Celune may not have unlocked yet
        if self.locked:
            self.log("Tried to speak during generation.", "warning")
            self.error_callback("Celune is currently busy")
            return False

        self.locked = True
        self.status_callback("Normalizing")
        normalized = self.normalize(text)

        # if normalization did not return a meaningful result, Celune says raw text
        # Celune will also say raw text if normalization is disabled
        # don't set CELUNE_NORMALIZE=1 to disable normalization
        if normalized is None:
            self.text_queue.put(text)  # say as is
        else:
            self.text_queue.put(normalized)  # from CeluneNorm
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

    @staticmethod
    def _resample_audio(
        audio: np.ndarray, source_sr: int, target_sr: int = 48000
    ) -> np.ndarray:
        """Resample the given audio to the given sample rate."""
        audio = np.asarray(audio, dtype=np.float32)

        if source_sr == target_sr:
            return audio

        factor = math.gcd(source_sr, target_sr)
        up = target_sr // factor
        down = source_sr // factor

        return np.asarray(
            resample_poly(audio, up=up, down=down, axis=0), dtype=np.float32
        )

    def _to_48khz(self, audio: np.ndarray, source_sr: int) -> np.ndarray:
        """Cast an audio chunk to 48 kHz stereo format."""
        audio = self._resample_audio(audio, source_sr, 48000)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        elif audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)

        return audio

    @staticmethod
    def _soften_onset(
        audio: np.ndarray, sr: int, duration: float = 0.2, start_gain: float = 0.5
    ) -> np.ndarray:
        """Soften the leading audio."""
        samples = int(sr * duration)
        samples = min(samples, len(audio))

        ramp = np.linspace(start_gain, 1.0, samples, dtype=np.float32)

        audio[:samples, 0] *= ramp
        audio[:samples, 1] *= ramp

        return audio

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
                start_time = time.perf_counter()
                self.log(f"[GEN] {text}")
                speech_len = 0.0

                chunks = self._split_text(text)

                for chunk_index, chunk_text in enumerate(chunks):
                    if self._exit_requested:
                        break

                    is_first_chunk = chunk_index == 0

                    for (
                        audio_chunk,
                        sr,
                        timing,
                    ) in self.model.generate_voice_clone_streaming(
                        text=chunk_text,
                        language=self.language,
                        ref_audio=self.ref_audio,
                        ref_text=self.ref_text,
                        chunk_size=self.chunk_size,
                    ):
                        if self._exit_requested:
                            break

                        if hasattr(audio_chunk, "cpu"):
                            audio_chunk = audio_chunk.cpu().numpy()

                        audio_chunk = self._to_48khz(audio_chunk, sr)

                        if is_first_chunk:
                            audio_chunk = self._soften_onset(audio_chunk, 48000)

                        if self._exit_requested:
                            break

                        self.audio_queue.put((audio_chunk, 48000, timing))

                        chunk_dur = len(audio_chunk) / 48000
                        speech_len += chunk_dur

                    if self._exit_requested:
                        break

                    self.audio_queue.put(self._utterance_done)

                end_time = time.perf_counter()
                generation_time = end_time - start_time

                if self._exit_requested:
                    self.cur_state = "idle"
                    self.locked = False
                    continue

                self.log(
                    f"[GEN] {speech_len:.2f} seconds, took {generation_time:.2f} seconds, "
                    f"RTF: {speech_len / generation_time:.2f}"
                )
                self.status_callback("Speaking")
                self.cur_state = "speaking"
                self.log("[GEN] done")
                self.queue_avail_callback()

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

                    avail, total = tuple(v / 1024 ** 3 for v in torch.cuda.mem_get_info(0))
                    if avail <= 2:
                        self.log(f"Celune is running out of VRAM ({avail:.2f}/{total:.2f} GB available).", "warning")
                        self.log("Please close any memory-resident applications to improve performance.", "warning")
                    else:
                        self.log(f"Available VRAM: {avail:.2f}/{total:.2f} GB")
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
                        self.log("Celune could not initialize the audio stream.", "error")
                        self.log("No suitable audio device is available.", "error")
                        self.error_callback("No suitable audio devices")
                    self._audio_unavailable = True

            if sr != self._current_sr:  # Celune audio stream must be 48 kHz
                raise RuntimeError(
                    f"Sample rate changed from {self._current_sr} to {sr}"
                )

            if self._exit_requested:
                continue

            self._stream.write(audio_chunk)
