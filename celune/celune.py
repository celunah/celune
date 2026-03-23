# pylint: disable=R0902, R0913, R0917, W0718
"""Celune's backend layer."""

import re
import sys
import time
import queue
import random
import platform
import threading
import traceback
from typing import Optional, Callable

import torch
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from faster_qwen3_tts import FasterQwen3TTS

import celune
from .extensions.base import CeluneContext
from .extensions.manager import CeluneExtensionManager


class Celune:
    """The character engine for Celune."""

    def __init__(
        self,
        model_name: str,
        ref_audio: str,
        ref_text: str,
        chunk_size: int = 24,
        prebuffer_chunks: int = 2,
        sentences_per_chunk: int = 4,
        language: str = "Auto",
        log_callback: Optional[Callable[[str], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
        queue_avail_callback: Optional[Callable[[], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
        voice_changed_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_name = model_name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.chunk_size = chunk_size
        self.prebuffer_chunks = prebuffer_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.language = language
        self.log_callback = log_callback or (lambda msg: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.error_callback = error_callback or (lambda error: None)
        self.status_callback = status_callback or (lambda msg, severity: None)
        self.voice_changed_callback = voice_changed_callback or (lambda name: None)
        self.current_voice = "neutral"
        self.voices: dict[str, tuple[str, str, int]] = {}

        self.model = None
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self._stop = False
        self._sentinel = object()
        self._playback_thread = None
        self._generation_thread = None
        self._stream: Optional[sd.OutputStream] = None
        self._current_sr: Optional[int] = None
        self._utterance_done = object()
        self.locked = True
        self.loaded = False
        self.cur_state = "init"
        self.extension_manager: CeluneExtensionManager | None = None

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
            version="2.0",
        )
        self.extension_manager = CeluneExtensionManager(ctx)
        self.extension_manager.autoload("extensions")

        self.log(f"[EXT] Loaded extensions: {self.extension_manager.list_extensions()}")

    def log(self, msg: str) -> None:
        """Log a message."""
        if self._stop:
            return

        self.log_callback(msg)

    def change_voice(self, voice: tuple[str, str, int]) -> None:
        """Change Celune's voice parameters."""
        if self._stop:
            return

        self.log_callback("Currently selected voice data:")
        self.log_callback(", ".join([voice[0], voice[1]]))
        self.log_callback(f"Seed changed to {voice[2]}")

        self.ref_audio = voice[0]
        self.ref_text = voice[1]

        random.seed(voice[2])
        np.random.seed(voice[2])
        torch.cuda.manual_seed_all(voice[2])

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load(self):
        """Load and initialize Celune."""
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
            f"Celune {celune.__version__}, "
            f"Python {platform.python_version()}, "
            f"PyTorch {torch.__version__}, "
            f"CUDA {torch.version.cuda}"
        )

        if sys.version_info < (3, 12):
            self.log("Celune does not work on this version of Python")
            return False

        cuda_avail = torch.cuda.is_available()
        self.log(f"CUDA available: {cuda_avail}")

        if not cuda_avail:
            self.log("No GPUs found.")
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
        return True

    def _warmup(self):
        """Warm up Celune's speech capabilities."""
        self.log("[WARMUP] Warming up...")
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
            self.log(f"[WARMUP ERROR] {traceback.format_exc()}")
            self.cur_state = "error"
            self.error_callback(f"{e.__class__.__name__}: {e}")
            return False

    def say(self, text: str) -> bool:
        """Queue text for Celune to say."""
        if not self.loaded:
            self.log("Tried to speak before model was loaded.")
            self.error_callback("Celune is not currently ready")
            return False

        if self.locked:
            self.log("Tried to speak during generation.")
            self.error_callback("Celune is currently busy")
            return False

        self.locked = True
        self.text_queue.put(text)
        return True

    def close(self):
        """Shut off Celune and exit."""
        self.log("Exiting.")
        self._stop = True
        self.text_queue.put(self._sentinel)
        self.audio_queue.put(self._sentinel)

        if self._generation_thread is not None:
            self._generation_thread.join(timeout=2)

        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2)

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def _48khz_chunk(chunk):
        """Cast an audio chunk to 48 kHz stereo format."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        upsampled = np.asarray(resample_poly(chunk, up=2, down=1), dtype=np.float32)
        stereo = np.column_stack((upsampled, upsampled))
        return stereo

    @staticmethod
    def _soften_onset(audio, sr, duration=0.2, start_gain=0.5):
        """Soften the leading audio."""
        samples = int(sr * duration)

        samples = min(samples, len(audio))

        ramp = np.linspace(start_gain, 1.0, samples, dtype=np.float32)

        audio[:samples, 0] *= ramp
        audio[:samples, 1] *= ramp

        return audio

    def _split_text(self, text: str):
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

    def _generation_worker(self):
        """Generate audio tokens and send them to the audio pipeline."""
        while True:
            text = self.text_queue.get()
            if text is self._sentinel:
                self.audio_queue.put(self._sentinel)
                break

            try:
                start_time = time.perf_counter()
                self.log(f"[GEN] {text}")
                speech_len = 0

                chunks = self._split_text(text)

                for chunk_index, chunk_text in enumerate(chunks):
                    is_first_chunk = chunk_index == 0

                    for (
                        audio_chunk,
                        _,
                        timing,
                    ) in self.model.generate_voice_clone_streaming(
                        text=chunk_text,
                        language=self.language,
                        ref_audio=self.ref_audio,
                        ref_text=self.ref_text,
                        chunk_size=self.chunk_size,
                    ):
                        if hasattr(audio_chunk, "cpu"):
                            audio_chunk = audio_chunk.cpu().numpy()

                        audio_chunk = self._48khz_chunk(audio_chunk)

                        if is_first_chunk:
                            audio_chunk = self._soften_onset(audio_chunk, 48000)

                        self.audio_queue.put((audio_chunk, 48000, timing))

                        chunk_dur = len(audio_chunk) / 48000
                        speech_len += chunk_dur

                    self.audio_queue.put(self._utterance_done)

                end_time = time.perf_counter()
                generation_time = end_time - start_time
                self.log(
                    f"[GEN] {speech_len:.2f} seconds, took {generation_time:.2f} seconds, "
                    f"RTF: {speech_len / generation_time:.2f}"
                )
                self.cur_state = "speaking"
                self.log("[GEN] done")
                self.queue_avail_callback()

            except Exception as e:
                self.log(f"[GEN ERROR] {traceback.format_exc()}")
                self.cur_state = "error"
                self.error_callback(e)

    def _playback_worker(self):
        """Receive audio chunks and play them."""

        started = False

        while True:
            if not started:
                while self.audio_queue.qsize() < self.prebuffer_chunks:
                    if self._stop:
                        return
                    time.sleep(0.01)

            item = self.audio_queue.get()
            if item is self._sentinel:
                break

            if item is self._utterance_done:
                more_pending = (not self.audio_queue.empty()) or (
                    not self.text_queue.empty()
                )

                if more_pending:
                    silence = np.zeros((48000, 2), dtype=np.float32)
                    self._stream.write(silence)
                else:
                    self.log("[STATE] idle")
                    self.idle_callback()
                    self.log("Ready to speak.")
                continue

            audio_chunk, sr, _ = item

            if self._stream is None:
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

            if sr != self._current_sr:
                raise RuntimeError(
                    f"Sample rate changed from {self._current_sr} to {sr}"
                )

            self._stream.write(audio_chunk)
