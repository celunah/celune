# pylint: disable=R0902, R0913, R0917, W0718
"""
Celune - A celestial TTS engine.
She has three tones, and can change them on the fly.
"""

import threading
import time
import hashlib
import os
import sys
import re
import queue
import random
import traceback
import platform
import itertools
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from faster_qwen3_tts import FasterQwen3TTS

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import TextArea, RichLog, Label, Button

from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()


class LogRedirect:
    """Redirect logs to the logger."""

    def __init__(self, write_callback):
        self.write_callback = write_callback
        self._buffer = ""

    def write(self, text):
        """Write text to the logger."""
        if not text:
            return

        if "`torch_dtype` is deprecated" in text:
            return

        self._buffer += text

        while "\n" in self._buffer or "\r" in self._buffer:
            newline_pos = self._buffer.find("\n") if "\n" in self._buffer else 10**9
            cr_pos = self._buffer.find("\r") if "\r" in self._buffer else 10**9
            pos = min(newline_pos, cr_pos)

            chunk = self._buffer[:pos].strip()
            self._buffer = self._buffer[pos + 1 :]

            if chunk:
                self.write_callback(chunk)

    def flush(self):
        """Flush the buffers."""
        if self._buffer.strip():
            self.write_callback(self._buffer.strip())
        self._buffer = ""


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
        queue_maxsize: int = 32,
        language: str = "English",
        log_callback: Optional[Callable[[str], None]] = None,
        idle_callback: Optional[Callable[[], None]] = None,
        queue_avail_callback: Optional[Callable[[], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model_name = model_name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.chunk_size = chunk_size
        self.prebuffer_chunks = prebuffer_chunks
        self.sentences_per_chunk = sentences_per_chunk
        self.queue_maxsize = queue_maxsize
        self.language = language
        self.log_callback = log_callback or (lambda msg: None)
        self.idle_callback = idle_callback or (lambda: None)
        self.queue_avail_callback = queue_avail_callback or (lambda: None)
        self.error_callback = error_callback or (lambda error: None)
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
        self.cur_state = "init"
        self.llm = None

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

        warmup_result = self._warmup()
        if warmup_result:
            self.log("Ready to speak.")
            self.locked = False
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

    def say(self, text: str):
        """Queue text for Celune to say."""
        self.text_queue.put(text)
        self.log(f"[QUEUE] {text}")

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

    @staticmethod
    def _normalize_input(text: str):
        """LLM text normalization stub."""
        return text

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

                    chunk_text = self._normalize_input(chunk_text)

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


class CeluneUI(App):
    """Celune's user interface."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #logs {
        height: 1fr;
        border: round #ceaaff;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1;
    }

    /* give scrollbar colors only to the elements that will have a scrollbar */
    #logs, #input {
        scrollbar-color: #ceaaff;
        scrollbar-color-hover: #debaff;
        scrollbar-color-active: #eecaff;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
    }

    #logs:focus {
        border: round #ceaaff;
        background: transparent;
    }

    #input {
        min-height: 3;
        height: 3;
        width: 1fr;
        border: round #ceaaff;
    }

    #style {
        width: 12;
        height: 3;
        border: round #ceaaff;
        margin-right: 1;
    }

    #input:focus {
        border: round #ceaaff;
        background-tint: #ceaaff 10%;
    }

    #logs, #input {
        margin-left: 1;
        margin-right: 1;
    }

    #status {
        height: 1;
        background: $surface;
        width: 1fr;
        margin-left: 2;
        margin-bottom: 1;
        color: #ceaaff;
    }

    #header {
        height: 3;
        width: 1fr;
        text-align: center;
        border: round transparent;  /* v-align hack */
        color: #ceaaff;
    }

    #controls {
        height: auto;
    }
    """

    def __init__(self):
        super().__init__()
        self.logs = None
        self.input_box = None
        self.style_button = None
        self.status = None
        self.tts = None
        self.tts_ready = False
        self.cur_state = "active"
        self.tts_styles = itertools.cycle(["Neutral", "Calm", "Energetic"])
        self.tts_voices = None
        self.cur_style = next(self.tts_styles)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = None
        self._log_stderr = None

    def compose(self) -> ComposeResult:
        """Define the UI."""
        with Vertical(id="container"):
            yield Label("Celune", id="header")
            yield RichLog(id="logs", wrap=True, markup=False)
            with Horizontal(id="controls"):
                yield TextArea(id="input", placeholder="Please wait")
                yield Button("Neutral", id="style", disabled=True)
            yield Label("Initializing", id="status")

    def on_mount(self) -> None:
        """Prepare Celune."""
        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.style_button = self.query_one("#style", Button)

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._log_stdout = LogRedirect(self.safe_log)
        self._log_stderr = LogRedirect(self.safe_log)

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr

        self.call_after_refresh(self.start_background_init)

    def start_background_init(self) -> None:
        """Run Celune's initialization function."""
        self.load_tts()

    @work(thread=True, exclusive=True)
    def load_tts(self) -> None:
        """Load Celune."""
        try:
            tts_voices = {
                "neutral": (
                    "refs/neutral.wav",
                    "My name is Celune, pronounced Celune. It is a pleasure to meet you.",
                    4243102495,
                ),
                "calm": (
                    "refs/calm.wav",
                    "My name is... Celune... It is so... quiet.",
                    418977738,
                ),
                "energetic": (
                    "refs/energetic.wav",
                    "My name is Celune! Let's do this, we have to get it done!",
                    590298652,
                ),
            }

            self.tts_voices = itertools.cycle(tts_voices.values())
            tts_hashes = {
                "neutral": "",
                "calm": "",
                "energetic": "",
            }

            for voice_name, (
                voice_path,
                _,
                _,
            ) in tts_voices.items():  # ignore seed and ref text
                if not os.path.exists(voice_path):
                    self.safe_log(f"Reference voice '{voice_name}' not found.")
                    self.safe_status(f"Missing reference voice '{voice_name}'")
                    return

                checksum_path = f"{os.path.splitext(voice_path)[0]}.sha256"

                if os.path.exists(checksum_path):
                    with open(checksum_path, "r", encoding="utf-8") as f:
                        tts_hashes[voice_name] = f.read().strip()

                    with open(voice_path, "rb") as f:
                        voice_hash = hashlib.file_digest(f, "sha256").hexdigest()

                    if voice_hash != tts_hashes[voice_name]:
                        self.safe_log(
                            f"Voice file mismatch, voice '{voice_name}' may be affected."
                        )
                else:
                    self.safe_log(f"Reference voice '{voice_name}' has no checksum.")

            voice_data = next(self.tts_voices)
            self.tts = Celune(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                ref_audio=voice_data[0],
                ref_text=voice_data[1],
                chunk_size=24,
                prebuffer_chunks=2,
                sentences_per_chunk=4,
                log_callback=self.tts_log,
                idle_callback=self.tts_idle,
                queue_avail_callback=self.tts_queue_avail,
                error_callback=self.tts_error,
            )

            if self.tts.load():
                self.tts_ready = True
                self.safe_status("Ready")
                self.style_button.disabled = False
                self.input_box.placeholder = "Enter text to speak here"

        except Exception as e:
            self.safe_log(f"[INIT ERROR] {traceback.format_exc()}")
            self.safe_status(f"{e.__class__.__name__}: {e}")
            self.cur_state = "error"

    def safe_status(self, msg: str) -> None:
        """Update current status."""
        if self.cur_state == "exiting" or self.logs is None:
            return

        if threading.current_thread() is threading.main_thread():
            self.status.update(msg)
        else:
            self.call_from_thread(self.status.update, msg)

    def safe_log(self, msg: str) -> None:
        """Log a message."""
        if self.cur_state == "exiting" or self.logs is None:
            return

        if threading.current_thread() is threading.main_thread():
            self.logs.write(msg)
        else:
            self.call_from_thread(self.logs.write, msg)

    def tts_log(self, msg: str) -> None:
        """Set status from TTS log."""
        if self.cur_state == "exiting":
            return

        if msg.startswith("[QUEUE]"):
            self.safe_status("Queued")
        elif msg.startswith("[GEN]"):
            self.safe_status("Generating")
        elif msg.startswith("[WARMUP]"):
            self.safe_status("Warming up")

        self.safe_log(msg)

    def on_key(self, event: events.Key) -> None:
        """Accept input and send text to Celune."""
        if self.cur_state == "exiting":
            return

        if event.key == "ctrl+j":
            if not self.tts or self.tts.locked:
                return

            text = self.input_box.text.strip()

            if not text:
                return

            self.logs.write(f"[INPUT] {text}")
            self.status.update("Queued")
            self.tts.locked = True
            self.style_button.disabled = True
            self.input_box.placeholder = "Please wait"
            self.tts.say(text)
            self.input_box.load_text("")
            event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Change Celune's tone."""
        if self.cur_state == "exiting":
            return

        if event.button != self.style_button:
            return

        self.style_button.label = next(self.tts_styles)
        self.tts.change_voice(next(self.tts_voices))

    def on_unmount(self) -> None:
        """Unload Celune."""
        self.cur_state = "exiting"

        if self.tts is not None:
            self.tts.close()

        if hasattr(self, "_old_stdout"):
            sys.stdout = self._old_stdout
        if hasattr(self, "_old_stderr"):
            sys.stderr = self._old_stderr

    def tts_idle(self):
        """Reset UI state after Celune stops talking."""
        if self.cur_state == "exiting":
            return
        self.tts.locked = False
        self.tts.cur_state = "idle"
        self.input_box.placeholder = "Enter text to speak here"
        self.safe_status("Idle")

    def tts_queue_avail(
        self,
    ):  # allow enqueuing new text while speaking but after generation
        """Unlock input queueing after Celune completes the generation."""
        if self.cur_state == "exiting":
            return
        self.tts.locked = False
        self.safe_status("Speaking")
        self.input_box.placeholder = "Enter text to speak here"
        self.style_button.disabled = False

    def tts_error(self, error: str) -> None:
        """Set UI status to the error message."""
        if self.cur_state == "exiting":
            return
        self.safe_status(error)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Expand the input size as needed."""
        if self.cur_state == "exiting":
            return

        if event.text_area.id != "input":
            return

        text = event.text_area.text
        line_count = text.count("\n") + 1
        min_lines = 1
        max_lines = 8

        visible_lines = max(min_lines, min(line_count, max_lines))
        event.text_area.styles.height = visible_lines + 2


if __name__ == "__main__":
    sys.stdout.write("\x1b]2;Celune\x07")
    sys.stdout.flush()

    CeluneUI().run()