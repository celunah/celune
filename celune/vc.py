# SPDX-License-Identifier: Apache-2.0
"""Shared runtime helpers for Celune voice conversion."""

from __future__ import annotations

import math
import queue as queue_module
import shutil
import tempfile
import importlib
import threading
import contextlib
from typing import TYPE_CHECKING, Literal, Optional, cast
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import torch
import soundfile as sf
from scipy import signal

from .i18n import string
from .paths import app_data_dir
from .binding import install_class_functions
from .typing.aliases import AudioChunk
from .typing.backends import StreamingSpeechModel
from .dataclasses.pipeline import AudioOutput, AudioInputRequest

if TYPE_CHECKING:
    from .celune import Celune

VC_VAD_RMS_THRESHOLD = 0.005
VC_VAD_HANGOVER_SECONDS = 0.3
VC_VAD_PREROLL_SECONDS = 0.18
VC_LIVE_CHUNK_SECONDS = 0.18
VC_LIVE_CHUNK_OVERLAP_SECONDS = 0.0
VC_PLAYBACK_ATTENUATION_DB = 2.0
VC_PLAYBACK_GAIN = 10.0 ** (-VC_PLAYBACK_ATTENUATION_DB / 20.0)
_VC_AUDIO_PEAK = 0.95

_LIVE_VAD_TARGET_SAMPLE_RATE = 16000
_LIVE_VAD_FRAME_SAMPLES = 512
_LIVE_VAD_THRESHOLD = 0.50
_LIVE_VAD_NEGATIVE_THRESHOLD = 0.35

__all__ = [
    "VC_LIVE_CHUNK_OVERLAP_SECONDS",
    "VC_LIVE_CHUNK_SECONDS",
    "VC_PLAYBACK_ATTENUATION_DB",
    "VC_PLAYBACK_GAIN",
    "VC_VAD_HANGOVER_SECONDS",
    "VC_VAD_PREROLL_SECONDS",
    "VC_VAD_RMS_THRESHOLD",
    "LiveVoiceActivityDetector",
    "VCDebugCapture",
    "VCDebugSaveResult",
    "convert_audio",
    "convert_live_audio",
    "create_live_voice_activity_detector",
    "normalize_vc_audio",
    "vc_input_has_voice",
    "vc_input_rms",
    "vc_live_chunk_frames",
    "vc_live_chunk_overlap_frames",
    "vc_vad_hangover_frames",
    "vc_vad_preroll_frames",
]


def normalize_vc_audio(audio: AudioChunk) -> AudioChunk:
    """Return finite float32 audio safe for a CEDTS VC payload.

    Args:
        audio: Audio samples that will cross the voice-conversion boundary.

    Returns:
        AudioChunk: A finite float32 array with a small peak headroom margin.
    """
    normalized = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if normalized.size == 0:
        return normalized

    peak = float(np.max(np.abs(normalized)))
    if peak > _VC_AUDIO_PEAK:
        normalized = normalized * (_VC_AUDIO_PEAK / peak)
    return np.asarray(normalized, dtype=np.float32)


def convert_audio(
    engine: Celune,
    audio: np.ndarray,
    sample_rate: int,
    label: str = "audio input",
    f0_condition: Optional[bool] = None,
) -> Optional[AudioOutput]:
    """Convert submitted audio and return the generated VC output."""
    if not engine.is_voice_conversion_mode():
        engine.log(string("celune.audio_conversion_unavailable"), "warning")
        engine.error_callback(string("celune.not_possible"))
        engine.progress_callback(0, 1)
        return None

    from .speech import convert_audio_input

    return convert_audio_input(
        engine,
        AudioInputRequest(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            label=label,
            f0_condition=f0_condition,
        ),
    )


def convert_live_audio(
    engine: Celune,
    audio: np.ndarray,
    sample_rate: int,
    label: str = "audio input",
    f0_condition: Optional[bool] = None,
) -> Optional[AudioOutput]:
    """Convert one low-latency live block through the active VC backend."""
    if not engine.is_voice_conversion_mode():
        return engine.convert_audio(
            audio,
            sample_rate,
            label=label,
            f0_condition=f0_condition,
        )

    from .speech import convert_audio_input

    return convert_audio_input(
        engine,
        AudioInputRequest(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sample_rate,
            label=label,
            f0_condition=f0_condition,
        ),
        live=True,
    )


def install(target):
    """Install public VC entrypoints on ``Celune``."""
    install_class_functions(
        target,
        {name: globals()[name] for name in ("convert_audio", "convert_live_audio")},
    )


def vc_input_rms(audio: AudioChunk) -> float:
    """Return RMS energy for one microphone callback buffer.

    Args:
        audio: One live microphone callback buffer.

    Returns:
        float: The RMS energy of the provided buffer.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))


def vc_vad_hangover_frames(sample_rate: int) -> int:
    """Return tolerated trailing silent frames before one VC flush.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many silent frames to retain before ending one speech chunk.
    """
    return max(1, int(sample_rate * VC_VAD_HANGOVER_SECONDS))


def vc_vad_preroll_frames(sample_rate: int) -> int:
    """Return retained pre-speech frames before one VC onset.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many pre-speech frames to retain for chunk onset recovery.
    """
    return max(1, int(sample_rate * VC_VAD_PREROLL_SECONDS))


def vc_input_has_voice(audio: AudioChunk) -> bool:
    """Return whether one live callback buffer likely contains voice.

    Args:
        audio: One live microphone callback buffer.

    Returns:
        bool: Whether the buffer meets the fallback RMS speech threshold.
    """
    return vc_input_rms(audio) >= VC_VAD_RMS_THRESHOLD


def vc_live_chunk_frames(sample_rate: int) -> int:
    """Return the active-speech chunk size for low-latency live VC.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: How many frames to accumulate before flushing one mid-speech chunk.
    """
    return max(1, int(sample_rate * VC_LIVE_CHUNK_SECONDS))


def vc_live_chunk_overlap_frames(sample_rate: int) -> int:
    """Return the retained tail overlap between adjacent live VC chunks.

    Args:
        sample_rate: Sample rate used by the live VC input stream.

    Returns:
        int: Zero when no overlap is configured because Seed-VC's native live path
            owns crossfade alignment; otherwise the configured overlap in frames.
    """
    if VC_LIVE_CHUNK_OVERLAP_SECONDS <= 0:
        return 0
    return max(1, int(sample_rate * VC_LIVE_CHUNK_OVERLAP_SECONDS))


def _normalize_live_audio(audio: AudioChunk) -> AudioChunk:
    """Return one mono float32 waveform for live input helpers."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim != 2:
        raise ValueError(f"expected 1D or 2D live audio, got {normalized.shape}")
    if normalized.shape[1] == 1:
        return normalized[:, 0]
    return np.asarray(
        np.mean(normalized, axis=1, dtype=np.float32),
        dtype=np.float32,
    )


def _resample_audio(
    audio: AudioChunk,
    source_sample_rate: int,
    target_sample_rate: int,
) -> AudioChunk:
    """Resample one mono waveform when the sample rate differs."""
    if source_sample_rate == target_sample_rate:
        return np.asarray(audio, dtype=np.float32)

    gcd = int(np.gcd(source_sample_rate, target_sample_rate))
    up = target_sample_rate // gcd
    down = source_sample_rate // gcd
    return np.asarray(
        signal.resample_poly(audio, up, down),
        dtype=np.float32,
    )


def _torch_probability(output: torch.Tensor) -> float:
    """Convert one model output tensor into a scalar probability."""
    values = output.detach().reshape(-1)
    if values.numel() <= 0:
        return 0.0
    return float(values[-1].item())


class LiveVoiceActivityDetector:
    """Stateful Silero-based speech detector for live VC capture."""

    def __init__(
        self,
        model: StreamingSpeechModel,
        *,
        threshold: float = _LIVE_VAD_THRESHOLD,
        negative_threshold: float = _LIVE_VAD_NEGATIVE_THRESHOLD,
        target_sample_rate: int = _LIVE_VAD_TARGET_SAMPLE_RATE,
        frame_samples: int = _LIVE_VAD_FRAME_SAMPLES,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self.target_sample_rate = target_sample_rate
        self.frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_active = False
        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._generation = 0
        self._input_queue: queue_module.Queue[Optional[tuple[AudioChunk, int]]] = (
            queue_module.Queue(maxsize=2)
        )
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

        reset_states = getattr(self.model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def _ensure_worker(self) -> None:
        """Start the detector worker lazily when live audio first arrives."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            return

        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run,
            name="celune-live-vad",
            daemon=True,
        )
        self._worker.start()

    def _reset_locked(self) -> None:
        """Reset detector buffers while holding the state lock."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._speech_active = False

    def _reset_model(self) -> None:
        """Reset the model's streaming state outside the audio callback."""
        reset_states = getattr(self.model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def _process_audio(self, audio: AudioChunk, sample_rate: int) -> None:
        """Process one queued callback buffer on the detector worker."""
        mono_audio = _normalize_live_audio(audio)
        resampled = _resample_audio(
            mono_audio,
            sample_rate,
            self.target_sample_rate,
        )
        with self._state_lock:
            if self._pending.size > 0:
                resampled = np.concatenate((self._pending, resampled))
            generation = self._generation
            speech_active = self._speech_active

            complete_frames = (
                len(resampled) // self.frame_samples
            ) * self.frame_samples
            if complete_frames <= 0:
                self._pending = resampled
                return

        with self._model_lock, torch.inference_mode():
            for start in range(0, complete_frames, self.frame_samples):
                frame = resampled[start : start + self.frame_samples]
                probability = _torch_probability(
                    self.model(
                        torch.from_numpy(frame),
                        self.target_sample_rate,
                    )
                )
                if speech_active:
                    if probability < self.negative_threshold:
                        speech_active = False
                elif probability >= self.threshold:
                    speech_active = True

        with self._state_lock:
            if generation != self._generation:
                return
            self._speech_active = speech_active
            self._pending = np.asarray(resampled[complete_frames:], dtype=np.float32)

    def _run(self) -> None:
        """Consume microphone buffers without running inference in PortAudio."""
        while not self._stop_event.is_set():
            item = self._input_queue.get()
            if item is None:
                if self._stop_event.is_set():
                    return
                with self._model_lock:
                    self._reset_model()
                continue
            audio, sample_rate = item
            try:
                self._process_audio(audio, sample_rate)
            except (RuntimeError, AssertionError, ValueError):
                with self._state_lock:
                    self._reset_locked()
                    self._speech_active = vc_input_has_voice(audio)
                with self._model_lock:
                    self._reset_model()

    def reset(self) -> None:
        """Reset buffered audio and the streaming detector state."""
        with self._state_lock:
            self._generation += 1
            self._reset_locked()

        worker = self._worker
        if worker is None or not worker.is_alive():
            self._reset_model()
            return

        with contextlib.suppress(queue_module.Full):
            self._input_queue.put_nowait(None)
            return
        with contextlib.suppress(queue_module.Empty):
            self._input_queue.get_nowait()
        with contextlib.suppress(queue_module.Full):
            self._input_queue.put_nowait(None)

    def close(self) -> None:
        """Stop the detector worker and release its queued callback buffers."""
        self._stop_event.set()
        with contextlib.suppress(queue_module.Full):
            self._input_queue.put_nowait(None)
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)
        self._worker = None

    def has_voice(
        self,
        audio: AudioChunk,
        sample_rate: int,
    ) -> bool:
        """Return whether one live microphone callback likely contains speech.

        Args:
            audio: Live microphone audio to inspect.
            sample_rate: Sample rate of ``audio``.

        Returns:
            bool: Whether speech appears active in the current callback window.
        """
        self._ensure_worker()
        queued_audio = np.asarray(audio, dtype=np.float32)
        try:
            self._input_queue.put_nowait((queued_audio, sample_rate))
        except queue_module.Full:
            with contextlib.suppress(queue_module.Empty):
                self._input_queue.get_nowait()
            with contextlib.suppress(queue_module.Full):
                self._input_queue.put_nowait((queued_audio, sample_rate))
        with self._state_lock:
            return self._speech_active


def create_live_voice_activity_detector() -> Optional[LiveVoiceActivityDetector]:
    """Return the AI VAD used by live capture when its optional backend is available.

    Returns:
        Optional[LiveVoiceActivityDetector]: The live detector, or ``None`` when
            Silero VAD is unavailable and the caller must use its RMS fallback.
    """
    try:
        silero_vad = importlib.import_module("silero_vad")
    except ImportError:
        return None

    try:
        load_silero_vad = getattr(silero_vad, "load_silero_vad", None)
        if not callable(load_silero_vad):
            return None
        return LiveVoiceActivityDetector(cast(StreamingSpeechModel, load_silero_vad()))
    except (RuntimeError, AssertionError, ValueError, ImportError):
        return None


_DEBUG_SAMPLE_RATE = 48000
_CaptureKind = Literal["microphone", "converted"]
_CaptureItem = tuple[_CaptureKind, AudioChunk, int]


@dataclass(frozen=True)
class VCDebugSaveResult:
    """Paths and errors returned when one VC debug capture is finalized."""

    directory: Path
    microphone_path: Optional[Path]
    converted_path: Optional[Path]
    errors: tuple[str, ...]


class VCDebugCapture:
    """Write live VC input and output to a temporary diagnostic directory."""

    def __init__(self) -> None:
        """Create a temporary capture directory and start its disk writer."""
        temporary_root = app_data_dir(create=True)
        self.directory = Path(tempfile.mkdtemp(prefix="vc-debug-", dir=temporary_root))
        self._queue: queue_module.Queue[Optional[_CaptureItem]] = queue_module.Queue()
        self._lock = threading.Lock()
        self._accepting = True
        self._stop_queued = False
        self._next_token = 0
        self._pending_tokens: set[int] = set()
        self._errors: list[str] = []
        self._on_complete: Optional[Callable[[VCDebugSaveResult], None]] = None
        self._writer = threading.Thread(
            target=self._write_capture,
            name="celune-vc-debug-writer",
        )
        try:
            self._writer.start()
        except Exception:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise

    def record_microphone(self, audio: AudioChunk, sample_rate: int) -> None:
        """Queue one raw microphone block without changing its signal level.

        Args:
            audio: Captured microphone samples in mono or multichannel form.
            sample_rate: Sample rate of ``audio`` in hertz.
        """
        prepared = self._prepare_audio(audio, sample_rate)
        if prepared is None:
            self._record_error("Invalid microphone audio block")
            return

        with self._lock:
            if not self._accepting:
                return
            self._queue.put(("microphone", prepared[0], prepared[1]))

    def begin_conversion(self) -> Optional[int]:
        """Track an in-flight conversion so stopping waits for its output.

        Returns:
            Optional[int]: Token to complete after conversion, or ``None`` if
                capture has stopped.
        """
        with self._lock:
            if not self._accepting:
                return None
            token = self._next_token
            self._next_token += 1
            self._pending_tokens.add(token)
            return token

    def record_conversion(
        self,
        token: Optional[int],
        audio: Optional[AudioChunk] = None,
        sample_rate: int = 0,
    ) -> None:
        """Queue converted audio for a tracked conversion and release its token.

        Args:
            token: Token returned by :meth:`begin_conversion`.
            audio: Converted samples, if conversion produced audio.
            sample_rate: Sample rate of ``audio`` in hertz.
        """
        if token is None:
            return

        prepared = None
        if audio is not None and len(audio):
            prepared = self._prepare_audio(audio, sample_rate)
            if prepared is None:
                self._record_error("Invalid converted audio block")

        with self._lock:
            if token not in self._pending_tokens:
                return
            if prepared is not None:
                self._queue.put(("converted", prepared[0], prepared[1]))
            self._pending_tokens.remove(token)
            self._queue_stop_if_ready_locked()

    def stop(
        self,
        on_complete: Optional[Callable[[VCDebugSaveResult], None]] = None,
        *,
        discard_pending: bool = False,
    ) -> None:
        """Finish accepting audio and export the capture on the writer thread.

        Args:
            on_complete: Optional callback invoked with the saved file paths and
                any capture errors after export finishes.
            discard_pending: Whether to drop outstanding conversions during
                application shutdown instead of waiting for them.
        """
        with self._lock:
            if self._stop_queued:
                return
            self._accepting = False
            if on_complete is not None:
                self._on_complete = on_complete
            if discard_pending:
                self._pending_tokens.clear()
            self._queue_stop_if_ready_locked()

    @staticmethod
    def _prepare_audio(
        audio: AudioChunk,
        sample_rate: int,
    ) -> Optional[tuple[AudioChunk, int]]:
        """Copy one supported audio block without changing its signal level."""
        if sample_rate <= 0:
            return None
        copied = np.array(audio, dtype=np.float32, copy=True)
        if copied.ndim not in {1, 2} or copied.size == 0:
            return None
        np.nan_to_num(
            copied,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        )
        return copied, sample_rate

    def _record_error(self, message: str) -> None:
        """Record a diagnostic write error for the final report."""
        with self._lock:
            self._errors.append(message)

    def _queue_stop_if_ready_locked(self) -> None:
        """Queue the writer sentinel after all accepted conversions finish."""
        if not self._accepting and not self._pending_tokens and not self._stop_queued:
            self._stop_queued = True
            self._queue.put(None)

    def _write_capture(self) -> None:
        """Append queued blocks to temporary WAVs, then export final FLACs."""
        spool_paths = {
            "microphone": self.directory / "microphone-capture.wav",
            "converted": self.directory / "converted-capture.wav",
        }
        writers: dict[_CaptureKind, sf.SoundFile] = {}
        writer_specs: dict[_CaptureKind, tuple[int, int]] = {}
        failed_kinds: set[_CaptureKind] = set()

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                kind, audio, sample_rate = item
                block = audio[:, None] if audio.ndim == 1 else audio
                spec = (sample_rate, block.shape[1])

                if kind in failed_kinds:
                    continue
                if kind in writer_specs and writer_specs[kind] != spec:
                    failed_kinds.add(kind)
                    self._record_error(
                        f"{kind} audio changed sample rate or channel count"
                    )
                    continue

                try:
                    writer = writers.get(kind)
                    if writer is None:
                        writer = sf.SoundFile(
                            spool_paths[kind],
                            mode="w",
                            samplerate=sample_rate,
                            channels=block.shape[1],
                            format="WAV",
                            subtype="FLOAT",
                        )
                        writers[kind] = writer
                        writer_specs[kind] = spec
                    writer.write(block)
                except Exception as exc:
                    failed_kinds.add(kind)
                    self._record_error(f"Could not write {kind} audio: {exc}")
        finally:
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    self._record_error(f"Could not close a capture file: {exc}")

        microphone_path = self._export_capture(
            "microphone",
            spool_paths["microphone"],
        )
        converted_path = self._export_capture(
            "converted",
            spool_paths["converted"],
        )
        with self._lock:
            has_errors = bool(self._errors)
        if microphone_path is None and converted_path is None and not has_errors:
            shutil.rmtree(self.directory, ignore_errors=True)

        with self._lock:
            result = VCDebugSaveResult(
                directory=self.directory,
                microphone_path=microphone_path,
                converted_path=converted_path,
                errors=tuple(self._errors),
            )
            callback = self._on_complete
        if callback is not None:
            callback(result)

    def _export_capture(
        self,
        kind: _CaptureKind,
        spool_path: Path,
    ) -> Optional[Path]:
        """Convert one temporary capture to the required FLAC format."""
        if not spool_path.exists() or spool_path.stat().st_size == 0:
            return None
        output_path = self.directory / f"{kind}.flac"
        try:
            audio, sample_rate = sf.read(
                spool_path,
                dtype="float32",
                always_2d=True,
            )
            if sample_rate != _DEBUG_SAMPLE_RATE:
                divisor = math.gcd(sample_rate, _DEBUG_SAMPLE_RATE)
                audio = signal.resample_poly(
                    audio,
                    _DEBUG_SAMPLE_RATE // divisor,
                    sample_rate // divisor,
                    axis=0,
                ).astype(np.float32, copy=False)
            sf.write(
                output_path,
                audio,
                _DEBUG_SAMPLE_RATE,
                format="FLAC",
                subtype="PCM_24",
            )
        except Exception as exc:
            self._record_error(f"Could not export {kind} FLAC: {exc}")
            return None

        spool_path.unlink(missing_ok=True)
        return output_path
