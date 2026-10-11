# SPDX-License-Identifier: Apache-2.0
"""Capture paired live voice-conversion audio for troubleshooting."""

from __future__ import annotations

import math
import queue
import shutil
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional
from collections.abc import Callable

import numpy as np
import soundfile as sf
from scipy import signal

from .paths import app_data_dir
from .typing.aliases import AudioChunk

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
        self._queue: queue.Queue[Optional[_CaptureItem]] = queue.Queue()
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
