# SPDX-License-Identifier: Apache-2.0
"""Optional IPA-based caption alignment for generated speech."""

from __future__ import annotations

import json
import queue
import importlib
import itertools
import threading
import subprocess
import unicodedata
import string as string_module

from pathlib import Path
from collections.abc import Mapping, Callable, Sequence
from typing import TYPE_CHECKING, Optional, Protocol, cast

import numpy as np
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue

from .i18n import string
from .constants import BASE_SR
from .utils import format_error_message
from .audio.resampling import resample_audio
from .dataclasses.pipeline import SpeechTiming
from .typing.pipeline import SpeechStreamQueue
from .typing.aliases import LogLevel, AudioChunks

if TYPE_CHECKING:
    from .celune import Celune

_ALIGNMENT_MODEL_ID = "sadda-speech/wav2vec2-espeak-ctc"
_MODEL_SAMPLE_RATE = 16000
_MODEL_FRAME_RATE = 50.0
_ONNX_LOG_SEVERITY_ERROR = 3
_ALIGNER_LOCK = threading.Lock()
_ALIGNER: Optional[CaptionAligner] = None
_EDGE_PUNCTUATION = string_module.punctuation + "…—–«»¡¿"
_STRESS_MARKS = "ˈˌ"
_IPA_TIE = "͡"


class _NamedModelValue(Protocol):
    """ONNX Runtime input or output metadata."""

    name: str


class _InferenceSession(Protocol):
    """Small ONNX Runtime surface used by the alignment model."""

    def get_inputs(self) -> Sequence[_NamedModelValue]:
        """Return model input metadata."""

    def get_outputs(self) -> Sequence[_NamedModelValue]:
        """Return model output metadata."""

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]:
        """Evaluate the ONNX graph."""


class _EspeakExecutableNotFound(FileNotFoundError):
    """Identify eSpeak NG launch failures for a concise user warning."""


class _OnnxRuntimeNotFound(ImportError):
    """Identify a missing ONNX Runtime install for a concise user warning."""


class CaptionAligner:
    """Lazily load the IPA acoustic model and align transcript words."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._preload_lock = threading.Lock()
        self._preload_started = False
        self._session: Optional[_InferenceSession] = None
        self._input_name = ""
        self._output_name = ""
        self._vocab: Optional[dict[str, int]] = None

    def preload(
        self,
        logger: Callable[[str, str], None],
        log_level: LogLevel,
    ) -> None:
        """Prepare the acoustic model asynchronously before first speech."""
        with self._preload_lock:
            if self._preload_started or self._session is not None:
                return
            self._preload_started = True

        logger(string("ui.caption_alignment_loading"), "info")

        def _worker() -> None:
            try:
                self._load_model()
            except _OnnxRuntimeNotFound:
                with self._preload_lock:
                    self._preload_started = False
                logger(string("ui.caption_onnxruntime_not_found"), "warning")
            except Exception as error:
                with self._preload_lock:
                    self._preload_started = False
                logger(
                    format_error_message(
                        string("ui.caption_alignment_failed"),
                        error,
                        log_level,
                    ),
                    "warning",
                )
            else:
                logger(string("ui.caption_alignment_ready"), "info")

        threading.Thread(
            target=_worker,
            name="celune-caption-model-preload",
            daemon=True,
        ).start()

    def align_words(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
        language: Optional[str] = None,
    ) -> tuple[tuple[float, float], ...]:
        """Return word intervals by aligning eSpeak IPA phones to audio.

        Args:
            audio: Mono normalized waveform.
            sample_rate: Sample rate of ``audio``.
            transcript: Exact text generated for the supplied audio.
            language: eSpeak language or dialect identifier for phonemization.

        Returns:
            tuple[tuple[float, float], ...]: Word start and end times in seconds.

        Raises:
            ValueError: Audio, sample rate, transcript, or IPA tokens are invalid.
            ImportError: The optional ONNX Runtime dependency is unavailable.
            FileNotFoundError: eSpeak NG is unavailable on ``PATH``.
            RuntimeError: Model loading, inference, or alignment failed.
        """
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError("caption alignment requires mono audio")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not transcript.strip() or waveform.size == 0:
            return ()

        words = _phonemize(transcript, _espeak_voice(language))
        if not words:
            return ()
        session, vocab = self._load_model()
        model_audio = resample_audio(
            waveform,
            sample_rate,
            _MODEL_SAMPLE_RATE,
        )
        normalized_audio = model_audio - np.mean(model_audio, dtype=np.float32)
        normalized_audio /= np.std(model_audio, dtype=np.float32) + 1e-7
        output = session.run(
            [self._output_name],
            {self._input_name: normalized_audio[None, :]},
        )
        if not output:
            raise RuntimeError("the caption acoustic model returned no logits")
        logits = np.asarray(output[0], dtype=np.float32)
        if logits.ndim == 3:
            if logits.shape[0] != 1:
                raise RuntimeError(
                    "the caption acoustic model returned an invalid batch"
                )
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[1] != len(vocab):
            raise RuntimeError("the caption acoustic model returned invalid logits")
        log_probs = _log_softmax(logits)

        target_tokens: list[int] = []
        word_token_ranges: list[tuple[int, int]] = []
        for word in words:
            token_start = len(target_tokens)
            target_tokens.extend(_tokenize_ipa(word, vocab))
            word_token_ranges.append((token_start, len(target_tokens)))

        frame_spans = _ctc_token_spans(log_probs, target_tokens, vocab["<pad>"])
        duration = len(model_audio) / _MODEL_SAMPLE_RATE
        return tuple(
            (
                min(duration, start / _MODEL_FRAME_RATE),
                min(duration, end / _MODEL_FRAME_RATE),
            )
            for start, end in _word_frame_spans(frame_spans, word_token_ranges)
        )

    def _load_model(self) -> tuple[_InferenceSession, dict[str, int]]:
        """Download and open the model assets once per process."""
        with self._load_lock:
            if self._session is None or self._vocab is None:
                hub = importlib.import_module("huggingface_hub")
                try:
                    runtime = importlib.import_module("onnxruntime")
                except ModuleNotFoundError as error:
                    if error.name != "onnxruntime":
                        raise
                    raise _OnnxRuntimeNotFound("onnxruntime") from error
                download = cast(Callable[[str, str], str], hub.hf_hub_download)
                model_path = download(_ALIGNMENT_MODEL_ID, "model.onnx")
                vocab_path = download(_ALIGNMENT_MODEL_ID, "vocab.json")
                vocab = cast(
                    dict[str, int],
                    json.loads(Path(vocab_path).read_text(encoding="utf-8")),
                )
                session_options = runtime.SessionOptions()
                session_options.log_severity_level = _ONNX_LOG_SEVERITY_ERROR
                session = cast(
                    _InferenceSession,
                    runtime.InferenceSession(
                        model_path,
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    ),
                )
                self._input_name = session.get_inputs()[0].name
                self._output_name = session.get_outputs()[0].name
                self._session = session
                self._vocab = vocab
            return self._session, self._vocab


def get_caption_aligner() -> CaptionAligner:
    """Return the process-wide lazy caption aligner."""
    global _ALIGNER
    with _ALIGNER_LOCK:
        if _ALIGNER is None:
            _ALIGNER = CaptionAligner()
        return _ALIGNER


def preload_caption_aligner(
    logger: Callable[[str, str], None],
    log_level: LogLevel,
) -> None:
    """Start preparing the process-wide aligner without blocking startup."""
    get_caption_aligner().preload(logger, log_level)


def align_chunk_word_start_frames(
    audio_chunks: Sequence[np.ndarray],
    sample_rate: int,
    transcript: str,
    display_words: tuple[str, ...],
    language: Optional[str],
    logger: Callable[[str, str], None],
    log_level: LogLevel,
) -> Optional[tuple[int, ...]]:
    """Align one generated chunk and return word starts in its audio frames."""
    try:
        chunk_audio = np.concatenate(audio_chunks)
        if chunk_audio.ndim == 2:
            chunk_audio = np.asarray(
                np.mean(chunk_audio, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
        aligned_words = get_caption_aligner().align_words(
            chunk_audio,
            sample_rate,
            transcript,
            language,
        )
        aligned_timings = map_word_timings(
            display_words,
            aligned_words,
            len(chunk_audio) / sample_rate,
        )
        if not aligned_timings or len(aligned_timings) != len(display_words):
            raise ValueError("caption alignment returned incomplete word timings")
        return tuple(
            max(0, min(len(chunk_audio), round(start * sample_rate)))
            for start, _end in aligned_timings
        )
    except _EspeakExecutableNotFound:
        logger(string("ui.caption_espeak_not_found"), "warning")
        return None
    except _OnnxRuntimeNotFound:
        logger(string("ui.caption_onnxruntime_not_found"), "warning")
        return None
    except Exception as error:
        logger(
            format_error_message(
                string("ui.caption_alignment_failed"),
                error,
                log_level,
            ),
            "warning",
        )
        return None


def map_word_timings(
    display_words: tuple[str, ...],
    aligned_words: tuple[tuple[float, float], ...],
    audio_duration: float,
) -> tuple[tuple[float, float], ...]:
    """Map transcript word timings onto displayed words within one chunk."""
    if not display_words or not aligned_words or audio_duration <= 0.0:
        return ()

    source_ranges: list[tuple[float, float]] = []
    previous_end = 0.0
    for raw_start, raw_end in aligned_words:
        start = max(previous_end, 0.0, min(audio_duration, raw_start))
        end = max(start, min(audio_duration, raw_end))
        source_ranges.append((start, end))
        previous_end = end

    if len(display_words) == len(source_ranges):
        return tuple(source_ranges)

    source_word_count = len(source_ranges)
    source_boundaries = [source_ranges[0][0]]
    source_boundaries.extend(
        (left[1] + right[0]) / 2 for left, right in itertools.pairwise(source_ranges)
    )
    source_boundaries.append(source_ranges[-1][1])

    def boundary_at(position: float) -> float:
        lower_index = min(int(position), source_word_count - 1)
        fraction = position - lower_index
        lower_time = source_boundaries[lower_index]
        upper_time = source_boundaries[lower_index + 1]
        return lower_time + (upper_time - lower_time) * fraction

    mapped: list[tuple[float, float]] = []
    for index in range(len(display_words)):
        start_position = index * source_word_count / len(display_words)
        end_position = (index + 1) * source_word_count / len(display_words)
        start = boundary_at(start_position)
        end = boundary_at(end_position)
        mapped.append((start, max(start, end)))
    return tuple(mapped)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Convert model logits into stable CTC log probabilities."""
    maxima = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - maxima
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _ctc_token_spans(
    log_probs: np.ndarray,
    target: list[int],
    blank_id: int,
) -> tuple[tuple[int, int], ...]:
    """Return frame spans for a fixed IPA token sequence using CTC Viterbi."""
    if not target:
        return ()
    frame_count = log_probs.shape[0]
    if frame_count < len(target):
        raise ValueError("caption audio is too short for its IPA transcript")

    labels = np.full(len(target) * 2 + 1, blank_id, dtype=np.int64)
    labels[1::2] = target
    state_count = len(labels)
    backpointers = np.zeros((frame_count, state_count), dtype=np.uint8)
    previous = np.full(state_count, -np.inf, dtype=np.float32)
    previous[0] = log_probs[0, blank_id]
    previous[1] = log_probs[0, target[0]]

    skip_allowed = np.zeros(state_count, dtype=np.bool_)
    if state_count > 2:
        skip_allowed[2:] = (labels[2:] != blank_id) & (labels[2:] != labels[:-2])

    state_indexes = np.arange(state_count)
    for frame in range(1, frame_count):
        advance = np.full(state_count, -np.inf, dtype=np.float32)
        advance[1:] = previous[:-1]
        skip = np.full(state_count, -np.inf, dtype=np.float32)
        skip[2:] = previous[:-2]
        skip[~skip_allowed] = -np.inf
        options = np.stack((previous, advance, skip))
        choices = np.argmax(options, axis=0)
        previous = options[choices, state_indexes] + log_probs[frame, labels]
        backpointers[frame] = choices

    end_state = state_count - 1
    if state_count > 1 and previous[-2] > previous[-1]:
        end_state -= 1
    if not np.isfinite(previous[end_state]):
        raise ValueError("caption audio could not be aligned to its IPA transcript")

    state_path = np.empty(frame_count, dtype=np.int64)
    state_path[-1] = end_state
    for frame in range(frame_count - 1, 0, -1):
        state_path[frame - 1] = state_path[frame] - int(
            backpointers[frame, state_path[frame]]
        )

    spans: list[tuple[int, int]] = []
    for token_index in range(len(target)):
        token_frames = np.flatnonzero(state_path == token_index * 2 + 1)
        if token_frames.size == 0:
            raise ValueError("caption audio could not be aligned to its IPA transcript")
        spans.append((int(token_frames[0]), int(token_frames[-1]) + 1))
    return tuple(spans)


def _word_frame_spans(
    token_spans: tuple[tuple[int, int], ...],
    word_token_ranges: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Group aligned IPA token spans into transcript words."""
    word_spans: list[tuple[int, int]] = []
    for token_start, token_end in word_token_ranges:
        spans = token_spans[token_start:token_end]
        word_spans.append(
            (
                min(start for start, _end in spans),
                max(end for _start, end in spans),
            )
        )
    return tuple(word_spans)


def _phonemize(transcript: str, voice: str) -> tuple[tuple[str, ...], ...]:
    """Return the eSpeak IPA phones for each whitespace-separated word."""
    words: list[tuple[str, ...]] = []
    for token in transcript.split():
        word = token.strip(_EDGE_PUNCTUATION)
        if not word:
            continue
        try:
            result = subprocess.run(
                ["espeak-ng", "-q", "--ipa", "-v", voice, word],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as error:
            raise _EspeakExecutableNotFound("espeak-ng") from error
        ipa = result.stdout.replace("\n", " ").translate(
            {ord(mark): None for mark in _STRESS_MARKS}
        )
        phones = _split_ipa_phones(ipa.strip())
        if not phones:
            raise ValueError(f"eSpeak produced no IPA phones for {word!r}")
        words.append(phones)
    return tuple(words)


def _split_ipa_phones(ipa: str) -> tuple[str, ...]:
    """Group combining and modifier characters with their IPA base phone."""
    phones: list[str] = []
    for character in ipa:
        if character.isspace():
            continue
        category = unicodedata.category(character)
        joins_previous = phones and (
            category in {"Mn", "Lm"} or phones[-1].endswith(_IPA_TIE)
        )
        if joins_previous:
            phones[-1] += character
        else:
            phones.append(character)
    return tuple(phones)


def _tokenize_ipa(phones: tuple[str, ...], vocab: dict[str, int]) -> tuple[int, ...]:
    """Resolve IPA phones to the longest matching model vocabulary tokens."""
    keys = sorted((key for key in vocab if key), key=len, reverse=True)
    token_ids: list[int] = []
    for phone in phones:
        position = 0
        while position < len(phone):
            for key in keys:
                if phone.startswith(key, position):
                    token_ids.append(vocab[key])
                    position += len(key)
                    break
            else:
                raise ValueError(f"IPA phone is not in the model vocabulary: {phone!r}")
    return tuple(token_ids)


def _espeak_voice(language: Optional[str]) -> str:
    """Normalize common Celune language values to eSpeak voice names."""
    normalized = (language or "en-us").strip().casefold().replace("_", "-")
    if normalized in {"", "auto", "unknown", "und"}:
        return "en-us"

    if normalized in {"zh", "zh-cn", "zh-tw", "chinese"}:
        return "cmn"
    if normalized in {"english", "en"}:
        return "en-us"

    if "-" in normalized:
        base, region = normalized.split("-", maxsplit=1)
        return f"{_iso_language_code(base)}-{region}"
    return _iso_language_code(normalized)


def _iso_language_code(language: str) -> str:
    """Convert an ISO language name or code to the short eSpeak form."""
    from iso639 import Lang

    try:
        parsed = Lang(language)
    except (DeprecatedLanguageValue, InvalidLanguageValue):
        return language
    return parsed.pt1 or language


class CaptionAlignmentWorker:
    """Align speech chunks off the generation thread and queue them in order."""

    def __init__(
        self,
        logger: Callable[[str, str], None],
        log_level: LogLevel,
    ) -> None:
        """Start a daemon worker for queued caption alignment and playback tasks."""
        self._logger = logger
        self._log_level = log_level
        self._tasks: queue.Queue[Optional[Callable[[], None]]] = queue.Queue()
        self._state_lock = threading.Lock()
        self._cancelled_sources: set[int] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="celune-caption-alignment",
            daemon=True,
        )
        self._thread.start()

    def submit_task(self, task: Callable[[], None]) -> None:
        """Queue work after previously submitted caption tasks."""
        self._tasks.put(task)

    def cancel_source(self, source_id: int) -> None:
        """Discard pending aligned audio when speech generation fails."""
        with self._state_lock:
            self._cancelled_sources.add(source_id)

    def submit_chunk(
        self,
        engine: Celune,
        source_id: int,
        audio_chunks: AudioChunks,
        speech_timing: SpeechTiming,
        stream_queue: Optional[SpeechStreamQueue],
        caption_text: str,
        transcript: str,
        display_words: tuple[str, ...],
        language: Optional[str],
        word_start: int,
        word_end: int,
        timing_words: tuple[str, ...],
        alignment_failed: list[bool],
        pushed_audio: list[bool],
        sample_rate: int = BASE_SR,
    ) -> None:
        """Align and queue one transcript chunk without blocking generation."""
        self.submit_task(
            lambda: self._align_and_flush_chunk(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                stream_queue,
                caption_text,
                transcript,
                display_words,
                language,
                word_start,
                word_end,
                timing_words,
                alignment_failed,
                pushed_audio,
                sample_rate,
            )
        )

    def submit_audio(
        self,
        engine: Celune,
        source_id: int,
        audio_chunks: AudioChunks,
        speech_timing: SpeechTiming,
        stream_queue: Optional[SpeechStreamQueue],
        pushed_audio: list[bool],
        caption_text: Optional[str] = None,
    ) -> None:
        """Queue unaligned speech audio behind any pending aligned chunks."""
        self.submit_task(
            lambda: self._flush_audio(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                stream_queue,
                pushed_audio,
                caption_text,
            )
        )

    def submit_done(
        self,
        engine: Celune,
        source_id: int,
        stream_queue: Optional[SpeechStreamQueue],
        *,
        release_pipeline_when_finished: bool = True,
        notify_idle_when_finished: bool = True,
        saved_path: Optional[str] = None,
        analysis_audio: Optional[np.ndarray] = None,
        finish_stream: bool = True,
    ) -> None:
        """Queue a source completion marker after its pending audio chunks."""

        def finish() -> None:
            from .playback import _queue_playback_done

            _queue_playback_done(
                engine,
                source_id,
                release_pipeline_when_finished=release_pipeline_when_finished,
                notify_idle_when_finished=notify_idle_when_finished,
                saved_path=saved_path,
                analysis_audio=analysis_audio,
            )
            with self._state_lock:
                self._cancelled_sources.discard(source_id)
            if finish_stream and stream_queue is not None:
                stream_queue.put(None)

        self.submit_task(finish)

    def submit_stream_error(
        self,
        stream_queue: SpeechStreamQueue,
        error: Exception,
    ) -> None:
        """Deliver a stream error after its already generated audio chunks."""

        def finish() -> None:
            stream_queue.put(error)
            stream_queue.put(None)

        self.submit_task(finish)

    def close(self, finalizer: Callable[[], None]) -> None:
        """Run final work after pending alignments, then stop the worker."""
        self.submit_task(finalizer)
        self._tasks.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                task()
            except Exception as error:
                self._logger(
                    format_error_message(
                        string("ui.caption_alignment_failed"),
                        error,
                        self._log_level,
                    ),
                    "warning",
                )

    def _align_and_flush_chunk(
        self,
        engine: Celune,
        source_id: int,
        audio_chunks: AudioChunks,
        speech_timing: SpeechTiming,
        stream_queue: Optional[SpeechStreamQueue],
        caption_text: str,
        transcript: str,
        display_words: tuple[str, ...],
        language: Optional[str],
        word_start: int,
        word_end: int,
        timing_words: tuple[str, ...],
        alignment_failed: list[bool],
        pushed_audio: list[bool],
        sample_rate: int,
    ) -> None:
        from .playback import (
            _playback_source_meta,
            _flush_buffered_speech_chunks,
            _record_caption_playback_segment,
        )

        with self._state_lock:
            if source_id in self._cancelled_sources:
                audio_chunks.clear()
                return
        with engine.queue_lock:
            source_meta = _playback_source_meta(engine).get(source_id)
            if not isinstance(source_meta, dict):
                audio_chunks.clear()
                return
            chunk_start = int(float(source_meta.get("total_frames", 0.0)))

        if alignment_failed[0]:
            word_start_frames: tuple[int, ...] = ()
        else:
            aligned_frames = align_chunk_word_start_frames(
                audio_chunks,
                sample_rate,
                transcript,
                display_words,
                language,
                self._logger,
                self._log_level,
            )
            if aligned_frames is None:
                alignment_failed[0] = True
                callback = getattr(engine, "caption_callback", None)
                if callable(callback):
                    callback(None)
                word_start_frames = ()
            else:
                word_start_frames = aligned_frames

        with self._state_lock:
            if source_id in self._cancelled_sources:
                audio_chunks.clear()
                return
            chunk_end = chunk_start + sum(len(chunk) for chunk in audio_chunks)
            if not alignment_failed[0]:
                _record_caption_playback_segment(
                    engine,
                    source_id,
                    chunk_start,
                    chunk_end,
                    word_start,
                    word_end,
                    timing_words,
                    word_start_frames,
                )
            pushed_audio[0] = _flush_buffered_speech_chunks(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                pushed_audio[0],
                stream_queue,
                caption_text=None if alignment_failed[0] else caption_text,
            )

    def _flush_audio(
        self,
        engine: Celune,
        source_id: int,
        audio_chunks: AudioChunks,
        speech_timing: SpeechTiming,
        stream_queue: Optional[SpeechStreamQueue],
        pushed_audio: list[bool],
        caption_text: Optional[str],
    ) -> None:
        from .playback import _flush_buffered_speech_chunks

        with self._state_lock:
            if source_id in self._cancelled_sources:
                audio_chunks.clear()
                return
            pushed_audio[0] = _flush_buffered_speech_chunks(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                pushed_audio[0],
                stream_queue,
                caption_text=caption_text,
            )
