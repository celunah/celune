# SPDX-License-Identifier: Apache-2.0
"""Caption alignment and playback progress tracking."""

from __future__ import annotations
from collections import abc, deque

import re
import json
import math
import queue
import inspect
import importlib
import itertools
import threading
import subprocess
import unicodedata
import string as string_module
from pathlib import Path
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Union, Optional, Protocol, cast

import numpy as np
from iso639.exceptions import InvalidLanguageValue, DeprecatedLanguageValue

from .timing import (
    monotonic_time,
)

from ..i18n import string
from ..constants import BASE_SR
from ..typing.common import JSON
from ..utils import format_error_message
from ..audio.resampling import resample_audio
from ..typing.pipeline import SpeechStreamQueue
from ..dataclasses.events import StateChangedEvent
from ..registry.consent import require_model_consent
from ..registry.other import CAPTION_ALIGNMENT_MODEL_ID
from ..typing.aliases import LogLevel, AudioChunk, AudioChunks
from ..exceptions import OnnxRuntimeNotFound, EspeakExecutableNotFound
from ..dataclasses.pipeline import (
    SpeechTiming,
    CaptionPlaybackState,
    CaptionPlaybackSection,
    CaptionPlaybackSegment,
    CaptionAlignmentSection,
)

if TYPE_CHECKING:
    from ..celune import Celune

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


class _HubDownload(Protocol):
    """Subset of the Hugging Face download helper used by the aligner."""

    def __call__(self, repo_id: str, filename: str, *, revision: str) -> str:
        """Download a repository file from an exact commit revision."""


class _InferenceSession(Protocol):
    """Small ONNX Runtime surface used by the alignment model."""

    def get_inputs(self) -> abc.Sequence[_NamedModelValue]:
        """Return model input metadata."""

    def get_outputs(self) -> abc.Sequence[_NamedModelValue]:
        """Return model output metadata."""

    def run(
        self,
        output_names: abc.Sequence[str],
        input_feed: abc.Mapping[str, np.ndarray],
    ) -> abc.Sequence[np.ndarray]:
        """Evaluate the ONNX graph."""


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
        logger: abc.Callable[[str, str], None],
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
            except OnnxRuntimeNotFound:
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
        alignment_sections: tuple[CaptionAlignmentSection, ...] = (),
    ) -> tuple[tuple[float, float], ...]:
        """Return word intervals by aligning eSpeak IPA phones to audio.

        Args:
            audio: Mono normalized waveform.
            sample_rate: Sample rate of ``audio``.
            transcript: Exact text generated for the supplied audio.
            language: eSpeak language or dialect identifier for phonemization.
            alignment_sections: Known audio and transcript ranges for passages.

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

        duration = len(model_audio) / _MODEL_SAMPLE_RATE
        if alignment_sections:
            return _align_words_in_sections(
                log_probs,
                target_tokens,
                word_token_ranges,
                vocab["<pad>"],
                alignment_sections,
                len(waveform),
                len(model_audio),
                duration,
            )

        frame_spans = _ctc_token_spans(log_probs, target_tokens, vocab["<pad>"])
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
                    raise OnnxRuntimeNotFound("onnxruntime") from error
                metadata = require_model_consent(CAPTION_ALIGNMENT_MODEL_ID)
                revision = metadata.revision
                download = cast(_HubDownload, hub.hf_hub_download)
                model_path = download(
                    CAPTION_ALIGNMENT_MODEL_ID,
                    "model.onnx",
                    revision=revision,
                )
                vocab_path = download(
                    CAPTION_ALIGNMENT_MODEL_ID,
                    "vocab.json",
                    revision=revision,
                )
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
    logger: abc.Callable[[str, str], None],
    log_level: LogLevel,
) -> None:
    """Start preparing the process-wide aligner without blocking startup."""
    get_caption_aligner().preload(logger, log_level)


def align_chunk_word_start_frames(
    audio_chunks: abc.Sequence[np.ndarray],
    sample_rate: int,
    transcript: str,
    display_words: tuple[str, ...],
    language: Optional[str],
    logger: abc.Callable[[str, str], None],
    log_level: LogLevel,
    alignment_sections: tuple[CaptionAlignmentSection, ...] = (),
) -> Optional[tuple[int, ...]]:
    """Align one chunk and return word starts in its audio frames.

    Known passage ranges constrain alignment while retaining the full transcript
    and waveform as one acoustic-model input.
    """
    try:
        chunk_audio = (
            audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)
        )
        if chunk_audio.ndim == 2:
            chunk_audio = np.asarray(
                np.mean(chunk_audio, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
        if alignment_sections:
            aligned_words = get_caption_aligner().align_words(
                chunk_audio,
                sample_rate,
                transcript,
                language,
                alignment_sections,
            )
            aligned_timings = map_section_word_timings(
                display_words,
                aligned_words,
                alignment_sections,
                len(chunk_audio) / sample_rate,
                sample_rate,
            )
        else:
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
    except EspeakExecutableNotFound:
        logger(string("ui.caption_espeak_not_found"), "warning")
        return None
    except OnnxRuntimeNotFound:
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


def build_caption_alignment_sections(
    display_text: str,
    audio_sections: abc.Sequence[tuple[int, int, str]],
) -> tuple[str, tuple[CaptionAlignmentSection, ...]]:
    """Build combined-transcript alignment ranges from generated text chunks."""
    spoken_sections = tuple(
        (audio_start, audio_end, text)
        for audio_start, audio_end, text in audio_sections
        if audio_end > audio_start and text.split()
    )
    if not spoken_sections:
        return "", ()

    transcript_chunks = tuple(section[2] for section in spoken_sections)
    display_ranges = caption_chunk_word_ranges(display_text, list(transcript_chunks))
    sections: list[CaptionAlignmentSection] = []
    transcript_word_start = 0
    for (audio_start, audio_end, text), (display_start, display_end) in zip(
        spoken_sections,
        display_ranges,
        strict=True,
    ):
        transcript_word_end = transcript_word_start + len(text.split())
        sections.append(
            CaptionAlignmentSection(
                audio_start,
                audio_end,
                transcript_word_start,
                transcript_word_end,
                display_start,
                display_end,
            )
        )
        transcript_word_start = transcript_word_end

    merged_sections: list[CaptionAlignmentSection] = []
    pending = sections[0]
    for section in sections[1:]:
        if pending.display_word_start == pending.display_word_end or (
            section.display_word_start == section.display_word_end
        ):
            pending = CaptionAlignmentSection(
                pending.audio_start_frame,
                section.audio_end_frame,
                pending.transcript_word_start,
                section.transcript_word_end,
                pending.display_word_start,
                section.display_word_end,
            )
        else:
            merged_sections.append(pending)
            pending = section
    if pending.display_word_start == pending.display_word_end and merged_sections:
        previous = merged_sections[-1]
        merged_sections[-1] = CaptionAlignmentSection(
            previous.audio_start_frame,
            pending.audio_end_frame,
            previous.transcript_word_start,
            pending.transcript_word_end,
            previous.display_word_start,
            pending.display_word_end,
        )
    else:
        merged_sections.append(pending)

    first = merged_sections[0]
    if first.audio_start_frame > 0:
        merged_sections[0] = CaptionAlignmentSection(
            0,
            first.audio_end_frame,
            first.transcript_word_start,
            first.transcript_word_end,
            first.display_word_start,
            first.display_word_end,
        )
    return " ".join(transcript_chunks), tuple(merged_sections)


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


def map_section_word_timings(
    display_words: tuple[str, ...],
    aligned_words: tuple[tuple[float, float], ...],
    sections: tuple[CaptionAlignmentSection, ...],
    audio_duration: float,
    sample_rate: int,
) -> tuple[tuple[float, float], ...]:
    """Map transcript and display word ranges independently per audio section.

    Args:
        display_words: Words shown by the caption UI.
        aligned_words: Transcript word intervals in the combined audio timeline.
        sections: Known audio and transcript word ranges for each passage.
        audio_duration: Total duration of the combined audio in seconds.
        sample_rate: Sample rate used by section audio frame offsets.

    Returns:
        tuple[tuple[float, float], ...]: Display word timings in the combined
        audio timeline.
    """
    if sample_rate <= 0 or audio_duration <= 0.0:
        raise ValueError("caption section timing requires positive audio duration")
    mapped: list[tuple[float, float]] = []
    previous_source_end = 0
    previous_display_end = 0
    previous_audio_end = 0
    for section in sections:
        source_start = section.transcript_word_start
        source_end = section.transcript_word_end
        display_start = section.display_word_start
        display_end = section.display_word_end
        if not (
            previous_source_end == source_start < source_end <= len(aligned_words)
            and 0 <= display_start < display_end <= len(display_words)
            and previous_display_end == display_start
            and previous_audio_end
            <= section.audio_start_frame
            < section.audio_end_frame
        ):
            raise ValueError("caption section word ranges are invalid")
        audio_start = section.audio_start_frame / sample_rate
        audio_end = section.audio_end_frame / sample_rate
        if audio_end > audio_duration:
            raise ValueError("caption section audio range exceeds the waveform")
        section_timings = aligned_words[source_start:source_end]
        local_timings = tuple(
            (start - audio_start, end - audio_start) for start, end in section_timings
        )
        section_display_words = display_words[display_start:display_end]
        mapped.extend(
            (start + audio_start, end + audio_start)
            for start, end in map_word_timings(
                section_display_words,
                local_timings,
                min(audio_duration, audio_end) - audio_start,
            )
        )
        previous_source_end = source_end
        previous_display_end = display_end
        previous_audio_end = section.audio_end_frame
    if previous_source_end != len(aligned_words) or previous_display_end != len(
        display_words
    ):
        raise ValueError("caption sections do not cover all display words")
    return tuple(mapped)


def _align_words_in_sections(
    log_probs: np.ndarray,
    target_tokens: list[int],
    word_token_ranges: list[tuple[int, int]],
    blank_id: int,
    sections: tuple[CaptionAlignmentSection, ...],
    waveform_samples: int,
    model_samples: int,
    duration: float,
) -> tuple[tuple[float, float], ...]:
    """Run CTC alignment inside known passage audio and transcript boundaries."""
    if not sections:
        return ()
    if sections[0].audio_start_frame != 0:
        raise ValueError("caption alignment sections must start at the waveform origin")
    timings: list[tuple[float, float]] = []
    previous_audio_end = 0
    previous_word_end = 0
    for section in sections:
        if not (
            previous_audio_end
            <= section.audio_start_frame
            < section.audio_end_frame
            <= waveform_samples
            and previous_word_end
            == section.transcript_word_start
            < section.transcript_word_end
            <= len(word_token_ranges)
        ):
            raise ValueError("caption alignment sections are invalid or out of order")
        token_start = word_token_ranges[section.transcript_word_start][0]
        token_end = word_token_ranges[section.transcript_word_end - 1][1]
        start_model_sample = (
            section.audio_start_frame * model_samples / waveform_samples
        )
        end_model_sample = section.audio_end_frame * model_samples / waveform_samples
        start_frame = int(start_model_sample * log_probs.shape[0] / model_samples)
        end_frame = int(np.ceil(end_model_sample * log_probs.shape[0] / model_samples))
        start_frame = max(0, min(log_probs.shape[0] - 1, start_frame))
        end_frame = max(start_frame + 1, min(log_probs.shape[0], end_frame))
        section_log_probs = log_probs[start_frame:end_frame]
        section_spans = _ctc_token_spans(
            section_log_probs,
            target_tokens[token_start:token_end],
            blank_id,
        )
        section_words = _word_frame_spans(
            section_spans,
            [
                (start - token_start, end - token_start)
                for start, end in word_token_ranges[
                    section.transcript_word_start : section.transcript_word_end
                ]
            ],
        )
        frame_offset = start_frame
        timings.extend(
            (
                min(duration, (frame_offset + start) / _MODEL_FRAME_RATE),
                min(duration, (frame_offset + end) / _MODEL_FRAME_RATE),
            )
            for start, end in section_words
        )
        previous_audio_end = section.audio_end_frame
        previous_word_end = section.transcript_word_end
    if previous_word_end != len(word_token_ranges):
        raise ValueError("caption sections do not cover all transcript words")
    return tuple(timings)


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
            raise EspeakExecutableNotFound("espeak-ng") from error
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
        logger: abc.Callable[[str, str], None],
        log_level: LogLevel,
    ) -> None:
        """Start a daemon worker for queued caption alignment and playback tasks."""
        self._logger = logger
        self._log_level = log_level
        self._tasks: queue.Queue[Optional[abc.Callable[[], None]]] = queue.Queue()
        self._state_lock = threading.Lock()
        self._cancelled_sources: set[int] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="celune-caption-alignment",
            daemon=True,
        )
        self._thread.start()

    def submit_task(self, task: abc.Callable[[], None]) -> None:
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
        alignment_sections: tuple[CaptionAlignmentSection, ...] = (),
    ) -> None:
        """Align and queue one transcript chunk without blocking generation.

        Optional passage ranges keep words within their known audio sections
        while the worker processes one combined transcript and waveform.
        """
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
                alignment_sections,
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
            from .buffering import queue_playback_done

            queue_playback_done(
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

    def close(self, finalizer: abc.Callable[[], None]) -> None:
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
        alignment_sections: tuple[CaptionAlignmentSection, ...],
    ) -> None:
        from .buffering import flush_buffered_speech_chunks

        with self._state_lock:
            if source_id in self._cancelled_sources:
                audio_chunks.clear()
                return
        with engine.queue_lock:
            source_meta = playback_source_meta(engine).get(source_id)
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
                alignment_sections,
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
            pushed_audio[0] = flush_buffered_speech_chunks(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                pushed_audio[0],
                stream_queue,
                caption_text=None if alignment_failed[0] else caption_text,
            )
            with engine.queue_lock:
                source_meta = playback_source_meta(engine).get(source_id)
                chunk_end = (
                    int(float(source_meta.get("total_frames", 0.0)))
                    if isinstance(source_meta, dict)
                    else chunk_start
                )
            if not alignment_failed[0]:
                record_caption_playback_segment(
                    engine,
                    source_id,
                    chunk_start,
                    chunk_end,
                    word_start,
                    word_end,
                    timing_words,
                    word_start_frames,
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
        from .buffering import flush_buffered_speech_chunks

        with self._state_lock:
            if source_id in self._cancelled_sources:
                audio_chunks.clear()
                return
            pushed_audio[0] = flush_buffered_speech_chunks(
                engine,
                source_id,
                audio_chunks,
                speech_timing,
                pushed_audio[0],
                stream_queue,
                caption_text=caption_text,
            )


def next_playback_source_id(engine: Celune) -> int:
    """Return the next monotonically increasing playback source id."""
    source_id = getattr(engine, "next_playback_source_id", 0) + 1
    engine.next_playback_source_id = source_id
    return source_id


def register_overlay_playback(
    engine: Celune,
) -> Optional[StateChangedEvent]:
    """Mark the mixer busy for a newly queued non-speech playback source."""
    return register_overlay_playback_state(
        engine,
        reset_ready_announcement=True,
    )


def register_overlay_playback_state(
    engine: Celune,
    *,
    reset_ready_announcement: bool,
) -> Optional[StateChangedEvent]:
    """Mark the mixer busy for overlay playback with optional ready reset."""
    state_event = None
    with engine.say_lock:
        if not engine.locked:
            state_event = engine.update_cur_state("speaking")
        engine.playback_done.clear()
        if reset_ready_announcement:
            engine._ready_announced = False
    return state_event


def playback_source_statuses(engine: Celune) -> dict[int, str]:
    """Return the mutable per-source playback status map."""
    statuses = getattr(engine, "playback_source_statuses", None)
    if isinstance(statuses, dict):
        return statuses

    statuses = {}
    engine.playback_source_statuses = statuses
    return statuses


def current_playback_status(engine: Celune) -> Optional[str]:
    """Return the most recently registered status for an active playback source."""
    statuses = playback_source_statuses(engine)
    try:
        return next(reversed(statuses.values()), None)
    except RuntimeError:
        return None


def playback_source_meta(
    engine: Celune,
) -> dict[int, dict[str, Union[str, float]]]:
    """Return per-source mixer metadata such as kind, gain state, and progress."""
    meta = getattr(engine, "playback_source_meta", None)
    if isinstance(meta, dict):
        return meta

    meta = {}
    engine.playback_source_meta = meta
    return meta


def playback_caption_states(engine: Celune) -> dict[int, CaptionPlaybackState]:
    """Return per-source caption frame ranges used during streamed playback."""
    states = getattr(engine, "playback_caption_states", None)
    if isinstance(states, dict):
        return states

    states = {}
    engine.playback_caption_states = states
    return states


def caption_chunk_word_ranges(
    display_text: str,
    chunks: list[str],
) -> tuple[tuple[int, int], ...]:
    """Map ordered synthesis chunks onto stable display-caption word ranges."""
    display_words = display_text.split()
    speech_words = [word for chunk in chunks for word in chunk.split()]
    if not display_words or not speech_words:
        return tuple((0, 0) for _chunk in chunks)

    def normalize_word(word: str) -> str:
        return re.sub(r"\W+", "", word.casefold())

    matcher = SequenceMatcher(
        a=[normalize_word(word) for word in speech_words],
        b=[normalize_word(word) for word in display_words],
    )
    speech_to_display = [0] * (len(speech_words) + 1)
    for (
        operation,
        speech_start,
        speech_end,
        display_start,
        display_end,
    ) in matcher.get_opcodes():
        speech_span = speech_end - speech_start
        display_span = display_end - display_start
        if operation == "equal":
            for offset in range(speech_span + 1):
                speech_to_display[speech_start + offset] = display_start + offset
        elif speech_span:
            for offset in range(speech_span + 1):
                speech_to_display[speech_start + offset] = display_start + round(
                    offset * display_span / speech_span
                )
        else:
            speech_to_display[speech_start] = display_end

    ranges: list[tuple[int, int]] = []
    previous_end = 0
    speech_word_end = 0
    for index, chunk in enumerate(chunks):
        speech_word_end += len(chunk.split())
        if index == len(chunks) - 1:
            word_end = len(display_words)
        elif speech_word_end < len(speech_to_display):
            word_end = speech_to_display[speech_word_end]
        else:
            word_end = len(display_words)
        word_end = max(previous_end, min(len(display_words), word_end))
        ranges.append((previous_end, word_end))
        previous_end = word_end

    return tuple(ranges)


def caption_tracking_modes(
    engine: Celune,
    *,
    tutorial_audio: bool,
    caption_worker_available: bool,
) -> tuple[bool, bool]:
    """Return whether section captions or spoken word captions are enabled."""
    if tutorial_audio or engine.config.get("captions") is not True:
        return False, False
    if getattr(engine.backend, "singing", False):
        return True, False
    return False, caption_worker_available


def caption_chunk_start_frame(
    engine: Celune,
    source_id: int,
    default_frame: int,
    *,
    use_queued_total: bool,
) -> int:
    """Resolve the starting frame for one spoken caption chunk."""
    if not use_queued_total or source_id not in playback_caption_states(engine):
        return default_frame
    with engine.queue_lock:
        source_meta = playback_source_meta(engine).get(source_id)
        if isinstance(source_meta, dict):
            return int(float(source_meta.get("total_frames", 0.0)))
    return default_frame


def register_playback_source(
    engine: Celune,
    source_id: int,
    *,
    kind: str,
    base_gain: float = 1.0,
    low_latency: bool = False,
    caption_word_total: int = 0,
    async_caption_audio: bool = False,
    track_caption_sections: bool = False,
) -> None:
    """Register one playback source for status and gain management."""
    clipped = float(np.clip(base_gain, 0.0, 1.0))
    playback_source_meta(engine)[source_id] = {
        "kind": kind,
        "base_gain": clipped,
        "current_gain": clipped,
        "total_frames": 0.0,
        "played_frames": 0.0,
        "total_frames_final": 0.0,
        "generation": float(getattr(engine, "_playback_generation", 0)),
        "low_latency": low_latency,
        "async_caption_audio": "true" if async_caption_audio else "",
    }
    if kind == "speech" and (caption_word_total > 0 or track_caption_sections):
        playback_caption_states(engine)[source_id] = CaptionPlaybackState(
            total_words=caption_word_total
        )


def record_caption_playback_segment(
    engine: Celune,
    source_id: int,
    start_frame: int,
    end_frame: int,
    word_start: int,
    word_end: int,
    timing_words: tuple[str, ...] = (),
    word_start_frames: tuple[int, ...] = (),
) -> None:
    """Record the exact queued audio span produced for one text chunk."""
    states = playback_caption_states(engine)
    state = states.get(source_id)
    if state is None or end_frame <= start_frame or word_end <= word_start:
        return

    with engine.queue_lock:
        state.segments.append(
            CaptionPlaybackSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                word_start=word_start,
                word_end=word_end,
                timing_words=timing_words,
                word_start_frames=word_start_frames,
            )
        )


def record_caption_playback_audio_chunk(
    engine: Celune,
    source_id: int,
    audio_frames: int,
    timing: Optional[JSON],
) -> bool:
    """Accumulate one sung audio chunk and record a completed section span."""
    section_final = timing is not None and timing.get("section_final") is True
    if audio_frames <= 0:
        return section_final
    state = playback_caption_states(engine).get(source_id)
    if state is None:
        return section_final

    display_text = timing.get("section_caption") if timing is not None else None
    with engine.queue_lock:
        if isinstance(display_text, str) and state.pending_section_start is None:
            state.pending_section_start = state.section_audio_frames
            state.pending_section_text = display_text
        state.section_audio_frames += audio_frames
        if (
            section_final
            and state.pending_section_start is not None
            and state.pending_section_text is not None
        ):
            _append_caption_playback_section(
                state,
                state.pending_section_start,
                state.section_audio_frames,
                state.pending_section_text,
            )
            state.pending_section_start = None
            state.pending_section_text = None
    return section_final


def _append_caption_playback_section(
    state: CaptionPlaybackState,
    start_frame: int,
    end_frame: int,
    display_text: str,
) -> None:
    """Append one section span, merging adjacent sections with identical text."""
    if end_frame <= start_frame:
        return
    previous = state.section_segments[-1] if state.section_segments else None
    if (
        previous is not None
        and previous.end_frame == start_frame
        and previous.display_text == display_text
    ):
        state.section_segments[-1] = CaptionPlaybackSection(
            previous.start_frame,
            end_frame,
            display_text,
        )
        return
    state.section_segments.append(
        CaptionPlaybackSection(start_frame, end_frame, display_text)
    )


def set_playback_source_status(engine: Celune, source_id: int, status: str) -> None:
    """Record and surface the current status for one active playback source."""
    statuses = playback_source_statuses(engine)
    statuses[source_id] = status
    engine.status_callback(status)


def clear_playback_source_status(engine: Celune, source_id: int) -> None:
    """Forget one playback-source status and restore the next active status."""
    statuses = playback_source_statuses(engine)
    statuses.pop(source_id, None)
    if statuses:
        engine.status_callback(next(reversed(statuses.values())))
    playback_source_meta(engine).pop(source_id, None)
    playback_caption_states(engine).pop(source_id, None)
    if getattr(engine, "_caption_source_id", None) == source_id:
        engine._caption_source_id = None


def _notify_caption_progress(
    engine: Celune,
    progress: Optional[float],
    total: Optional[float],
    visible_words: Optional[int] = None,
) -> None:
    """Notify caption consumers while preserving legacy two-argument callbacks."""
    callback = getattr(engine, "caption_progress_callback", None)
    if not callable(callback):
        return
    if visible_words is None:
        callback(progress, total)
        return

    try:
        parameters = tuple(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        callback(progress, total)
        return

    positional_count = sum(
        parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        for parameter in parameters
    )
    if positional_count >= 3 or any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        callback(progress, total, visible_words)
        return
    if any(
        parameter.name == "visible_words"
        and parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters
    ):
        callback(progress, total, visible_words=visible_words)
        return
    callback(progress, total)


def _caption_words_at_frame(
    state: CaptionPlaybackState,
    played_frames: float,
) -> int:
    """Return monotonically revealed caption words for the current audio frame."""
    visible_words = 0
    for segment in state.segments:
        if played_frames < segment.start_frame:
            break
        if played_frames >= segment.end_frame:
            visible_words = segment.word_end
            continue

        segment_frames = segment.end_frame - segment.start_frame
        segment_words = segment.word_end - segment.word_start
        elapsed_frames = max(0.0, played_frames - segment.start_frame)
        if len(segment.word_start_frames) == segment_words:
            visible_words = segment.word_start + sum(
                elapsed_frames >= start_frame
                for start_frame in segment.word_start_frames
            )
        else:
            visible_words = segment.word_start + math.ceil(
                elapsed_frames * segment_words / segment_frames
            )
        break

    return min(state.total_words, max(0, visible_words))


def notify_speech_playback_finished(engine: Celune, source_id: int) -> None:
    """Complete caption progress when one speech source drains independently."""
    source_meta = playback_source_meta(engine).get(source_id)
    if not isinstance(source_meta, dict) or source_meta.get("kind") != "speech":
        return
    caption_source_id = getattr(engine, "_caption_source_id", None)
    if caption_source_id is not None and caption_source_id != source_id:
        return

    caption_state = playback_caption_states(engine).get(source_id)
    if caption_state is not None and caption_state.section_segments:
        if caption_state.active_section_index is not None:
            callback = getattr(engine, "caption_callback", None)
            if callable(callback):
                callback(None)
        if caption_source_id == source_id:
            engine._caption_source_id = None
        return

    total_frames = max(1.0, float(source_meta.get("total_frames", 0.0)))
    _notify_caption_progress(engine, total_frames, total_frames)


def update_playback_progress(
    engine: Celune,
    source_buffers: Optional[
        abc.Mapping[int, deque[tuple[AudioChunk, Optional[SpeechTiming]]]]
    ] = None,
) -> None:
    """Reflect the active playback source position in the shared progress bar."""
    if source_buffers is not None and not source_buffers:
        return

    meta = playback_source_meta(engine)
    active_ids = [
        source_id
        for source_id in (source_buffers if source_buffers is not None else meta)
        if source_id in meta
    ]
    if not active_ids:
        return

    source_id = max(active_ids)
    source_meta = meta.get(source_id)
    if not isinstance(source_meta, dict):
        return

    total_frames = float(source_meta.get("total_frames", 0.0))
    played_frames = float(source_meta.get("played_frames", 0.0))
    if total_frames <= 0.0:
        return

    now = monotonic_time()
    last_emit_at = float(getattr(engine, "_playback_progress_last_emit_at", 0.0))
    last_source_id = getattr(engine, "_playback_progress_last_source_id", None)
    emit_interval = 0.08
    if last_source_id == source_id and (now - last_emit_at) < emit_interval:
        return

    engine._playback_progress_last_emit_at = now
    engine._playback_progress_last_source_id = source_id
    engine.progress_callback(min(played_frames, total_frames), total_frames)

    speech_ids = [
        active_source_id
        for active_source_id in active_ids
        if meta.get(active_source_id, {}).get("kind") == "speech"
    ]
    if not speech_ids:
        return
    caption_source_id = getattr(engine, "_caption_source_id", None)
    if caption_source_id is not None:
        if caption_source_id not in speech_ids:
            return
        speech_source_id = caption_source_id
    else:
        speech_source_id = max(speech_ids)
    speech_meta = meta.get(speech_source_id)
    if not isinstance(speech_meta, dict):
        return
    with engine.queue_lock:
        caption_state = playback_caption_states(engine).get(speech_source_id)
        caption_segments = (
            None if caption_state is None else tuple(caption_state.segments)
        )
        section_segments = (
            None if caption_state is None else tuple(caption_state.section_segments)
        )
    speech_played_frames = min(
        float(speech_meta.get("played_frames", 0.0)),
        float(speech_meta.get("total_frames", 0.0)),
    )
    if caption_state is not None and section_segments:
        section_index = None
        for index, section in enumerate(section_segments):
            if speech_played_frames < section.start_frame:
                break
            section_index = index
        with engine.queue_lock:
            current_state = playback_caption_states(engine).get(speech_source_id)
            changed = False
            if (
                current_state is not None
                and section_index is not None
                and current_state.active_section_index != section_index
            ):
                current_state.active_section_index = section_index
                changed = True
        if changed and section_index is not None:
            callback = getattr(engine, "caption_callback", None)
            if callable(callback):
                engine._caption_source_id = speech_source_id
                caption_text = section_segments[section_index].display_text
                callback(caption_text or None)
        return

    if caption_state is not None and caption_segments:
        snapshot = CaptionPlaybackState(
            total_words=caption_state.total_words,
            segments=list(caption_segments),
        )
        visible_words = _caption_words_at_frame(
            snapshot,
            speech_played_frames,
        )
        _notify_caption_progress(
            engine,
            speech_played_frames,
            float(speech_meta.get("total_frames", 0.0)),
            visible_words,
        )
    elif float(speech_meta.get("total_frames_final", 0.0)) >= 1.0:
        _notify_caption_progress(
            engine,
            speech_played_frames,
            float(speech_meta.get("total_frames", 0.0)),
        )
