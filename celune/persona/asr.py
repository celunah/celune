# SPDX-License-Identifier: Apache-2.0
"""Speech-to-text helpers for Persona microphone input."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from collections.abc import Mapping, Sequence, Callable
from typing import TYPE_CHECKING, Union, Optional, cast

import torch
import numpy as np

from ..dsp import resample_audio
from ..paths import huggingface_progress
from ..typing.aliases import AudioChunk
from ..typing.persona import (
    WhisperScalar,
    WhisperSegmentPayload,
    WhisperGenerationPayload,
    _WhisperModel,
    _WhisperProcessor,
)

if TYPE_CHECKING:
    # noinspection PyPep8Naming
    # noinspection PyPep8Naming
    from torch import dtype as DType
    from torch import device as Device


DEFAULT_PERSONA_SPEECH_MODEL_ID = "openai/whisper-large-v3-turbo"
PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS = 5.0
PERSONA_SPEECH_END_DELAY_SECONDS = 1.5
WHISPER_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WhisperWord:
    """One Whisper-aligned word with audio-relative timestamps."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class WhisperSegment:
    """One timestamped segment returned by Whisper."""

    text: str
    start: float
    end: float
    words: tuple[WhisperWord, ...] = ()


class WhisperTranscriber:
    """Lazily load and run one configurable Hugging Face Whisper model."""

    def __init__(
        self,
        model_id: str,
        language: Optional[str] = None,
        progress_callback: Optional[
            Callable[[Optional[float], Optional[float]], None]
        ] = None,
    ) -> None:
        self.model_id = model_id.strip() or DEFAULT_PERSONA_SPEECH_MODEL_ID
        self.language = language.strip() if language and language.strip() else None
        self._processor: Optional[_WhisperProcessor] = None
        self._model: Optional[_WhisperModel] = None
        self._device: Optional[Device] = None
        self._dtype: Optional[DType] = None
        self._is_multilingual = True
        self._load_lock = threading.Lock()
        self._progress_callback = progress_callback

    def _load_model(self) -> None:
        """Load the configured Whisper processor and model once."""
        if self._processor is not None and self._model is not None:
            return

        with self._load_lock:
            if self._processor is not None and self._model is not None:
                return

            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                AutoModelForSpeechSeq2Seq,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("can't load Whisper without a CUDA device")

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            with huggingface_progress(self._progress_callback):
                model = cast(
                    _WhisperModel,
                    AutoModelForSpeechSeq2Seq.from_pretrained(
                        self.model_id,
                        dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        quantization_config=bnb_config,
                        device_map="auto",
                    ),
                )
                self._processor = cast(
                    _WhisperProcessor, AutoProcessor.from_pretrained(self.model_id)
                )
            model.eval()
            generation_config = getattr(model, "generation_config", None)
            self._is_multilingual = bool(
                getattr(generation_config, "is_multilingual", True)
            )
            self._model = model
            self._device = next(model.parameters()).device
            self._dtype = torch.bfloat16

    def _decode(
        self,
        audio: AudioChunk,
        sample_rate: int,
        return_segments: bool = False,
    ) -> Union[str, tuple[WhisperSegment, ...]]:
        """Prepare audio, generate Whisper tokens, and decode the transcript."""
        self._load_model()
        processor = self._processor
        model = self._model
        device = self._device
        dtype = self._dtype
        if processor is None or model is None or device is None or dtype is None:
            raise RuntimeError("Whisper speech model did not initialize")

        processed = processor(
            np.asarray(audio, dtype=np.float32),
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = processed.input_features

        model_inputs: dict[str, Union[torch.Tensor, str]] = {
            "input_features": input_features.to(device=device, dtype=dtype),
        }
        attention_mask = processed.attention_mask
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(device=device)
        if self._is_multilingual:
            model_inputs["task"] = "transcribe"
            if self.language is not None and self.language.lower() != "auto":
                model_inputs["language"] = self.language

        with torch.inference_mode():
            if return_segments:
                generated = model.generate(
                    **model_inputs,
                    return_timestamps=True,
                    return_segments=True,
                    return_token_timestamps=True,
                )
            else:
                generated = model.generate(**model_inputs)

        if return_segments:
            return self._decode_segments(
                processor,
                cast(WhisperGenerationPayload, generated),
            )

        generated_ids = cast(torch.Tensor, generated)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    @staticmethod
    def _decode_segments(
        processor: _WhisperProcessor,
        generated: WhisperGenerationPayload,
    ) -> tuple[WhisperSegment, ...]:
        """Extract timestamped text segments from a Whisper generation result."""
        if not isinstance(generated, Mapping):
            return ()
        raw_segments = generated.get("segments")
        if not isinstance(raw_segments, Sequence) or not raw_segments:
            return ()
        if isinstance(raw_segments[0], Mapping):
            segment_items = raw_segments
        else:
            first_batch = raw_segments[0]
            if not isinstance(first_batch, Sequence):
                return ()
            segment_items = first_batch

        segments: list[WhisperSegment] = []
        for raw_segment in segment_items:
            if not isinstance(raw_segment, Mapping):
                continue
            segment = raw_segment
            start = segment.get("start")
            end = segment.get("end")
            tokens = segment.get("tokens")
            start_value = WhisperTranscriber._timestamp_value(start)
            end_value = WhisperTranscriber._timestamp_value(end)
            if start_value is None or end_value is None:
                continue
            text = ""
            if tokens is not None:
                try:
                    token_tensor = torch.as_tensor(tokens)
                    if token_tensor.ndim == 1:
                        token_tensor = token_tensor.unsqueeze(0)
                    decoded = processor.batch_decode(
                        token_tensor,
                        skip_special_tokens=True,
                    )
                except (RuntimeError, TypeError, ValueError):
                    decoded = []
                if decoded:
                    text = decoded[0].strip()
            if text and end_value > start_value:
                words = WhisperTranscriber._decode_words(
                    processor,
                    segment,
                    start_value,
                    end_value,
                )
                segments.append(
                    WhisperSegment(
                        text=text,
                        start=start_value,
                        end=end_value,
                        words=words,
                    )
                )
        return tuple(segments)

    @staticmethod
    def _decode_words(
        processor: _WhisperProcessor,
        raw_segment: WhisperSegmentPayload,
        segment_start: float,
        segment_end: float,
    ) -> tuple[WhisperWord, ...]:
        """Group Whisper token timestamps into word-level timing ranges."""
        tokens = raw_segment.get("tokens")
        indexes = raw_segment.get("idxs")
        result = raw_segment.get("result")
        if not isinstance(indexes, Sequence) or len(indexes) < 2:
            return ()
        if not isinstance(result, Mapping):
            return ()

        token_timestamps = result.get("token_timestamps")
        if token_timestamps is None or tokens is None:
            return ()

        try:
            start_index = int(indexes[0])
            end_index = int(indexes[1])
            timestamp_values = cast(
                Sequence[WhisperScalar], token_timestamps[start_index:end_index]
            )
            token_values = cast(Sequence[WhisperScalar], tokens)
            if len(token_values) != len(timestamp_values):
                return ()
        except (IndexError, TypeError, ValueError):
            return ()

        tokenizer = getattr(processor, "tokenizer", None)
        decode_token = getattr(tokenizer, "decode", None)
        if not callable(decode_token):
            return ()

        words: list[WhisperWord] = []
        current_text = ""
        current_start: Optional[float] = None

        for token, timestamp in zip(token_values, timestamp_values):
            token_id = WhisperTranscriber._timestamp_value(token)
            timestamp_value = WhisperTranscriber._timestamp_value(timestamp)
            if token_id is None or timestamp_value is None:
                continue

            try:
                token_text = decode_token(
                    [int(token_id)],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except (TypeError, ValueError):
                continue
            if not isinstance(token_text, str) or not token_text:
                continue

            timestamp_value = max(
                segment_start,
                min(segment_end, timestamp_value),
            )
            starts_word = bool(current_text) and token_text[:1].isspace()
            if starts_word:
                words.append(
                    WhisperWord(
                        text=current_text.strip(),
                        start=max(segment_start, current_start or segment_start),
                        end=max(
                            segment_start,
                            current_start or segment_start,
                            timestamp_value,
                        ),
                    )
                )
                current_text = ""
                current_start = None

            if current_start is None:
                current_start = max(segment_start, timestamp_value)
            current_text += token_text

        if current_text.strip():
            words.append(
                WhisperWord(
                    text=current_text.strip(),
                    start=max(segment_start, current_start or segment_start),
                    end=max(segment_start, current_start or segment_start, segment_end),
                )
            )

        return tuple(word for word in words if word.text and word.end >= word.start)

    @staticmethod
    def _timestamp_value(value: Optional[WhisperScalar]) -> Optional[float]:
        """Convert a Python or tensor scalar into a timestamp."""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if value is None or not isinstance(value, torch.Tensor):
            return None
        converted = value.item()
        return float(converted) if isinstance(converted, (int, float)) else None

    def transcribe(
        self,
        audio: AudioChunk,
        sample_rate: int,
    ) -> str:
        """Transcribe one captured microphone snapshot.

        Args:
            audio: Mono or multichannel microphone audio samples.
            sample_rate: Sample rate of the captured audio in hertz.

        Returns:
            str: The transcribed speech, or an empty string when no audio is available.

        Raises:
            ValueError: If the captured audio is not one- or two-dimensional.
        """
        if len(audio) <= 0:
            return ""

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = np.asarray(
                np.mean(mono_audio, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
        if mono_audio.ndim != 1:
            raise ValueError("Whisper speech input must be mono audio")
        if sample_rate != WHISPER_SAMPLE_RATE:
            resampled = resample_audio(
                mono_audio,
                sample_rate,
                WHISPER_SAMPLE_RATE,
            )
            mono_audio = np.asarray(
                np.mean(resampled, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
        decoded = self._decode(mono_audio, WHISPER_SAMPLE_RATE)
        return decoded if isinstance(decoded, str) else ""

    def transcribe_segments(
        self,
        audio: AudioChunk,
        sample_rate: int,
    ) -> tuple[WhisperSegment, ...]:
        """Transcribe speech and return timestamped Whisper segments.

        Args:
            audio: Mono or multichannel audio samples.
            sample_rate: Sample rate of the audio in hertz.

        Returns:
            tuple[WhisperSegment, ...]: Timestamped speech segments, or an empty
                tuple when Whisper does not provide segment timestamps.

        Raises:
            ValueError: If the audio is not one- or two-dimensional.
        """
        if len(audio) <= 0:
            return ()

        mono_audio = np.asarray(audio, dtype=np.float32)
        if mono_audio.ndim == 2:
            mono_audio = np.asarray(
                np.mean(mono_audio, axis=1, dtype=np.float32),
                dtype=np.float32,
            )
        if mono_audio.ndim != 1:
            raise ValueError("Whisper speech input must be mono audio")
        if sample_rate != WHISPER_SAMPLE_RATE:
            resampled = resample_audio(mono_audio, sample_rate, WHISPER_SAMPLE_RATE)
            mono_audio = np.asarray(
                np.mean(resampled, axis=1, dtype=np.float32)
                if np.asarray(resampled).ndim == 2
                else resampled,
                dtype=np.float32,
            )
        try:
            result = self._decode(
                mono_audio,
                WHISPER_SAMPLE_RATE,
                return_segments=True,
            )
        except (RuntimeError, TypeError, ValueError):
            return ()
        return result if isinstance(result, tuple) else ()
