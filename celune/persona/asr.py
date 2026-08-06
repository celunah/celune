# SPDX-License-Identifier: MIT
"""Speech-to-text helpers for Persona microphone input."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np
import torch

from ..dsp import resample_audio
from ..typing.aliases import AudioChunk
from ..typing.persona import _WhisperModel, _WhisperProcessor
from ..typing.persona import (  # pylint: disable=W0611
    _WhisperProcessorOutput,  # noqa: F401
)

if TYPE_CHECKING:
    # noinspection PyPep8Naming
    from torch import device as Device

    # noinspection PyPep8Naming
    from torch import dtype as DType


DEFAULT_PERSONA_SPEECH_MODEL_ID = "openai/whisper-large-v3-turbo"
PERSONA_SPEECH_END_DELAY_SECONDS = 1.5
WHISPER_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WhisperSegment:
    """One timestamped segment returned by Whisper."""

    text: str
    start: float
    end: float


class WhisperTranscriber:
    """Lazily load and run one configurable Hugging Face Whisper model."""

    def __init__(self, model_id: str, language: Optional[str] = None) -> None:
        self.model_id = model_id.strip() or DEFAULT_PERSONA_SPEECH_MODEL_ID
        self.language = language.strip() if language and language.strip() else None
        self._processor: Optional[_WhisperProcessor] = None
        self._model: Optional[_WhisperModel] = None
        self._device: Optional[Device] = None
        self._dtype: Optional[DType] = None
        self._is_multilingual = True
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        """Load the configured Whisper processor and model once."""
        if self._processor is not None and self._model is not None:
            return

        with self._load_lock:
            if self._processor is not None and self._model is not None:
                return

            from transformers import (
                AutoModelForSpeechSeq2Seq,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("can't load Whisper without a CUDA device")

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
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
            model.eval()
            generation_config = getattr(model, "generation_config", None)
            self._is_multilingual = bool(
                getattr(generation_config, "is_multilingual", True)
            )
            self._processor = cast(
                _WhisperProcessor, AutoProcessor.from_pretrained(self.model_id)
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
                )
            else:
                generated = model.generate(**model_inputs)

        if return_segments:
            return self._decode_segments(processor, generated)

        generated_ids = cast(torch.Tensor, generated)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    @staticmethod
    def _decode_segments(
        processor: _WhisperProcessor,
        generated: object,
    ) -> tuple[WhisperSegment, ...]:
        """Extract timestamped text segments from a Whisper generation result."""
        if not isinstance(generated, Mapping):
            return ()
        raw_segments = generated.get("segments")
        if not isinstance(raw_segments, Sequence) or not raw_segments:
            return ()
        first_batch = raw_segments[0]
        if not isinstance(first_batch, Sequence):
            return ()

        segments: list[WhisperSegment] = []
        for raw_segment in first_batch:
            if not isinstance(raw_segment, Mapping):
                continue
            start = raw_segment.get("start")
            end = raw_segment.get("end")
            tokens = raw_segment.get("tokens")
            start_value = WhisperTranscriber._timestamp_value(start)
            end_value = WhisperTranscriber._timestamp_value(end)
            if start_value is None or end_value is None:
                continue
            text = ""
            if tokens is not None:
                decoded = processor.batch_decode(
                    cast(torch.Tensor, tokens),
                    skip_special_tokens=True,
                )
                if decoded:
                    text = decoded[0].strip()
            if text and end_value > start_value:
                segments.append(
                    WhisperSegment(text=text, start=start_value, end=end_value)
                )
        return tuple(segments)

    @staticmethod
    def _timestamp_value(value: object) -> Optional[float]:
        """Convert a Python or tensor scalar into a timestamp."""
        if isinstance(value, (int, float)):
            return float(value)
        item = getattr(value, "item", None)
        if not callable(item):
            return None
        converted = item()
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
