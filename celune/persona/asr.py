# SPDX-License-Identifier: Apache-2.0
"""Speech-to-text helpers for Persona microphone input."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np

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

            import torch
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
    ) -> str:
        """Prepare audio, generate Whisper tokens, and decode the transcript."""
        import torch

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
            generated_ids = model.generate(**model_inputs)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

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
        return self._decode(mono_audio, WHISPER_SAMPLE_RATE)
