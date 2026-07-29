# SPDX-License-Identifier: MIT
"""Persona runtime protocols and type aliases."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Optional, Protocol, TypedDict, Union

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .common import JSONSerializable, VideoMetadataScalar

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Parameter

    from ..cevoice import CEVoicePersona
    from .aliases import AudioChunk
    from .common import Config

type Role = Literal["system", "user", "assistant"]
type VisionInput = Union[JSONSerializable, torch.Tensor, bytes, memoryview]
type ProcessorKwargValue = Union[VideoMetadataScalar, Sequence[VideoMetadataScalar]]
type ModelGenerateKwargValue = Union[torch.Tensor, int, float, bool]


class TextContentItem(TypedDict):
    """Text content block accepted by Persona chat messages."""

    type: Literal["text"]
    text: str


class ImageContentItem(TypedDict):
    """Image content block accepted by Persona chat messages."""

    type: Literal["image"]
    image: str


class VideoContentItem(TypedDict):
    """Video content block accepted by Persona chat messages."""

    type: Literal["video"]
    video: str


type ContentItem = Union[TextContentItem, ImageContentItem, VideoContentItem]
type VideoMetadata = dict[str, VideoMetadataScalar]
type VideoInputWithMetadata = tuple[VisionInput, VideoMetadata]
type VisionProcessorOutput = tuple[
    Optional[list[VisionInput]],
    Optional[list[VideoInputWithMetadata]],
    dict[str, ProcessorKwargValue],
]
type MessageContent = Union[str, list[ContentItem]]


class ChatMessagePayload(TypedDict):
    """Serialized chat message structure used by the Persona runtime."""

    role: Role
    content: MessageContent


type JSONDict = ChatMessagePayload


class ChatTemplateRenderer(Protocol):
    """Renderer supporting Hugging Face-style chat templates."""

    def apply_chat_template(
        self,
        conversation: Sequence[ChatMessagePayload],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        return_dict: bool = True,
        return_tensors: str = "pt",
    ) -> Union[str, BatchEncoding]:
        """Render or tokenize a chat conversation.

        Args:
            conversation: Persona chat history to render.
            tokenize: Whether the rendered conversation should be tokenized.
            add_generation_prompt: Whether to append an assistant generation turn.
            return_dict: Whether structured tensor output should be returned.
            return_tensors: Tensor backend requested by the caller.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class PersonaTokenizer(Protocol):
    """Tokenizer protocol used by the Persona runtime."""

    eos_token_id: Optional[int]

    def __call__(
        self,
        *,
        text: Union[str, Sequence[str]],
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
    ) -> BatchEncoding:
        """Tokenize text into a batch encoding."""
        raise NotImplementedError("protocol not defined")

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        """Decode generated token IDs into text.

        Args:
            token_ids: Generated token IDs to decode.
            skip_special_tokens: Whether special tokens should be omitted.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class PersonaProcessor(ChatTemplateRenderer, Protocol):
    """Processor protocol used by the Persona runtime."""

    tokenizer: Optional[PersonaTokenizer]

    def __call__(
        self,
        *,
        text: str,
        images: Optional[Sequence[VisionInput]] = None,
        videos: Optional[Sequence[VisionInput]] = None,
        video_metadata: Optional[Sequence[VideoMetadata]] = None,
        return_tensors: str,
        **kwargs: ProcessorKwargValue,
    ) -> BatchEncoding:
        """Build multimodal model inputs."""
        raise NotImplementedError("protocol not defined")


class PersonaModel(Protocol):
    """Model protocol used by the Persona runtime."""

    device: Union[torch.device, str]

    def generate(self, **kwargs: ModelGenerateKwargValue) -> torch.Tensor:
        """Generate token IDs from prepared inputs.

        Args:
            kwargs: Prepared model inputs and generation options.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")

    def eval(self) -> None:
        """Switch the model into eval mode.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class _WhisperProcessorOutput(Protocol):
    """Protocol for tensor fields returned by a Whisper processor."""

    input_features: Tensor
    attention_mask: Optional[Tensor]


class _WhisperProcessor(Protocol):
    """Protocol for the processor operations used by the transcriber."""

    def __call__(
        self,
        audio: AudioChunk,
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> _WhisperProcessorOutput:
        """Prepare one audio sample for Whisper."""

    def batch_decode(
        self,
        generated_ids: Tensor,
        *,
        skip_special_tokens: bool,
    ) -> list[str]:
        """Decode generated Whisper token IDs into text.

        Args:
            generated_ids: Token IDs generated by the Whisper model.
            skip_special_tokens: Whether to omit special control tokens.
        """


class _WhisperModel(Protocol):
    """Protocol for the model operations used by the transcriber."""

    def eval(self) -> None:
        """Switch the model to evaluation mode."""

    def generate(self, **kwargs: Union[Tensor, str]) -> Tensor:
        """Generate token IDs from prepared Whisper inputs.

        Args:
            kwargs: Tensor and text inputs forwarded to Whisper generation.
        """

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over the model parameters.

        Returns:
            Iterator[torch.nn.Parameter]: An iterator of model parameters.
        """


class _EmotionModelConfig(Protocol):
    """Protocol for model configs that expose emotion label mappings."""

    id2label: Mapping[Union[int, str], str]


class PersonaEngineView(Protocol):
    """Typed view of the engine fields consumed by Persona helpers."""

    config: Config
    current_character_persona: Optional[CEVoicePersona]
    current_character: Optional[str]
    voice_bundle_is_default: bool
    persona_history: list[dict[str, str]]
    persona_attachments: list[dict[str, str]]
    persona_session_summary: str


class PersonaClientResponse:
    """Small response shim matching the local HTTP client contract."""

    def __init__(self, payload: dict[str, JSONSerializable]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mirror the ``httpx`` response API for local in-process calls."""

    def json(self) -> dict[str, JSONSerializable]:
        """Return the stored response payload.

        Returns:
            str: The stored JSON response payload.
        """
        return dict(self._payload)
