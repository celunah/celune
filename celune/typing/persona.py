"""Persona runtime protocols and type aliases."""

from collections.abc import Sequence
from typing import Literal, Optional, Protocol, TypedDict, Union

import torch
from transformers.tokenization_utils_base import BatchEncoding

from .common import JSONSerializable, VideoMetadataScalar

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

        """
        raise NotImplementedError("protocol not defined")


class PersonaTokenizer(Protocol):
    """Tokenizer protocol used by the Persona runtime."""

    eos_token_id: Optional[int]

    def __call__(self, *, text: str, return_tensors: str) -> BatchEncoding:
        """Tokenize text into a batch encoding."""
        raise NotImplementedError("protocol not defined")

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        """Decode generated token IDs into text.

        Args:
            token_ids: Generated token IDs to decode.
            skip_special_tokens: Whether special tokens should be omitted.

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

        """
        raise NotImplementedError("protocol not defined")

    def eval(self) -> None:
        """Switch the model into eval mode."""
        raise NotImplementedError("protocol not defined")
