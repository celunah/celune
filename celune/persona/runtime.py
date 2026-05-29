# SPDX-License-Identifier: MIT
"""Shared Persona runtime helpers for Celune-managed generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import gc
import os
import threading
from typing import Literal, Optional, Protocol, TypedDict, Union, cast

import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..vram import resolve_vram_preset
from ..constants import JSONSerializable, PERSONA_MODEL_ID

Role = Literal["system", "user", "assistant"]
type VideoMetadataScalar = Optional[Union[bool, int, float, str]]
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
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        return_dict: bool = ...,
        return_tensors: str = ...,
    ) -> Union[str, BatchEncoding]:
        """Render or tokenize a chat conversation.

        Args:
            conversation: The current Persona conversation.
            tokenize: Whether the conversation should be tokenized.
            add_generation_prompt: Whether the generation prompt should be appended.
            return_dict: Whether a Python dict should be returned.
            return_tensors: Whether PyTorch tensors should be returned.

        Raises:
            NotImplementedError: The protocol was called directly.
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
            token_ids: The token IDs to decode.
            skip_special_tokens: Whether special token IDs should be skipped while decoding.

        Raises:
            NotImplementedError: The protocol was called directly.
        """
        raise NotImplementedError("protocol not defined")


class PersonaProcessor(ChatTemplateRenderer, Protocol):
    """Processor protocol used by the Persona runtime."""

    tokenizer: Optional[PersonaTokenizer]

    def __call__(
        self,
        *,
        text: str,
        images: Optional[Sequence[VisionInput]] = ...,
        videos: Optional[Sequence[VisionInput]] = ...,
        video_metadata: Optional[Sequence[VideoMetadata]] = ...,
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
            kwargs: Generation-specific keyword arguments to use.

        Raises:
            NotImplementedError: The protocol was called directly.
        """
        raise NotImplementedError("protocol not defined")

    def eval(self) -> None:
        """Switch the model into eval mode.

        Raises:
            NotImplementedError: The protocol was called directly.
        """
        raise NotImplementedError("protocol not defined")


@dataclass(slots=True)
class ChatMessage:
    """One OpenAI-style chat message."""

    role: Role
    content: MessageContent


@dataclass(slots=True)
class GenerateRequest:
    """Celune-to-Persona generation request."""

    model: Optional[str] = None
    quantization: Optional[str] = None
    quantized: bool = True
    system: Optional[str] = None
    user: Optional[str] = None
    messages: list[ChatMessage] = field(default_factory=list)
    max_new_tokens: int = 220
    temperature: float = 0.75
    top_p: float = 0.9
    repetition_penalty: float = 1.05


@dataclass(slots=True)
class GenerateResponse:
    """Persona generation response."""

    text: str
    response: str
    model: str
    quantization: str


def _render_chat_prompt(
    renderer: Union[ChatTemplateRenderer, PersonaTokenizer],
    messages: Sequence[ChatMessagePayload],
) -> str:
    """Render chat messages into one prompt string."""
    apply_chat_template = getattr(renderer, "apply_chat_template", None)
    if callable(apply_chat_template):
        rendered = cast(ChatTemplateRenderer, renderer).apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered

    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = message.get("content")
        if isinstance(content, list):
            rendered_items: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                raw_text = item.get("text")
                if item.get("type") == "text" and isinstance(raw_text, str):
                    text = raw_text.strip()
                    if text:
                        rendered_items.append(text)
            rendered_content = "\n".join(rendered_items)
        elif isinstance(content, str):
            rendered_content = content.strip()
        else:
            rendered_content = ""

        if rendered_content:
            parts.append(f"{role.title()}: {rendered_content}")

    parts.append("Assistant:")
    return "\n\n".join(parts)


def _payload_from_message(message: ChatMessage) -> ChatMessagePayload:
    """Convert one runtime chat message into the serialized payload shape."""
    return ChatMessagePayload(role=message.role, content=message.content)


class PersonaBackend:
    """Character-agnostic backend for Persona generation."""

    def __init__(self) -> None:
        self.model_id = ""
        self.quantization = ""
        self.processor: Optional[PersonaProcessor] = None
        self.tokenizer: Optional[PersonaTokenizer] = None
        self.model: Optional[PersonaModel] = None
        self.supports_vision = False

    def load(self, model_id: str, quantization: str) -> None:
        """Load the requested model, quantized by default.

        Args:
            model_id: The model ID to load.
            quantization: The requested quantization mode to use.

        Raises:
            ValueError: Quantization was requested when CUDA wasn't available.
        """
        if (
            self.model is not None
            and self.model_id == model_id
            and self.quantization == quantization
        ):
            return

        self.unload()

        normalized = quantization.casefold()
        if normalized in {"4bit", "nf4", "bnb4", "bitsandbytes-4bit"}:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Persona quantized loading requires a CUDA-enabled Torch build"
                )
            model = cast(
                PersonaModel,
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    ),
                ),
            )
        elif normalized in {"8bit", "bnb8", "bitsandbytes-8bit"}:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Persona quantized loading requires a CUDA-enabled Torch build"
                )
            model = cast(
                PersonaModel,
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                ),
            )
        elif normalized in {"none", "false", "off", "disabled"}:
            model = cast(
                PersonaModel,
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                ),
            )
        else:
            raise ValueError(f"unsupported Persona quantization mode: {quantization}")

        try:
            processor = cast(
                PersonaProcessor,
                AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                ),
            )
        except Exception as exc:
            raise ValueError(
                f"Persona processor failed to load for model '{model_id}'"
            ) from exc

        supports_vision = True
        tokenizer = processor.tokenizer
        if tokenizer is None:
            tokenizer = cast(
                PersonaTokenizer,
                AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                ),
            )

        model.eval()
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.quantization = quantization
        self.supports_vision = supports_vision

    def unload(self) -> None:
        """Unload the active Persona model and related processor state."""
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.model_id = ""
        self.quantization = ""
        self.supports_vision = False
        gc.collect()

        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a persona-formatted response.

        Args:
            request: The request to be processed by Persona.

        Returns:
            GenerateResponse: A response generated from Persona.

        Raises:
            ValueError: Persona request has no message or Persona was not loaded while the caller wanted to use it.
        """
        messages = request.messages or _messages_from_legacy_fields(request)
        if not messages:
            raise ValueError("Persona request has no messages")
        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise ValueError("Persona backend is not loaded")

        message_dicts = [_payload_from_message(message) for message in messages]
        used_vision = _messages_have_vision(message_dicts)
        inputs: Optional[BatchEncoding] = None
        model_inputs: Optional[dict[str, torch.Tensor]] = None
        output_ids: Optional[torch.Tensor] = None
        new_ids: Optional[torch.Tensor] = None
        try:
            inputs = self._build_inputs(message_dicts)
            model_inputs = {
                key: cast(torch.Tensor, value) for key, value in dict(inputs).items()
            }
            generation_kwargs: dict[str, int] = {}
            pad_token_id = tokenizer.eos_token_id
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            with torch.inference_mode():
                output_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=request.temperature > 0,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    **generation_kwargs,
                )

            input_ids = cast(torch.Tensor, inputs["input_ids"])
            new_ids = output_ids[0, input_ids.shape[1] :]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return GenerateResponse(
                text=text,
                response=text,
                model=self.model_id,
                quantization=self.quantization,
            )
        finally:
            # Vision requests can allocate substantial transient GPU memory for
            # decoded image/video inputs; drop those tensors as soon as the turn ends.
            del new_ids
            del output_ids
            del model_inputs
            del inputs
            if used_vision:
                gc.collect()
                if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        torch.cuda.synchronize()
                    with contextlib.suppress(Exception):
                        torch.cuda.empty_cache()

    def _build_inputs(self, messages: Sequence[ChatMessagePayload]) -> BatchEncoding:
        """Build model inputs, including optional image and video content."""
        model = self.model
        if model is None:
            raise ValueError("Persona backend is not loaded")

        if _messages_have_vision(messages):
            processor = self.processor
            if processor is None:
                raise ValueError(
                    "Persona backend for the current model does not support vision input"
                )
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError as exc:
                if _processor_supports_native_vision(processor):
                    encoded = cast(
                        ChatTemplateRenderer,
                        processor,
                    ).apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    return cast(BatchEncoding, encoded).to(model.device)
                raise ValueError(
                    "Persona vision requests require qwen-vl-utils or native "
                    "Qwen VL processor support in the Persona environment"
                ) from exc

            prompt = _render_chat_prompt(processor, messages)
            vision_messages = list(messages)
            vision_info = cast(
                VisionProcessorOutput,
                process_vision_info(
                    cast(
                        list[
                            dict[
                                str,
                                Union[
                                    TextContentItem, ImageContentItem, VideoContentItem
                                ],
                            ]
                        ],
                        vision_messages,
                    ),
                    return_video_kwargs=True,
                    return_video_metadata=True,
                ),
            )
            image_inputs, video_inputs, video_kwargs = vision_info
            processed_video_inputs: Optional[list[VisionInput]] = None
            video_metadata: Optional[list[VideoMetadata]] = None
            if video_inputs:
                raw_video_inputs, raw_video_metadata = zip(*video_inputs)
                processed_video_inputs = list(raw_video_inputs)
                video_metadata = list(raw_video_metadata)

            return processor(
                text=prompt,
                images=image_inputs,
                videos=processed_video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
                **video_kwargs,
            ).to(model.device)

        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Persona backend is not loaded")
        renderer: Union[ChatTemplateRenderer, PersonaTokenizer] = (
            self.processor if self.processor is not None else tokenizer
        )
        prompt = _render_chat_prompt(renderer, messages)
        return tokenizer(text=prompt, return_tensors="pt").to(model.device)


class PersonaRuntime:
    """Lazy Persona model runtime owned by the Celune process."""

    def __init__(
        self,
        config: Optional[Mapping[str, JSONSerializable]] = None,
    ) -> None:
        self.backend = PersonaBackend()
        self.config = config
        self.lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Return the currently loaded model identifier.

        Returns:
            str: The currently loaded Persona model ID, Qwen/Qwen2.5-VL-3B-Instruct or a derivative.
        """
        return self.backend.model_id

    @property
    def quantization(self) -> str:
        """Return the currently loaded quantization mode.

        Returns:
            str: The Persona quantization mode currently in use.
        """
        return self.backend.quantization

    def load(self, model_id: str, quantization: str) -> None:
        """Explicitly load the Persona backend with the requested model.

        Args:
            model_id: The model ID to load.
            quantization: The quantization mode to use.
        """
        quantization = self._allowed_quantization(quantization)
        with self.lock:
            self.backend.load(model_id, quantization)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a persona-formatted response.

        Args:
            request: The request to be processed by Persona.

        Returns:
            GenerateResponse: A generated Persona response.
        """
        model_id = request.model or os.getenv("PERSONA_MODEL") or PERSONA_MODEL_ID
        quantization = (
            request.quantization
            or os.getenv("PERSONA_QUANTIZATION")
            or ("4bit" if request.quantized else "none")
        )
        quantization = self._allowed_quantization(quantization)
        with self.lock:
            self.backend.load(model_id, quantization)
            return self.backend.generate(request)

    def close(self) -> None:
        """Unload the active Persona backend state."""
        with self.lock:
            self.backend.unload()

    def _allowed_quantization(self, requested: str) -> str:
        """Clamp Persona quantization to what the VRAM preset allows."""
        if self.config is None:
            return requested

        preset = resolve_vram_preset(self.config)
        if not preset.persona_enabled:
            raise ValueError(f"Persona is not available for VRAM tier '{preset.tier}'")

        allowed = preset.persona_quantization
        if requested.casefold() == allowed.casefold():
            return requested
        return allowed


def request_from_json(payload: dict[str, JSONSerializable]) -> GenerateRequest:
    """Convert a JSON-like payload into a Persona generation request.

    Args:
        payload: The JSON-serializable payload to process as a Persona request.

    Returns:
        GenerateRequest: A generated Persona request from JSON data.
    """
    raw_messages = payload.get("messages")
    messages: list[ChatMessage] = []
    model = payload.get("model")
    quantization = payload.get("quantization")
    system = payload.get("system")
    user = payload.get("user")
    max_new_tokens = payload.get("max_new_tokens")
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    repetition_penalty = payload.get("repetition_penalty")
    if isinstance(raw_messages, list):
        for item in raw_messages:
            message = _message_from_json(item)
            if message is not None:
                messages.append(message)

    return GenerateRequest(
        model=model if isinstance(model, str) else None,
        quantization=quantization if isinstance(quantization, str) else None,
        quantized=bool(payload.get("quantized", True)),
        system=system if isinstance(system, str) else None,
        user=user if isinstance(user, str) else None,
        messages=messages,
        max_new_tokens=int(max_new_tokens)
        if isinstance(max_new_tokens, (int, float))
        else 220,
        temperature=float(temperature)
        if isinstance(temperature, (int, float))
        else 0.75,
        top_p=float(top_p) if isinstance(top_p, (int, float)) else 0.9,
        repetition_penalty=(
            float(repetition_penalty)
            if isinstance(repetition_penalty, (int, float))
            else 1.05
        ),
    )


def response_to_json(response: GenerateResponse) -> dict[str, JSONSerializable]:
    """Convert a Persona generation response to a JSON-like payload.

    Args:
        response: The response to convert to a JSON-serializable payload.

    Returns:
        dict[str, JSONSerializable]: A JSON-serializable representation of the Persona response.
    """
    return {
        "text": response.text,
        "response": response.response,
        "model": response.model,
        "quantization": response.quantization,
    }


def _messages_from_legacy_fields(request: GenerateRequest) -> list[ChatMessage]:
    """Build chat messages from the flat compatibility fields."""
    messages: list[ChatMessage] = []
    if request.system and request.system.strip():
        messages.append(ChatMessage(role="system", content=request.system.strip()))
    if request.user and request.user.strip():
        messages.append(ChatMessage(role="user", content=request.user.strip()))
    return messages


def _messages_have_vision(messages: Sequence[ChatMessagePayload]) -> bool:
    """Return whether any chat message contains image or video content."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True

    return False


def _processor_supports_native_vision(processor: PersonaProcessor) -> bool:
    """Return whether the processor exposes native multimodal chat rendering."""
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    return callable(apply_chat_template)


def _content_from_json(value: JSONSerializable) -> Optional[MessageContent]:
    """Normalize serialized Persona message content."""
    if isinstance(value, str):
        return value

    if not isinstance(value, list):
        return None

    items: list[ContentItem] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                items.append(TextContentItem(type="text", text=text))
        elif item_type == "image":
            image = item.get("image")
            if isinstance(image, str):
                items.append(ImageContentItem(type="image", image=image))
        elif item_type == "video":
            video = item.get("video")
            if isinstance(video, str):
                items.append(VideoContentItem(type="video", video=video))

    return items


def _message_from_json(value: JSONSerializable) -> Optional[ChatMessage]:
    """Normalize one serialized Persona chat message."""
    if not isinstance(value, dict):
        return None

    role = value.get("role")
    normalized_role: Role
    if role == "system":
        normalized_role = "system"
    elif role == "user":
        normalized_role = "user"
    elif role == "assistant":
        normalized_role = "assistant"
    else:
        return None

    content = _content_from_json(value.get("content"))
    if content is None:
        return None

    return ChatMessage(role=normalized_role, content=content)
