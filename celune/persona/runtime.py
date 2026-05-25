# SPDX-License-Identifier: MIT
"""Shared Persona runtime helpers for Celune-managed generation."""

from __future__ import annotations

import os
import gc
import threading
import contextlib
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Literal, Optional, Protocol, Any, cast

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.tokenization_utils_base import BatchEncoding

from ..constants import JSONSerializable, PERSONA_MODEL_ID

Role = Literal["system", "user", "assistant"]
type JSONDict = dict[str, JSONSerializable]
MessageContent = str | list[JSONDict]


class ChatTemplateRenderer(Protocol):
    """Renderer supporting Hugging Face-style chat templates."""

    def apply_chat_template(
        self,
        conversation: object,
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        return_dict: bool = ...,
        return_tensors: str = ...,
    ) -> str | BatchEncoding:
        """Render or tokenize a chat conversation."""
        raise NotImplementedError("protocol method")


class PersonaTokenizer(Protocol):
    """Tokenizer protocol used by the Persona runtime."""

    eos_token_id: int | None

    def __call__(self, *, text: str, return_tensors: str) -> BatchEncoding:
        """Tokenize text into a batch encoding."""
        raise NotImplementedError("protocol method")

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
        """Decode generated token IDs into text."""
        raise NotImplementedError("protocol method")


class PersonaProcessor(ChatTemplateRenderer, Protocol):
    """Processor protocol used by the Persona runtime."""

    tokenizer: PersonaTokenizer | None

    def __call__(
        self,
        *,
        text: str,
        images: object = ...,
        videos: object = ...,
        video_metadata: object = ...,
        return_tensors: str,
        do_resize: bool = ...,
        **kwargs: object,
    ) -> BatchEncoding:
        """Build multimodal model inputs."""
        raise NotImplementedError("protocol method")


class PersonaModel(Protocol):
    """Model protocol used by the Persona runtime."""

    device: torch.device | str

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Generate token IDs from prepared inputs."""
        raise NotImplementedError("protocol method")

    def eval(self) -> object:
        """Switch the model into eval mode."""
        raise NotImplementedError("protocol method")


@dataclass(slots=True)
class ChatMessage:
    """One OpenAI-style chat message."""

    role: Role
    content: MessageContent


@dataclass(slots=True)
class GenerateRequest:
    """Celune-to-Persona generation request."""

    model: str | None = None
    quantization: str | None = None
    quantized: bool = True
    system: str | None = None
    user: str | None = None
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


def _render_chat_prompt(renderer: object, messages: Sequence[JSONDict]) -> str:
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
        """Load the requested model, quantized by default."""
        if (
            self.model is not None
            and self.model_id == model_id
            and self.quantization == quantization
        ):
            return

        self.unload()

        load_kwargs: dict[str, Any] = {"device_map": "auto"}
        normalized = quantization.casefold()
        if normalized in {"4bit", "nf4", "bnb4", "bitsandbytes-4bit"}:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Persona quantized loading requires a CUDA-enabled Torch build"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif normalized in {"8bit", "bnb8", "bitsandbytes-8bit"}:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Persona quantized loading requires a CUDA-enabled Torch build"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif normalized in {"none", "false", "off", "disabled"}:
            load_kwargs["torch_dtype"] = torch.bfloat16
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
        except Exception:
            processor = None

        model = cast(
            PersonaModel,
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                **load_kwargs,
            ),
        )
        supports_vision = processor is not None and _processor_supports_native_vision(
            processor
        )

        if processor is None:
            tokenizer = cast(
                PersonaTokenizer,
                AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                ),
            )
        else:
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
        """Generate a persona-formatted response."""
        messages = request.messages or _messages_from_legacy_fields(request)
        if not messages:
            raise ValueError("Persona request has no messages")
        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            raise ValueError("Persona backend is not loaded")

        message_dicts = cast(
            list[JSONDict],
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        )
        inputs = self._build_inputs(message_dicts)
        generation_kwargs: dict[str, object] = {}
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
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

    def _build_inputs(self, messages: Sequence[JSONDict]) -> BatchEncoding:
        """Build model inputs, including optional image and video content."""
        model = self.model
        if model is None:
            raise ValueError("Persona backend is not loaded")

        if _messages_have_vision(messages):
            processor = self.processor
            if not self.supports_vision or processor is None:
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
            vision_info = cast(
                tuple[
                    list[object] | None,
                    list[tuple[object, object]] | None,
                    dict[str, object],
                ],
                process_vision_info(
                    list(messages),
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                ),
            )
            image_inputs, video_inputs, video_kwargs = vision_info
            video_metadata = None
            if video_inputs:
                video_inputs, video_metadata = zip(*video_inputs)
                video_inputs = list(video_inputs)
                video_metadata = list(video_metadata)

            return processor(
                text=prompt,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            ).to(model.device)

        tokenizer = self.tokenizer
        renderer: object = self.processor if self.processor is not None else tokenizer
        if tokenizer is None:
            raise ValueError("Persona backend is not loaded")
        prompt = _render_chat_prompt(renderer, messages)
        return tokenizer(text=prompt, return_tensors="pt").to(model.device)


class PersonaRuntime:
    """Lazy Persona model runtime owned by the Celune process."""

    def __init__(self) -> None:
        self.backend = PersonaBackend()
        self.lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Return the currently loaded model identifier."""
        return self.backend.model_id

    @property
    def quantization(self) -> str:
        """Return the currently loaded quantization mode."""
        return self.backend.quantization

    def load(self, model_id: str, quantization: str) -> None:
        """Explicitly load the Persona backend with the requested model."""
        with self.lock:
            self.backend.load(model_id, quantization)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a persona-formatted response."""
        model_id = (
            request.model
            or os.getenv("PERSONA_MODEL")
            or os.getenv("PYOP_MODEL")
            or PERSONA_MODEL_ID
        )
        quantization = (
            request.quantization
            or os.getenv("PERSONA_QUANTIZATION")
            or os.getenv("PYOP_QUANTIZATION")
            or ("4bit" if request.quantized else "none")
        )
        self.load(model_id, quantization)
        with self.lock:
            return self.backend.generate(request)

    def close(self) -> None:
        """Unload the active Persona backend state."""
        with self.lock:
            self.backend.unload()


def request_from_json(payload: dict[str, JSONSerializable]) -> GenerateRequest:
    """Convert a JSON-like payload into a Persona generation request."""
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
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"system", "user", "assistant"} and isinstance(
                content, (str, list)
            ):
                messages.append(
                    ChatMessage(
                        role=cast(Role, role),
                        content=cast(MessageContent, content),
                    )
                )

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
    """Convert a Persona generation response to a JSON-like payload."""
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


def _messages_have_vision(messages: Sequence[JSONDict]) -> bool:
    """Return whether any chat message contains image or video content."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True

    return False


def _processor_supports_native_vision(processor: object) -> bool:
    """Return whether the processor exposes native multimodal chat rendering."""
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    return callable(apply_chat_template)
