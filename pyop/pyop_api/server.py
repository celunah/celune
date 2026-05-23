# SPDX-License-Identifier: MIT
"""FastAPI service for the detached PYOP model."""

from __future__ import annotations

import os
import threading
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import PYOP_MODEL_ID

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One OpenAI-style chat message."""

    role: Role
    content: str


class GenerateRequest(BaseModel):
    """Celune-to-PYOP generation request."""

    model: Optional[str] = None
    quantization: Optional[str] = None
    quantized: bool = True
    system: Optional[str] = None
    user: Optional[str] = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_new_tokens: int = Field(default=220, ge=1, le=2048)
    temperature: float = Field(default=0.75, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.0, le=3.0)


class GenerateResponse(BaseModel):
    """PYOP generation response."""

    text: str
    response: str
    model: str
    quantization: str


class PyopRuntime:
    """Lazy PYOP model runtime owned by the detached API process."""

    def __init__(self) -> None:
        self.model_id = ""
        self.quantization = ""
        self.tokenizer: Any = None
        self.model: Any = None
        self.lock = threading.Lock()

    def load(self, model_id: str, quantization: str) -> None:
        """Load the requested model, quantized by default."""
        if (
            self.model is not None
            and self.tokenizer is not None
            and self.model_id == model_id
            and self.quantization == quantization
        ):
            return

        with self.lock:
            if (
                self.model is not None
                and self.tokenizer is not None
                and self.model_id == model_id
                and self.quantization == quantization
            ):
                return

            load_kwargs: dict[str, Any] = {"device_map": "auto"}
            normalized = quantization.casefold()
            if normalized in {"4bit", "nf4", "bnb4", "bitsandbytes-4bit"}:
                if not torch.cuda.is_available():
                    raise ValueError(
                        "PYOP quantized loading requires a CUDA-enabled Torch build"
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
                        "PYOP quantized loading requires a CUDA-enabled Torch build"
                    )
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            elif normalized in {"none", "false", "off", "disabled"}:
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                raise ValueError(f"unsupported PYOP quantization mode: {quantization}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                **load_kwargs,
            )
            self.model.eval()
            self.model_id = model_id
            self.quantization = quantization

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate a persona-formatted response."""
        model_id = request.model or os.getenv("PYOP_MODEL", PYOP_MODEL_ID)
        quantization = (
            request.quantization
            or os.getenv("PYOP_QUANTIZATION")
            or ("4bit" if request.quantized else "none")
        )
        self.load(model_id, quantization)

        messages = request.messages or _messages_from_legacy_fields(request)
        if not messages:
            raise ValueError("PYOP request has no messages")

        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return GenerateResponse(
            text=text,
            response=text,
            model=model_id,
            quantization=quantization,
        )


def _messages_from_legacy_fields(request: GenerateRequest) -> list[ChatMessage]:
    """Build chat messages from the flat compatibility fields."""
    messages: list[ChatMessage] = []
    if request.system and request.system.strip():
        messages.append(ChatMessage(role="system", content=request.system.strip()))
    if request.user and request.user.strip():
        messages.append(ChatMessage(role="user", content=request.user.strip()))
    return messages


runtime = PyopRuntime()
app = FastAPI(title="PYOP", version="0.1.0")


@app.get("/")
def health() -> dict[str, str]:
    """Return a lightweight health response."""
    return {
        "status": "ok",
        "model": runtime.model_id or os.getenv("PYOP_MODEL", PYOP_MODEL_ID),
        "quantization": runtime.quantization or os.getenv("PYOP_QUANTIZATION", "4bit"),
        "cuda": str(torch.cuda.is_available()).lower(),
        "torch": torch.__version__,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a PYOP response for Celune."""
    try:
        return runtime.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
