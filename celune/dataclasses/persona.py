# SPDX-License-Identifier: MIT
"""Persona runtime dataclasses."""

from dataclasses import dataclass, field
from typing import Optional

from ..constants import PERSONA_CONTEXT_SPACE
from ..typing.persona import MessageContent, Role


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
    context_space: int = PERSONA_CONTEXT_SPACE

    def __post_init__(self) -> None:
        """Validate the requested Persona input context space."""
        if (
            isinstance(self.context_space, bool)
            or not isinstance(self.context_space, int)
            or self.context_space <= 0
        ):
            raise ValueError("Persona context_space must be a positive integer")


@dataclass(slots=True)
class GenerateResponse:
    """Persona generation response."""

    text: str
    response: str
    model: str
    quantization: str
