"""Backend-facing protocols and type aliases."""

from collections.abc import Iterator
from typing import Protocol, TypeVar

import torch


class BackendModel(Protocol):
    """Opaque backend model protocol for backend-independent storage."""


ModelT = TypeVar("ModelT", bound=BackendModel)
MiniPromptState = dict[str, dict[str, torch.Tensor]]


class MiniModel(Protocol):
    """Pocket TTS model surface used by Celune's mini backend."""

    sample_rate: int

    def get_state_for_audio_prompt(self, audio_conditioning: str) -> MiniPromptState:
        """Return a reusable prompt state for one reference audio path.

        Args:
            audio_conditioning: Backend-specific prompt descriptor for one voice sample.
        """
        raise NotImplementedError("protocol not defined")

    def generate_audio_stream(
        self,
        model_state: MiniPromptState,
        text_to_generate: str,
    ) -> Iterator[torch.Tensor]:
        """Yield streamed audio chunks for one prompt state and text.

        Args:
            model_state: Prompt state cached for the active voice.
            text_to_generate: Text content to synthesize.
        """
        raise NotImplementedError("protocol not defined")
