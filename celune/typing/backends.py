# SPDX-License-Identifier: Apache-2.0
"""Backend-facing protocols and type aliases."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

if TYPE_CHECKING:
    from .aliases import RuntimeValue, SeedVCArgument, SeedVCGenerator
    from .common import JSONSerializable


class BackendModel(Protocol):
    """Opaque backend model protocol for backend-independent storage."""


ModelT = TypeVar("ModelT", bound=BackendModel)
type MiniPromptState = dict[str, dict[str, torch.Tensor]]


class MiniModel(Protocol):
    """Pocket TTS model surface used by Celune's mini backend."""

    sample_rate: int

    def get_state_for_audio_prompt(self, audio_conditioning: str) -> MiniPromptState:
        """Return a reusable prompt state for one reference audio path.

        Args:
            audio_conditioning: Backend-specific prompt descriptor for one voice sample.

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
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

        Raises:
            NotImplementedError: If `NotImplementedError` needs to be raised.
        """
        raise NotImplementedError("protocol not defined")


class _StreamingSpeechModel(Protocol):
    """Protocol for stateful streaming speech detectors."""

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return one speech probability tensor."""

    def reset_states(self) -> None:
        """Reset the detector's internal streaming state."""


class GPTSoVITSPipeline(Protocol):
    """Subset of the official GPT-SoVITS pipeline used by Celune."""

    def run(self, inputs: dict[str, JSONSerializable]) -> Iterator[RuntimeValue]:
        """Run one GPT-SoVITS request and yield audio tuples.

        Args:
            inputs: Request dictionary containing text, language, reference audio, prompt metadata, and inference
                controls.

        Returns:
            Iterator[RuntimeValue]: GPT-SoVITS sample-rate/audio pairs.
        """

    def stop(self) -> None:
        """Stop the active inference operation."""


class _GPTSoVITSConfig(Protocol):
    """Constructor surface of GPT-SoVITS' ``TTS_Config`` class."""


class _SeedVCWrapper(Protocol):
    """Protocol for the dynamically loaded Seed-VC wrapper."""

    def convert_voice(self, **kwargs: SeedVCArgument) -> SeedVCGenerator:
        """Run Seed-VC and return its generator-style conversion result.

        Args:
            kwargs: String, numeric, and boolean conversion options accepted by Seed-VC.

        Returns:
            SeedVCGenerator: A generator whose return value is the converted waveform.
        """
