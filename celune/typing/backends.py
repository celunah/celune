# SPDX-License-Identifier: MIT
"""Backend-facing protocols and type aliases."""

from __future__ import annotations

from collections.abc import Mapping, Iterator
from typing import TYPE_CHECKING, TypeVar, Optional, Protocol, TypedDict

import torch
import numpy as np

from .aliases import AudioChunk
from .common import JSON, JSONSerializable

if TYPE_CHECKING:
    from .aliases import RuntimeValue, SeedVCArgument, SeedVCGenerator
    from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest


class _SeedVCRealtimeArguments(Protocol):
    """Arguments required by Seed-VC's native live model loader."""

    checkpoint_path: Optional[str]
    config_path: Optional[str]
    fp16: bool


class BackendModel(Protocol):
    """Opaque backend model protocol for backend-independent storage."""


ModelT = TypeVar("ModelT", bound=BackendModel)
type MiniPromptState = dict[str, dict[str, torch.Tensor]]
type BackendArgumentValue = JSONSerializable
type BackendArguments = dict[str, BackendArgumentValue]
type BackendGeneration = tuple[AudioChunk, int, Optional[JSON]]


class BackendDescription(TypedDict):
    """Static metadata exchanged by an isolated backend worker."""

    name: str
    chunk_rate: float
    supported_languages: tuple[str, ...]
    voice_models: Optional[Mapping[str, str]]
    default_voice: Optional[str]
    model_name: Optional[str]
    voices: list[str]
    clone_model_id: Optional[str]
    uses_voice_bundles: bool
    max_new_tokens: int
    is_fake: bool


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


class _StreamingSpeechModel(Protocol):  # noqa: PYI046
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


class _GPTSoVITSConfig(Protocol):  # noqa: PYI046
    """Constructor surface of GPT-SoVITS' ``TTS_Config`` class."""


class _SeedVCWrapper(Protocol):  # noqa: PYI046
    """Protocol for the dynamically loaded Seed-VC wrapper."""

    def convert_voice(self, **kwargs: SeedVCArgument) -> SeedVCGenerator:
        """Run Seed-VC and return its generator-style conversion result.

        Args:
            kwargs: String, numeric, and boolean conversion options accepted by Seed-VC.

        Returns:
            SeedVCGenerator: A generator whose return value is the converted waveform.
        """


class _BackendRuntime(Protocol):  # noqa: PYI046
    """Runtime method surface used by the generic worker loop."""

    model: Optional[BackendModel]

    def model_is_available_locally(
        self,
        **kwargs: BackendArgumentValue,
    ) -> tuple[bool, Optional[str]]:
        """Return whether a model is available."""

    def preload_models(self) -> None:
        """Preload backend models."""

    def load_model(self, **kwargs: BackendArgumentValue) -> BackendModel:
        """Load one backend model."""

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Unload backend models.

        Args:
            release_cuda_cache: Whether to synchronize CUDA and release cached accelerator blocks.
        """

    def generate_stream(
        self,
        model: BackendModel,
        **kwargs: BackendArgumentValue,
    ) -> Iterator[BackendGeneration]:
        """Generate streamed backend audio."""

    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert one voice-conversion request."""


class _LoguruLogger(Protocol):  # noqa: PYI046
    """Subset of Loguru's logger interface used by the DotsTTS backend."""

    def disable(self, name: str) -> None:
        """Disable one logger namespace."""

    def enable(self, name: str) -> None:
        """Enable one logger namespace."""


class _SeedVCRealtimeModule(Protocol):  # noqa: PYI046
    """Subset of Seed-VC's real-time module used by its backend."""

    device: torch.device
    fp16: bool

    def load_models(self, args: _SeedVCRealtimeArguments) -> tuple[object, ...]:
        """Load Seed-VC's native real-time model set."""

    def custom_infer(
        self,
        model_set: tuple[object, ...],
        reference_wav: np.ndarray,
        new_reference_wav_name: str,
        input_wav_res: torch.Tensor,
        block_frame_16k: int,
        skip_head: int,
        skip_tail: int,
        return_length: int,
        diffusion_steps: int,
        inference_cfg_rate: float,
        max_prompt_length: float,
        cd_difference: float = 2.0,
    ) -> torch.Tensor:
        """Convert one rolling input buffer through Seed-VC's live path."""
