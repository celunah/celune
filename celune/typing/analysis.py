"""Analysis-specific protocols and type aliases."""

from collections.abc import Mapping
from typing import Protocol, TypedDict, Union

import numpy as np
import numpy.typing as npt
import torch

type TextConfigValue = Union[str, dict[str, "TextConfigValue"]]
type TextConfig = dict[str, TextConfigValue]
type EmbeddingPayload = Union[
    torch.Tensor,
    npt.NDArray[np.float32],
    list[float],
    Mapping[str, "EmbeddingPayload"],
]


class EmbeddingOutput(Protocol):
    """Speaker embedding model output used by Celune analysis."""

    last_hidden_state: EmbeddingPayload


class EmbeddingProcessor(Protocol):
    """Processor callable returned by the embedding model package."""

    def __call__(
        self,
        y: npt.NDArray[np.float32],
        *,
        sampling_rate: int,
    ) -> Mapping[str, torch.Tensor]:
        """Prepare model inputs from a waveform."""
        raise NotImplementedError("protocol not defined")


class EmbeddingModel(Protocol):
    """Embedding model behavior used by Celune analysis."""

    def eval(self) -> None:
        """Switch the model into evaluation mode."""
        raise NotImplementedError("protocol not defined")

    def to(self, device: torch.device) -> torch.nn.Module:
        """Move the model to a device.

        Args:
            device: Destination device for the embedding model.

        """
        raise NotImplementedError("protocol not defined")

    def __call__(self, **inputs: torch.Tensor) -> EmbeddingOutput:
        """Run embedding inference."""
        raise NotImplementedError("protocol not defined")


class VoiceMatch(TypedDict):
    """Similarity score for one reference voice."""

    voice: str
    cosine: float
    percent: float
