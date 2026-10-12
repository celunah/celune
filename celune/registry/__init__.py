# SPDX-License-Identifier: Apache-2.0
"""Central catalogs for model metadata and pinned runtime selections."""

from typing import Optional

from .asr import ASR_MODELS
from .llm import LLM_MODELS
from .tts import TTS_MODELS
from .other import OTHER_MODELS
from .types import ModelCategory, ModelMetadata

__all__ = [
    "ALL_MODELS",
    "ASR_MODELS",
    "LLM_MODELS",
    "MODEL_BY_ID",
    "OTHER_MODELS",
    "TTS_MODELS",
    "ModelCategory",
    "ModelMetadata",
    "get_model_metadata",
]

ALL_MODELS: tuple[ModelMetadata, ...] = (
    *TTS_MODELS,
    *LLM_MODELS,
    *ASR_MODELS,
    *OTHER_MODELS,
)

MODEL_BY_ID = {metadata.model_id: metadata for metadata in ALL_MODELS}
if len(MODEL_BY_ID) != len(ALL_MODELS):
    raise RuntimeError("model registry contains duplicate model IDs")


def get_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    """Return the catalog entry for one exact Hugging Face repository ID."""
    return MODEL_BY_ID.get(model_id)
