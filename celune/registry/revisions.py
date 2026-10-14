# SPDX-License-Identifier: Apache-2.0
"""Revision lookup helpers backed by Celune's model catalogs."""

from typing import Optional

from .asr import ASR_MODELS
from .llm import LLM_MODELS
from .other import OTHER_MODELS


def remote_code_model_revision(model_id: str) -> Optional[str]:
    """Return a pinned revision for a registered remote model.

    Args:
        model_id: Full Hugging Face model repository ID.

    Returns:
        Optional[str]: The registered commit revision, or ``None`` when the
            model is unknown or has no pinned revision.
    """
    for model in (*LLM_MODELS, *ASR_MODELS, *OTHER_MODELS):
        if model.model_id == model_id:
            return model.revision
    return None
