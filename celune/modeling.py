"""Normalizer loading helpers for Celune."""

from typing import Callable, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .backends import CeluneBackend
from .constants import NORMALIZER_MODEL_ID


def load_normalizer_components(
    log: Callable[[str, str], None],
    backend: Union[CeluneBackend, type(CeluneBackend)],
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load CeluneNorm and return its tokenizer and model."""
    available, path = backend.model_is_available_locally(NORMALIZER_MODEL_ID)
    model_ref = path if available and path is not None else NORMALIZER_MODEL_ID

    if available:
        log("Normalizer is already available in cache", "info")

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    llm = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    return tokenizer, llm
