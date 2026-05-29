# SPDX-License-Identifier: MIT
"""Normalizer loading helpers for Celune."""

from collections.abc import Mapping
from typing import Callable, Union, Optional

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .backends import CeluneBackend
from .vram import resolve_vram_preset
from .constants import JSONSerializable, NORMALIZER_MODEL_ID

NORMALIZER_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<NORM>")
NORMALIZER_DEVICE = "cpu"


def normalizer_device(
    config: Optional[Mapping[str, JSONSerializable]] = None,
) -> str:
    """Return the runtime device used for CeluneNorm loading.

    Args:
        config: Celune's current configuration.

    Returns:
        str: The selected normalizer device based on current VRAM preset.
    """
    if config is None:
        return NORMALIZER_DEVICE
    return resolve_vram_preset(config).normalizer_device


def load_normalizer_components(
    log: Callable[[str, str], None],
    backend: Union[CeluneBackend, type[CeluneBackend]],
    config: Optional[Mapping[str, JSONSerializable]] = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load CeluneNorm and return its tokenizer and model.

    Args:
        log: Logging callback used to report cache and loading progress.
        backend: Backend type or instance used to resolve model cache helpers.
        config: Celune configuration used to resolve the target device.

    Returns:
        tuple[PreTrainedTokenizerBase, PreTrainedModel]: The loaded tokenizer and causal language model.
    """
    available, path = backend.model_is_available_locally(NORMALIZER_MODEL_ID)
    model_ref = path if available and path is not None else NORMALIZER_MODEL_ID

    if available:
        log("Normalizer is already available in cache", "info")

    tokenizer = AutoTokenizer.from_pretrained(model_ref, extra_special_tokens={})
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(NORMALIZER_SPECIAL_TOKENS)},
        replace_additional_special_tokens=False,
    )

    device = normalizer_device(config)
    supported_dispatch = {"auto", "balanced", "balanced_low_0", "sequential"}
    device_map = device if device in supported_dispatch else {"": device}
    llm = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    return tokenizer, llm
