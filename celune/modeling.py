"""Model and normalizer loading helpers for Celune."""

import os
import glob
from typing import Callable, Optional

import torch
from faster_qwen3_tts import FasterQwen3TTS
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .constants import NORMALIZER_MODEL_ID


def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
    """Check if a model is already available in the Hugging Face cache."""
    base = HF_HUB_CACHE
    model_dir = os.path.join(base, f"models--{model.replace('/', '--')}")

    refs_main = os.path.join(model_dir, "refs", "main")
    snapshots_dir = os.path.join(model_dir, "snapshots")

    expected_files = [
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "tokenizer_config.json",
    ]

    if not os.path.exists(refs_main):
        return False, None

    with open(refs_main, encoding="utf-8") as f:
        commit = f.read().strip()

    snapshot_path = os.path.join(snapshots_dir, commit)

    if not os.path.isdir(snapshot_path):
        return False, None

    if all(
        glob.glob(os.path.join(snapshot_path, pattern)) for pattern in expected_files
    ):
        return True, snapshot_path

    return False, None


def preload_models(model_ids: list[str], log: Callable[[str, str], None]) -> None:
    """Ensure all requested TTS models are cached locally."""
    for model_id in model_ids:
        available, _ = model_is_available_locally(model_id)
        if not available:
            log(f"Downloading {model_id}...", "info")
            os.environ["HF_HUB_OFFLINE"] = "0"
            snapshot_download(repo_id=model_id)
        else:
            log(f"{model_id} is already available.", "info")


def load_tts_model(model_id: str, log: Callable[[str, str], None]) -> FasterQwen3TTS:
    """Load a TTS model from cache when available, otherwise download it."""
    available, path = model_is_available_locally(model_id)

    if available and path is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        model = FasterQwen3TTS.from_pretrained(path)
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        log("Downloading TTS model...", "info")
        model = FasterQwen3TTS.from_pretrained(model_id)

    return model


def load_normalizer_components(
    log: Callable[[str, str], None],
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load CeluneNorm and return its tokenizer and model."""
    available, path = model_is_available_locally(NORMALIZER_MODEL_ID)
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
