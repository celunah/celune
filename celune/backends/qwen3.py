"""Qwen3 backend implementation for Celune."""

from __future__ import annotations

import glob
import os
from typing import Callable, Optional

from faster_qwen3_tts import FasterQwen3TTS
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from .base import CeluneBackend


class Qwen3(CeluneBackend):
    """Celune Qwen3-TTS backend."""

    name = "qwen3"
    voice_models = {
        "balanced": "lunahr/Celune-1.7B-Neutral",
        "calm": "lunahr/Celune-1.7B-Calm",
        "bold": "lunahr/Celune-1.7B-Energetic",
        "upbeat": "lunahr/Celune-1.7B-Upbeat",
    }
    default_voice = "balanced"

    @staticmethod
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
            glob.glob(os.path.join(snapshot_path, pattern))
            for pattern in expected_files
        ):
            return True, snapshot_path

        return False, None

    def preload_models(self, log: Callable[[str, str], None]) -> None:
        """Ensure all known Qwen3 voice models are cached locally."""
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                log(f"Downloading {model_id}...", "info")
                os.environ["HF_HUB_OFFLINE"] = "0"
                snapshot_download(repo_id=model_id)
            else:
                log(f"{model_id} is already available.", "info")

    def load_model(
        self, model_id: str, log: Callable[[str, str], None]
    ) -> FasterQwen3TTS:
        """Load a Qwen3 TTS model from cache when available."""
        available, path = self.model_is_available_locally(model_id)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = FasterQwen3TTS.from_pretrained(path)
            return self.model

        os.environ["HF_HUB_OFFLINE"] = "0"
        log("Downloading TTS model...", "info")
        self.model = FasterQwen3TTS.from_pretrained(model_id)
        return self.model

    def generate_stream(self, model: FasterQwen3TTS, **kwargs):
        """Delegate unified streaming generation to FasterQwen3TTS."""
        kwargs.pop("voice", None)  # Celune has native voices in this backend
        yield from model.generate_custom_voice_streaming(speaker="celune", **kwargs)
