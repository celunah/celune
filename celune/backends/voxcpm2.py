"""VoxCPM2 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
import contextlib
from pathlib import Path
from typing import Callable, Optional

from voxcpm import VoxCPM
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from . import get_version
from .base import CeluneBackend
from ..exceptions import BackendError


class VoxCPM2(CeluneBackend):
    """Celune VoxCPM2 backend."""

    name = "voxcpm2"
    voice_models = {
        "balanced": "openbmb/VoxCPM2",
        "calm": "openbmb/VoxCPM2",
        "bold": "openbmb/VoxCPM2",
        "upbeat": "openbmb/VoxCPM2",
    }
    reference_wavs = {
        "balanced": "refs/balanced.wav",
        "calm": "refs/calm.wav",
        "bold": "refs/bold.wav",
        "upbeat": "refs/upbeat.wav",
    }
    reference_transcripts = {
        "balanced": "My name is Celune, pronounced Celune. It is a pleasure to meet you.",
        "calm": "My name is... Celune... It is so... quiet.",
        "bold": "My name is Celune! Let's do this, we have to get it done!",
        "upbeat": "Hehehe... Hi, I'm Celune. Look, I have something to tell... might as well make it fun. Shall we?",
    }
    default_voice = "balanced"

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output():
        """Suppress unnecessary backend output."""
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    yield

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache."""
        base = HF_HUB_CACHE
        model_dir = os.path.join(base, f"models--{model.replace('/', '--')}")

        refs_main = os.path.join(model_dir, "refs", "main")
        snapshots_dir = os.path.join(model_dir, "snapshots")

        expected_files = [
            "config.json",
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
        """Ensure all known voice models are cached locally."""
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                log(f"Downloading {model_id}...", "info")
                os.environ["HF_HUB_OFFLINE"] = "0"
                snapshot_download(repo_id=model_id)
            else:
                log(f"{model_id} is already available.", "info")

    def load_model(
        self,
        model_id: str,
        log: Callable[[str, str], None],
        load_denoiser: bool = False,
    ) -> VoxCPM:
        """Load the given voice model."""
        available, path = self.model_is_available_locally(model_id)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            with self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(path, load_denoiser=load_denoiser)
            return self.model

        os.environ["HF_HUB_OFFLINE"] = "0"
        log("Downloading TTS model...", "info")
        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(model_id, load_denoiser=load_denoiser)
        return self.model

    def generate_stream(self, model: VoxCPM, **kwargs):
        """Generate Celune compatible audio chunks."""
        # convert/remove invalid params
        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        kwargs.pop("chunk_size", None)

        try:
            ref_wav = Path(os.getcwd()) / self.reference_wavs[voice]
            ref_text = self.reference_transcripts[voice]
        except KeyError as e:
            raise BackendError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        text = kwargs.pop("text")

        if instruct:
            # if this includes "music" or "singing", Celune may sing
            text = f"({instruct}) {text}"

        if hasattr(model, "generate_streaming"):
            with self._suppress_backend_output():
                for chunk in model.generate_streaming(
                    text,
                    prompt_wav_path=ref_wav,
                    prompt_text=ref_text,
                ):  # Celune wants (audio, sr, timing)
                    yield chunk, 48000, None
        else:
            version = get_version("voxcpm")
            raise BackendError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )
