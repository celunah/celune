"""VoxCPM2 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
import hashlib
import contextlib
from pathlib import Path
from typing import Callable, Optional, Generator

import torch
import numpy as np
import numpy.typing as npt
from voxcpm import VoxCPM
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from . import get_version
from .base import CeluneBackend
from ..exceptions import BackendError


class VoxCPM2(CeluneBackend):
    """Celune VoxCPM2 backend."""

    name: str = "voxcpm2"
    voice_models: dict[str, str] = {
        "balanced": "openbmb/VoxCPM2",
        "calm": "openbmb/VoxCPM2",
        "bold": "openbmb/VoxCPM2",
        "upbeat": "openbmb/VoxCPM2",
    }
    reference_wavs: dict[str, str] = {
        "balanced": "refs/balanced.wav",
        "calm": "refs/calm.wav",
        "bold": "refs/bold.wav",
        "upbeat": "refs/upbeat.wav",
    }
    voice_cfg: dict[str, float] = {
        "balanced": 2.4,
        "calm": 3.6,
        "bold": 2.4,
        "upbeat": 2.4,
    }
    default_voice: str = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self.log = log
        self.optimize_enabled = True
        self._validate_refs()

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output():
        """Suppress unnecessary backend output.

        Returns:
            Generator[None, None, None]: A context manager that silences stdout
            and stderr while backend code executes.
        """
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    yield

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Check if a model is already available in the Hugging Face cache.

        Args:
            model: The Hugging Face repository ID to inspect.

        Returns:
            tuple[bool, Optional[str]]: A flag indicating cache availability and
            the resolved snapshot path when present.
        """
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

    def preload_models(self) -> None:
        """Ensure all known voice models are cached locally.

        Returns:
            None: This method downloads any missing backend models.
        """
        for model_id in self.all_model_ids:
            available, _ = self.model_is_available_locally(model_id)
            if not available:
                self.log(f"Downloading {model_id}...", "info")
                snapshot_download(repo_id=model_id)
            else:
                self.log(f"{model_id} is already available.", "info")

    def _validate_refs(self) -> None:
        for name, ref in self.reference_wavs.items():
            full_path = Path(__file__).resolve().parents[1] / ref
            try:
                with open(full_path, "rb") as f:
                    checksum = hashlib.file_digest(f, "sha256").hexdigest()
            except FileNotFoundError as e:
                raise BackendError(f"reference audio for '{name}' not found") from e
            except PermissionError as e:
                raise BackendError(f"cannot access reference audio for '{name}'") from e

            checksum_path = f"{os.path.splitext(full_path)[0]}.sha256"
            if os.path.exists(checksum_path):
                with open(checksum_path, "r", encoding="utf-8") as f:
                    expected = f.read().strip()

                if checksum != expected:
                    self.log(
                        f"Checksum mismatch for '{name}', output may be affected.",
                        "warning",
                    )
                else:
                    self.log(
                        f"Checksum not found for '{name}', skipping checksum verification.",
                        "warning",
                    )

    def load_model(
        self,
        model_id: str,
        load_denoiser: bool = False,
        optimize: Optional[bool] = None,
    ) -> VoxCPM:
        """Load the given voice model.

        Args:
            model_id: The VoxCPM model repository ID to load.
            load_denoiser: Whether to enable the backend denoiser component.
            optimize: Whether to attempt optimization. When omitted, reuse the
                backend's current optimization setting.

        Returns:
            VoxCPM: The loaded VoxCPM model instance.
        """
        available, path = self.model_is_available_locally(model_id)
        if optimize is None:
            optimize = self.optimize_enabled
        else:
            self.optimize_enabled = optimize

        torch.cuda.manual_seed_all(3584181039)
        torch.backends.cudnn.deterministic = True

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            with self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=load_denoiser,
                    optimize=optimize,
                )
            return self.model

        self.log("Downloading TTS model...", "info")
        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=load_denoiser,
                optimize=optimize,
            )
        return self.model

    def generate_stream(
            self, model: VoxCPM, **kwargs
    ) -> Generator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded VoxCPM model instance.
            **kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterable[tuple]: An iterator of ``(audio, sample_rate, timing)``
            tuples suitable for Celune's playback pipeline.
        """
        # convert/remove invalid params
        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        kwargs.pop("chunk_size", None)

        try:
            ref_wav = Path(__file__).resolve().parents[1] / self.reference_wavs[voice]
            cfg = self.voice_cfg[voice]
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        text = kwargs.pop("text", None)
        if not text:
            raise ValueError("expected input, nothing found")

        if instruct:
            # if this includes "music" or "singing", Celune may sing
            text = f"({instruct}) {text}"

        if hasattr(model, "generate_streaming"):
            with self._suppress_backend_output():
                for chunk in model.generate_streaming(
                    text,
                    reference_wav_path=ref_wav,
                    inference_timesteps=6,
                    cfg_value=cfg,
                ):  # Celune wants (audio, sr, timing)
                    yield chunk, 48000, None
        else:
            version = get_version("voxcpm")
            raise NotImplementedError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )
