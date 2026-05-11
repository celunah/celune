# SPDX-License-Identifier: MIT
"""VoxCPM2 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
import random
import secrets
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
    chunk_rate: float = 6.25
    voice_models: dict[str, str] = {
        "balanced": "openbmb/VoxCPM2",
        "calm": "openbmb/VoxCPM2",
        "bold": "openbmb/VoxCPM2",
        "upbeat": "openbmb/VoxCPM2",
    }
    reference_waves: dict[str, str] = {
        "balanced": "refs/balanced.wav",
        "calm": "refs/calm.wav",
        "bold": "refs/bold.wav",
        "upbeat": "refs/upbeat.wav",
    }

    # the sane default CFG is 2.4 for most voices,
    # `calm` needs a higher CFG of 3.0 to capture the nuances without distorting
    voice_cfg: dict[str, float] = {
        "balanced": 2.4,
        "calm": 3.0,
        "bold": 2.4,
        "upbeat": 2.4,
    }
    default_voice: str = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        """Initialize the VoxCPM2 backend.

        Args:
            log: Logger callback used by the backend.

        Returns:
            None: This constructor prepares backend state and validates
                reference audio.
        """
        super().__init__(log=log)
        self.log = log
        self.optimize_enabled = False
        self.random_seed = True
        self._validate_refs()

    def _apply_seed(self) -> None:
        """Seed all generation RNGs for the next backend operation."""
        if self.random_seed:
            self.current_seed = secrets.randbits(32)

        if self.current_seed is None:
            return

        random.seed(self.current_seed)
        np.random.seed(self.current_seed)
        torch.cuda.manual_seed_all(self.current_seed)
        torch.manual_seed(self.current_seed)

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Generator[None, None, None]:
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
        """Validate bundled reference audio files.

        Returns:
            None: This method checks that reference files are accessible and logs
                checksum status when checksums exist.
        """
        for name, ref in self.reference_waves.items():
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

    def load_model(self, model_id: str, **kwargs) -> VoxCPM:
        """Load the given voice model.

        Args:
            model_id: The VoxCPM model repository ID to load.
            **kwargs:
                - load_denoiser: Whether to load the denoiser model.
                - optimize: Whether to try to optimize the model.

        Returns:
            VoxCPM: The loaded VoxCPM model instance.
        """
        available, path = self.model_is_available_locally(model_id)

        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            with self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
            return self.model

        self.log("Downloading TTS model...", "info")
        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(
                model_id,
                load_denoiser=kwargs.get("load_denoiser", False),
                optimize=kwargs.get("optimize", False),
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
            Generator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of
                ``(audio, sample_rate, timing)`` tuples suitable for Celune's playback pipeline.

        Raises:
            ValueError: The requested voice is unknown or input text is empty.
        """
        # convert/remove invalid params
        voice = kwargs.pop("voice", self.default_voice)
        instruct = kwargs.pop("instruct", None)
        kwargs.pop("language", None)
        chunk_size = kwargs.pop("chunk_size", 1)

        try:
            ref_wav = Path(__file__).resolve().parents[1] / self.reference_waves[voice]
            cfg = self.voice_cfg[voice]
        except KeyError as e:
            raise ValueError(
                f"unknown voice '{voice}' for backend '{self.name}'"
            ) from e

        text = kwargs.pop("text", None)
        if not text:
            # saying nothing makes no sense
            raise ValueError("expected text to say")

        if instruct:
            # if this includes "music" or "singing", Celune may sing
            # these instructions can also be injected manually
            text = f"({instruct}) {text}"

        # Random seeding causes regenerations of Celune's output to be unique,
        # while a custom seed makes the next output reproducible.
        self._apply_seed()

        chunks_per_batch = max(1, round(chunk_size / (1 / self.chunk_rate)))
        if hasattr(model, "generate_streaming"):
            backend_stream = None
            try:
                with self._suppress_backend_output():
                    backend_stream = model.generate_streaming(
                        text,
                        reference_wav_path=ref_wav,
                        inference_timesteps=6,
                        cfg_value=cfg,
                    )

                batch = []
                chunk_index = 0
                pending_audio: Optional[npt.NDArray[np.float32]] = None
                pending_timing: Optional[dict] = None
                while True:
                    with self._suppress_backend_output():
                        try:
                            chunk = next(backend_stream)
                        except StopIteration:
                            break

                    batch.append(chunk)
                    if len(batch) >= chunks_per_batch:
                        if pending_audio is not None and pending_timing is not None:
                            yield pending_audio, 48000, pending_timing

                        audio = np.concatenate(batch)
                        pending_timing = {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": len(batch),
                            "is_final": False,
                        }
                        pending_audio = audio
                        batch.clear()
                        chunk_index += 1

                if batch:  # push remaining
                    if pending_audio is not None and pending_timing is not None:
                        yield pending_audio, 48000, pending_timing

                    audio = np.concatenate(batch)
                    timing = {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": len(batch),
                        "is_final": True,
                    }
                    yield audio, 48000, timing
                elif pending_audio is not None and pending_timing is not None:
                    pending_timing["is_final"] = True
                    yield pending_audio, 48000, pending_timing
            finally:
                if backend_stream is not None and hasattr(backend_stream, "close"):
                    with contextlib.suppress(Exception):
                        backend_stream.close()
        else:
            version = get_version("voxcpm")
            raise NotImplementedError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )
