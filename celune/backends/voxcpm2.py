# SPDX-License-Identifier: MIT
"""VoxCPM2 backend implementation for Celune."""

from __future__ import annotations

import os
import contextlib
from typing import Callable, Optional
from collections.abc import Iterator

import torch
import numpy as np
import numpy.typing as npt
from voxcpm import VoxCPM

from . import get_version
from .base import CeluneBackend, cached_hf_snapshot_path
from ..cevoice import default_loader
from ..constants import BASE_SR


class VoxCPM2(CeluneBackend):
    """Celune VoxCPM2 backend."""

    name: str = "voxcpm2"
    uses_voice_bundles: bool = True
    chunk_rate: float = 6.25
    max_new_tokens: int = 2048
    supported_languages: tuple[str, ...] = (
        "ar",
        "my",
        "zh-cn",
        "da",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "he",
        "hi",
        "id",
        "it",
        "ja",
        "km",
        "ko",
        "lo",
        "ms",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "sw",
        "sv",
        "tl",
        "th",
        "tr",
        "vi",
    )

    voice_models: dict[str, str] = {
        "balanced": "openbmb/VoxCPM2",
        "calm": "openbmb/VoxCPM2",
        "bold": "openbmb/VoxCPM2",
        "upbeat": "openbmb/VoxCPM2",
    }

    # the sane default CFG is 2.4 for most voices,
    # `calm` needs a higher CFG of 3.0 to capture the nuances without distorting
    # however the max chunk length has to be limited to reduce the distortions over time
    voice_cfg: dict[str, float] = {
        "balanced": 2.4,
        "calm": 3.0,
        "bold": 2.4,
        "upbeat": 2.4,
    }
    default_voice: str = "balanced"

    def __init__(self, log: Callable[[str, str], None]) -> None:
        super().__init__(log=log)
        self.log = log
        self.optimize_enabled = False
        self._validate_refs()

    @staticmethod
    @contextlib.contextmanager
    def _suppress_backend_output() -> Iterator:
        """Suppress unnecessary backend output.

        Returns:
            Iterator: A context manager that silences stdout
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
        return cached_hf_snapshot_path(
            model,
            [
                "config.json",
                "model*.safetensors",
                "tokenizer_config.json",
            ],
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
    ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded VoxCPM model instance.
            **kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]: An iterator of
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
            loader = default_loader()
            if loader is not None:
                ref_wav = loader.materialize(voice, "wav")
            else:
                ref_wav = self._reference_wave_path(voice)
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

        # random seeding causes regenerations of Celune's output to be unique
        # while a custom seed makes the next output reproducible
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
                        # the longer you speak, the higher the drift risk over time
                        # 2048 tokens is also used by Qwen3-TTS
                        # consistent context lengths help to combat drift, and consume less VRAM
                        max_len=self.max_new_tokens,
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
                            yield pending_audio, BASE_SR, pending_timing

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
                        yield pending_audio, BASE_SR, pending_timing

                    audio = np.concatenate(batch)
                    timing = {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": len(batch),
                        "is_final": True,
                    }
                    yield audio, BASE_SR, timing
                elif pending_audio is not None and pending_timing is not None:
                    pending_timing["is_final"] = True
                    yield pending_audio, BASE_SR, pending_timing
            finally:
                if backend_stream is not None and hasattr(backend_stream, "close"):
                    with contextlib.suppress(Exception):
                        backend_stream.close()
        else:
            version = get_version("voxcpm")
            raise NotImplementedError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )
