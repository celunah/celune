"""VoxCPM2 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
import random
import secrets
import hashlib
import contextlib
import time
from pathlib import Path
from typing import Any, Callable, Optional, Generator

import torch
import numpy as np
import numpy.typing as npt
from voxcpm import VoxCPM
from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig, get_dtype
from voxcpm.modules.audiovae import AudioVAEV2
from transformers import LlamaTokenizerFast
from safetensors.torch import load_file as load_safetensors_file
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from . import get_version
from .base import BackendTiming, CeluneBackend
from ..modeling import (
    VOXCPM2_SAFE_TRANSFORMER_INT8_PROFILE,
    inspect_hybrid_tts_checkpoint,
    load_hybrid_tts_checkpoint,
    qwen3_int8_state_dict,
    replace_int8_linear_modules,
)
from ..exceptions import BackendError


class VoxCPM2(CeluneBackend):
    """Celune VoxCPM2 backend."""

    name: str = "voxcpm2"
    chunk_rate: float = 6.25

    # if int8 load was requested this will be overriden with the hybrid models
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

    def __init__(
        self,
        log: Callable[[str, str], None],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the VoxCPM2 backend.

        Args:
            log: Logger callback used by the backend.
            config: Loaded Celune configuration dictionary.

        Returns:
            None: This constructor prepares backend state and validates
                reference audio.
        """
        super().__init__(log=log, config=config)

        if self.config.get("int8"):
            self.voice_models = {
                voice: "lunahr/VoxCPM2-hybrid-int8" for voice in self.voice_models
            }
            self.model_name = self.voice_models[self.default_voice]

        self.log = log
        self.optimize_enabled = False
        self.random_seed = True
        self._validate_refs()

    def _apply_seed(self) -> None:
        """Seed all generation RNGs for the next backend operation.

        Returns:
            None: This method seeds Celune's RNG.
        """
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
            if Path(model_id).expanduser().exists():
                self.log(f"{model_id} was found locally.", "info")
                continue

            available, _ = self.model_is_available_locally(model_id)
            if not available:
                self.log(f"Downloading {model_id}...", "info")
                snapshot_download(repo_id=model_id)
            else:
                self.log(f"{model_id} is available in cache.", "info")

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
        local_path = Path(model_id).expanduser()
        if local_path.exists():
            self.log("Loading TTS model from local path...", "info")
            model_path = str(local_path.resolve())
            if self._checkpoint_needs_hybrid_load(model_path):
                self.model = self._load_hybrid_int8_model(
                    model_path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
                return self.model

            with self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(
                    model_path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
            return self.model

        available, path = self.model_is_available_locally(model_id)

        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            if self._checkpoint_needs_hybrid_load(path):
                self.model = self._load_hybrid_int8_model(
                    path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
                return self.model

            with self._suppress_backend_output():
                self.model = VoxCPM.from_pretrained(
                    path,
                    load_denoiser=kwargs.get("load_denoiser", False),
                    optimize=kwargs.get("optimize", False),
                )
            return self.model

        self.log("Downloading TTS model...", "info")
        path = snapshot_download(repo_id=model_id)
        if self._checkpoint_needs_hybrid_load(path):
            self.model = self._load_hybrid_int8_model(
                path,
                load_denoiser=kwargs.get("load_denoiser", False),
                optimize=kwargs.get("optimize", False),
            )
            return self.model

        with self._suppress_backend_output():
            self.model = VoxCPM.from_pretrained(
                path,
                load_denoiser=kwargs.get("load_denoiser", False),
                optimize=kwargs.get("optimize", False),
            )
        return self.model

    @staticmethod
    def _checkpoint_needs_hybrid_load(model_path: str) -> bool:
        """Return whether a VoxCPM2 checkpoint contains actual INT8 tensors.

        Returns:
            bool: The result of the hybrid model inspection.
        """
        return inspect_hybrid_tts_checkpoint(model_path).has_int8_tensors

    def _load_hybrid_int8_model(
        self,
        model_path: str,
        load_denoiser: bool,
        optimize: bool,
    ) -> VoxCPM:
        """Load a VoxCPM2 hybrid INT8/BF16 checkpoint through Celune's path.

        Args:
            model_path: The model path or Hugging Face ID to load.
            load_denoiser: Whether to load the denoiser.
            optimize: Whether to try optimizing the model.

        Returns:
            VoxCPM: The underlying model object.

        Raises:
            BackendError: The hybrid model checkpoint is invalid or Celune could not load it.
        """
        load_start = time.perf_counter()
        path = Path(model_path)
        with open(path / "config.json", encoding="utf-8") as f:
            config = VoxCPMConfig.model_validate_json(f.read())

        tokenizer = LlamaTokenizerFast.from_pretrained(str(path))
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = (
            AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        )

        audiovae_safetensors_path = path / "audiovae.safetensors"
        audiovae_pth_path = path / "audiovae.pth"
        if audiovae_safetensors_path.exists():
            vae_state_dict = load_safetensors_file(
                str(audiovae_safetensors_path), device="cpu"
            )
        elif audiovae_pth_path.exists():
            checkpoint = torch.load(
                audiovae_pth_path,
                map_location="cpu",
                weights_only=True,
            )
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise BackendError(
                "VoxCPM2 AudioVAE checkpoint not found for hybrid INT8 load"
            )

        with self._suppress_backend_output():
            model = VoxCPM2Model(config, tokenizer, audio_vae)
        model = model.to(get_dtype(model.config.dtype))
        model.audio_vae = model.audio_vae.to(torch.float32)

        loaded = load_hybrid_tts_checkpoint(path, keep_state_dict=True)
        if loaded.state_dict is None:
            raise BackendError(
                "VoxCPM2 INT8 checkpoint scan did not retain the state dict"
            )

        int8_linear_count = replace_int8_linear_modules(
            model,
            loaded.index,
            loaded.state_dict,
            VOXCPM2_SAFE_TRANSFORMER_INT8_PROFILE,
        )
        state_dict = qwen3_int8_state_dict(loaded.state_dict)
        for key, value in vae_state_dict.items():
            state_dict[f"audio_vae.{key}"] = value

        state_dict = {
            name: tensor.to(model.device) if tensor.dtype is torch.int8 else tensor
            for name, tensor in state_dict.items()
        }
        incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
        unexpected = [key for key in incompatible.unexpected_keys if "lora_" not in key]
        if unexpected:
            preview = ", ".join(unexpected[:5])
            raise BackendError(
                f"VoxCPM2 INT8 checkpoint had unexpected keys: {preview}"
            )
        del state_dict
        loaded.state_dict = None

        cache_start = time.perf_counter()
        cache_count = 0
        model = model.to(model.device).eval().optimize(disable=not optimize)
        dequant_cache_ms = (time.perf_counter() - cache_start) * 1000

        pipeline = VoxCPM.__new__(VoxCPM)
        pipeline.tts_model = model
        pipeline.text_normalizer = None
        pipeline.denoiser = None
        if load_denoiser:
            from voxcpm.zipenhancer import ZipEnhancer

            pipeline.denoiser = ZipEnhancer(
                "iic/speech_zipenhancer_ans_multiloss_16k_base"
            )

        setattr(pipeline, "_celune_int8_linear_count", int8_linear_count)
        setattr(pipeline, "_celune_int8_runtime_cache_count", cache_count)
        setattr(
            pipeline, "_celune_int8_load_ms", (time.perf_counter() - load_start) * 1000
        )
        setattr(pipeline, "_celune_int8_dequant_cache_ms", dequant_cache_ms)
        return pipeline

    def generate_stream(
        self, model: VoxCPM, **kwargs
    ) -> Generator[tuple[npt.NDArray[np.float32], int, BackendTiming]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded VoxCPM model instance.
            **kwargs: Streaming generation arguments passed to the backend.

        Returns:
            Generator[tuple[npt.NDArray[np.float32], int, BackendTiming]]: An iterator of
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

        # random seeding causes regenerations of Celune's output to be unique,
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
                    )

                batch = []
                batch_start = time.perf_counter()
                chunk_index = 0
                while True:
                    with self._suppress_backend_output():
                        try:
                            chunk = next(backend_stream)
                        except StopIteration:
                            break

                    batch.append(chunk)
                    if len(batch) >= chunks_per_batch:
                        yield_ms = (time.perf_counter() - batch_start) * 1000
                        audio = np.concatenate(batch)
                        timing: BackendTiming = {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "chunk_steps": len(batch),
                            "total_wall_ms": yield_ms,
                            "is_final": False,
                        }
                        yield audio, 48000, timing
                        batch.clear()
                        batch_start = time.perf_counter()
                        chunk_index += 1

                if batch:  # push remaining
                    yield_ms = (time.perf_counter() - batch_start) * 1000
                    audio = np.concatenate(batch)
                    timing = {
                        "backend": self.name,
                        "chunk_index": chunk_index,
                        "chunk_steps": len(batch),
                        "total_wall_ms": yield_ms,
                        "is_final": True,
                    }
                    yield audio, 48000, timing
            finally:
                if backend_stream is not None and hasattr(backend_stream, "close"):
                    with contextlib.suppress(Exception):
                        backend_stream.close()
        else:
            version = get_version("voxcpm")
            raise NotImplementedError(
                f"streaming support not available (requires voxcpm>=1.5.0, installed: {version})"
            )
