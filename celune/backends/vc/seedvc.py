# SPDX-License-Identifier: MIT
"""Seed-VC backend for Celune voice conversion."""

import gc
import importlib
import tempfile
import contextlib
import threading
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Optional, Callable, Protocol, Union, cast
from collections.abc import Generator

import numpy as np
import numpy.typing as npt
import soundfile as sf

from .base import CeluneVCBackend
from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from ...i18n import string
from ...paths import huggingface_hub_cache_dir

__all__ = ["CeluneSeedVCBackend"]

type SeedVCArgument = Union[str, int, float, bool]
type SeedVCGenerator = Generator[
    Optional[npt.NDArray[np.float32]], None, npt.NDArray[np.float32]
]


class _SeedVCWrapper(Protocol):
    """Protocol for the dynamically loaded Seed-VC wrapper."""

    def convert_voice(self, **kwargs: SeedVCArgument) -> SeedVCGenerator:
        """Run Seed-VC and return its generator-style conversion result.

        Args:
            kwargs: String, numeric, and boolean conversion options accepted by Seed-VC.

        Returns:
            SeedVCGenerator: A generator whose return value is the converted waveform.
        """


class _TemporaryWaveFile:
    """Temporary WAV file helper used for backends that require file paths."""

    def __init__(self, audio: npt.NDArray[np.float32], sample_rate: int) -> None:
        self.path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                self.path = Path(temp_file.name)
            sf.write(self.path, audio, sample_rate)
        except Exception:
            if self.path is not None:
                with contextlib.suppress(OSError):
                    self.path.unlink(missing_ok=True)
            raise

    def close(self) -> None:
        """Delete the underlying temporary WAV file."""
        if self.path is None:
            return
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)
        self.path = None

    def __enter__(self) -> Path:
        assert self.path is not None
        return self.path

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()


class CeluneSeedVCBackend(CeluneVCBackend):
    """Seed-VC backend using the forked package wrapper confirmed by Celune tests."""

    name = "seed-vc"

    def __init__(
        self,
        log: Callable[[str, str], None],
        diffusion_steps: int = 30,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.5,
        f0_condition: bool = False,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
    ) -> None:
        super().__init__(log=log)
        self.diffusion_steps = diffusion_steps
        self.length_adjust = length_adjust
        self.inference_cfg_rate = inference_cfg_rate
        self.f0_condition = f0_condition
        self.auto_f0_adjust = auto_f0_adjust
        self.pitch_shift = pitch_shift
        self._wrapper: Optional[_SeedVCWrapper] = None
        self._wrapper_lock = threading.Lock()

    @staticmethod
    def _seedvc_huggingface_cache_dir(create: bool = False) -> Path:
        """Return the shared Hugging Face cache directory used for Seed-VC assets."""
        return huggingface_hub_cache_dir(create=create)

    @classmethod
    def _configure_seedvc_downloads(
        cls,
        hf_utils_module: ModuleType,
        wrapper_module: ModuleType,
    ) -> None:
        """Redirect Seed-VC's hardcoded checkpoint downloads into Celune's cache."""
        cache_dir = cls._seedvc_huggingface_cache_dir(create=True)
        hf_hub_download = getattr(hf_utils_module, "hf_hub_download")

        def load_custom_model_from_hf(
            repo_id: str,
            model_filename: str = "pytorch_model.bin",
            config_filename: Optional[str] = None,
        ):
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_filename,
                cache_dir=str(cache_dir),
            )
            if config_filename is None:
                return model_path

            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=config_filename,
                cache_dir=str(cache_dir),
            )
            return model_path, config_path

        setattr(hf_utils_module, "load_custom_model_from_hf", load_custom_model_from_hf)
        setattr(wrapper_module, "load_custom_model_from_hf", load_custom_model_from_hf)

    @classmethod
    def _load_wrapper_type(cls) -> type[_SeedVCWrapper]:
        """Import and return the Seed-VC wrapper class."""
        try:
            hf_utils_module = importlib.import_module("seed_vc.hf_utils")
            wrapper_module = importlib.import_module("seed_vc.seed_vc_wrapper")
        except ImportError as e:
            raise ImportError(
                string("seedvc.package_required", package="seed_vc")
            ) from e

        cls._configure_seedvc_downloads(hf_utils_module, wrapper_module)
        return cast(type[_SeedVCWrapper], getattr(wrapper_module, "SeedVCWrapper"))

    def _get_wrapper(self) -> _SeedVCWrapper:
        """Return a cached Seed-VC wrapper instance."""
        with self._wrapper_lock:
            if self._wrapper is None:
                wrapper_type = self._load_wrapper_type()
                self.log(string("seedvc.loading_models"), "info")
                self._wrapper = wrapper_type()
            return self._wrapper

    @staticmethod
    def _mix_to_mono(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return a mono float32 waveform for Seed-VC inference."""
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim == 1:
            return np.clip(mono, -1.0, 1.0)
        if mono.ndim == 2:
            return np.clip(np.mean(mono, axis=1, dtype=np.float32), -1.0, 1.0)
        raise ValueError("Seed-VC expects one-dimensional or two-dimensional audio")

    @staticmethod
    def _drain_generator_return_value(
        generator: SeedVCGenerator,
    ) -> npt.NDArray[np.float32]:
        """Run a generator to completion and return its final value."""
        try:
            while True:
                next(generator)
        except StopIteration as exc:
            return exc.value

    @property
    def output_sample_rate(self) -> int:
        """Return the sample rate produced by the backend's default Seed-VC mode.

        Returns:
            int: ``44100`` when f0 conditioning is enabled, otherwise ``22050``.
        """
        return 44100 if self.f0_condition else 22050

    @staticmethod
    def _resolve_f0_condition(request: VoiceConversionRequest, default: bool) -> bool:
        """Return the effective f0 conditioning mode for one conversion request."""
        if isinstance(request.f0_condition, bool):
            return request.f0_condition
        return default

    def preload_models(self) -> None:
        """Load Seed-VC eagerly so VC mode is ready before the first request."""
        self._get_wrapper()

    def unload_model(self) -> None:
        """Release the cached Seed-VC wrapper and best-effort GPU memory."""
        with self._wrapper_lock:
            self._wrapper = None

        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert source audio into the currently selected reference voice.

        Args:
            request: Voice conversion input containing source audio and a target reference WAV path.

        Returns:
            AudioOutput: Converted audio packaged for the Celune pipeline.

        Raises:
            ValueError: The request does not include a target reference WAV.
        """
        if not request.target_references:
            raise ValueError(string("seedvc.target_reference_required"))

        target_reference = request.target_references[0]
        wrapper = self._get_wrapper()
        source_audio = self._mix_to_mono(request.source_audio)
        f0_condition = self._resolve_f0_condition(request, self.f0_condition)

        with _TemporaryWaveFile(source_audio, request.sample_rate) as source_path:
            result = self._drain_generator_return_value(
                wrapper.convert_voice(
                    source=str(source_path),
                    target=str(target_reference),
                    diffusion_steps=self.diffusion_steps,
                    length_adjust=self.length_adjust,
                    inference_cfg_rate=self.inference_cfg_rate,
                    f0_condition=f0_condition,
                    auto_f0_adjust=self.auto_f0_adjust,
                    pitch_shift=request.pitch_shift
                    if request.pitch_shift is not None
                    else 0,
                    stream_output=False,
                )
            )

        audio = np.asarray(result, dtype=np.float32).reshape(-1)
        audio = np.clip(audio, -1.0, 1.0)
        return AudioOutput(
            audio=audio,
            sample_rate=44100 if f0_condition else 22050,
            label=request.label,
        )
