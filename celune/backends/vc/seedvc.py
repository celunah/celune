# SPDX-License-Identifier: MIT
"""Seed-VC backend for Celune voice conversion."""

import contextlib
import gc
import importlib
import importlib.metadata
import importlib.util
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace, TracebackType
from typing import Optional, cast

import numpy as np
import soundfile as sf
import torch
import torchaudio

from ...dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from ...i18n import string
from ...paths import huggingface_hub_cache_dir
from ...typing.aliases import AudioChunk, SeedVCGenerator
from ...typing.backends import _SeedVCRealtimeModule, _SeedVCWrapper
from .base import CeluneVCBackend

__all__ = ["CeluneSeedVCBackend"]

_LIVE_BLOCK_SECONDS = 0.18
_LIVE_CROSSFADE_SECONDS = 0.04
_LIVE_EXTRA_CONTEXT_SECONDS = 2.5
_LIVE_EXTRA_CONTEXT_RIGHT_SECONDS = 0.02
_LIVE_CONTEXT_DIFFERENCE_SECONDS = 2.0
_LIVE_DIFFUSION_STEPS = 10
_LIVE_INFERENCE_CFG_RATE = 0.7
_LIVE_MAX_PROMPT_SECONDS = 3.0


class _TemporaryWaveFile:
    """Temporary WAV file helper used for backends that require file paths."""

    def __init__(self, audio: AudioChunk, sample_rate: int) -> None:
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
        self._live_module: Optional[_SeedVCRealtimeModule] = None
        self._live_model_set: Optional[tuple[object, ...]] = None
        self._live_session_key: Optional[tuple[Path, int]] = None
        self._live_reference_path: Optional[Path] = None
        self._live_reference_wav = np.zeros(0, dtype=np.float32)
        self._live_model_sample_rate = 0
        self._live_block_frame = 0
        self._live_block_frame_16k = 0
        self._live_crossfade_frame = 0
        self._live_sola_buffer_frame = 0
        self._live_sola_search_frame = 0
        self._live_extra_frame = 0
        self._live_extra_frame_right = 0
        self._live_skip_head = 0
        self._live_skip_tail = 0
        self._live_return_length = 0
        self._live_input_pending = np.zeros(0, dtype=np.float32)
        self._live_input_wav: Optional[torch.Tensor] = None
        self._live_sola_buffer: Optional[torch.Tensor] = None
        self._live_fade_in_window: Optional[torch.Tensor] = None
        self._live_fade_out_window: Optional[torch.Tensor] = None
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
        hf_hub_download = hf_utils_module.hf_hub_download

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

    def _get_live_runtime(self) -> tuple[_SeedVCRealtimeModule, tuple[object, ...]]:
        """Return the cached native Seed-VC real-time module and model set."""
        with self._wrapper_lock:
            if self._live_module is None or self._live_model_set is None:
                realtime_module = self._load_live_module()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                setattr(realtime_module, "device", device)
                setattr(realtime_module, "fp16", device.type == "cuda")
                args = SimpleNamespace(
                    checkpoint_path=None,
                    config_path=None,
                    fp16=device.type == "cuda",
                )
                self.log(string("seedvc.loading_live_models"), "info")
                model_set = realtime_module.load_models(args)
                self._live_module = realtime_module
                self._live_model_set = model_set
            return self._live_module, self._live_model_set

    @classmethod
    def _load_live_module(cls) -> _SeedVCRealtimeModule:
        """Load Seed-VC's native real-time module without launching its GUI."""
        script_path = cls._find_live_script()
        if script_path is None:
            for module_name in (
                "seed_vc.real_time",
                "seed_vc.realtime",
                "seed_vc.real_time_gui",
            ):
                try:
                    module = importlib.util.find_spec(module_name)
                except (ImportError, ModuleNotFoundError, ValueError):
                    module = None
                if module is not None:
                    imported = __import__(module_name, fromlist=["*"])
                    return cast(_SeedVCRealtimeModule, imported)
            raise ImportError(string("seedvc.live_path_required"))

        module_name = "celune_seedvc_realtime"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(string("seedvc.live_path_required"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        source_root = str(script_path.parent)
        sys.path.insert(0, source_root)
        try:
            spec.loader.exec_module(module)
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(source_root)

        try:
            hf_utils = __import__("hf_utils", fromlist=["*"])
            cls._configure_seedvc_downloads(hf_utils, hf_utils)
            setattr(
                module, "load_custom_model_from_hf", hf_utils.load_custom_model_from_hf
            )
        except ImportError:
            pass
        return cast(_SeedVCRealtimeModule, module)

    @staticmethod
    def _find_live_script() -> Optional[Path]:
        """Find Seed-VC's native real-time script in the installed distribution."""
        for distribution_name in ("seed-vc", "seed_vc"):
            try:
                distribution = importlib.metadata.distribution(distribution_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            files = distribution.files or ()
            for file in files:
                if Path(file).name == "real-time-gui.py":
                    candidate = Path(str(distribution.locate_file(file)))
                    if candidate.is_file():
                        return candidate

        package_spec = importlib.util.find_spec("seed_vc")
        if package_spec is None or package_spec.submodule_search_locations is None:
            return None
        package_dirs = [Path(path) for path in package_spec.submodule_search_locations]
        for package_dir in package_dirs:
            for candidate in (
                package_dir / "real-time-gui.py",
                package_dir / "real_time_gui.py",
                package_dir.parent / "real-time-gui.py",
                package_dir.parent / "real_time_gui.py",
            ):
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _get_live_model_sample_rate(model_set: tuple[object, ...]) -> int:
        """Return the sample rate declared by Seed-VC's live model set."""
        if len(model_set) < 6 or not isinstance(model_set[-1], dict):
            raise RuntimeError(string("seedvc.live_model_invalid"))
        sample_rate = model_set[-1].get("sampling_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise RuntimeError(string("seedvc.live_model_invalid"))
        return sample_rate

    @staticmethod
    def _aligned_live_frames(sample_rate: int, seconds: float) -> int:
        """Return a duration rounded to Seed-VC's 50 Hz codec frame grid."""
        frame = max(1, sample_rate // 50)
        return max(frame, round(seconds * sample_rate / frame) * frame)

    @staticmethod
    def _read_live_reference(path: Path, target_sample_rate: int) -> AudioChunk:
        """Read and normalize a Seed-VC live reference waveform."""
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        reference = np.asarray(audio, dtype=np.float32)
        if reference.ndim == 2:
            reference = np.mean(reference, axis=1, dtype=np.float32)
        if reference.ndim != 1:
            raise ValueError(string("seedvc.reference_audio_invalid"))
        if sample_rate != target_sample_rate:
            waveform = torch.from_numpy(reference).unsqueeze(0)
            reference = np.asarray(
                torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    target_sample_rate,
                )
                .squeeze(0)
                .numpy(),
                dtype=np.float32,
            )
        return reference

    @staticmethod
    def _resample_live_audio(
        audio: AudioChunk,
        source_sample_rate: int,
        target_sample_rate: int,
    ) -> AudioChunk:
        """Resample one live mono waveform with Seed-VC's tensor resampler."""
        if source_sample_rate == target_sample_rate:
            return np.asarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        resampled = torchaudio.functional.resample(
            waveform,
            source_sample_rate,
            target_sample_rate,
        )
        return np.asarray(resampled.squeeze(0).numpy(), dtype=np.float32)

    def _clear_live_session(self) -> None:
        """Clear live buffers while retaining any loaded Seed-VC models."""
        self._live_session_key = None
        self._live_reference_path = None
        self._live_reference_wav = np.zeros(0, dtype=np.float32)
        self._live_model_sample_rate = 0
        self._live_block_frame = 0
        self._live_block_frame_16k = 0
        self._live_crossfade_frame = 0
        self._live_sola_buffer_frame = 0
        self._live_sola_search_frame = 0
        self._live_extra_frame = 0
        self._live_extra_frame_right = 0
        self._live_skip_head = 0
        self._live_skip_tail = 0
        self._live_return_length = 0
        self._live_input_pending = np.zeros(0, dtype=np.float32)
        self._live_input_wav = None
        self._live_sola_buffer = None
        self._live_fade_in_window = None
        self._live_fade_out_window = None

    def _initialize_live_session(
        self,
        realtime_module: _SeedVCRealtimeModule,
        model_set: tuple[object, ...],
        reference_path: Path,
        input_sample_rate: int,
    ) -> None:
        """Initialize native live buffers for one reference and input rate."""
        self._clear_live_session()
        model_sample_rate = self._get_live_model_sample_rate(model_set)
        self._live_reference_path = reference_path
        self._live_reference_wav = self._read_live_reference(
            reference_path,
            model_sample_rate,
        )
        self._live_model_sample_rate = model_sample_rate

        zc = max(1, model_sample_rate // 50)
        self._live_block_frame = self._aligned_live_frames(
            model_sample_rate,
            _LIVE_BLOCK_SECONDS,
        )
        self._live_block_frame_16k = 320 * self._live_block_frame // zc
        self._live_crossfade_frame = self._aligned_live_frames(
            model_sample_rate,
            _LIVE_CROSSFADE_SECONDS,
        )
        self._live_sola_buffer_frame = min(self._live_crossfade_frame, 4 * zc)
        self._live_sola_search_frame = zc
        self._live_extra_frame = self._aligned_live_frames(
            model_sample_rate,
            _LIVE_EXTRA_CONTEXT_SECONDS,
        )
        self._live_extra_frame_right = self._aligned_live_frames(
            model_sample_rate,
            _LIVE_EXTRA_CONTEXT_RIGHT_SECONDS,
        )
        self._live_skip_head = self._live_extra_frame // zc
        self._live_skip_tail = self._live_extra_frame_right // zc
        self._live_return_length = (
            self._live_block_frame
            + self._live_sola_buffer_frame
            + self._live_sola_search_frame
        ) // zc
        self._live_input_pending = np.zeros(0, dtype=np.float32)

        buffer_frames = (
            self._live_extra_frame
            + self._live_crossfade_frame
            + self._live_sola_search_frame
            + self._live_block_frame
            + self._live_extra_frame_right
        )
        device = realtime_module.device
        self._live_input_wav = torch.zeros(buffer_frames, device=device)
        self._live_sola_buffer = torch.zeros(
            self._live_sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )
        fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self._live_sola_buffer_frame,
                    device=device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self._live_fade_in_window = fade_in_window
        self._live_fade_out_window = 1 - fade_in_window
        self._live_session_key = (reference_path, input_sample_rate)

        setattr(realtime_module, "prompt_condition", None)
        setattr(realtime_module, "mel2", None)
        setattr(realtime_module, "style2", None)
        setattr(realtime_module, "reference_wav_name", "")
        setattr(realtime_module, "prompt_len", 0)
        setattr(
            realtime_module,
            "ce_dit_difference",
            _LIVE_CONTEXT_DIFFERENCE_SECONDS,
        )

    def _append_live_model_audio(self, audio: AudioChunk) -> None:
        """Append one model-rate block to Seed-VC's rolling live buffer."""
        if self._live_input_wav is None:
            raise RuntimeError(string("seedvc.live_model_invalid"))
        tensor = torch.from_numpy(audio).to(
            device=self._live_input_wav.device,
            dtype=self._live_input_wav.dtype,
        )
        self._live_input_wav[: -self._live_block_frame] = self._live_input_wav[
            self._live_block_frame :
        ].clone()
        self._live_input_wav[-self._live_block_frame :] = tensor

    def _apply_live_sola(self, generated: torch.Tensor) -> torch.Tensor:
        """Align and crossfade one native Seed-VC output block."""
        if (
            self._live_sola_buffer is None
            or self._live_fade_in_window is None
            or self._live_fade_out_window is None
        ):
            raise RuntimeError(string("seedvc.live_model_invalid"))
        output = generated.reshape(-1)
        required = (
            self._live_sola_buffer_frame
            + self._live_sola_search_frame
            + self._live_block_frame
        )
        if output.numel() < required:
            output = torch.nn.functional.pad(output, (0, required - output.numel()))
        conv_input = output[
            : self._live_sola_buffer_frame + self._live_sola_search_frame
        ][None, None, :]
        correlation = torch.nn.functional.conv1d(
            conv_input,
            self._live_sola_buffer[None, None, :],
        )
        denominator = torch.sqrt(
            torch.nn.functional.conv1d(
                conv_input**2,
                torch.ones(
                    1,
                    1,
                    self._live_sola_buffer_frame,
                    device=output.device,
                ),
            )
            + 1e-8
        )
        offset = int(torch.argmax(correlation[0, 0] / denominator[0, 0]).item())
        aligned = output[offset:]
        aligned[: self._live_sola_buffer_frame] *= self._live_fade_in_window
        aligned[: self._live_sola_buffer_frame] += (
            self._live_sola_buffer * self._live_fade_out_window
        )
        self._live_sola_buffer[:] = aligned[
            self._live_block_frame : self._live_block_frame
            + self._live_sola_buffer_frame
        ]
        return aligned[: self._live_block_frame]

    def _convert_live_block(self, model_audio: AudioChunk) -> AudioChunk:
        """Run one fixed-size block through Seed-VC's native live function."""
        if self._live_module is None or self._live_model_set is None:
            raise RuntimeError(string("seedvc.live_model_invalid"))
        self._append_live_model_audio(model_audio)
        if self._live_input_wav is None or self._live_reference_path is None:
            raise RuntimeError(string("seedvc.live_model_invalid"))
        input_wav_res = torchaudio.functional.resample(
            self._live_input_wav.unsqueeze(0),
            self._live_model_sample_rate,
            16000,
        ).squeeze(0)
        output = self._live_module.custom_infer(
            self._live_model_set,
            self._live_reference_wav,
            str(self._live_reference_path),
            input_wav_res,
            self._live_block_frame_16k,
            self._live_skip_head,
            self._live_skip_tail,
            self._live_return_length,
            _LIVE_DIFFUSION_STEPS,
            _LIVE_INFERENCE_CFG_RATE,
            _LIVE_MAX_PROMPT_SECONDS,
            _LIVE_CONTEXT_DIFFERENCE_SECONDS,
        )
        return np.asarray(
            self._apply_live_sola(output).detach().cpu().numpy(),
            dtype=np.float32,
        )

    def _convert_live_audio(self, audio: AudioChunk, sample_rate: int) -> AudioChunk:
        """Convert one live input block and return audio at the input rate."""
        source = self._mix_to_mono(audio)
        source = self._resample_live_audio(
            source,
            sample_rate,
            self._live_model_sample_rate,
        )
        self._live_input_pending = np.concatenate((self._live_input_pending, source))
        outputs: list[AudioChunk] = []
        while len(self._live_input_pending) >= self._live_block_frame:
            source_block = self._live_input_pending[: self._live_block_frame]
            self._live_input_pending = self._live_input_pending[
                self._live_block_frame :
            ]
            converted = self._convert_live_block(source_block)
            outputs.append(
                self._resample_live_audio(
                    converted,
                    self._live_model_sample_rate,
                    sample_rate,
                )
            )
        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32, copy=False)

    @staticmethod
    def _mix_to_mono(audio: AudioChunk) -> AudioChunk:
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
    ) -> AudioChunk:
        """Run a generator to completion and return its final value."""
        try:
            while True:
                next(generator)
        except StopIteration as exc:
            return exc.value

    @property
    def output_sample_rate(self) -> int:  # noqa
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
        if self.f0_condition:
            self._get_wrapper()
        else:
            self._get_live_runtime()

    def unload_model(self) -> None:
        """Release the cached Seed-VC wrapper and best-effort GPU memory."""
        with self._wrapper_lock:
            self._wrapper = None
            self._clear_live_session()
            self._live_module = None
            self._live_model_set = None

        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def stop_live(self) -> None:
        """Reset the native Seed-VC live session without unloading its models."""
        with self._wrapper_lock:
            self._clear_live_session()

    def convert_live(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert one live block through Seed-VC's native real-time path."""
        if request.f0_condition:
            return self.convert(request)
        if not request.target_references:
            raise ValueError(string("seedvc.target_reference_required"))

        reference_path = request.target_references[0]
        realtime_module, model_set = self._get_live_runtime()
        session_key = (reference_path, request.sample_rate)
        with self._wrapper_lock:
            if self._live_session_key != session_key:
                self._initialize_live_session(
                    realtime_module,
                    model_set,
                    reference_path,
                    request.sample_rate,
                )
            audio = self._convert_live_audio(
                request.source_audio,
                request.sample_rate,
            )

        return AudioOutput(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=request.sample_rate,
            label=request.label,
        )

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
