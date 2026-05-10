"""Qwen3 backend implementation for Celune."""

from __future__ import annotations

import os
import glob
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Generator, Literal, Optional, cast

import numpy as np
import numpy.typing as npt
import torch
from accelerate import init_empty_weights
from torch import nn

# imported with a name because __version__ is reserved to Celune
# it's not in __all__, but that's not Celune's job, so we have to ignore the warning
from faster_qwen3_tts import FasterQwen3TTS, __version__ as qwen3_ver
from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.talker_graph import TalkerGraph
from faster_qwen3_tts.utils import suppress_flash_attn_warning
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, AutoProcessor

with suppress_flash_attn_warning():
    from qwen_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
    )

    # this apparently is also not in __all__, we can't fix that either
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTokenizer
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from .base import BackendTiming, CeluneBackend
from ..modeling import (
    QWEN3_SAFE_TRANSFORMER_INT8_PROFILE,
    enable_qwen3_int8_runtime_cache,
    inspect_hybrid_tts_checkpoint,
    load_hybrid_tts_checkpoint,
    qwen3_int8_state_dict,
    replace_int8_linear_modules,
)
from ..exceptions import BackendError


def _load_faster_qwen3_tts_int8(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    max_seq_len: int = 2048,
) -> FasterQwen3TTS:
    """Load a Qwen3 hybrid INT8/BF16 checkpoint as FasterQwen3TTS.

    Args:
        model_path: Local checkpoint directory to load.
        device: Target device for model tensors and CUDA graphs.
        dtype: Runtime dtype used for non-INT8 computation.
        attn_implementation: Attention implementation requested from Qwen.
        max_seq_len: Maximum sequence length for the talker CUDA graph.
    Returns:
        FasterQwen3TTS: Loaded FasterQwen3TTS wrapper.

    Raises:
        BackendError: The checkpoint cannot be retained or does not match the model.
    """
    load_start = time.perf_counter()
    loaded = load_hybrid_tts_checkpoint(model_path, keep_state_dict=True)
    if loaded.state_dict is None:
        raise BackendError("INT8 checkpoint scan did not retain the state dict")

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    config = cast(Any, AutoConfig.from_pretrained(model_path))
    config._attn_implementation = attn_implementation

    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)

    int8_linear_count = replace_int8_linear_modules(
        model,
        loaded.index,
        loaded.state_dict,
        QWEN3_SAFE_TRANSFORMER_INT8_PROFILE,
    )

    state_dict = qwen3_int8_state_dict(loaded.state_dict)
    state_dict = {
        name: tensor.to(device)
        for name, tensor in state_dict.items()
        if not tensor.is_meta
    }
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        missing = ", ".join(incompatible.missing_keys[:5])
        unexpected = ", ".join(incompatible.unexpected_keys[:5])
        raise BackendError(
            "Qwen3 INT8 checkpoint did not match the model "
            f"(missing: {missing or 'none'}; unexpected: {unexpected or 'none'})"
        )

    cast(nn.Module, model).to(device)
    model.eval()
    cache_start = time.perf_counter()
    runtime_cache_count = enable_qwen3_int8_runtime_cache(
        model,
        device=device,
        dtype=dtype,
        offload_quantized_buffers=True,
        module_filter=None,
    )
    dequant_cache_ms = (time.perf_counter() - cache_start) * 1000
    del state_dict
    loaded.state_dict = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(Path(model_path) / "speech_tokenizer"),
        device_map=device,
        dtype=dtype,
    )
    model.load_speech_tokenizer(speech_tokenizer)

    with open(Path(model_path) / "generation_config.json", encoding="utf-8") as f:
        model.load_generate_config(json.load(f))

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    base_model = Qwen3TTSModel(
        model=model, processor=processor, generate_defaults=model.generate_config
    )

    talker = base_model.model.talker
    talker_config = base_model.model.config.talker_config
    predictor = talker.code_predictor
    pred_config = predictor.model.config

    predictor_graph = PredictorGraph(
        predictor,
        pred_config,
        talker_config.hidden_size,
        device=device,
        dtype=dtype,
        do_sample=True,
        top_k=50,
        temperature=0.9,
    )
    talker_graph = TalkerGraph(
        talker.model,
        talker_config,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )
    tts = FasterQwen3TTS(
        base_model=base_model,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )
    setattr(tts, "_celune_int8_linear_count", int8_linear_count)
    setattr(tts, "_celune_int8_runtime_cache_count", runtime_cache_count)
    setattr(tts, "_celune_int8_load_ms", (time.perf_counter() - load_start) * 1000)
    setattr(tts, "_celune_int8_dequant_cache_ms", dequant_cache_ms)
    return tts


def _qwen3_checkpoint_needs_hybrid_load(model_path: str) -> bool:
    """Return whether a Qwen3 checkpoint needs hybrid INT8 loading.

    Args:
        model_path: Local checkpoint directory to inspect.

    Returns:
        bool: ``True`` when the checkpoint contains hybrid INT8/BF16 weights.
    """
    index = inspect_hybrid_tts_checkpoint(model_path)
    return index.has_int8_tensors


def _load_faster_qwen3_tts_auto(
    model_path: str,
) -> FasterQwen3TTS:
    """Load Qwen3 using hybrid or normal loading based on checkpoint contents.

    Args:
        model_path: Local checkpoint directory to load.
    Returns:
        FasterQwen3TTS: Loaded FasterQwen3TTS wrapper.
    """
    if _qwen3_checkpoint_needs_hybrid_load(model_path):
        return _load_faster_qwen3_tts_int8(model_path)
    return FasterQwen3TTS.from_pretrained(model_path)


def _timed_qwen3_stream(
    stream: Generator[tuple[npt.NDArray[np.float32], int, Optional[dict]], None, None],
) -> Generator[tuple[npt.NDArray[np.float32], int, BackendTiming], None, None]:
    """Attach wall-clock yield timing to Qwen3 streaming chunks."""
    chunk_index = 0
    while True:
        start = time.perf_counter()
        try:
            audio_chunk, sr, timing = next(stream)
        except StopIteration:
            return

        yield_ms = (time.perf_counter() - start) * 1000
        timing_out: BackendTiming = dict(timing) if isinstance(timing, dict) else {}
        timing_out.setdefault("backend", "qwen3")
        timing_out.setdefault("chunk_index", chunk_index)
        model_ms = float(timing_out.get("prefill_ms", 0.0)) + float(
            timing_out.get("decode_ms", 0.0)
        )
        timing_out["total_wall_ms"] = yield_ms
        codec_ms = max(0.0, yield_ms - model_ms)
        if codec_ms > 0.0:
            timing_out["codec_ms_approx"] = codec_ms
        chunk_index += 1
        yield audio_chunk, sr, timing_out


class Qwen3(CeluneBackend):
    """Celune Qwen3-TTS backend."""

    name: str = "qwen3"
    chunk_rate: float = 12.5
    clone_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    supported_modes: tuple[str, ...] = ("native", "clone")
    voice_models: dict[str, str] = {
        "balanced": "lunahr/Celune-1.7B-Neutral",
        "calm": "lunahr/Celune-1.7B-Calm",
        "bold": "lunahr/Celune-1.7B-Energetic",
        "upbeat": "lunahr/Celune-1.7B-Upbeat",
    }
    reference_waves: dict[str, str] = {
        "balanced": "refs/balanced.wav",
        "calm": "refs/calm.wav",
        "bold": "refs/bold.wav",
        "upbeat": "refs/upbeat.wav",
    }
    reference_texts: dict[str, str] = {
        "balanced": (
            "My name is Celune, pronounced Celune. It is a pleasure to meet you."
        ),
        "calm": "My name is... Celune... It is so... quiet.",
        "bold": "My name is Celune! Let's do this, we have to get it done!",
        "upbeat": (
            "Hehehe... Hi, I'm Celune. Look, I have something to tell... "
            "might as well make it fun. Shall we?"
        ),
    }
    default_voice: str = "balanced"

    def __init__(
        self,
        log: Callable[[str, str], None],
        mode: Literal["native", "clone"] = "native",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the Qwen3 backend.

        Args:
            log: Logger callback used by the backend.
            mode: Qwen3 generation mode to use.
            config: Loaded Celune configuration dictionary.

        Returns:
            None: This constructor validates and stores the active mode.

        Raises:
            ValueError: The requested Qwen3 generation mode is unsupported.
        """
        if mode not in self.supported_modes:
            raise ValueError(
                f"unsupported qwen3 mode '{mode}' "
                f"(available: {', '.join(self.supported_modes)})"
            )

        super().__init__(log=log, config=config)

        if self.config.get("int8"):
            self.clone_model = "lunahr/Qwen3-TTS-12Hz-1.7B-Base-hybrid-int8"

            self.voice_models = {
                "balanced": "lunahr/Celune-1.7B-Neutral-hybrid-int8",
                "calm": "lunahr/Celune-1.7B-Calm-hybrid-int8",
                "bold": "lunahr/Celune-1.7B-Energetic-hybrid-int8",
                "upbeat": "lunahr/Celune-1.7B-Upbeat-hybrid-int8",
            }
            self.model_name = self.voice_models[self.default_voice]

        self.mode = mode
        if self.mode == "clone":
            self.model_name = self.clone_model
            self._validate_refs()

    @property
    def default_model_id(self) -> str:
        """Return the model loaded by default for the active Qwen3 mode.

        Returns:
            str: The default Qwen3 model identifier.
        """
        if self.mode == "clone":
            if self.int8_model_path is not None:
                return self.int8_model_path
            return self.clone_model
        return super().default_model_id

    @property
    def all_model_ids(self) -> list[str]:
        """Return every model required by the active Qwen3 mode.

        Returns:
            list[str]: The model identifiers needed by the selected mode.
        """
        if self.mode == "clone":
            if self.int8_model_path is not None:
                return [self.int8_model_path]
            return [self.clone_model]
        return super().all_model_ids

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a Celune voice to the model required by the active Qwen3 mode.

        Args:
            voice: The Celune voice name to resolve.

        Returns:
            str: The model identifier for the requested voice.

        Raises:
            ValueError: Clone mode cannot resolve the requested voice.
        """
        if self.int8_model_path is not None:
            if voice not in self.voice_models:
                raise ValueError(
                    f"{self.name} cannot resolve a model for voice '{voice}'"
                )
            return self.int8_model_path

        if self.mode == "clone":
            if voice not in self.voice_models:
                raise ValueError(
                    f"{self.name} cannot resolve a model for voice '{voice}'"
                )
            return self.clone_model

        return super().model_id_for_voice(voice)

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

    def preload_models(self) -> None:
        """Ensure all known Qwen3 voice models are cached locally.

        Returns:
            None: This method downloads any missing voice models.
        """
        for model_id in self.all_model_ids:
            if Path(model_id).expanduser().exists():
                self.log(f"{model_id} is available locally.", "info")
                continue

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

    def load_model(self, model_id: str, **kwargs) -> FasterQwen3TTS:
        """Load the given voice model.

        Args:
            model_id: The Qwen3 model repository ID to load.
            **kwargs: Additional keyword arguments to use.

        Returns:
            FasterQwen3TTS: The loaded Qwen3 TTS model instance.
        """
        local_path = Path(model_id).expanduser()
        if local_path.exists():
            self.log("Loading TTS model from local path...", "info")
            model_path = str(local_path.resolve())
            self.model = _load_faster_qwen3_tts_auto(model_path)
            self._log_int8_runtime_stats(self.model)
            return self.model

        available, path = self.model_is_available_locally(model_id)

        if available and path is not None:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = _load_faster_qwen3_tts_auto(path)
            self._log_int8_runtime_stats(self.model)
            return self.model

        self.log("Downloading TTS model...", "info")
        path = snapshot_download(repo_id=model_id)
        self.model = _load_faster_qwen3_tts_auto(path)
        self._log_int8_runtime_stats(self.model)
        return self.model

    def _log_int8_runtime_stats(self, model: FasterQwen3TTS) -> None:
        """Log Celune INT8 runtime details when the model exposes them."""
        int8_linear_count = getattr(model, "_celune_int8_linear_count", None)
        if int8_linear_count is None:
            return

        cache_count = getattr(model, "_celune_int8_runtime_cache_count", 0)
        load_ms = float(getattr(model, "_celune_int8_load_ms", 0.0))
        dequant_cache_ms = float(getattr(model, "_celune_int8_dequant_cache_ms", 0.0))
        self.log(
            "Qwen3 INT8 VRAM runtime: "
            f"{int8_linear_count} linear layers replaced, "
            f"{cache_count} cached for replay, "
            f"load {load_ms:.1f} ms, dequant/cache {dequant_cache_ms:.1f} ms.",
            "info",
        )

    def generate_stream(
        self, model: FasterQwen3TTS, **kwargs
    ) -> Generator[tuple[npt.NDArray[np.float32], int, BackendTiming]]:
        """Generate Celune compatible audio chunks.

        Args:
            model: The loaded Qwen3 model instance.
            **kwargs: Streaming generation keyword arguments to use.

        Returns:
            Generator[tuple[npt.NDArray[np.float32], int, BackendTiming]]: An iterator of Qwen3 streaming audio chunks.

        Raises:
            ValueError: The current Qwen3 mode and/or requested voice is unsupported, or input text is empty.
        """
        if not kwargs.get("text", None):
            raise ValueError("expected text to say")

        # if faster_qwen3_tts >= 0.2.5 use instructions, else remove this arg
        major, minor, patch = (int(num) for num in qwen3_ver.split("."))
        if not (major >= 0 and minor >= 2 and patch >= 5):
            kwargs.pop("instruct", None)

        if self.mode == "native":
            # we are not using the voice param here, as the model defines only one
            # and you have to reload the model to apply voice settings
            kwargs.pop("voice", None)
            self.log("Qwen3 generation path: native custom voice.", "debug")
            # Celune natively works with Qwen-formatted chunks
            yield from _timed_qwen3_stream(
                model.generate_custom_voice_streaming(speaker="celune", **kwargs)
            )
        elif self.mode == "clone":
            # we are using the voice param here as it tells Celune which reference to use
            voice = kwargs.pop("voice", self.default_voice)

            try:
                # this path resolves to celune/refs/[voice].wav
                ref_wav = (
                    Path(__file__).resolve().parents[1] / self.reference_waves[voice]
                )
                ref_text = self.reference_texts[voice]
            except KeyError as e:
                raise ValueError(
                    f"unknown voice '{voice}' for backend '{self.name}'"
                ) from e

            self.log("Qwen3 generation path: voice clone.", "debug")
            yield from _timed_qwen3_stream(
                model.generate_voice_clone_streaming(
                    ref_audio=ref_wav,
                    ref_text=ref_text,
                    non_streaming_mode=False,  # VERY IMPORTANT ON >=0.2.5
                    **kwargs,
                )
            )
        else:
            raise ValueError(f"unsupported qwen3 mode '{self.mode}'")
