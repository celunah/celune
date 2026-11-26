# SPDX-License-Identifier: Apache-2.0
"""Streaming Celune adapter for Zyphra's ZONOS3-Mini reference model."""

import gc
import os
import queue
import base64
import warnings
import threading
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Iterator
from typing import Optional, Protocol, Union, cast

import torch
import numpy as np
from huggingface_hub import snapshot_download, try_to_load_from_cache

from ...registry.tts import (
    TTS_DEFAULT_VOICE,
    ZONOS3_TTS_MODEL_ID,
    ZONOS3_TTS_REVISION,
)
from .base import CeluneBackend
from ...typing.aliases import AudioChunk
from ...cevoice import CEVoiceLoader, default_loader
from ...registry.consent import require_model_consent
from ...paths import huggingface_hub_cache_dir, huggingface_progress
from ...typing.backends import BackendArgumentValue, BackendArguments


_SAMPLE_RATE = 44_100
_MODEL_PATH = "lm/zonos3_mini"
_MODEL_WEIGHTS = f"{_MODEL_PATH}/model.pth"
_TEXT_ROWS = f"{_MODEL_PATH}/text_rows/pause_tokens_s5000.pt"
_ASSET_ALLOW_PATTERNS = (
    f"{_MODEL_PATH}/model.pth",
    f"{_MODEL_PATH}/params.json",
    f"{_MODEL_PATH}/tokenizer/superbpe8192.json",
    _TEXT_ROWS,
    "autoencoder/generator.pt",
    "autoencoder/teacher/**",
    "autoencoder/heads/keep2_plain/**",
    "speaker_encoders/ownmect_s2/ckpt_0037.pth",
)
_SUPPORTED_LANGUAGES = (
    "en",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "ja",
    "zh",
    "ko",
    "ru",
    "ar",
    "bn",
    "bg",
    "cs",
    "da",
    "nl",
    "fi",
    "ka",
    "el",
    "gu",
    "he",
    "hi",
    "hu",
    "id",
    "kn",
    "ms",
    "ml",
    "mr",
    "no",
    "or",
    "pl",
    "pa",
    "ro",
    "sk",
    "sv",
    "tl",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "vi",
)
_LANGUAGE_ALIASES = {
    "en": "en_us",
    "en-us": "en_us",
    "en-gb": "en_gb",
    "fr": "fr_fr",
    "fr-fr": "fr_fr",
    "pt": "pt_br",
    "pt-br": "pt_br",
    "zh": "cmn",
    "zh-cn": "cmn",
}
_QUEUE_CAPACITY = 8


def _resolve_codec_device(device: torch.device) -> tuple[torch.device, bool]:
    """Resolve the codec device override or use the language model's device.

    Args:
        device: Device selected for the ZONOS3 language model.

    Returns:
        tuple[torch.device, bool]: Selected device and whether automatic fallback is enabled.
    """
    configured_device = os.environ.get("ZONOS3_AUDIO_CODEC_DEVICE", "auto").strip()
    automatic = not configured_device or configured_device.casefold() == "auto"
    if automatic:
        return device, True
    return torch.device(configured_device), False


def _load_codec_decoder_with_fallback(
    load_decoder: Callable[[torch.device], torch.nn.Module],
    device: torch.device,
    *,
    automatic: bool,
    log: Callable[[str, str], None],
) -> tuple[torch.nn.Module, torch.device]:
    """Retry automatic CUDA decoder loading on CPU when VRAM is exhausted.

    Args:
        load_decoder: Load one decoder on the supplied device.
        device: Initially selected codec device.
        automatic: Whether device placement allows CPU fallback.
        log: Backend diagnostic callback.

    Returns:
        tuple[torch.nn.Module, torch.device]: Loaded decoder and its device.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of "
                r"`torch\.nn\.utils\.parametrizations\.weight_norm`\."
            ),
            category=FutureWarning,
        )
        try:
            decoder_model = load_decoder(device)
        except torch.cuda.OutOfMemoryError as error:
            if not automatic or device.type != "cuda":
                raise
            error.__traceback__ = None
            del error
            gc.collect()
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            log(
                "[ZONOS3] Matched codec decoder did not fit on CUDA, retrying on CPU",
                "warning",
            )
            decoder_model = load_decoder(device)
    return decoder_model, device


def _snapshot_asset_path(
    snapshot_path: Path,
    value: Union[str, Path],
) -> Optional[Path]:
    """Map a release URI into a pinned snapshot without resolving file symlinks.

    Args:
        snapshot_path: Root of the downloaded Hugging Face snapshot.
        value: Upstream release URI or an unrelated local path.

    Returns:
        Optional[Path]: Snapshot path for a matching URI, or ``None`` otherwise.

    Raises:
        ValueError: If a matching URI attempts to traverse outside the snapshot.
    """
    value_string = str(value)
    uri_root = f"hf://{ZONOS3_TTS_MODEL_ID}"
    uri_prefix = f"{uri_root}/"
    if value_string == uri_root:
        return snapshot_path.resolve()
    if not value_string.startswith(uri_prefix):
        return None

    relative_path = Path(value_string.removeprefix(uri_prefix))
    if (
        relative_path.is_absolute()
        or relative_path.drive
        or ".." in relative_path.parts
    ):
        raise ValueError(
            f"ZONOS3 release asset path escapes the pinned snapshot: {value_string}"
        )
    return snapshot_path.resolve() / relative_path


class _Zonos3LanguageModel(Protocol):
    """The model device and dtype needed by upstream synthesis."""

    @property
    def device(self) -> torch.device:
        """Return the language model's active device."""

    @property
    def dtype(self) -> torch.dtype:
        """Return the language model's active weight dtype."""


class _ReferenceAssets(Protocol):
    """Release hook implemented by Celune's lazy upstream asset wrapper."""

    def release(self) -> None:
        """Clear cached codec and speaker encoder modules."""


@dataclass(slots=True)
class Zonos3Runtime:
    """Hold one loaded ZONOS3 language model and its lazy reference assets."""

    lm: Optional[_Zonos3LanguageModel]
    assets: Optional[_ReferenceAssets]
    snapshot_path: Path
    precision: str = "bf16"

    def close(self) -> None:
        """Release model, speaker encoder, and decoder caches."""
        assets = self.assets
        self.assets = None
        self.lm = None
        if assets is not None:
            assets.release()


@dataclass(slots=True)
class _AudioStreamEvent:
    """Carry one audio chunk, error, or completion signal between threads."""

    pcm: Optional[AudioChunk] = None
    error: Optional[Exception] = None
    finished: bool = False


class _SynthesisCancelled(Exception):
    """Stop upstream synthesis after a consumer closes the stream."""


class Zonos3(CeluneBackend[Zonos3Runtime]):
    """ZONOS3-Mini TTS with streaming PCM and CEVOICE voice references."""

    name: str = "zonos3"
    chunk_rate: float = 15.0
    max_new_tokens: int = 4096
    supported_languages: tuple[str, ...] = _SUPPORTED_LANGUAGES
    uses_voice_bundles: bool = True
    default_voice: Optional[str] = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
        quantize_kv_cache: bool = True,
    ) -> None:
        super().__init__(
            log=log,
            fatal=fatal,
            quantize=quantize,
            quantize_kv_cache=quantize_kv_cache,
        )
        self.model_name = ZONOS3_TTS_MODEL_ID
        self.quantization_mode = None

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for ZONOS3."""
        return default_loader()

    @property
    def default_model_id(self) -> str:
        """Return the pinned ZONOS3-Mini model identifier."""
        return ZONOS3_TTS_MODEL_ID

    @property
    def all_model_ids(self) -> list[str]:
        """Return the only model identifier used by this backend."""
        return [ZONOS3_TTS_MODEL_ID]

    @property
    def voices(self) -> list[str]:
        """Return the compatible voice names in the active voice pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice-pack entry to the shared ZONOS3-Mini model."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return ZONOS3_TTS_MODEL_ID
        _, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(f"unknown voice '{voice}' for backend '{self.name}'")
        return ZONOS3_TTS_MODEL_ID

    def model_is_available_locally(
        self,
        model: str,
        lang: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check for every pinned model and reference asset in the Hub cache."""
        del lang
        if model != ZONOS3_TTS_MODEL_ID:
            return False, None

        cached_model = try_to_load_from_cache(
            repo_id=ZONOS3_TTS_MODEL_ID,
            filename=_MODEL_WEIGHTS,
            revision=ZONOS3_TTS_REVISION,
            cache_dir=str(huggingface_hub_cache_dir(create=True)),
        )
        if not isinstance(cached_model, str):
            return False, None

        snapshot_path = Path(cached_model).parents[2]
        required_paths = (
            _MODEL_WEIGHTS,
            f"{_MODEL_PATH}/params.json",
            f"{_MODEL_PATH}/tokenizer/superbpe8192.json",
            _TEXT_ROWS,
            "autoencoder/generator.pt",
            "autoencoder/teacher/config.json",
            "autoencoder/teacher/model.safetensors",
            "autoencoder/heads/keep2_plain/config_full_unfreezeall_mr_drop30_4x10.json",
            "autoencoder/heads/keep2_plain/step_0050000.pt",
            "speaker_encoders/ownmect_s2/ckpt_0037.pth",
        )
        if not all((snapshot_path / path).is_file() for path in required_paths):
            return False, None
        return True, str(snapshot_path)

    def preload_models(self) -> None:
        """Download the pinned model, tokenizer, codec, and speaker encoder files."""
        if not torch.cuda.is_available():
            raise RuntimeError("ZONOS3-Mini requires an available CUDA device")

        metadata = require_model_consent(ZONOS3_TTS_MODEL_ID)
        available, _ = self.model_is_available_locally(ZONOS3_TTS_MODEL_ID)
        if available:
            return

        with huggingface_progress(self.report_progress):
            snapshot_download(
                repo_id=ZONOS3_TTS_MODEL_ID,
                revision=metadata.revision,
                cache_dir=str(huggingface_hub_cache_dir(create=True)),
                allow_patterns=list(_ASSET_ALLOW_PATTERNS),
            )

        available, snapshot_path = self.model_is_available_locally(ZONOS3_TTS_MODEL_ID)
        if not available or snapshot_path is None:
            raise RuntimeError(
                "ZONOS3-Mini snapshot is missing required codec or model files"
            )

    @staticmethod
    def _pin_upstream_assets(snapshot_path: Path) -> None:
        """Resolve upstream release URIs against the pinned local snapshot."""
        from zonos3.utils import release

        original_file_resolver = release.resolve_file
        original_directory_resolver = release.resolve_dir_or_local

        def resolve_file(value: Union[str, Path]) -> str:
            """Use the pinned snapshot for release files from the model repository."""
            local = _snapshot_asset_path(snapshot_path, value)
            if local is None:
                return original_file_resolver(value)
            if not local.is_file():
                raise FileNotFoundError(f"ZONOS3 release asset is missing: {local}")
            return str(local)

        def resolve_dir_or_local(value: Union[str, Path]) -> str:
            """Use the pinned snapshot for release directories from the model repository."""
            local = _snapshot_asset_path(snapshot_path, value)
            if local is None:
                return original_directory_resolver(value)
            if not local.is_dir():
                raise FileNotFoundError(f"ZONOS3 release directory is missing: {local}")
            return str(local)

        vars(release).update(
            {
                "resolve_file": resolve_file,
                "resolve_dir_or_local": resolve_dir_or_local,
            }
        )

    def load_model(
        self, model_id: str, **kwargs: BackendArgumentValue
    ) -> Zonos3Runtime:
        """Load the pinned BF16 reference model and its lazy codec assets."""
        del kwargs
        if model_id != ZONOS3_TTS_MODEL_ID:
            raise ValueError(f"unsupported ZONOS3 model identifier: {model_id}")
        if not torch.cuda.is_available():
            raise RuntimeError("ZONOS3-Mini requires an available CUDA device")

        require_model_consent(model_id)
        available, snapshot_path_value = self.model_is_available_locally(model_id)
        if not available or snapshot_path_value is None:
            raise RuntimeError("ZONOS3-Mini assets are not downloaded")

        snapshot_path = Path(snapshot_path_value)
        model_directory = snapshot_path / _MODEL_PATH
        device = torch.device("cuda", torch.cuda.current_device())
        self.report_progress(0.0, 1.0)
        self._pin_upstream_assets(snapshot_path)

        from zonos3.models.text_extra import TextExtraRows
        from zonos3.reference.lm import Zonos3LM
        from zonos3.reference.synth import ReferenceAssets

        backend_log = self.log

        class CeluneReferenceAssets(ReferenceAssets):
            """Load the matched decoder beside the language model when memory allows."""

            _celune_codec_device: Optional[torch.device] = None

            def decoder(
                self,
                codec: str,
                device: torch.device,
                decoder: Optional[str] = None,
            ) -> torch.nn.Module:
                """Load one codec decoder on Celune's configured device."""
                codec_device, automatic = _resolve_codec_device(device)
                if automatic and self._celune_codec_device is not None:
                    codec_device = self._celune_codec_device

                from zonos3.tokenizer import matched_codec

                cache_key = (
                    codec,
                    matched_codec.decoder_id(codec, decoder),
                    str(codec_device),
                )
                was_cached = cache_key in self._decoders
                if not was_cached:
                    backend_log(
                        f"[ZONOS3] Loading matched codec decoder on {codec_device}",
                        "info",
                    )

                base_decoder = super().decoder

                def load_decoder(target_device: torch.device) -> torch.nn.Module:
                    """Load this matched codec decoder on the requested device."""
                    return base_decoder(codec, target_device, decoder)

                codec_model, codec_device = _load_codec_decoder_with_fallback(
                    load_decoder,
                    codec_device,
                    automatic=automatic,
                    log=backend_log,
                )

                if automatic:
                    self._celune_codec_device = codec_device
                if not was_cached:
                    backend_log(
                        f"[ZONOS3] Matched codec decoder ready on {codec_device}",
                        "info",
                    )
                return codec_model

            def release(self) -> None:
                """Clear lazy model assets and the upstream codec's process cache."""
                self._decoders.clear()
                self._speaker_encoders.clear()
                self._normalizer = None
                self._celune_codec_device = None
                from zonos3.tokenizer import matched_codec

                matched_codec._CACHE.clear()
                matched_codec._GRAPHS.clear()

        lm = Zonos3LM.load(model_directory, device=device, dtype=torch.bfloat16)
        self.report_progress(0.7, 1.0)
        lm.append_text_rows(TextExtraRows.load(snapshot_path / _TEXT_ROWS))
        assets = CeluneReferenceAssets.load()
        assets.grow(lm)
        assets.decoder(lm.config.audio_codec, device)
        self.report_progress(0.95, 1.0)
        runtime = Zonos3Runtime(lm, cast(_ReferenceAssets, assets), snapshot_path)
        self.quantization_attempted = False
        self.quantization_active = False
        self.quantization_mode = None
        self.model = runtime
        self.report_progress(1.0, 1.0)
        return runtime

    def resolve_generation_language(self, lang: Optional[str]) -> str:
        """Normalize a Celune language code to a ZONOS3 request language."""
        if not lang:
            return "en_us"
        normalized = lang.strip().casefold().replace("_", "-")
        direct = _LANGUAGE_ALIASES.get(normalized)
        if direct is not None:
            return direct
        base = normalized.split("-", 1)[0]
        if base in _SUPPORTED_LANGUAGES:
            return base
        return "en_us"

    @staticmethod
    def _run_synthesis(
        runtime: Zonos3Runtime,
        request: BackendArguments,
        on_audio: Callable[[np.ndarray], None],
    ) -> None:
        """Run one blocking upstream request while forwarding decoded PCM callbacks."""
        if runtime.lm is None or runtime.assets is None:
            raise RuntimeError("ZONOS3-Mini runtime has been unloaded")

        from zonos3.reference.lm import Zonos3LM
        from zonos3.reference.synth import ReferenceAssets, synthesize

        language_model = cast(Zonos3LM, runtime.lm)
        reference_assets = cast(ReferenceAssets, runtime.assets)

        synthesize(
            language_model,
            request,
            assets=reference_assets,
            device=language_model.device,
            dtype=language_model.dtype,
            on_audio=on_audio,
        )

    def generate_stream(
        self,
        model: Zonos3Runtime,
        **kwargs: BackendArgumentValue,
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Yield progressive ZONOS3 PCM chunks as they are decoded."""
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("expected text to say")

        voice_value = kwargs.pop("voice", self.default_voice)
        voice = voice_value if isinstance(voice_value, str) else self.default_voice
        if voice is None:
            raise ValueError("ZONOS3-Mini requires a selected voice")
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(f"unknown voice '{voice}' for backend '{self.name}'")

        reference_path = self._truncate_reference(loader.materialize(voice, "wav"))
        language_value = kwargs.pop("language", kwargs.pop("lang", None))
        if language_value is not None and not isinstance(language_value, str):
            raise TypeError("language must be a string")
        language = self.resolve_generation_language(language_value)

        self._apply_seed()
        maximum = kwargs.pop("max_new_tokens", self.max_new_tokens)
        if not isinstance(maximum, (int, float, str)):
            raise TypeError("max_new_tokens must be an integer")
        max_tokens = int(maximum)
        if max_tokens < 1:
            raise ValueError("max_new_tokens must be greater than zero")

        request: BackendArguments = {
            "text": text.strip(),
            "language": language,
            "speaker_mode": "embedding",
            "speaker_audio_base64": base64.b64encode(
                reference_path.read_bytes()
            ).decode("ascii"),
            "speaker_audio_name": reference_path.name,
            "max_tokens": max_tokens,
            "seed": self.current_seed or 0,
            "stream": True,
            "stream_decode": "prefix",
        }
        for option, request_option in (
            ("temperature", "temperature"),
            ("top_k", "topk"),
            ("top_p", "top_p"),
            ("speaker_cfg_scale", "speaker_cfg_scale"),
        ):
            value = kwargs.pop(option, None)
            if value is not None:
                request[request_option] = value

        output: queue.Queue[_AudioStreamEvent] = queue.Queue(_QUEUE_CAPACITY)
        cancelled = threading.Event()
        chunk_index = 0
        total_samples = 0
        pending_audio: Optional[AudioChunk] = None

        def enqueue(event: _AudioStreamEvent) -> bool:
            """Apply bounded backpressure until the consumer receives an event."""
            while not cancelled.is_set():
                try:
                    output.put(event, timeout=0.1)
                except queue.Full:
                    continue
                return True
            return False

        def on_audio(pcm: np.ndarray) -> None:
            """Queue one validated upstream audio chunk for the CEDTS consumer."""
            if cancelled.is_set():
                raise _SynthesisCancelled
            audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
            if audio.size and not enqueue(_AudioStreamEvent(pcm=audio.copy())):
                raise _SynthesisCancelled

        def synthesize_in_thread() -> None:
            """Run synthesis off-thread so callbacks can be yielded immediately."""
            error: Optional[Exception] = None
            try:
                self._run_synthesis(model, request, on_audio)
            except _SynthesisCancelled:
                return
            except Exception as exception:
                error = exception
            finally:
                if error is not None:
                    enqueue(_AudioStreamEvent(error=error))
                enqueue(_AudioStreamEvent(finished=True))

        producer = threading.Thread(
            target=synthesize_in_thread,
            name="celune-zonos3-synthesis",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                event = output.get()
                if event.error is not None:
                    raise event.error
                if event.finished:
                    break
                if event.pcm is None or not event.pcm.size:
                    continue

                audio = event.pcm
                if not np.isfinite(audio).all():
                    raise RuntimeError("ZONOS3-Mini returned non-finite audio")
                np.clip(audio, -1.0, 1.0, out=audio)
                total_samples += audio.size
                if pending_audio is not None:
                    yield (
                        pending_audio,
                        _SAMPLE_RATE,
                        {
                            "backend": self.name,
                            "chunk_index": chunk_index,
                            "is_final": False,
                        },
                    )
                    chunk_index += 1
                pending_audio = audio
        finally:
            cancelled.set()
            # CEDTS enforces a bounded drain and terminates this worker if upstream
            # synthesis does not reach its next callback after cancellation.
            producer.join()

        if total_samples == 0:
            raise RuntimeError("ZONOS3-Mini returned empty audio")
        if pending_audio is not None:
            yield (
                pending_audio,
                _SAMPLE_RATE,
                {
                    "backend": self.name,
                    "chunk_index": chunk_index,
                    "is_final": True,
                },
            )

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Release ZONOS3 language, speaker, and decoder weights."""
        self.quantization_active = False
        self.quantization_mode = None
        super().unload_model(release_cuda_cache=release_cuda_cache)
