# SPDX-License-Identifier: Apache-2.0
"""Streaming Ming 3.0 TTS backend powered by audio.cpp."""

import os
from typing import Union, Optional
from pathlib import Path
from collections.abc import Callable, Iterator

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from .base import CeluneBackend
from ...i18n import string
from ...paths import (
    app_data_dir,
    huggingface_progress,
    huggingface_hub_cache_dir,
)
from ...utils import discard
from ...cevoice import CEVoiceLoader, default_loader
from .audio_cpp import (
    AudioCppRuntime,
    archive_sha256,
    runtime_assets,
    start_audio_cpp_server,
    ensure_audio_cpp_server,
)
from ...registry.tts import (
    MING3_TTS_Q8_FILE,
    MING3_TTS_Q8_SIZE,
    TTS_DEFAULT_VOICE,
    MING3_TTS_MODEL_ID,
    MING3_TTS_BF16_FILE,
    MING3_TTS_BF16_SIZE,
    MING3_TTS_GGUF_REPO,
    MING3_TTS_Q8_SHA256,
    MING3_TTS_BF16_SHA256,
    MING3_TTS_GGUF_REVISION,
    ModelContract,
)
from ...registry.tts import (
    model_contract as resolve_model_contract,
)
from ...typing.aliases import AudioChunk
from ...typing.backends import BackendArgumentValue
from ...registry.consent import require_model_consent

_AUDIOCPP_RELEASE = "v0.9.3"
_SAMPLE_RATE = 44_100
_MAX_TOKENS = 512
_MAX_AUDIO_SECONDS = 120.0
_MIN_AUDIO_SECONDS = 0.1
_MAX_AUDIO_SECONDS_PER_CHARACTER = 0.75

_LANGUAGE_ALIASES = {
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "chinese": "zh-cn",
    "en": "en",
    "english": "en",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "de": "de",
    "german": "de",
    "fr": "fr",
    "french": "fr",
    "es": "es",
    "spanish": "es",
}


def _numeric_option(value: BackendArgumentValue, name: str) -> Union[int, float]:
    """Convert a numeric backend option while rejecting nonnumeric JSON values."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{name} must be numeric")


def _integer_option(value: BackendArgumentValue, name: str) -> int:
    """Convert an integer backend option while rejecting nonnumeric JSON values."""
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"{name} must be an integer")


MingRuntime = AudioCppRuntime


class Ming3(CeluneBackend[MingRuntime]):
    """Celune's native-streaming Ming 3.0 voice-cloning backend."""

    name: str = "ming3"
    supports_audio_cpp = True
    requires_audio_cpp = True
    uses_voice_bundles: bool = True
    allow_quantization_fallback: bool = False
    chunk_rate: float = 12.5
    max_new_tokens: int = _MAX_TOKENS
    supported_languages: tuple[str, ...] = (
        "zh-cn",
        "en",
        "ja",
        "ko",
        "de",
        "fr",
        "es",
    )
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
        self.model_name = MING3_TTS_MODEL_ID
        self.quantization_mode = None
        self._verified_model_artifacts: set[tuple[str, int, int, str]] = set()
        self._validate_refs()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE/CECHAR loader for Ming3."""
        return default_loader()

    @property
    def default_model_id(self) -> str:
        """Return the registered Ming model identifier."""
        return MING3_TTS_MODEL_ID

    @property
    def all_model_ids(self) -> list[str]:
        """Return the single model identifier required by this backend."""
        return [MING3_TTS_MODEL_ID]

    @property
    def voices(self) -> list[str]:
        """Return the voice names exposed by the active voice pack."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        _, voice_names = compatible_bundle
        return list(voice_names)

    def _validate_refs(self) -> None:
        """Materialize pack WAVs so invalid cloning references fail early."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        for voice in voice_names:
            loader.materialize(voice, "wav")

    def model_id_for_voice(self, voice: str) -> str:
        """Resolve a voice-pack entry to Ming's shared model."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return MING3_TTS_MODEL_ID
        _, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(f"unknown voice '{voice}' for backend '{self.name}'")
        return MING3_TTS_MODEL_ID

    @property
    def package_precision(self) -> str:
        """Choose Q8_0 for Celune quantization or BF16 when it is disabled."""
        return "q8_0" if self.quantization_requested else "bf16"

    def _package(self) -> tuple[str, int, str]:
        """Return the pinned GGUF filename, byte size, and SHA-256."""
        if self.package_precision == "q8_0":
            return MING3_TTS_Q8_FILE, MING3_TTS_Q8_SIZE, MING3_TTS_Q8_SHA256
        return MING3_TTS_BF16_FILE, MING3_TTS_BF16_SIZE, MING3_TTS_BF16_SHA256

    def model_contract(self, model_id: str, **kwargs: object) -> ModelContract:
        """Return the pinned GGUF contract for the selected precision."""
        discard(kwargs)
        if model_id != MING3_TTS_MODEL_ID:
            raise ValueError(f"unsupported Ming model identifier: {model_id}")
        return resolve_model_contract(
            self.name,
            MING3_TTS_GGUF_REPO,
            variant=self.package_precision,
        )

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check whether the selected, pinned GGUF is in the Hub cache."""
        discard(lang)
        if model != MING3_TTS_MODEL_ID:
            return False, None

        filename, expected_size, _ = self._package()
        cached_path = try_to_load_from_cache(
            repo_id=MING3_TTS_GGUF_REPO,
            filename=filename,
            revision=MING3_TTS_GGUF_REVISION,
        )
        if not isinstance(cached_path, str):
            return False, None
        path = Path(cached_path)
        if not path.is_file() or path.stat().st_size != expected_size:
            return False, None
        return True, str(path)

    def _download_model(self) -> Path:
        """Download the chosen GGUF from the pinned audio.cpp package."""
        require_model_consent(MING3_TTS_MODEL_ID)
        available, cached_path = self.model_is_available_locally(MING3_TTS_MODEL_ID)
        filename, expected_size, expected_sha256 = self._package()
        if available and cached_path is not None:
            path = Path(cached_path)
            try:
                self._verify_model_artifact(path, expected_size, expected_sha256)
            except RuntimeError:
                pass
            else:
                return self._model_runtime_path(path, filename)
        self.log(
            string("tts.model_downloading", model_id=MING3_TTS_MODEL_ID),
            "info",
        )
        with huggingface_progress(self.report_progress):
            path = Path(
                hf_hub_download(
                    repo_id=MING3_TTS_GGUF_REPO,
                    filename=filename,
                    revision=MING3_TTS_GGUF_REVISION,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                    force_download=available,
                )
            )

        self._verify_model_artifact(path, expected_size, expected_sha256)
        return self._model_runtime_path(path, filename)

    @staticmethod
    def _model_runtime_path(path: Path, filename: str) -> Path:
        """Give a cached GGUF a stable filename without copying its contents."""
        runtime_root = app_data_dir(create=True) / "backends" / "ming3" / "models"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_path = runtime_root / filename
        source_path = path.resolve(strict=True)
        if (
            runtime_path.is_file()
            and not runtime_path.is_symlink()
            and os.path.samefile(source_path, runtime_path)
        ):
            return runtime_path

        runtime_path.unlink(missing_ok=True)
        try:
            os.link(source_path, runtime_path)
        except OSError as error:
            raise RuntimeError(
                f"could not create filename-preserving Ming GGUF link: {runtime_path}"
            ) from error
        return runtime_path

    def _verify_model_artifact(
        self, path: Path, expected_size: int, expected_sha256: str
    ) -> None:
        """Reject an incomplete or unrecognized Ming GGUF package."""
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Ming GGUF has an unexpected size: {path} "
                f"(expected {expected_size} bytes)"
            )
        stat = path.stat()
        identity = (
            str(path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            expected_sha256,
        )
        if identity in self._verified_model_artifacts:
            return
        actual_sha256 = self._archive_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Ming GGUF checksum mismatch: {path} "
                f"(expected {expected_sha256}, got {actual_sha256})"
            )
        self._verified_model_artifacts.add(identity)

    @staticmethod
    def _runtime_assets() -> tuple[tuple[str, str], ...]:
        """Expose the shared pinned archives through the existing Ming API."""
        return runtime_assets()

    @staticmethod
    def _archive_sha256(path: Path) -> str:
        """Hash model artifacts with the shared audio.cpp integrity helper."""
        return archive_sha256(path)

    def _ensure_runtime(self) -> Path:
        """Use the shared pinned runtime, including the earlier Ming cache."""
        return ensure_audio_cpp_server(
            report_progress=self.report_progress,
            legacy_cache_root=app_data_dir(create=True) / "backends" / "ming3",
            log=self.log,
            backend_name=self.name,
        )

    def _start_server(self, server_path: Path, model_path: Path) -> MingRuntime:
        """Start the shared audio.cpp server eagerly with the Ming GGUF."""
        return start_audio_cpp_server(
            server_path,
            model_path=model_path,
            precision=self.package_precision,
            sample_rate=_SAMPLE_RATE,
            model_config={
                "id": "ming3",
                "family": "ming3_tts",
                "path": model_path.as_posix(),
                "task": "clon",
                "mode": "streaming",
                "lazy": False,
            },
            torch_root=Path(torch.__file__).resolve().parent,
            device=torch.cuda.current_device(),
            backend_name=self.name,
            log=self.log,
            report_progress=self.report_progress,
        )

    def preload_models(self) -> None:
        """Download the selected GGUF and the pinned audio.cpp runtime."""
        if not torch.cuda.is_available():
            raise RuntimeError("Ming3 requires an available CUDA device")
        require_model_consent(MING3_TTS_MODEL_ID)
        self._ensure_runtime()
        self._download_model()

    def inference_metadata(self, model: MingRuntime) -> dict[str, str]:
        """Describe Ming's loaded audio.cpp model and GGUF precision."""
        quantization = model.precision.upper()
        if quantization == "BF16":
            quantization = "full"
        return {
            "model": self.model_name or self.default_model_id,
            "type": "gguf",
            "quant": quantization,
        }

    def load_model(
        self,
        model_id: str,
        **kwargs: BackendArgumentValue,
    ) -> MingRuntime:
        """Start the CUDA server with Celune's selected GGUF precision."""
        del kwargs
        if model_id != MING3_TTS_MODEL_ID:
            raise ValueError(f"unsupported Ming model identifier: {model_id}")
        if not torch.cuda.is_available():
            raise RuntimeError("Ming3 requires an available CUDA device")

        require_model_consent(model_id)
        self.quantization_attempted = self.quantization_requested
        model_path = self._download_model()
        server_path = self._ensure_runtime()
        runtime = self._start_server(server_path, model_path)
        self.quantization_active = runtime.precision == "q8_0"
        self.model = runtime
        return runtime

    def resolve_generation_language(self, lang: Optional[str]) -> str:
        """Normalize Ming's language and default unsupported values to English."""
        if not lang:
            return "en"
        normalized = lang.strip().casefold().replace("_", "-")
        language = _LANGUAGE_ALIASES.get(normalized)
        if language is None and "-" in normalized:
            language = _LANGUAGE_ALIASES.get(normalized.split("-", 1)[0])
        return language or "en"

    @staticmethod
    def _maximum_audio_seconds(text: str) -> float:
        """Return a generous duration limit based on utterance length."""
        return min(
            _MAX_AUDIO_SECONDS,
            max(10.0, len(text.strip()) * _MAX_AUDIO_SECONDS_PER_CHARACTER),
        )

    def generate_stream(
        self,
        model: MingRuntime,
        **kwargs: BackendArgumentValue,
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Yield native SSE PCM chunks while Ming is still generating audio."""
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("expected text to say")

        voice = kwargs.pop("voice", self.default_voice)
        if not isinstance(voice, str):
            voice = self.default_voice
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return
        loader, voice_names = compatible_bundle
        if voice not in voice_names:
            raise ValueError(f"unknown voice '{voice}' for backend '{self.name}'")

        reference_path = self._truncate_reference(loader.materialize(voice, "wav"))
        reference_text = loader.bundle.voices[voice].get("reference_text")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError(f"Ming3 requires reference_text for voice '{voice}'")

        language_value = kwargs.pop("language", None)
        if language_value is None:
            language_value = kwargs.pop("lang", None)
        else:
            kwargs.pop("lang", None)
        if language_value is not None and not isinstance(language_value, str):
            raise TypeError("language must be a string")
        requested_language = language_value
        language = self.resolve_generation_language(requested_language)
        language_code = "zh" if language == "zh-cn" else language

        self._apply_seed()
        max_new_tokens = _integer_option(
            kwargs.pop("max_new_tokens", self.max_new_tokens), "max_new_tokens"
        )
        options: dict[str, object] = {
            "language": language_code,
            "max_tokens": min(max_new_tokens, _MAX_TOKENS),
            "seed": self.current_seed or 0,
        }
        for name in ("temperature", "sigma", "guidance_scale"):
            value = kwargs.pop(name, None)
            if value is not None:
                options[name] = float(_numeric_option(value, name))
        if "guidance_scale" not in options:
            cfg_scale = kwargs.pop("cfg_scale", None)
            if cfg_scale is not None:
                options["guidance_scale"] = float(
                    _numeric_option(cfg_scale, "cfg_scale")
                )
        else:
            kwargs.pop("cfg_scale", None)
        instruction = kwargs.pop("instruct", kwargs.pop("instructions", None))
        if isinstance(instruction, str) and instruction.strip():
            options["instruct"] = instruction.strip()

        request_body: dict[str, object] = {
            "model": "ming3",
            "input": text,
            "voice_ref": reference_path.resolve().as_posix(),
            "reference_text": reference_text.strip(),
            "response_format": "pcm",
            "stream_format": "sse",
            "options": options,
        }
        yield from model.stream_speech(request_body, default_sample_rate=_SAMPLE_RATE)
