# SPDX-License-Identifier: Apache-2.0
"""Streaming Ming 3.0 TTS backend powered by audio.cpp."""

import os
import sys
import json
import time
import base64
import shutil
import socket
import hashlib
import tarfile
import zipfile
import platform
import tempfile
import sysconfig
import threading
import contextlib
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass, field
from collections.abc import Callable, Iterator

import torch
import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from ...paths import (
    app_data_dir,
    huggingface_hub_cache_dir,
    huggingface_progress,
    temp_data_dir,
)
from ...i18n import string
from ...registry.tts import (
    MING3_TTS_BF16_FILE,
    MING3_TTS_BF16_SHA256,
    MING3_TTS_BF16_SIZE,
    MING3_TTS_GGUF_REPO,
    MING3_TTS_GGUF_REVISION,
    MING3_TTS_MODEL_ID,
    MING3_TTS_Q8_FILE,
    MING3_TTS_Q8_SHA256,
    MING3_TTS_Q8_SIZE,
    ModelContract,
    model_contract as resolve_model_contract,
    TTS_DEFAULT_VOICE,
)
from .base import CeluneBackend
from ...typing.aliases import AudioChunk
from ...typing.backends import BackendArgumentValue
from ...cevoice import CEVoiceLoader, default_loader
from ...registry.consent import require_model_consent

_AUDIOCPP_RELEASE = "v0.9.3"
_AUDIOCPP_RELEASE_URL = (
    "https://github.com/0xShug0/audio.cpp/releases/download/" + _AUDIOCPP_RELEASE
)
_WINDOWS_RUNTIME = (
    (
        "audio-v0.9.3-bin-windows-x64-cuda13.3.zip",
        "0bd3988be10c31ec078c40df2bd0d4266e660804f6d5083aeb05d1309d091e43",
    ),
    (
        "audio-v0.9.3-cudart-windows-x64-cuda13.3.zip",
        "839eb20f684a8aab64895b1e466b32ce9b29dfc19dc1ce15b2ef35218512ac0a",
    ),
)
_LINUX_RUNTIME = (
    "audio-v0.9.3-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
    "5be3e8d515fb244387de87958668ebb729d9442de6a53df12e4c718ea918419d",
)
_SAMPLE_RATE = 44_100
_MAX_TOKENS = 512
_MAX_AUDIO_SECONDS = 120.0
_MIN_AUDIO_SECONDS = 0.1
_MAX_AUDIO_SECONDS_PER_CHARACTER = 0.75
_SERVER_START_TIMEOUT_SECONDS = 600.0
_SERVER_REQUEST_TIMEOUT_SECONDS = 300.0

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


@dataclass(slots=True)
class MingRuntime:
    """Own one audio.cpp server and its streaming HTTP endpoint."""

    process: subprocess.Popen[bytes]
    base_url: str
    log_path: Path
    model_path: Path
    precision: str
    request_lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        """Stop the server and release the model's CUDA allocations."""
        if self.process.poll() is not None:
            return

        with contextlib.suppress(OSError):
            self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self.process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=5)


class Ming3(CeluneBackend[MingRuntime]):
    """Celune's native-streaming Ming 3.0 voice-cloning backend."""

    name: str = "ming3"
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
        del kwargs
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
        del lang
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
        """Select the pinned CUDA runtime archive for the current platform."""
        machine = platform.machine().casefold()
        if not machine:
            machine = sysconfig.get_platform().rsplit("-", 1)[-1].casefold()
        if machine not in {"amd64", "x86_64"}:
            raise RuntimeError(
                f"Ming's audio.cpp runtime requires x86_64, got {machine}"
            )
        if sys.platform == "win32":
            return _WINDOWS_RUNTIME
        if sys.platform == "linux":
            return (_LINUX_RUNTIME,)
        raise RuntimeError(f"Ming's CUDA runtime is unavailable on {sys.platform}")

    @staticmethod
    def _remove_owned_tree(path: Path, root: Path) -> None:
        """Remove a runtime directory only when it resolves inside its cache root."""
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
            raise RuntimeError(
                f"refusing to remove a path outside the Ming runtime cache: {path}"
            )
        shutil.rmtree(resolved_path, ignore_errors=True)

    @staticmethod
    def _archive_sha256(path: Path) -> str:
        """Hash a downloaded runtime archive before extraction."""
        with path.open("rb") as archive:
            return hashlib.file_digest(archive, "sha256").hexdigest()

    def _download_runtime_archive(self, path: Path, filename: str, digest: str) -> None:
        """Download and verify one pinned audio.cpp release archive."""
        if path.is_file() and self._archive_sha256(path) == digest:
            return
        path.unlink(missing_ok=True)
        partial_path = path.with_suffix(path.suffix + ".part")
        partial_path.unlink(missing_ok=True)
        url = f"{_AUDIOCPP_RELEASE_URL}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "Celune"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content_length = response.headers.get("Content-Length")
                total = int(content_length) if content_length else None
                received = 0
                with partial_path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        self.report_progress(received, total)
            if self._archive_sha256(partial_path) != digest:
                raise RuntimeError(
                    f"audio.cpp release archive checksum mismatch: {filename}"
                )
            partial_path.replace(path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _extract_runtime_archive(archive_path: Path, destination: Path) -> None:
        """Safely unpack the pinned Windows ZIP or Linux tarball."""
        if archive_path.suffix == ".zip":
            root = destination.resolve()
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise RuntimeError(
                            f"audio.cpp archive contains an unsafe path: {member.filename}"
                        )
                archive.extractall(destination)
            return

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination, filter="data")

    @staticmethod
    def _find_server(root: Path) -> Path:
        """Find the server executable inside an extracted release archive."""
        executable = (
            "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"
        )
        matches = tuple(root.rglob(executable))
        if len(matches) != 1 or not matches[0].is_file():
            raise RuntimeError(
                f"audio.cpp {_AUDIOCPP_RELEASE} archive did not contain one {executable}"
            )
        return matches[0]

    def _ensure_runtime(self) -> Path:
        """Install the pinned CUDA server in Celune's persistent user data."""
        assets = self._runtime_assets()
        filename, _ = assets[0]
        cache_root = app_data_dir(create=True) / "backends" / "ming3"
        cache_root.mkdir(parents=True, exist_ok=True)
        if filename.endswith(".zip"):
            runtime_name = filename.removesuffix(".zip")
        else:
            runtime_name = filename.removesuffix(".tar.gz")
        runtime_root = cache_root / runtime_name
        marker = runtime_root / ".celune-ready"
        digest = ":".join(asset_digest for _, asset_digest in assets)

        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
            return self._find_server(runtime_root)

        if runtime_root.exists():
            self._remove_owned_tree(runtime_root, cache_root)

        self.log(string("tts.ming3_runtime_downloading"), "info")
        self.report_progress(None, None)
        archive_paths = []
        for asset_name, asset_digest in assets:
            archive_path = cache_root / asset_name
            self._download_runtime_archive(archive_path, asset_name, asset_digest)
            archive_paths.append(archive_path)
        staging = Path(tempfile.mkdtemp(prefix="ming3-runtime-", dir=cache_root))
        try:
            for archive_path in archive_paths:
                self._extract_runtime_archive(archive_path, staging)
            self._find_server(staging)
            if sys.platform == "linux":
                for executable in staging.rglob("audiocpp_server"):
                    executable.chmod(executable.stat().st_mode | 0o111)
            (staging / ".celune-ready").write_text(digest, encoding="utf-8")
            os.replace(staging, runtime_root)
            for archive_path in archive_paths:
                archive_path.unlink(missing_ok=True)
        except Exception:
            self._remove_owned_tree(staging, cache_root)
            raise
        return self._find_server(runtime_root)

    @staticmethod
    def _native_environment(server_path: Path) -> dict[str, str]:
        """Expose PyTorch's CUDA runtime libraries to audio.cpp's child process."""
        environment = os.environ.copy()
        torch_root = Path(torch.__file__).resolve().parent
        site_packages = torch_root.parent
        library_paths = [server_path.parent, torch_root / "lib"]
        if sys.platform == "win32":
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                library_paths.append(Path(cuda_path) / "bin")
            package_library = "bin"
        else:
            package_library = "lib"
            library_paths.append(Path("/usr/local/cuda/lib64"))

        for package in (site_packages / "nvidia").glob("*"):
            candidate = package / package_library
            if candidate.is_dir():
                library_paths.append(candidate)

        variable = "PATH" if sys.platform == "win32" else "LD_LIBRARY_PATH"
        existing = environment.get(variable)
        paths = [str(path) for path in dict.fromkeys(library_paths) if path.is_dir()]
        if existing:
            paths.append(existing)
        environment[variable] = os.pathsep.join(paths)
        return environment

    @staticmethod
    def _server_json(base_url: str, endpoint: str, timeout: float) -> dict[str, object]:
        """Return one JSON response from the loopback server."""
        with urllib.request.urlopen(
            f"{base_url}{endpoint}", timeout=timeout
        ) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise TypeError(f"audio.cpp returned invalid JSON from {endpoint}")
        return payload

    @staticmethod
    def _server_log_tail(path: Path) -> str:
        """Return the most recent audio.cpp diagnostics for a failed operation."""
        if not path.is_file():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        )

    def _start_server(self, server_path: Path, model_path: Path) -> MingRuntime:
        """Start audio.cpp eagerly and wait until its Ming model is loaded."""
        with socket.socket() as candidate_socket:
            candidate_socket.bind(("127.0.0.1", 0))
            port = candidate_socket.getsockname()[1]

        base_url = f"http://127.0.0.1:{port}"
        temp_dir = temp_data_dir(create=True)
        config_path = temp_dir / f"ming3-server-{os.getpid()}-{port}.json"
        log_path = temp_dir / f"ming3-server-{os.getpid()}-{port}.log"
        config = {
            "host": "127.0.0.1",
            "port": port,
            "backend": "cuda",
            "device": torch.cuda.current_device(),
            "threads": min(os.cpu_count() or 1, 4),
            "lazy_load": False,
            "max_loaded_models": 1,
            "models": [
                {
                    "id": "ming3",
                    "family": "ming3_tts",
                    "path": model_path.as_posix(),
                    "task": "clon",
                    "mode": "streaming",
                    "lazy": False,
                }
            ],
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.log(string("tts.ming3_runtime_starting"), "info")
        self.report_progress(None, None)

        with log_path.open("ab") as log_file:
            process = subprocess.Popen(
                [
                    str(server_path),
                    "--config",
                    str(config_path),
                    "--no-ui",
                    "--log",
                    "--log-file",
                    str(log_path),
                ],
                cwd=str(server_path.parent),
                env=self._native_environment(server_path),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )

        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        try:
            model_seen = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        "audio.cpp exited before Ming became ready "
                        f"(exit code {process.returncode}):\n{self._server_log_tail(log_path)}"
                    )
                try:
                    health = self._server_json(base_url, "/health", timeout=2.0)
                    if health.get("status") != "ok":
                        time.sleep(0.25)
                        continue
                    models = self._server_json(base_url, "/v1/models", timeout=2.0)
                    entries = models.get("data", models.get("models", []))
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict) and entry.get("id") == "ming3":
                                model_seen = True
                                if entry.get("loaded") is True:
                                    return MingRuntime(
                                        process=process,
                                        base_url=base_url,
                                        log_path=log_path,
                                        model_path=model_path,
                                        precision=self.package_precision,
                                    )
                                break
                except (
                    OSError,
                    urllib.error.URLError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    pass
                time.sleep(0.25)

            raise TimeoutError(
                "audio.cpp did not load Ming within "
                f"{_SERVER_START_TIMEOUT_SECONDS:.0f} seconds "
                f"(model listed: {model_seen}):\n{self._server_log_tail(log_path)}"
            )
        except Exception:
            with contextlib.suppress(OSError):
                process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
            raise
        finally:
            config_path.unlink(missing_ok=True)

    def preload_models(self) -> None:
        """Download the selected GGUF and the pinned audio.cpp runtime."""
        if not torch.cuda.is_available():
            raise RuntimeError("Ming3 requires an available CUDA device")
        require_model_consent(MING3_TTS_MODEL_ID)
        self._download_model()
        self._ensure_runtime()

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

        requested_language = kwargs.pop("language", None)
        if requested_language is None:
            requested_language = kwargs.pop("lang", None)
        else:
            kwargs.pop("lang", None)
        language = self.resolve_generation_language(requested_language)
        language_code = "zh" if language == "zh-cn" else language
        max_audio_seconds = self._maximum_audio_seconds(text)

        self._apply_seed()
        options: dict[str, Union[str, int, float]] = {
            "language": language_code,
            "max_tokens": min(
                int(kwargs.pop("max_new_tokens", self.max_new_tokens)), _MAX_TOKENS
            ),
            "seed": int(self.current_seed or 0),
        }
        for name in ("temperature", "sigma", "guidance_scale"):
            value = kwargs.pop(name, None)
            if value is not None:
                options[name] = float(value)
        if "guidance_scale" not in options:
            cfg_scale = kwargs.pop("cfg_scale", None)
            if cfg_scale is not None:
                options["guidance_scale"] = float(cfg_scale)
        else:
            kwargs.pop("cfg_scale", None)
        instruction = kwargs.pop("instruct", kwargs.pop("instructions", None))
        if isinstance(instruction, str) and instruction.strip():
            options["instruct"] = instruction.strip()

        request_body = {
            "model": "ming3",
            "input": text,
            "voice_ref": reference_path.resolve().as_posix(),
            "reference_text": reference_text.strip(),
            "response_format": "pcm",
            "stream_format": "sse",
            "options": options,
        }
        request = urllib.request.Request(
            f"{model.base_url}/v1/audio/speech",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with model.request_lock:
            if model.process.poll() is not None:
                raise RuntimeError(
                    "audio.cpp Ming server is no longer running:\n"
                    + self._server_log_tail(model.log_path)
                )
            total_samples = 0
            received_audio = False
            stream_finished = False
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=_SERVER_REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    for line in response:
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            stream_finished = True
                            break
                        if not data:
                            continue
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("type")
                        if event_type == "error" or "error" in event:
                            raise RuntimeError(
                                f"audio.cpp Ming generation failed: {event.get('error', event)}"
                            )
                        if event_type != "speech.audio.delta":
                            continue
                        encoded_audio = event.get("audio")
                        if not isinstance(encoded_audio, str):
                            raise TypeError("audio.cpp returned an invalid PCM event")
                        pcm = base64.b64decode(encoded_audio, validate=True)
                        if len(pcm) % 2:
                            raise RuntimeError(
                                "audio.cpp returned an incomplete PCM sample"
                            )
                        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
                        audio *= 1.0 / 32768.0
                        if audio.size == 0:
                            continue
                        total_samples += int(audio.size)
                        if total_samples / _SAMPLE_RATE > max_audio_seconds:
                            raise RuntimeError(
                                "Ming generated implausibly long audio for the request "
                                f"({max_audio_seconds:.1f} second limit)"
                            )
                        received_audio = True
                        yield audio, _SAMPLE_RATE, None
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"audio.cpp rejected the Ming request ({error.code}): {details}"
                ) from error

            if not stream_finished:
                raise RuntimeError(
                    "audio.cpp Ming stream ended before its completion event"
                )
            if not received_audio:
                raise RuntimeError("Ming returned empty audio")
            duration = total_samples / _SAMPLE_RATE
            if duration < _MIN_AUDIO_SECONDS:
                raise RuntimeError(
                    f"Ming returned implausibly short audio ({duration:.3f} seconds)"
                )
