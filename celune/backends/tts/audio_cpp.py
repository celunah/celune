# SPDX-License-Identifier: Apache-2.0
"""Shared process, download, and streaming support for audio.cpp TTS adapters."""

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
from dataclasses import field, dataclass
from collections.abc import Callable, Iterator
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download

from ...i18n import string
from ...paths import (
    app_data_dir,
    huggingface_hub_cache_dir,
    huggingface_progress,
    temp_data_dir,
    temp_directory_environment,
)
from ...typing.aliases import AudioChunk
from .audio_cpp_models import (
    AUDIOCPP_GGUF_REPO,
    AUDIOCPP_GGUF_REVISION,
    AudioCppModelArtifact,
)

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
    (
        "audio-v0.9.3-bin-ubuntu-x64-cuda12.8-colab.tar.gz",
        "5be3e8d515fb244387de87958668ebb729d9442de6a53df12e4c718ea918419d",
    ),
)
_SERVER_START_TIMEOUT_SECONDS = 600.0
_SERVER_REQUEST_TIMEOUT_SECONDS = 300.0
_MAX_AUDIO_SECONDS = 120.0
_MAX_AUDIO_SECONDS_PER_CHARACTER = 0.75
_MIN_AUDIO_SECONDS = 0.1
_VERIFIED_GGUF_ARTIFACTS: set[tuple[str, int, int, str]] = set()
_VERIFIED_GGUF_LOCK = threading.Lock()


@dataclass(slots=True)
class AudioCppRuntime:
    """Own one pinned audio.cpp server and its active GGUF model."""

    process: subprocess.Popen[bytes]
    base_url: str
    log_path: Path
    model_path: Path
    precision: str
    default_sample_rate: int = 44_100
    server_model_id: str = "celune"
    request_lock: threading.Lock = field(default_factory=threading.Lock)

    def close(self) -> None:
        """Stop the server process and release the model's accelerator memory."""
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

    def stream_speech(
        self,
        request_body: dict[str, object],
        *,
        default_sample_rate: Optional[int] = None,
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Yield PCM deltas from one audio.cpp server-sent event stream."""
        text = request_body.get("input")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("expected text to say")
        sample_rate_default = default_sample_rate or self.default_sample_rate
        maximum_audio_seconds = min(
            _MAX_AUDIO_SECONDS,
            max(10.0, len(text.strip()) * _MAX_AUDIO_SECONDS_PER_CHARACTER),
        )
        request = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with self.request_lock:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "audio.cpp server is no longer running:\n"
                    + server_log_tail(self.log_path)
                )
            total_duration = 0.0
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
                                "audio.cpp generation failed: "
                                f"{event.get('error', event)}"
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
                        event_sample_rate = event.get("sample_rate")
                        sample_rate = (
                            event_sample_rate
                            if isinstance(event_sample_rate, int)
                            and not isinstance(event_sample_rate, bool)
                            and event_sample_rate > 0
                            else sample_rate_default
                        )
                        total_duration += audio.size / sample_rate
                        if total_duration > maximum_audio_seconds:
                            raise RuntimeError(
                                "audio.cpp generated implausibly long audio for the "
                                f"request ({maximum_audio_seconds:.1f} second limit)"
                            )
                        received_audio = True
                        yield audio, sample_rate, None
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"audio.cpp rejected the request ({error.code}): {details}"
                ) from error

            if not stream_finished:
                raise RuntimeError("audio.cpp stream ended before its completion event")
            if not received_audio:
                raise RuntimeError("audio.cpp returned empty audio")
            if total_duration < _MIN_AUDIO_SECONDS:
                raise RuntimeError(
                    "audio.cpp returned implausibly short audio "
                    f"({total_duration:.3f} seconds)"
                )


def runtime_assets() -> tuple[tuple[str, str], ...]:
    """Select the pinned audio.cpp CUDA server archives for this platform."""
    machine = platform.machine().casefold()
    if not machine:
        machine = sysconfig.get_platform().rsplit("-", 1)[-1].casefold()
    if machine not in {"amd64", "x86_64"}:
        raise RuntimeError(f"audio.cpp requires x86_64, got {machine}")
    if sys.platform == "win32":
        return _WINDOWS_RUNTIME
    if sys.platform == "linux":
        return _LINUX_RUNTIME
    raise RuntimeError(f"audio.cpp CUDA runtime is unavailable on {sys.platform}")


def ensure_audio_cpp_server(
    *,
    report_progress: Callable[[Optional[float], Optional[float]], None],
    legacy_cache_root: Optional[Path] = None,
    log: Optional[Callable[[str, str], None]] = None,
    backend_name: str = "TTS",
) -> Path:
    """Download, verify, and extract the pinned audio.cpp server on demand."""
    assets = runtime_assets()
    filename = assets[0][0]
    runtime_name = filename.removesuffix(".zip").removesuffix(".tar.gz")
    cache_root = app_data_dir(create=True) / "backends" / "audio_cpp"
    cache_root.mkdir(parents=True, exist_ok=True)
    runtime_root = cache_root / runtime_name
    marker = runtime_root / ".celune-ready"
    digest = ":".join(asset_digest for _, asset_digest in assets)

    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
        return find_server(runtime_root)

    if legacy_cache_root is not None:
        legacy_root = legacy_cache_root / runtime_name
        legacy_marker = legacy_root / ".celune-ready"
        if (
            legacy_marker.is_file()
            and legacy_marker.read_text(encoding="utf-8").strip() == digest
        ):
            return find_server(legacy_root)

    if runtime_root.exists():
        remove_owned_tree(runtime_root, cache_root)

    if log is not None:
        log(
            string("inference.audio_cpp_runtime_downloading", backend=backend_name),
            "info",
        )
    report_progress(None, None)
    archive_paths: list[Path] = []
    for asset_name, asset_digest in assets:
        archive_path = cache_root / asset_name
        download_runtime_archive(
            archive_path,
            asset_name,
            asset_digest,
            report_progress=report_progress,
        )
        archive_paths.append(archive_path)

    staging = Path(tempfile.mkdtemp(prefix="audio-cpp-runtime-", dir=cache_root))
    try:
        for archive_path in archive_paths:
            extract_runtime_archive(archive_path, staging)
        find_server(staging)
        if sys.platform == "linux":
            for executable in staging.rglob("audiocpp_server"):
                executable.chmod(executable.stat().st_mode | 0o111)
        (staging / ".celune-ready").write_text(digest, encoding="utf-8")
        os.replace(staging, runtime_root)
        for archive_path in archive_paths:
            archive_path.unlink(missing_ok=True)
    except Exception:
        remove_owned_tree(staging, cache_root)
        raise
    return find_server(runtime_root)


def load_audio_cpp_tts_model(
    artifact: AudioCppModelArtifact,
    *,
    backend_id: str,
    sample_rate: int,
    torch_root: Path,
    device: int,
    log: Callable[[str, str], None],
    report_progress: Callable[[Optional[float], Optional[float]], None],
    legacy_cache_root: Optional[Path] = None,
    task: str = "tts",
    load_options: Optional[dict[str, object]] = None,
    session_options: Optional[dict[str, object]] = None,
    default_request_options: Optional[dict[str, object]] = None,
) -> AudioCppRuntime:
    """Download a verified GGUF and eagerly load it in the shared server."""
    server_path = ensure_audio_cpp_server(
        report_progress=report_progress,
        legacy_cache_root=legacy_cache_root,
        log=log,
        backend_name=backend_id,
    )
    downloaded_path = download_verified_gguf(
        artifact.filename,
        size=artifact.size,
        sha256=artifact.sha256,
        report_progress=report_progress,
    )
    model_path = materialize_audio_cpp_gguf(downloaded_path, artifact, backend_id)
    server_model_id = f"celune-{backend_id}"
    model_config: dict[str, object] = {
        "id": server_model_id,
        "family": artifact.family,
        "path": model_path.as_posix(),
        "task": task,
        "mode": "streaming",
        "lazy": False,
    }
    if load_options:
        model_config["load_options"] = load_options
    if session_options:
        model_config["session_options"] = session_options
    if default_request_options:
        model_config["default_request_options"] = default_request_options
    return start_audio_cpp_server(
        server_path,
        model_path=model_path,
        precision=artifact.quantization,
        sample_rate=sample_rate,
        model_config=model_config,
        torch_root=torch_root,
        device=device,
        backend_name=backend_id,
        log=log,
        report_progress=report_progress,
    )


def download_verified_gguf(
    filename: str,
    *,
    size: int,
    sha256: str,
    report_progress: Callable[[Optional[float], Optional[float]], None],
) -> Path:
    """Download the pinned package file and verify its SHA-256 and size."""
    cache_dir = huggingface_hub_cache_dir(create=True)
    with huggingface_progress(report_progress):
        path = Path(
            hf_hub_download(
                repo_id=AUDIOCPP_GGUF_REPO,
                filename=filename,
                revision=AUDIOCPP_GGUF_REVISION,
                cache_dir=str(cache_dir),
                force_download=False,
            )
        )
    if not _verified_artifact(path, size=size, sha256=sha256):
        with huggingface_progress(report_progress):
            path = Path(
                hf_hub_download(
                    repo_id=AUDIOCPP_GGUF_REPO,
                    filename=filename,
                    revision=AUDIOCPP_GGUF_REVISION,
                    cache_dir=str(cache_dir),
                    force_download=True,
                )
            )
    if not _verified_artifact(path, size=size, sha256=sha256):
        raise RuntimeError(f"audio.cpp GGUF integrity check failed: {filename}")
    return path


def materialize_audio_cpp_gguf(
    cached_path: Path,
    artifact: AudioCppModelArtifact,
    backend_id: str,
) -> Path:
    """Expose one verified GGUF at a stable path that keeps its file extension."""
    source = cached_path.resolve(strict=True)
    if not _verified_artifact(
        source,
        size=artifact.size,
        sha256=artifact.sha256,
    ):
        raise RuntimeError(
            f"audio.cpp GGUF integrity check failed: {artifact.filename}"
        )

    relative_path = Path(artifact.filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError("audio.cpp GGUF filename contains an unsafe path")
    runtime_root = (
        app_data_dir(create=True)
        / "backends"
        / "audio_cpp"
        / "models"
        / backend_id
        / AUDIOCPP_GGUF_REVISION
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    target = runtime_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    root_resolved = runtime_root.resolve()
    parent_resolved = target.parent.resolve()
    if (
        parent_resolved != root_resolved
        and root_resolved not in parent_resolved.parents
    ):
        raise RuntimeError("audio.cpp GGUF cache path escapes its runtime directory")
    if target.is_symlink():
        raise RuntimeError("audio.cpp GGUF runtime path cannot be a symbolic link")
    if target.is_file() and _verified_artifact(
        target,
        size=artifact.size,
        sha256=artifact.sha256,
    ):
        return target
    if target.exists() and not target.is_file():
        raise RuntimeError("audio.cpp GGUF runtime path is not a regular file")
    target.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=target.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    temporary_path.unlink()
    try:
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copyfile(source, temporary_path)
        if not _verified_artifact(
            temporary_path,
            size=artifact.size,
            sha256=artifact.sha256,
        ):
            raise RuntimeError(
                f"audio.cpp GGUF materialization failed verification: {artifact.filename}"
            )
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _verified_artifact(path: Path, *, size: int, sha256: str) -> bool:
    """Check one candidate file against its pinned package metadata."""
    if not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != size:
        return False
    identity = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, sha256)
    with _VERIFIED_GGUF_LOCK:
        if identity in _VERIFIED_GGUF_ARTIFACTS:
            return True
    if archive_sha256(path) != sha256:
        return False
    with _VERIFIED_GGUF_LOCK:
        _VERIFIED_GGUF_ARTIFACTS.add(identity)
    return True


def download_runtime_archive(
    path: Path,
    filename: str,
    digest: str,
    *,
    report_progress: Callable[[Optional[float], Optional[float]], None],
) -> None:
    """Download and verify one pinned audio.cpp release archive."""
    if path.is_file() and archive_sha256(path) == digest:
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
                    report_progress(received, total)
        if archive_sha256(partial_path) != digest:
            raise RuntimeError(
                f"audio.cpp release archive checksum mismatch: {filename}"
            )
        partial_path.replace(path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def extract_runtime_archive(archive_path: Path, destination: Path) -> None:
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


def find_server(root: Path) -> Path:
    """Find the pinned server executable inside an extracted release archive."""
    executable = "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"
    matches = tuple(root.rglob(executable))
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(
            f"audio.cpp {_AUDIOCPP_RELEASE} archive did not contain one {executable}"
        )
    return matches[0]


def native_environment(server_path: Path, torch_root: Path) -> dict[str, str]:
    """Expose the worker's CUDA runtime libraries to the audio.cpp child."""
    environment = os.environ.copy()
    environment.update(temp_directory_environment())
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


def server_json(base_url: str, endpoint: str, timeout: float) -> dict[str, object]:
    """Return one JSON response from the loopback server."""
    with urllib.request.urlopen(f"{base_url}{endpoint}", timeout=timeout) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise TypeError(f"audio.cpp returned invalid JSON from {endpoint}")
    return payload


def server_log_tail(path: Path) -> str:
    """Return the most recent audio.cpp diagnostics for a failed operation."""
    if not path.is_file():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    )


def start_audio_cpp_server(
    server_path: Path,
    *,
    model_path: Path,
    precision: str,
    sample_rate: int,
    model_config: dict[str, object],
    torch_root: Path,
    device: int,
    backend_name: str,
    log: Callable[[str, str], None],
    report_progress: Callable[[Optional[float], Optional[float]], None],
) -> AudioCppRuntime:
    """Start audio.cpp with one eager model and wait for model readiness."""
    with socket.socket() as candidate_socket:
        candidate_socket.bind(("127.0.0.1", 0))
        port = candidate_socket.getsockname()[1]

    base_url = f"http://127.0.0.1:{port}"
    temp_dir = temp_data_dir(create=True)
    config_path = temp_dir / f"audio-cpp-{os.getpid()}-{port}.json"
    log_path = temp_dir / f"audio-cpp-{os.getpid()}-{port}.log"
    config = {
        "host": "127.0.0.1",
        "port": port,
        "backend": "cuda",
        "device": device,
        "threads": min(os.cpu_count() or 1, 4),
        "lazy_load": False,
        "max_loaded_models": 1,
        "models": [model_config],
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    log(string("inference.audio_cpp_runtime_starting", backend=backend_name), "info")
    report_progress(None, None)

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
            env=native_environment(server_path, torch_root),
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
                    "audio.cpp exited before the model became ready "
                    f"(exit code {process.returncode}):\n{server_log_tail(log_path)}"
                )
            try:
                health = server_json(base_url, "/health", timeout=2.0)
                if health.get("status") != "ok":
                    time.sleep(0.25)
                    continue
                models = server_json(base_url, "/v1/models", timeout=2.0)
                entries = models.get("data", models.get("models", []))
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get(
                            "id"
                        ) == model_config.get("id"):
                            model_seen = True
                            if entry.get("loaded"):
                                return AudioCppRuntime(
                                    process=process,
                                    base_url=base_url,
                                    log_path=log_path,
                                    model_path=model_path,
                                    precision=precision,
                                    default_sample_rate=sample_rate,
                                    server_model_id=str(
                                        model_config.get("id", "celune")
                                    ),
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
            "audio.cpp did not load the model within "
            f"{_SERVER_START_TIMEOUT_SECONDS:.0f} seconds "
            f"(model listed: {model_seen}):\n{server_log_tail(log_path)}"
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


def remove_owned_tree(path: Path, root: Path) -> None:
    """Remove an audio.cpp cache tree only when it resolves inside its root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(
            f"refusing to remove a path outside the audio.cpp cache: {path}"
        )
    shutil.rmtree(resolved_path, ignore_errors=True)


def archive_sha256(path: Path) -> str:
    """Hash a downloaded GGUF or runtime archive with SHA-256."""
    with path.open("rb") as archive:
        return hashlib.file_digest(archive, "sha256").hexdigest()
