# SPDX-License-Identifier: Apache-2.0
"""Pinned llama.cpp runtime and verified Persona GGUF packages."""

import os
import sys
import json
import time
import base64
import hashlib
import http.client
import mimetypes
import platform
import shutil
import socket
import stat
import tarfile
import tempfile
import threading
import subprocess
import urllib.request
import zipfile
import contextlib
from pathlib import Path
from collections import abc, deque
from typing import Optional
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

from ..paths import app_data_dir, huggingface_hub_cache_dir
from ..dataclasses.persona import GenerateRequest, GenerateResponse

LLAMA_CPP_RELEASE = "b12110"
LLAMA_CPP_COMMIT = "cd959bae99ff4bed22bef2604f45241c09bbb8c8"
QWEN3_VL_GGUF_REPO = "Qwen/Qwen3-VL-4B-Instruct-GGUF"
QWEN3_VL_GGUF_REVISION = "1cd86afb9a95c410a6038ab3b40d8b578c892266"
_QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
_VERIFIED_ARTIFACTS: set[tuple[str, int, int, str]] = set()
_VERIFIED_ARTIFACTS_LOCK = threading.Lock()
_MAX_SERVER_LOG_LINES = 200
_SERVER_START_TIMEOUT_SECONDS = 120.0
_HTTP_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class PersonaGGUFVariant:
    """Describe one exact language-model and vision-projector pair."""

    request_quantization: str
    quantization: str
    model_file: str
    model_size: int
    model_sha256: str
    projector_file: str
    projector_size: int
    projector_sha256: str


QWEN3_VL_GGUF_VARIANTS = {
    "4bit": PersonaGGUFVariant(
        request_quantization="4bit",
        quantization="Q4_K_M",
        model_file="Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        model_size=2_497_281_664,
        model_sha256="66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a",
        projector_file="mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        projector_size=453_974_304,
        projector_sha256="30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
    ),
    "8bit": PersonaGGUFVariant(
        request_quantization="8bit",
        quantization="Q8_0",
        model_file="Qwen3VL-4B-Instruct-Q8_0.gguf",
        model_size=4_280_406_144,
        model_sha256="054721f478bc5fa6beffb7f38eae575d45298f88cbb8d2f83ef675a727863eb1",
        projector_file="mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
        projector_size=453_974_304,
        projector_sha256="30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d",
    ),
}


@dataclass(frozen=True, slots=True)
class _RuntimeArchive:
    """Pinned llama.cpp binary archive for one supported platform."""

    filename: str
    size: int
    sha256: str


_RUNTIME_ARCHIVES = {
    ("win32", "amd64"): (
        _RuntimeArchive(
            filename="llama-b12110-bin-win-cuda-12.4-x64.zip",
            size=287_014_657,
            sha256="1ce755bc3a061889234048f88e33328e17971eaad73c4666cfeb26026b317fee",
        ),
        _RuntimeArchive(
            filename="cudart-llama-bin-win-cuda-12.4-x64.zip",
            size=391_443_627,
            sha256="8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        ),
    ),
    ("linux", "x86_64"): (
        _RuntimeArchive(
            filename="llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
            size=187_314_621,
            sha256="4af1e1da23661db4a5b84b29aae16ebc053df689c5b8b1427c4670f2c6406d8c",
        ),
        _RuntimeArchive(
            filename="cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz",
            size=594_371_477,
            sha256="6f1d562086d84ae85df880d3711079ec225602b51d530608e5001600625067dd",
        ),
    ),
}


def persona_gguf_variant(
    model_id: str,
    quantization: str,
) -> Optional[PersonaGGUFVariant]:
    """Return a verified GGUF package for the exact selected Persona model."""
    if model_id != _QWEN3_VL_MODEL_ID:
        return None
    normalized = quantization.strip().casefold()
    if normalized in {"4bit", "nf4", "bnb4", "bitsandbytes-4bit"}:
        return QWEN3_VL_GGUF_VARIANTS["4bit"]
    if normalized in {"8bit", "bnb8", "bitsandbytes-8bit"}:
        return QWEN3_VL_GGUF_VARIANTS["8bit"]
    return None


def llama_cpp_archives() -> tuple[_RuntimeArchive, ...]:
    """Return pinned llama.cpp binaries and CUDA libraries for this platform."""
    machine = platform.machine().casefold()
    if machine in {"x86_64", "amd64"}:
        machine = "amd64" if sys.platform == "win32" else "x86_64"
    archives = _RUNTIME_ARCHIVES.get((sys.platform, machine))
    if archives is None:
        raise RuntimeError(
            f"llama.cpp {LLAMA_CPP_RELEASE} has no pinned CUDA build for "
            f"{sys.platform}/{machine}"
        )
    return archives


def _sha256(path: Path) -> str:
    """Hash one model or runtime file with bounded memory use."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _verify_artifact(path: Path, size: int, digest: str) -> None:
    """Validate one downloaded model artifact against its pinned identity."""
    if not path.is_file() or path.stat().st_size != size:
        raise RuntimeError(f"GGUF artifact has an unexpected size: {path}")
    stat_result = path.stat()
    identity = (
        str(path.resolve()),
        stat_result.st_size,
        stat_result.st_mtime_ns,
        digest,
    )
    with _VERIFIED_ARTIFACTS_LOCK:
        if identity in _VERIFIED_ARTIFACTS:
            return
    if _sha256(path) != digest:
        raise RuntimeError(f"GGUF artifact checksum mismatch: {path}")
    with _VERIFIED_ARTIFACTS_LOCK:
        _VERIFIED_ARTIFACTS.add(identity)


def download_persona_gguf(variant: PersonaGGUFVariant) -> tuple[Path, Path]:
    """Download and verify a pinned Qwen3-VL model/projector pair."""
    from huggingface_hub import hf_hub_download

    cache_dir = str(huggingface_hub_cache_dir(create=True))
    model_path = Path(
        hf_hub_download(
            repo_id=QWEN3_VL_GGUF_REPO,
            filename=variant.model_file,
            revision=QWEN3_VL_GGUF_REVISION,
            cache_dir=cache_dir,
        )
    )
    projector_path = Path(
        hf_hub_download(
            repo_id=QWEN3_VL_GGUF_REPO,
            filename=variant.projector_file,
            revision=QWEN3_VL_GGUF_REVISION,
            cache_dir=cache_dir,
        )
    )
    _verify_artifact(model_path, variant.model_size, variant.model_sha256)
    _verify_artifact(projector_path, variant.projector_size, variant.projector_sha256)
    return model_path, projector_path


def _safe_target(root: Path, member_name: str) -> Path:
    """Resolve one archive member while rejecting traversal and absolute paths."""
    relative = Path(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"llama.cpp archive contains an unsafe path: {member_name}")
    root_resolved = root.resolve()
    target = (root / relative).resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise RuntimeError(f"llama.cpp archive contains an unsafe path: {member_name}")
    return target


def _extract_runtime_archive(archive_path: Path, target: Path) -> None:
    """Extract regular files and directories from the pinned binary archive."""
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            for member in members:
                destination = _safe_target(target, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError("llama.cpp ZIP archive contains a symbolic link")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        return

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            destination = _safe_target(target, member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    "llama.cpp TAR archive contains a link or special file"
                )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("llama.cpp TAR archive contains an unreadable file")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if member.mode & 0o111:
                destination.chmod(destination.stat().st_mode | 0o111)


def _remove_owned_tree(path: Path, root: Path) -> None:
    """Remove an extracted runtime only when it remains under its cache root."""
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(
            f"refusing to remove a path outside the llama.cpp runtime cache: {path}"
        )
    shutil.rmtree(resolved_path, ignore_errors=True)


def ensure_llama_server() -> Path:
    """Download, verify, and cache Celune's pinned llama-server build."""
    archives = llama_cpp_archives()
    runtime_root = (
        app_data_dir(create=True)
        / "backends"
        / "llama_cpp"
        / LLAMA_CPP_RELEASE
        / Path(sys.platform)
    )
    executable_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    marker = runtime_root / ".complete"
    runtime_identity = "\n".join(archive.sha256 for archive in archives)
    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == runtime_identity
    ):
        executables = list(runtime_root.rglob(executable_name))
        if len(executables) == 1 and executables[0].is_file():
            return executables[0]

    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    archive_paths: list[Path] = []
    for archive in archives:
        archive_path = runtime_root.parent / archive.filename
        if not archive_path.is_file() or archive_path.stat().st_size != archive.size:
            archive_path.unlink(missing_ok=True)
            partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
            partial_path.unlink(missing_ok=True)
            request = urllib.request.Request(
                f"https://github.com/ggml-org/llama.cpp/releases/download/"
                f"{LLAMA_CPP_RELEASE}/{archive.filename}",
                headers={"User-Agent": "Celune"},
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial_path.open("wb") as output,
            ):
                shutil.copyfileobj(response, output, 1024 * 1024)
            if (
                partial_path.stat().st_size != archive.size
                or _sha256(partial_path) != archive.sha256
            ):
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "pinned llama.cpp runtime archive failed integrity verification"
                )
            partial_path.replace(archive_path)

        if _sha256(archive_path) != archive.sha256:
            raise RuntimeError("cached llama.cpp runtime archive checksum mismatch")
        archive_paths.append(archive_path)

    staging = Path(tempfile.mkdtemp(prefix="llama-cpp-", dir=runtime_root.parent))
    try:
        for archive_path in archive_paths:
            _extract_runtime_archive(archive_path, staging)
        executables = list(staging.rglob(executable_name))
        if len(executables) != 1:
            raise RuntimeError(
                f"pinned llama.cpp archive must contain one {executable_name}"
            )
        (staging / ".complete").write_text(runtime_identity, encoding="utf-8")
        if runtime_root.exists():
            _remove_owned_tree(runtime_root, runtime_root.parent)
        staging.replace(runtime_root)
    except Exception:
        _remove_owned_tree(staging, runtime_root.parent)
        raise
    executables = list(runtime_root.rglob(executable_name))
    if len(executables) != 1:
        raise RuntimeError(f"cached llama.cpp runtime has no {executable_name}")
    return executables[0]


def _local_image_path(source: str) -> Path:
    """Resolve a Persona attachment URI to a local file path."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2:
            raw_path = raw_path[1:]
        return Path(raw_path)
    if parsed.scheme:
        raise ValueError("GGUF Persona accepts local image attachments only")
    return Path(source)


def _image_data_uri(source: str) -> str:
    """Encode a verified local image as an OpenAI-compatible data URI."""
    path = _local_image_path(source)
    if path.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("GGUF Persona supports JPEG and PNG image attachments")
    if not path.is_file():
        raise ValueError("GGUF Persona image attachment is unavailable on disk")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in {"image/jpeg", "image/png"}:
        raise ValueError("GGUF Persona supports JPEG and PNG image attachments")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class LlamaCppPersonaRuntime:
    """Manage one pinned llama-server process for Persona generation."""

    def __init__(
        self,
        model_id: str,
        variant: PersonaGGUFVariant,
        model_path: Path,
        projector_path: Path,
        context_size: int,
        cancellation_event: threading.Event,
        log: Optional[abc.Callable[[str, str], None]] = None,
        server_path: Optional[Path] = None,
    ) -> None:
        self.model_id = model_id
        self.quantization = variant.quantization
        self._cancellation_event = cancellation_event
        self._log_callback = log
        self._diagnostics: deque[str] = deque(maxlen=_MAX_SERVER_LOG_LINES)
        self._diagnostics_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._active_connection: Optional[http.client.HTTPConnection] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._log_thread: Optional[threading.Thread] = None
        self._closed = False
        self._port = self._available_port()
        server_path = server_path or ensure_llama_server()
        command = [
            str(server_path),
            "--model",
            str(model_path),
            "--mmproj",
            str(projector_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--ctx-size",
            str(max(2048, context_size)),
            "--n-gpu-layers",
            "auto",
            "--jinja",
            "--no-webui",
            "--alias",
            model_id,
        ]
        if sys.platform == "win32":
            self._process = subprocess.Popen(  # pylint: disable=R1732
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            self._process = subprocess.Popen(  # pylint: disable=R1732
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        self._log_thread = threading.Thread(
            target=self._drain_server_log,
            name="celune-llama-server-log",
            daemon=True,
        )
        self._log_thread.start()
        try:
            self._wait_until_ready()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _available_port() -> int:
        """Ask the operating system for an unused loopback TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(("127.0.0.1", 0))
            return int(server_socket.getsockname()[1])

    def _drain_server_log(self) -> None:
        """Drain child output so diagnostics cannot block the server process."""
        process = self._process
        if process is None or process.stdout is None:
            return
        for raw_line in iter(process.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            with self._diagnostics_lock:
                self._diagnostics.append(line)
            callback = self._log_callback
            if callable(callback):
                callback(f"[PERSONA][llama.cpp] {line}", "warning")

    def _diagnostic_tail(self) -> str:
        """Return recent server output for one initialization or API failure."""
        with self._diagnostics_lock:
            return "\n".join(self._diagnostics)

    def _wait_until_ready(self) -> None:
        """Wait until llama-server reports that both model components are loaded."""
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._cancellation_event.is_set():
                self.close()
                raise InterruptedError("Persona initialization was cancelled")
            process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited before loading the Persona model: "
                    + self._diagnostic_tail()
                )
            try:
                connection = http.client.HTTPConnection(
                    "127.0.0.1", self._port, timeout=1.0
                )
                connection.request("GET", "/health")
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.25)
        self.close()
        raise TimeoutError(
            "llama-server did not become ready before the startup timeout: "
            + self._diagnostic_tail()
        )

    def _request_json(
        self,
        endpoint: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Send one local JSON request and preserve direct cancellation errors."""
        if self._cancellation_event.is_set():
            raise InterruptedError("Persona generation was cancelled")
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout=_HTTP_TIMEOUT_SECONDS
        )
        with self._connection_lock:
            self._active_connection = connection
        try:
            encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
            connection.request(
                "POST",
                endpoint,
                body=encoded,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read()
            if self._cancellation_event.is_set():
                raise InterruptedError("Persona generation was cancelled")
            try:
                decoded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("llama-server returned invalid JSON") from error
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(
                    f"llama-server rejected {endpoint} ({response.status}): {decoded}"
                )
            if not isinstance(decoded, dict):
                raise TypeError("llama-server returned a non-object JSON response")
            return decoded
        except OSError as error:
            if self._cancellation_event.is_set():
                raise InterruptedError("Persona generation was cancelled") from error
            raise RuntimeError(
                f"llama-server request failed: {error}\n{self._diagnostic_tail()}"
            ) from error
        finally:
            with self._connection_lock:
                if self._active_connection is connection:
                    self._active_connection = None
            connection.close()

    def _messages(self, request: GenerateRequest) -> list[dict[str, object]]:
        """Convert user conversation history to the llama.cpp chat shape."""
        messages = request.messages
        if not messages:
            from .runtime import _messages_from_legacy_fields

            messages = _messages_from_legacy_fields(request)
        converted = []
        for message in messages:
            content = message.content
            if isinstance(content, str):
                converted_content: object = content
            else:
                content_blocks: list[dict[str, object]] = []
                for item in content:
                    if item.get("type") == "video":
                        raise ValueError(
                            "The active GGUF Persona model does not support video attachments"
                        )
                    if item.get("type") == "image":
                        image_source = item.get("image")
                        if not isinstance(image_source, str):
                            raise TypeError("Persona image attachment is invalid")
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_uri(image_source)},
                            }
                        )
                    elif item.get("type") == "text":
                        text_content = item.get("text")
                        if isinstance(text_content, str):
                            content_blocks.append(
                                {"type": "text", "text": text_content}
                            )
                    else:
                        raise ValueError("Persona message contains unsupported content")
                converted_content = content_blocks
            converted.append({"role": message.role, "content": converted_content})
        if not converted:
            raise ValueError("Persona request has no messages")
        return converted

    def _count_tokens(self, messages: list[dict[str, object]]) -> int:
        """Use llama-server's chat token endpoint for exact prompt accounting."""
        result = self._request_json(
            "/v1/chat/completions/input_tokens",
            {"model": self.model_id, "messages": messages},
        )
        token_count = result.get("input_tokens")
        if not isinstance(token_count, int) or isinstance(token_count, bool):
            raise TypeError("llama-server omitted the prompt token count")
        return token_count

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate one response, preserving usage counts and cancellation."""
        messages = self._messages(request)
        max_new_tokens = min(request.max_new_tokens, request.context_space - 1)
        if max_new_tokens <= 0:
            raise ValueError("Persona prompt leaves no room for a response")
        input_budget = request.context_space - max_new_tokens
        prompt_tokens = self._count_tokens(messages)
        while prompt_tokens > input_budget:
            removable = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if message.get("role") != "system" and index < len(messages) - 1
                ),
                None,
            )
            if removable is None:
                raise ValueError("Persona prompt leaves no room for a response")
            del messages[removable]
            prompt_tokens = self._count_tokens(messages)
        max_new_tokens = min(max_new_tokens, request.context_space - prompt_tokens)
        if max_new_tokens <= 0:
            raise ValueError("Persona prompt leaves no room for a response")

        result = self._request_json(
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repeat_penalty": request.repetition_penalty,
            },
        )
        choices = result.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise RuntimeError("llama-server returned no assistant choice")
        choice = choices[0].get("message")
        if not isinstance(choice, dict):
            raise TypeError("llama-server returned an invalid assistant message")
        content = choice.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
        else:
            raise TypeError("llama-server returned an invalid assistant response")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            raise TypeError("llama-server omitted actual token usage")
        prompt_count = usage.get("prompt_tokens")
        completion_count = usage.get("completion_tokens")
        if (
            not isinstance(prompt_count, int)
            or isinstance(prompt_count, bool)
            or not isinstance(completion_count, int)
            or isinstance(completion_count, bool)
        ):
            raise TypeError("llama-server returned invalid token usage")
        from ..utils import normalize_special_characters

        normalized_text = normalize_special_characters(text.strip())
        return GenerateResponse(
            text=normalized_text,
            response=normalized_text,
            model=self.model_id,
            quantization=self.quantization,
            prompt_tokens=prompt_count,
            completion_tokens=completion_count,
        )

    def interrupt(self) -> None:
        """Close the active HTTP request so Celune cancellation reaches the server."""
        with self._connection_lock:
            connection = self._active_connection
        if connection is not None:
            connection.close()

    def close(self) -> None:
        """Stop the managed llama-server process and its log reader."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        if (
            self._log_thread is not None
            and self._log_thread is not threading.current_thread()
        ):
            self._log_thread.join(timeout=1)
