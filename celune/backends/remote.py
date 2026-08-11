# SPDX-License-Identifier: MIT
"""Proxy objects for backends running in isolated Python processes."""

import os
import json
import builtins
import threading
import subprocess
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import IO, Optional, cast

from .. import exceptions as celune_exceptions
from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from ..exceptions import BackendError
from ..i18n import string
from ..paths import project_root
from ..typing.backends import (
    BackendArgumentValue,
    BackendArguments,
    BackendDescription,
    BackendGeneration,
)
from ..typing.worker import WorkerResponse, WorkerValue
from .environment import (
    BackendEnvironment,
    BackendEnvironmentManager,
    BackendManifest,
)
from .tts.base import CeluneBackend
from .worker_protocol import (
    WorkerProtocolError,
    receive_message,
    send_message,
)
from .vc.base import CeluneVCBackend

__all__ = ["RemoteBackendProxy", "RemoteModelHandle", "RemoteVCBackendProxy"]


def _worker_exception(error_type: Optional[str], message: str) -> Exception:
    """Recreate a safe built-in or Celune exception reported by a worker."""
    if error_type:
        module_name, _, class_name = error_type.rpartition(".")
        namespace: Optional[object] = None
        if module_name == "builtins":
            namespace = builtins
        elif module_name == "celune.exceptions":
            namespace = celune_exceptions
        if namespace is not None:
            candidate = getattr(namespace, class_name, None)
            if isinstance(candidate, type) and issubclass(candidate, Exception):
                try:
                    return candidate(message)
                except Exception:
                    pass
        return BackendError(f"{error_type}: {message}")
    return BackendError(message)


@dataclass(frozen=True)
class RemoteModelHandle:
    """Reference to a model owned by a backend worker process."""

    identifier: int


class RemoteBackendProxy(CeluneBackend[RemoteModelHandle]):
    """Expose one backend worker through the normal Celune TTS interface."""

    def __init__(
        self,
        manifest: BackendManifest,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        environment_manager: Optional[BackendEnvironmentManager] = None,
        **backend_kwargs: BackendArgumentValue,
    ) -> None:
        self._manifest = manifest
        self._environment_manager = environment_manager or BackendEnvironmentManager()
        self._fatal_callback = fatal
        self._log_callback = log
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._protocol_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._worker_stderr: deque[str] = deque(maxlen=200)
        self._worker_stderr_lock = threading.Lock()
        environment = self._environment_manager.ensure(manifest)
        try:
            self._start_worker(environment, log, backend_kwargs)
            super().__init__(log=log, fatal=fatal)
            self._load_description()
        except Exception:
            self.close()
            raise

    def _start_worker(
        self,
        environment: BackendEnvironment,
        log: Callable[[str, str], None],
        backend_kwargs: BackendArguments,
    ) -> None:
        """Start the worker process and forward its stderr logs."""
        environment_variables = os.environ.copy()
        # discard inherited pip and uv variables so Celune's base configuration
        # cannot affect dependency resolution for Celune backends
        environment_variables.pop("PYTHONHOME", None)
        project_path = str(project_root())
        existing_python_path = environment_variables.get("PYTHONPATH")
        environment_variables["PYTHONPATH"] = os.pathsep.join(
            item for item in (project_path, existing_python_path) if item
        )
        self._process = subprocess.Popen(  # pylint: disable=R1732
            [
                str(environment.python),
                str(project_root() / "celune" / "backend_worker_bootstrap.py"),
                "--backend",
                self._manifest.backend_id,
                "--backend-kwargs",
                json.dumps(backend_kwargs),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment_variables,
        )
        if self._process.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._read_worker_logs,
                args=(self._process.stderr, log),
                daemon=True,
            )
            self._stderr_thread.start()

    def _read_worker_logs(
        self,
        stream: IO[bytes],
        log: Callable[[str, str], None],
    ) -> None:
        """Forward worker stderr lines to Celune's logging callback."""
        traceback_active = False
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            with self._worker_stderr_lock:
                self._worker_stderr.append(text)
            severity, explicit, message = self._split_worker_log(text)
            if text.startswith("Traceback (most recent call last):"):
                traceback_active = True
            if traceback_active and not explicit:
                severity = "error"
            if traceback_active and self._is_traceback_exception_line(text):
                traceback_active = False
            log(message, severity)

    @staticmethod
    def _split_worker_log(text: str) -> tuple[str, bool, str]:
        """Split one worker log line into its severity and message."""
        if text.startswith("[") and "] " in text:
            severity, _, message = text[1:].partition("] ")
            if severity in {"info", "warning", "error"}:
                return severity, True, message
        return "info", False, text

    @staticmethod
    def _is_traceback_exception_line(text: str) -> bool:
        """Return whether one worker stderr line contains a traceback's final exception."""
        if text.startswith(("Traceback", "File ", "During ", "The above ")):
            return False
        return text.startswith(
            (
                "ArithmeticError:",
                "AssertionError:",
                "AttributeError:",
                "EOFError:",
                "FileExistsError:",
                "FileNotFoundError:",
                "ImportError:",
                "IndexError:",
                "KeyError:",
                "LookupError:",
                "MemoryError:",
                "ModuleNotFoundError:",
                "NameError:",
                "NotImplementedError:",
                "OSError:",
                "OverflowError:",
                "RuntimeError:",
                "StopIteration:",
                "SyntaxError:",
                "SystemError:",
                "TypeError:",
                "UnboundLocalError:",
                "UnicodeError:",
                "ValueError:",
                "ZeroDivisionError:",
            )
        )

    def _worker_error_detail(self) -> str:
        """Return recent worker stderr for a failed protocol operation."""
        with self._worker_stderr_lock:
            recent_lines = list(self._worker_stderr)[-20:]
        return "\n".join(recent_lines)

    def _notify_fatal(self) -> None:
        """Forward one worker fatal notification to the Celune runtime."""
        if self._fatal_callback is not None:
            self._fatal_callback()

    def _read_response(self, process: subprocess.Popen[bytes]) -> WorkerResponse:
        """Read responses while handling out-of-band worker notifications."""
        assert process.stdout is not None
        while True:
            try:
                response = cast(WorkerResponse, receive_message(process.stdout))
            except WorkerProtocolError as error:
                detail = self._worker_error_detail()
                if detail:
                    raise WorkerProtocolError(f"{error}:\n{detail}") from error
                raise
            if response.get("fatal", False):
                self._notify_fatal()
                continue
            return response

    def _load_description(self) -> None:
        """Load static backend metadata from the worker."""
        description = cast(BackendDescription, self._request("describe"))
        self.name = description["name"]
        self.chunk_rate = description["chunk_rate"]
        self.supported_languages = description["supported_languages"]
        self.voice_models = description["voice_models"]
        self.default_voice = description["default_voice"]
        self.model_name = description["model_name"]
        self._remote_voices = description["voices"]
        self.clone_model_id = description["clone_model_id"]
        self.uses_voice_bundles = description["uses_voice_bundles"]
        self.max_new_tokens = description["max_new_tokens"]
        self.is_fake = description["is_fake"]

    def _request(self, operation: str, **arguments: WorkerValue) -> WorkerValue:
        """Send one request and return its response value."""
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError(
                string(
                    "backends.worker_not_running",
                    backend=self._manifest.backend_id,
                )
            )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError(string("backends.worker_streams_unavailable"))
        with self._protocol_lock:
            send_message(
                process.stdin, {"operation": operation, "arguments": arguments}
            )
            response = self._read_response(process)
        if not response.get("ok", False):
            raise _worker_exception(
                response.get("error_type"),
                response.get("error", "backend worker failed"),
            )
        return response.get("value")

    def _stream_request(
        self,
        operation: str,
        **arguments: WorkerValue,
    ) -> Iterator[WorkerValue]:
        """Send one streaming request and yield worker values."""
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError(
                f"backend worker '{self._manifest.backend_id}' is not running"
            )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("backend worker protocol streams are unavailable")
        with self._protocol_lock:
            send_message(
                process.stdin, {"operation": operation, "arguments": arguments}
            )
            completed = False
            stream_frame_count = 0
            try:
                while True:
                    try:
                        response = self._read_stream_frame(process)
                    except Exception:
                        completed = True
                        raise
                    if response.get("done", False):
                        self._log_callback(
                            f"[STREAM] proxy completed frames={stream_frame_count}",
                            "info",
                        )
                        completed = True
                        return
                    if response.get("stream", False):
                        stream_frame_count += 1
                        if stream_frame_count <= 3:
                            self._log_callback(
                                f"[STREAM] proxy received frame={stream_frame_count}",
                                "info",
                            )
                        yield response.get("value")
            finally:
                if not completed:
                    self._drain_stream(process)

    def _read_stream_frame(
        self,
        process: subprocess.Popen[bytes],
    ) -> WorkerResponse:
        """Read one streaming frame and raise on worker-reported failures."""
        response = self._read_response(process)
        if not response.get("ok", False):
            raise _worker_exception(
                response.get("error_type"),
                response.get("error", "backend worker failed"),
            )
        return response

    def _drain_stream(self, process: subprocess.Popen[bytes]) -> None:
        """Consume remaining stream frames so the next request stays aligned."""
        assert process.stdout is not None
        with suppress(WorkerProtocolError, OSError):
            while True:
                response = self._read_response(process)
                if response.get("done", False) or not response.get("ok", True):
                    return

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Check model availability inside the worker environment."""
        value = self._request("model_is_available_locally", model=model, lang=lang)
        return cast(tuple[bool, Optional[str]], value)

    @property
    def voices(self) -> list[str]:
        """Return voice names reported by the worker backend."""
        return list(self._remote_voices)

    def load_model(
        self,
        model_id: str,
        **kwargs: BackendArgumentValue,
    ) -> RemoteModelHandle:
        """Load a model in the worker and return an opaque handle."""
        value = self._request("load_model", model_id=model_id, **kwargs)
        return RemoteModelHandle(cast(int, value))

    def preload_models(self) -> None:
        """Ensure backend models are available inside the worker environment."""
        self._request("preload_models")

    def unload_model(self) -> None:
        """Unload worker-side models and clear the local handle."""
        self._request("unload_model")
        self.model = None

    def generate_stream(
        self,
        model: RemoteModelHandle,
        **kwargs: BackendArgumentValue,
    ) -> Iterator[BackendGeneration]:
        """Generate audio by streaming worker results through the proxy."""
        arguments = dict(kwargs)
        arguments["model_id"] = model.identifier
        for value in self._stream_request("generate_stream", **arguments):
            yield cast(BackendGeneration, value)

    def resolve_generation_language(self, lang: Optional[str]) -> Optional[str]:
        """Resolve a language using the worker backend implementation."""
        return cast(
            Optional[str],
            self._request("call", method="resolve_generation_language", lang=lang),
        )

    def should_reload_for_language(self, lang: Optional[str]) -> bool:
        """Ask the worker whether a language change needs a model reload."""
        return cast(
            bool,
            self._request("call", method="should_reload_for_language", lang=lang),
        )

    def close(self) -> None:
        """Stop the worker process and release its model resources."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None and process.stdin is not None:
            with self._protocol_lock:
                with suppress(OSError, BrokenPipeError):
                    send_message(process.stdin, {"operation": "shutdown"})
        try:
            if process.poll() is None:
                process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
        stderr_thread = self._stderr_thread
        self._stderr_thread = None
        if (
            stderr_thread is not None
            and stderr_thread is not threading.current_thread()
        ):
            stderr_thread.join(timeout=2)


class RemoteVCBackendProxy(CeluneVCBackend):
    """Expose a voice-conversion worker through the VC backend interface."""

    def __init__(
        self,
        manifest: BackendManifest,
        log: Callable[[str, str], None],
        environment_manager: Optional[BackendEnvironmentManager] = None,
    ) -> None:
        self._worker = RemoteBackendProxy(
            manifest,
            log=log,
            environment_manager=environment_manager,
        )
        super().__init__(log=log)
        self.name = self._worker.name
        self.is_fake = self._worker.is_fake

    def preload_models(self) -> None:
        """Ensure voice-conversion assets are available in the worker."""
        self._worker.preload_models()

    def unload_model(self) -> None:
        """Release voice-conversion assets in the worker."""
        self._worker.unload_model()

    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert audio in the isolated voice-conversion worker."""
        return cast(
            AudioOutput,
            self._worker._request("convert", request=request),
        )

    def convert_live(self, request: VoiceConversionRequest) -> AudioOutput:
        """Convert one block through the isolated backend's live session."""
        return cast(
            AudioOutput,
            self._worker._request("call", method="convert_live", request=request),
        )

    def stop_live(self) -> None:
        """Reset the isolated backend's live conversion session."""
        self._worker._request("call", method="stop_live")

    def close(self) -> None:
        """Stop the voice-conversion worker process."""
        self._worker.close()
