# SPDX-License-Identifier: Apache-2.0
"""Proxy objects for backends running in isolated Python processes."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import os
import re
import json
import time
import select
import threading
import subprocess
from uuid import uuid4
from typing import IO, Optional, cast
from contextlib import suppress
from collections import OrderedDict, deque
from dataclasses import dataclass
from collections.abc import Callable, Iterator

from ..i18n import string
from ..paths import (
    configure_numba_cache,
    huggingface_home_dir,
    huggingface_hub_cache_dir,
    project_root,
)
from ..terminal import RUNTIME_LOG_FILTER_MESSAGES
from ..backends.vc.base import CeluneVCBackend
from ..backends.tts.base import CeluneBackend
from ..exceptions import (
    CEDTSError,
    BackendError,
    CEDTSEOFError,
    CEDTSStreamError,
    CEDTSPayloadError,
    CEDTSTimeoutError,
    CEDTSProtocolError,
)
from ..backends.environment import (
    BackendManifest,
    BackendEnvironment,
    BackendEnvironmentManager,
)
from ..typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerResponse,
    WorkerPayloadDescriptor,
)
from ..typing.aliases import LogLevel, LogCallback
from .protocol import (
    CEDTS_VERSION,
    CORE_CAPABILITIES,
    DEFAULT_CEDTS_LIMITS,
    CEDTSLimits,
    build_packet,
    send_message,
    send_payloads,
    decode_message,
    encode_message,
    receive_message,
    receive_payloads,
    limits_from_capabilities,
)
from ..typing.backends import (
    BackendArguments,
    BackendGeneration,
    BackendDescription,
    BackendArgumentValue,
)
from ..dataclasses.pipeline import AudioOutput, VoiceConversionRequest

__all__ = ["RemoteBackendProxy", "RemoteModelHandle", "RemoteVCBackendProxy"]


def _worker_protocol_error(key: str, **kwargs: str) -> CEDTSError:
    """Create a localized, typed worker proxy CEDTS error."""
    message = string(f"backends.worker_proxy.{key}", **kwargs)
    if key == "worker_payload_descriptors_are_invalid":
        return CEDTSPayloadError(message)
    if key in {
        "worker_binary_input_stream_is_unavailable",
        "worker_binary_output_stream_is_unavailable",
    }:
        return CEDTSStreamError(message)
    return CEDTSProtocolError(message)


_CANCEL_ACK_TIMEOUT_SECONDS = 5.0
_STREAM_DRAIN_TIMEOUT_SECONDS = 5.0
_WORKER_THREAD_JOIN_TIMEOUT_SECONDS = 2.0
_SHUTDOWN_ACK_TIMEOUT_SECONDS = 5.0
_BACKEND_MODEL_OPERATION_TIMEOUT_SECONDS = 900.0
_BACKEND_MODEL_LOAD_TIMEOUT_SECONDS = _BACKEND_MODEL_OPERATION_TIMEOUT_SECONDS
_MAX_RESPONSE_QUEUE_ITEMS = 128
_MAX_RESPONSE_QUEUE_BYTES = 16 * 1024 * 1024
# Retain recent packet IDs to reject replayed packets without growing state for
# the lifetime of a long-running proxy. IDs outside this replay window may be
# reused by a peer, while active request correlation remains separately tracked.
_MESSAGE_ID_REPLAY_WINDOW = 4096
_CANCELLATION_TOMBSTONE_WINDOW = 4096
_WORKER_ENVIRONMENT_VARIABLES = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "USERNAME",
    "USER",
    "LOGNAME",
    "LNAME",
    "HOMEDRIVE",
    "HOMEPATH",
    "TEMP",
    "TMP",
    "TMPDIR",
    "NUMBA_CACHE_DIR",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_PATH_V12_8",
    "CUDA_PATH_V13_0",
    "CUDA_CACHE_PATH",
    "CUDA_MODULE_LOADING",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_DISABLE_PROGRESS_BARS",
    "GPT_SOVITS_ROOT",
    "NLTK_DATA",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


@dataclass
class _RequestCancellationState:
    """Track cancellation and terminal state for one streaming request."""

    request_id: str
    terminal: bool = False
    cancel_packet_id: Optional[str] = None
    cancel_ack_event: Optional[threading.Event] = None
    cancel_ack_result: Optional[bool] = None
    cancel_sent: bool = False


def _approximate_value_size(value: WorkerValue) -> int:
    """Estimate the memory retained by one decoded worker value."""
    if value is None:
        return 1
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (bool, int, float)):
        return len(repr(value).encode("utf-8"))
    if isinstance(value, AudioOutput):
        return _approximate_value_size(cast(WorkerValue, value.audio)) + len(
            value.label.encode("utf-8")
        )
    if isinstance(value, VoiceConversionRequest):
        return _approximate_value_size(cast(WorkerValue, value.source_audio)) + sum(
            len(reference.as_posix().encode("utf-8"))
            for reference in value.target_references
        )
    if isinstance(value, dict):
        return sum(
            len(key.encode("utf-8")) + _approximate_value_size(cast(WorkerValue, item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_approximate_value_size(item) for item in value)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    return len(repr(value).encode("utf-8"))


def _approximate_response_size(response: WorkerResponse) -> int:
    """Estimate the memory retained by one decoded worker response."""
    return _approximate_value_size(cast(WorkerValue, response))


def _worker_environment(environment: BackendEnvironment) -> dict[str, str]:
    """Build the bounded environment passed to an isolated backend worker."""
    parent_environment = os.environ
    worker_environment: dict[str, str] = {}
    for variable in _WORKER_ENVIRONMENT_VARIABLES:
        value = next(
            (
                candidate
                for name, candidate in parent_environment.items()
                if name.casefold() == variable.casefold()  # pylint: disable=E1101
            ),
            None,
        )
        if value is not None:
            worker_environment[variable] = value

    worker_environment.setdefault("HF_HOME", str(huggingface_home_dir()))
    worker_environment.setdefault(
        "HF_HUB_CACHE",
        str(huggingface_hub_cache_dir()),
    )

    backend_bin = environment.python.resolve().parent
    parent_path = worker_environment.get("PATH")
    worker_environment["PATH"] = os.pathsep.join(
        item for item in (str(backend_bin), parent_path) if item
    )
    worker_environment["NUMBA_CACHE_DIR"] = str(configure_numba_cache())
    worker_environment["PYTHONPATH"] = str(project_root().resolve())
    worker_environment["PYTHONNOUSERSITE"] = "1"
    return worker_environment


def _set_windows_handle_inheritable(handle: int, inheritable: bool) -> None:
    """Set inheritance on one native Windows handle."""
    set_handle_inheritable = getattr(os, "set_handle_inheritable", None)
    if not callable(set_handle_inheritable):
        raise OSError("Windows handle inheritance is unavailable")
    set_handle_inheritable(handle, inheritable)  # pylint: disable=E1102


def _emit_log(
    callback: Callable[..., None],
    message: str,
    severity: str = "info",
    loglevel: LogLevel = "info",
) -> None:
    """Call a log callback while retaining compatibility with two-argument hooks."""
    try:
        callback(message, severity, loglevel=loglevel)
    except TypeError as error:
        if "loglevel" not in str(error):
            raise
        callback(message, severity)


_WORKER_ERROR_CODE = "backend_worker_error"


def _worker_exception(error_type: Optional[str], message: str) -> Exception:
    """Convert worker error data into a typed CEDTS or backend error."""
    cedts_error_types: dict[str, type[CEDTSError]] = {
        f"{error_class.__module__}.{error_class.__qualname__}": error_class
        for error_class in (
            CEDTSError,
            CEDTSEOFError,
            CEDTSTimeoutError,
            CEDTSProtocolError,
            CEDTSPayloadError,
            CEDTSStreamError,
        )
    }
    cedts_error_type = cedts_error_types.get(error_type or "")
    if cedts_error_type is not None:
        if cedts_error_type is CEDTSTimeoutError:
            return CEDTSTimeoutError("worker response", 0.0, message=message)
        return cedts_error_type(message)
    safe_error_type = error_type or string(
        "backends.worker_proxy.unknown_worker_error_type"
    )
    return BackendError(
        string(
            "backends.worker_proxy.worker_error",
            error_code=_WORKER_ERROR_CODE,
            error_type=safe_error_type,
            detail=message,
        ),
        error_code=_WORKER_ERROR_CODE,
        error_type=error_type,
    )


@dataclass(frozen=True)
class RemoteModelHandle:
    """Reference to a model owned by a backend worker process."""

    identifier: int


class RemoteBackendProxy(CeluneBackend[RemoteModelHandle]):
    """Expose one backend worker through the normal Celune TTS interface."""

    def __init__(
        self,
        manifest: BackendManifest,
        log: LogCallback,
        fatal: Optional[Callable[[], None]] = None,
        environment_manager: Optional[BackendEnvironmentManager] = None,
        event_callback: Optional[Callable[[WorkerMessage], None]] = None,
        **backend_kwargs: BackendArgumentValue,
    ) -> None:
        self._manifest = manifest
        self._environment_manager = environment_manager or BackendEnvironmentManager()
        self._fatal_callback = fatal
        self._log_callback = log
        self._event_callback = event_callback
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._binary_input: Optional[IO[bytes]] = None
        self._binary_output: Optional[IO[bytes]] = None
        self._protocol_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._worker_stderr: deque[str] = deque(maxlen=200)
        self._worker_stderr_lock = threading.Lock()
        self._stream_active = threading.Event()
        self._active_request_lock = threading.Lock()
        self._active_request_id: Optional[str] = None
        self._cancel_target: Optional[str] = None
        self._cancel_packet_id: Optional[str] = None
        self._cancel_ack_event: Optional[threading.Event] = None
        self._cancel_ack_result: Optional[bool] = None
        self._cancel_sent = False
        self._request_cancellation_states: dict[str, _RequestCancellationState] = {}
        self._terminal_cancellation_states: OrderedDict[
            str, _RequestCancellationState
        ] = OrderedDict()
        self._received_message_ids: OrderedDict[str, None] = OrderedDict()
        self._negotiated_capabilities: dict[str, WorkerValue] = {}
        self._cedts_limits = DEFAULT_CEDTS_LIMITS
        self._reader_stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_error: Optional[Exception] = None
        self._response_condition = threading.Condition()
        self._pending_reply_ids: set[str] = set()
        self._response_queues: dict[str, deque[WorkerResponse]] = {}
        self._event_condition = threading.Condition()
        self._event_queue: deque[WorkerMessage] = deque(maxlen=256)
        self._response_queue_item_sizes = {}
        self._response_queue_bytes = {}

        environment = self._environment_manager.ensure(manifest)
        try:
            self._start_worker(environment, log, backend_kwargs)
            self._handshake()
            super().__init__(log=log, fatal=fatal)
            self._start_packet_reader()
            self._load_description()
        except Exception:
            with suppress(Exception):
                self.close()
            raise

    def _start_worker(
        self,
        environment: BackendEnvironment,
        log: LogCallback,
        backend_kwargs: BackendArguments,
    ) -> None:
        """Start the worker process and forward its stderr logs."""
        environment_variables = _worker_environment(environment)
        worker_binary_input = -1
        core_binary_input = -1
        core_binary_output = -1
        worker_binary_output = -1
        windows_binary_handles: tuple[int, ...] = ()
        try:
            worker_binary_input, core_binary_input = os.pipe()
            core_binary_output, worker_binary_output = os.pipe()
            command = [
                str(environment.python),
                str(project_root() / "celune" / "cedts" / "bootstrap.py"),
                "--backend",
                self._manifest.backend_id,
                "--backend-kwargs",
                json.dumps(backend_kwargs),
            ]
            if os.name == "nt":
                import msvcrt  # pylint: disable=E0401

                windows_binary_handles = tuple(
                    msvcrt.get_osfhandle(descriptor)
                    for descriptor in (worker_binary_input, worker_binary_output)
                )
                for handle in windows_binary_handles:
                    _set_windows_handle_inheritable(handle, True)
                startup_info = subprocess.STARTUPINFO()
                startup_info.lpAttributeList = {
                    "handle_list": list(windows_binary_handles)
                }
                command.extend(
                    (
                        "--binary-input-handle",
                        str(windows_binary_handles[0]),
                        "--binary-output-handle",
                        str(windows_binary_handles[1]),
                    )
                )
                self._process = subprocess.Popen(  # pylint: disable=R1732
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment_variables,
                    close_fds=True,
                    startupinfo=startup_info,
                )
            else:
                command.extend(
                    (
                        "--binary-input-fd",
                        str(worker_binary_input),
                        "--binary-output-fd",
                        str(worker_binary_output),
                    )
                )
                self._process = subprocess.Popen(  # pylint: disable=R1732
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment_variables,
                    pass_fds=(worker_binary_input, worker_binary_output),
                )
            for handle in windows_binary_handles:
                with suppress(OSError):
                    _set_windows_handle_inheritable(handle, False)
            for descriptor in (worker_binary_input, worker_binary_output):
                with suppress(OSError):
                    os.close(descriptor)
            worker_binary_input = -1
            worker_binary_output = -1
            self._binary_input = os.fdopen(core_binary_input, "wb", buffering=0)
            core_binary_input = -1
            self._binary_output = os.fdopen(core_binary_output, "rb", buffering=0)
            core_binary_output = -1
            _emit_log(
                log,
                string(
                    "backends.worker_proxy.worker_started",
                    backend=self._manifest.backend_id,
                    pid=getattr(self._process, "pid", None),
                    python=environment.python,
                ),
                loglevel="debug",
            )
            if self._process.stderr is not None:
                self._stderr_thread = threading.Thread(
                    target=self._read_worker_logs,
                    args=(self._process.stderr, log),
                    daemon=True,
                )
                self._stderr_thread.start()
        except Exception:
            for handle in windows_binary_handles:
                with suppress(OSError):
                    _set_windows_handle_inheritable(handle, False)
            for descriptor in (
                worker_binary_input,
                core_binary_input,
                core_binary_output,
                worker_binary_output,
            ):
                if descriptor >= 0:
                    with suppress(OSError):
                        os.close(descriptor)
            process = self._process
            self._process = None
            if process is not None:
                with suppress(Exception):
                    if process.poll() is None:
                        self._terminate_process(process)
                with suppress(Exception):
                    self._close_process_streams(process)
            raise

    def _read_worker_logs(
        self,
        stream: IO[bytes],
        log: LogCallback,
    ) -> None:
        """Forward worker stderr lines to Celune's logging callback."""
        traceback_active = False
        for line in iter(stream.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            if not text:
                continue
            with self._worker_stderr_lock:
                self._worker_stderr.append(text)
            if any(
                filtered_message in text
                for filtered_message in RUNTIME_LOG_FILTER_MESSAGES
            ):
                continue
            severity, explicit, message, loglevel = self._split_worker_log(text)
            if text.startswith("Traceback (most recent call last):"):
                traceback_active = True
            if traceback_active and not explicit:
                severity = "error"
            if traceback_active and self._is_traceback_exception_line(text):
                traceback_active = False
            _emit_log(log, message, severity, loglevel)

    @staticmethod
    def _split_worker_log(text: str) -> tuple[str, bool, str, LogLevel]:
        """Split one worker log line into its severity and message."""
        if text.startswith("["):
            first, separator, remainder = text[1:].partition("]")
            if (
                separator
                and first in {"info", "warning", "error"}
                and remainder.startswith(" ")
            ):
                return first, True, remainder[1:], "info"
            if (
                separator
                and first in {"verbose", "debug"}
                and remainder.startswith("[")
            ):
                severity, severity_separator, message = remainder[1:].partition("]")
                if (
                    severity_separator
                    and severity in {"info", "warning", "error"}
                    and message.startswith(" ")
                ):
                    return severity, True, message[1:], first
        return "info", False, text, "info"

    @staticmethod
    def _is_traceback_exception_line(text: str) -> bool:
        """Return whether one worker stderr line contains a traceback's final exception."""
        if text.startswith(("Traceback", "File ", "During ", "The above ")):
            return False
        return bool(re.match(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt):", text))

    def _worker_error_detail(self) -> str:
        """Return recent worker stderr for a failed protocol operation."""
        with self._worker_stderr_lock:
            recent_lines = list(self._worker_stderr)[-20:]
        return "\n".join(recent_lines)

    def _abort_after_reader_failure(self) -> None:
        """Terminate a worker after its response stream becomes undecodable."""
        process = getattr(self, "_process", None)
        if process is None or not callable(getattr(process, "poll", None)):
            return
        with suppress(Exception):
            self.abort()

    def _notify_fatal(self) -> None:
        """Forward one worker fatal notification to the Celune runtime."""
        fatal_callback = getattr(self, "_fatal_callback", None)
        if fatal_callback is not None:
            fatal_callback()

    def _stop_packet_reader(self) -> None:
        """Mark the packet reader as intentionally stopped before closing a worker."""
        reader_stop = getattr(self, "_reader_stop", None)
        if reader_stop is not None:
            reader_stop.set()

    def _start_packet_reader(self) -> None:
        """Start the sole reader for the worker control and binary streams."""
        process = self._process
        if process is None or process.stdout is None:
            raise _worker_protocol_error("worker_streams_are_unavailable_for_reader")
        self._ensure_response_state()
        self._reader_stop.clear()
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._read_worker_packets,
            args=(process,),
            daemon=True,
            name=f"celune-worker-reader-{process.pid}",
        )
        self._reader_thread.start()

    def _ensure_response_state(self) -> None:
        """Initialize dispatcher response state for lightweight test doubles."""
        if not hasattr(self, "_response_condition"):
            self._response_condition = threading.Condition()
        if not hasattr(self, "_pending_reply_ids"):
            self._pending_reply_ids = set()
        if not hasattr(self, "_response_queues"):
            self._response_queues = {}
        if not hasattr(self, "_response_queue_item_sizes"):
            self._response_queue_item_sizes = {}
        if not hasattr(self, "_response_queue_bytes"):
            self._response_queue_bytes = {}

    def _ensure_event_state(self) -> None:
        """Initialize dispatcher event state for lightweight test doubles."""
        if not hasattr(self, "_event_condition"):
            self._event_condition = threading.Condition()
        if not hasattr(self, "_event_queue"):
            self._event_queue = deque(maxlen=256)

    def _clear_response_queue_locked(self, reply_to: str) -> None:
        """Discard one response queue and all of its memory accounting."""
        self._response_queues.pop(reply_to, None)
        self._response_queue_item_sizes.pop(reply_to, None)
        self._response_queue_bytes.pop(reply_to, None)

    def _clear_runtime_state(self) -> None:
        """Release all connection state and wake consumers during shutdown."""
        self._ensure_response_state()
        with self._response_condition:
            self._pending_reply_ids.clear()
            self._response_queues.clear()
            self._response_queue_item_sizes.clear()
            self._response_queue_bytes.clear()
            if getattr(self, "_reader_error", None) is None:
                self._reader_error = CEDTSEOFError()
            self._response_condition.notify_all()

        self._ensure_event_state()
        with self._event_condition:
            self._event_queue.clear()
            self._event_condition.notify_all()

        self._ensure_cancellation_state()
        with self._active_request_lock:
            for state in self._request_cancellation_states.values():
                state.terminal = True
                if state.cancel_ack_event is not None:
                    state.cancel_ack_event.set()
            self._request_cancellation_states.clear()
            for state in self._terminal_cancellation_states.values():
                state.terminal = True
                if state.cancel_ack_event is not None:
                    state.cancel_ack_event.set()
            self._terminal_cancellation_states.clear()
            self._active_request_id = None
            self._cancel_target = None
            self._cancel_packet_id = None
            self._cancel_ack_event = None
            self._cancel_ack_result = None
            self._cancel_sent = False
        stream_active = getattr(self, "_stream_active", None)
        if stream_active is not None:
            stream_active.clear()

    def _ensure_cancellation_state(self) -> None:
        """Initialize request cancellation state for lightweight test doubles."""
        if not hasattr(self, "_active_request_lock"):
            self._active_request_lock = threading.Lock()
            self._active_request_id = None
            self._cancel_target = None
            self._cancel_packet_id = None
            self._cancel_ack_event = None
            self._cancel_ack_result = None
            self._cancel_sent = False
        if not hasattr(self, "_request_cancellation_states"):
            self._request_cancellation_states = {}
        if not hasattr(self, "_terminal_cancellation_states"):
            self._terminal_cancellation_states = OrderedDict()
        active_request_id = getattr(self, "_active_request_id", None)
        if (
            isinstance(active_request_id, str)
            and active_request_id not in self._request_cancellation_states
            and active_request_id not in self._terminal_cancellation_states
        ):
            self._request_cancellation_states[active_request_id] = (
                _RequestCancellationState(
                    request_id=active_request_id,
                    cancel_packet_id=getattr(self, "_cancel_packet_id", None),
                    cancel_ack_event=getattr(self, "_cancel_ack_event", None),
                    cancel_ack_result=getattr(self, "_cancel_ack_result", None),
                    cancel_sent=getattr(self, "_cancel_sent", False),
                )
            )

    def _sync_active_cancellation_fields(
        self, state: _RequestCancellationState
    ) -> None:
        """Keep legacy active-request fields aligned with the request state."""
        if getattr(self, "_active_request_id", None) != state.request_id:
            return
        self._cancel_target = state.request_id
        self._cancel_packet_id = state.cancel_packet_id
        self._cancel_ack_event = state.cancel_ack_event
        self._cancel_ack_result = state.cancel_ack_result
        self._cancel_sent = state.cancel_sent

    def _mark_request_terminal_locked(self, request_id: str) -> None:
        """Mark a request terminal while holding the active-request lock."""
        state = self._request_cancellation_states.pop(request_id, None)
        if state is not None:
            state.terminal = True
            if (
                state.cancel_sent
                and state.cancel_packet_id is not None
                and state.cancel_ack_event is not None
                and not state.cancel_ack_event.is_set()
            ):
                self._terminal_cancellation_states[request_id] = state
                self._terminal_cancellation_states.move_to_end(request_id)
                while (
                    len(self._terminal_cancellation_states)
                    > _CANCELLATION_TOMBSTONE_WINDOW
                ):
                    self._terminal_cancellation_states.popitem(last=False)

    def _read_worker_packets(self, process: subprocess.Popen[bytes]) -> None:
        """Read and dispatch every packet from a worker without competing readers."""
        try:
            while not self._reader_stop.is_set():
                try:
                    packet = self._receive_packet(process)
                    if self._reader_stop.is_set():
                        return
                    self._register_packet(packet)
                    self._dispatch_packet(packet)
                except Exception as error:
                    if not self._reader_stop.is_set():
                        detail = self._worker_error_detail()
                        if detail and isinstance(error, CEDTSProtocolError):
                            error = _worker_protocol_error(
                                "error_with_detail",
                                error=str(error),
                                detail=detail,
                            )
                        with self._response_condition:
                            self._reader_error = error
                            self._response_condition.notify_all()
                        with self._event_condition:
                            self._event_condition.notify_all()
                        log_callback = getattr(self, "_log_callback", None)
                        if callable(log_callback):
                            with suppress(Exception):
                                _emit_log(
                                    cast(Callable[..., None], log_callback),
                                    string(
                                        "backends.worker_proxy.packet_reader_failed",
                                        error=str(error),
                                    ),
                                    "error",
                                )
                        threading.Thread(
                            target=self._abort_after_reader_failure,
                            name="celune-worker-reader-abort",
                            daemon=True,
                        ).start()
                    return
        finally:
            if not self._reader_stop.is_set():
                with suppress(Exception):
                    if process.poll() is not None:
                        self._close_process_streams(process)

    def _dispatch_packet(self, packet: WorkerMessage) -> None:
        """Dispatch one validated worker packet to its event or response consumer."""
        if packet.get("cedts_version") != CEDTS_VERSION:
            raise _worker_protocol_error("worker_packet_version_is_unsupported")
        kind = packet.get("kind")
        if kind == "cancel_ack":
            self._handle_cancel_ack(packet)
            return
        if kind in {"progress", "callback"}:
            reply_to = packet.get("reply_to")
            with self._response_condition:
                if (
                    not isinstance(reply_to, str)
                    or reply_to not in self._pending_reply_ids
                ):
                    raise _worker_protocol_error("worker_event_correlation_is_invalid")
            self._dispatch_event(packet)
            return
        if kind == "event":
            self._dispatch_event(packet)
            return
        if kind not in {"response", "error", "shutdown_ack"}:
            raise _worker_protocol_error("worker_sent_an_unexpected_packet_kind")
        reply_to = packet.get("reply_to")
        if not isinstance(reply_to, str):
            raise _worker_protocol_error("worker_response_correlation_is_invalid")
        response = self._response_from_packet(packet)
        with self._response_condition:
            if reply_to not in self._pending_reply_ids:
                raise _worker_protocol_error("worker_response_correlation_is_unknown")
            response_queue = self._response_queues.get(reply_to)
            response_size = _approximate_response_size(response)
            queued_bytes = self._response_queue_bytes.get(reply_to, 0)
            if (
                response_queue is not None
                and len(response_queue) >= _MAX_RESPONSE_QUEUE_ITEMS
            ) or queued_bytes + response_size > _MAX_RESPONSE_QUEUE_BYTES:
                raise _worker_protocol_error("worker_response_queue_limit_exceeded")
            if response_queue is None:
                response_queue = deque()
                self._response_queues[reply_to] = response_queue
            response_queue.append(response)
            self._response_queue_item_sizes.setdefault(reply_to, deque()).append(
                response_size
            )
            self._response_queue_bytes[reply_to] = queued_bytes + response_size
            self._response_condition.notify_all()
        if kind == "shutdown_ack":
            self._stop_packet_reader()

    def _handle_cancel_ack(self, packet: WorkerMessage) -> None:
        """Resolve a request-scoped cancellation acknowledgement."""
        data = packet.get("data")
        if not isinstance(data, dict):
            raise _worker_protocol_error(
                "worker_cancellation_acknowledgement_data_is_invalid"
            )
        cancel_data = cast(dict[str, WorkerValue], data)
        target = cancel_data.get("target_message_id")
        self._ensure_cancellation_state()
        with self._active_request_lock:
            if not isinstance(target, str):
                raise _worker_protocol_error(
                    "worker_cancellation_acknowledgement_is_invalid"
                )
            state = self._request_cancellation_states.get(target)
            if state is None:
                state = self._terminal_cancellation_states.get(target)
            if state is None or packet.get("reply_to") != state.cancel_packet_id:
                raise _worker_protocol_error(
                    "worker_cancellation_acknowledgement_is_invalid"
                )
            if state.terminal:
                state.cancel_ack_result = cancel_data.get("cancelled") is True
                if state.cancel_ack_event is not None:
                    state.cancel_ack_event.set()
                self._terminal_cancellation_states.pop(target, None)
                self._sync_active_cancellation_fields(state)
                return
            if state.cancel_ack_event is None:
                raise _worker_protocol_error(
                    "worker_cancellation_acknowledgement_is_invalid"
                )
            state.cancel_ack_result = cancel_data.get("cancelled") is True
            state.cancel_ack_event.set()
            self._sync_active_cancellation_fields(state)

    def _dispatch_event(self, packet: WorkerMessage) -> None:
        """Queue and forward one correlated worker event or callback packet."""
        if not hasattr(self, "_event_condition"):
            self._event_condition = threading.Condition()
            self._event_queue = deque(maxlen=256)
        with self._event_condition:
            self._event_queue.append(packet)
            self._event_condition.notify_all()
        if packet.get("kind") == "progress":
            data = packet.get("data")
            if isinstance(data, dict):
                step = data.get("step")
                total = data.get("total")
                if isinstance(step, int) and not isinstance(step, bool):
                    self.report_progress(
                        float(step),
                        float(total)
                        if isinstance(total, int)
                        and not isinstance(total, bool)
                        and total > 0
                        else None,
                    )
        if packet.get("operation") == "fatal":
            log_callback = getattr(self, "_log_callback", None)
            if log_callback is not None:
                _emit_log(
                    log_callback,
                    string("backends.worker_proxy.fatal_notification"),
                    loglevel="debug",
                )
            self._notify_fatal()
        callback = getattr(self, "_event_callback", None)
        if callback is not None:
            try:
                callback(packet)
            except Exception as error:
                log_callback = getattr(self, "_log_callback", None)
                if log_callback is not None:
                    _emit_log(
                        log_callback,
                        string(
                            "backends.worker_proxy.event_callback_failed",
                            error=str(error),
                        ),
                        severity="error",
                    )

    def get_worker_event(
        self, timeout: Optional[float] = None
    ) -> Optional[WorkerMessage]:
        """Return the next queued worker event, progress update, or callback.

        Args:
            timeout: Maximum seconds to wait for an event, or ``None`` to wait indefinitely.

        Returns:
            Optional[WorkerMessage]: The next event packet, or ``None`` when the timeout expires.

        Raises:
            CEDTSError: If the worker packet reader has failed.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._event_condition:
            while True:
                if self._event_queue:
                    return self._event_queue.popleft()
                with self._response_condition:
                    reader_error = getattr(self, "_reader_error", None)
                if reader_error is not None:
                    raise reader_error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._event_condition.wait(remaining)

    def _response_from_packet(self, packet: WorkerMessage) -> WorkerResponse:
        """Convert a response packet into the proxy's response mapping."""
        data = packet.get("data")
        if not isinstance(data, dict):
            raise _worker_protocol_error("worker_response_data_is_invalid")
        response = cast(WorkerResponse, data)
        if response.get("fatal", False):
            raise _worker_protocol_error("worker_fatal_flag_must_use_an_event_packet")
        if packet.get("kind") == "error":
            response["ok"] = False
        log_callback = getattr(self, "_log_callback", None)
        if log_callback is not None:
            _emit_log(
                log_callback,
                string(
                    "backends.worker_proxy.response_received",
                    ok=response.get("ok"),
                    stream=response.get("stream", False),
                    done=response.get("done", False),
                ),
                loglevel="debug",
            )
        return response

    def _raise_handshake_error(self, packet: WorkerMessage, hello_id: str) -> None:
        """Raise the inert backend error carried by a correlated handshake packet."""
        if packet.get("kind") != "error":
            return
        if packet.get("reply_to") != hello_id:
            raise _worker_protocol_error("worker_response_correlation_is_invalid")
        response = self._response_from_packet(packet)
        error = response.get(
            "error", string("backends.worker_proxy.backend_worker_failed")
        )
        error_type = response.get("error_type")
        raise _worker_exception(
            error_type if isinstance(error_type, str) else None,
            error if isinstance(error, str) else str(error),
        )

    def _receive_handshake_packet(
        self,
        process: subprocess.Popen[bytes],
    ) -> WorkerMessage:
        """Receive one handshake packet while forwarding worker events."""
        while True:
            packet = self._receive_packet(process)
            self._register_packet(packet)
            if packet.get("kind") == "event":
                self._dispatch_event(packet)
                continue
            return packet

    def _receive_packet(self, process: subprocess.Popen[bytes]) -> WorkerMessage:
        """Read and decode one CEDTS packet from the worker."""
        assert process.stdout is not None
        limits = getattr(self, "_cedts_limits", DEFAULT_CEDTS_LIMITS)
        control = receive_message(process.stdout, limits=limits)
        descriptors = control.get("payloads", [])
        if not isinstance(descriptors, list):
            raise _worker_protocol_error("worker_payload_descriptors_are_invalid")
        typed_descriptors = cast(list[WorkerPayloadDescriptor], descriptors)
        binary_output = getattr(self, "_binary_output", None)
        if binary_output is None:
            if descriptors:
                raise _worker_protocol_error(
                    "worker_binary_output_stream_is_unavailable"
                )
            payloads = {}
        else:
            payloads = receive_payloads(
                binary_output,
                typed_descriptors,
                limits=limits,
            )
        decoded = decode_message(control, payloads, limits=limits)
        decoded["cedts_version"] = control["cedts_version"]
        decoded["kind"] = control["kind"]
        decoded["message_id"] = control["message_id"]
        decoded["reply_to"] = control["reply_to"]
        decoded["operation"] = control["operation"]
        return cast(WorkerMessage, decoded)

    def _register_packet(self, packet: WorkerMessage) -> None:
        """Reject duplicate worker packet identities."""
        message_id = packet.get("message_id")
        received_message_ids = self._received_message_ids
        if not isinstance(message_id, str) or message_id in received_message_ids:
            raise _worker_protocol_error(
                "worker_packet_message_id_is_duplicate_or_invalid"
            )
        received_message_ids[message_id] = None
        received_message_ids.move_to_end(message_id)
        if len(received_message_ids) > _MESSAGE_ID_REPLAY_WINDOW:
            received_message_ids.popitem(last=False)

    def _clear_received_message_ids(self) -> None:
        """Release packet replay state when the worker connection closes."""
        received_message_ids = getattr(self, "_received_message_ids", None)
        if isinstance(received_message_ids, OrderedDict):
            received_message_ids.clear()

    def _read_response(
        self,
        process: subprocess.Popen[bytes],
        reply_to: Optional[str] = None,
        timeout: Optional[float] = None,
        packet_name: Optional[str] = None,
    ) -> WorkerResponse:
        """Return a correlated response from the dispatcher-owned response queue."""
        if getattr(self, "_reader_thread", None) is None:
            return self._read_response_direct(
                process,
                reply_to,
                timeout,
                packet_name,
            )
        self._ensure_response_state()
        if reply_to is None:
            raise _worker_protocol_error("worker_response_correlation_is_missing")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._response_condition:
                response_queue = self._response_queues.get(reply_to)
                if response_queue:
                    response = response_queue.popleft()
                    item_sizes = self._response_queue_item_sizes.get(reply_to)
                    response_size = (
                        item_sizes.popleft()
                        if item_sizes
                        else _approximate_response_size(response)
                    )
                    queued_bytes = self._response_queue_bytes.get(reply_to, 0)
                    self._response_queue_bytes[reply_to] = max(
                        0, queued_bytes - response_size
                    )
                    if not response_queue:
                        self._clear_response_queue_locked(reply_to)
                    return response
                if self._reader_error is not None:
                    raise self._reader_error
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise CEDTSTimeoutError(
                        packet_name or "worker response",
                        timeout or 0.0,
                    )
                self._response_condition.wait(remaining)

    def _read_response_direct(
        self,
        process: subprocess.Popen[bytes],
        reply_to: Optional[str] = None,
        timeout: Optional[float] = None,
        packet_name: Optional[str] = None,
    ) -> WorkerResponse:
        """Read a correlated response while handling out-of-band notifications."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None:
                if remaining <= 0:
                    raise CEDTSTimeoutError(
                        packet_name or "worker response",
                        timeout or 0.0,
                    )
                assert process.stdout is not None
                try:
                    ready, _, _ = select.select(
                        [process.stdout],
                        [],
                        [],
                        remaining,
                    )
                except (OSError, ValueError):
                    # The direct path is only used before the dispatcher starts;
                    # Windows subprocess pipes do not support select().
                    ready = [process.stdout]
                if not ready:
                    raise CEDTSTimeoutError(
                        packet_name or "worker response",
                        timeout or 0.0,
                    )
            try:
                packet = self._receive_packet(process)
                self._register_packet(packet)
            except CEDTSProtocolError as error:
                detail = self._worker_error_detail()
                if detail:
                    raise _worker_protocol_error(
                        "error_with_detail", error=str(error), detail=detail
                    ) from error
                raise
            if packet.get("cedts_version") != CEDTS_VERSION:
                raise _worker_protocol_error("worker_packet_version_is_unsupported")
            kind = packet.get("kind")
            packet_reply_to = packet.get("reply_to")
            if kind == "cancel_ack":
                self._handle_cancel_ack(packet)
                continue
            if kind in {"event", "progress", "callback"}:
                if packet_reply_to not in {None, reply_to}:
                    raise _worker_protocol_error("worker_event_correlation_is_invalid")
                self._dispatch_event(packet)
                continue
            if kind not in {"response", "error", "shutdown_ack"}:
                raise _worker_protocol_error("worker_sent_an_unexpected_packet_kind")
            if reply_to is not None and packet_reply_to != reply_to:
                raise _worker_protocol_error("worker_response_correlation_is_invalid")
            return self._response_from_packet(packet)

    def _handshake(self) -> None:
        """Negotiate CEDTS capabilities before using backend operations."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise _worker_protocol_error("worker_streams_are_unavailable_for_handshake")
        hello_id = self._send_packet(
            process.stdin,
            "hello",
            "handshake",
            cast(
                dict[str, WorkerValue],
                {
                    "versions": [CEDTS_VERSION],
                    "capabilities": CORE_CAPABILITIES,
                    "required_capabilities": {
                        "streaming": True,
                        "cancellation": True,
                    },
                },
            ),
        )
        hello_ack = self._receive_handshake_packet(process)
        self._raise_handshake_error(hello_ack, hello_id)
        if (
            hello_ack.get("kind") != "hello_ack"
            or hello_ack.get("reply_to") != hello_id
            or hello_ack.get("cedts_version") != CEDTS_VERSION
        ):
            raise _worker_protocol_error("worker_hello_acknowledgement_is_invalid")
        ack_data = hello_ack.get("data")
        if not isinstance(ack_data, dict):
            raise _worker_protocol_error("worker_hello_acknowledgement_data_is_invalid")
        selected_version = ack_data.get("cedts_version")
        capabilities = ack_data.get("capabilities")
        if selected_version != CEDTS_VERSION or not isinstance(capabilities, dict):
            raise _worker_protocol_error("worker_capabilities_are_incompatible")
        if capabilities.get("streaming") is not True:
            raise _worker_protocol_error("worker_does_not_support_streaming")
        if capabilities.get("cancellation") is not True:
            raise _worker_protocol_error("worker_does_not_support_cancellation")
        supported_operations = capabilities.get("supported_operations")
        supported_media_types = capabilities.get("supported_media_types")
        if (
            not isinstance(supported_operations, list)
            or "describe" not in supported_operations
            or not isinstance(supported_media_types, list)
        ):
            raise _worker_protocol_error("worker_capability_negotiation_is_incomplete")
        self._negotiated_capabilities = cast(dict[str, WorkerValue], capabilities)
        self._cedts_limits = limits_from_capabilities(capabilities)
        ready = self._receive_handshake_packet(process)
        self._raise_handshake_error(ready, hello_id)
        if (
            ready.get("kind") != "ready"
            or ready.get("reply_to") != hello_id
            or ready.get("cedts_version") != CEDTS_VERSION
        ):
            raise _worker_protocol_error("worker_ready_packet_is_invalid")
        ready_data = ready.get("data")
        if (
            not isinstance(ready_data, dict)
            or ready_data.get("capabilities") != capabilities
        ):
            raise _worker_protocol_error("worker_ready_capabilities_do_not_match")

    def _send_packet(
        self,
        stream: IO[bytes],
        kind: str,
        operation: str,
        data: Optional[dict[str, WorkerValue]] = None,
        *,
        reply_to: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Encode and send one CEDTS packet with its binary payloads."""
        limits: CEDTSLimits = getattr(self, "_cedts_limits", DEFAULT_CEDTS_LIMITS)
        packet = build_packet(
            kind,
            operation,
            data,
            reply_to=reply_to,
            message_id=message_id,
        )
        if kind in {"request", "shutdown"}:
            self._ensure_response_state()
            with self._response_condition:
                self._pending_reply_ids.add(cast(str, packet["message_id"]))
        send_lock = getattr(self, "_send_lock", None)
        if send_lock is None:
            control, payloads = encode_message(packet, limits=limits)
            send_message(stream, control, limits=limits)
            if self._binary_input is None:
                if payloads:
                    raise _worker_protocol_error(
                        "worker_binary_input_stream_is_unavailable"
                    )
            else:
                send_payloads(self._binary_input, payloads, limits=limits)
        else:
            with send_lock:
                control, payloads = encode_message(packet, limits=limits)
                send_message(stream, control, limits=limits)
                if self._binary_input is None:
                    if payloads:
                        raise _worker_protocol_error(
                            "worker_binary_input_stream_is_unavailable"
                        )
                else:
                    send_payloads(self._binary_input, payloads, limits=limits)
        return cast(str, packet["message_id"])

    def _load_description(self) -> None:
        """Load static backend metadata from the worker."""
        description = cast(BackendDescription, self._request("describe"))
        self.name = description["name"]
        self.chunk_rate = description["chunk_rate"]
        self.supported_languages = tuple(description["supported_languages"])
        self.voice_models = description["voice_models"]
        self.default_voice = description["default_voice"]
        self.model_name = description["model_name"]
        self._remote_voices = description["voices"]
        self.clone_model_id = description["clone_model_id"]
        self.uses_voice_bundles = description["uses_voice_bundles"]
        self.max_new_tokens = description["max_new_tokens"]
        self.is_fake = description["is_fake"]

    def _request(
        self,
        operation: str,
        *,
        response_timeout: Optional[float] = None,
        **arguments: WorkerValue,
    ) -> WorkerValue:
        """Send one request and return its response value."""
        self._ensure_response_state()
        process = self._process
        if (
            process is None
            or process.poll() is not None
            or getattr(self, "_closing", False)
        ):
            raise RuntimeError(
                string(
                    "backends.worker_not_running",
                    backend=self._manifest.backend_id,
                )
            )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError(string("backends.worker_streams_unavailable"))
        self._ensure_cancellation_state()
        with self._protocol_lock:
            _emit_log(
                self._log_callback,
                string(
                    "backends.worker_proxy.request_sent",
                    operation=operation,
                    arguments=tuple(arguments),
                ),
                loglevel="debug",
            )
            request_id = uuid4().hex
            with self._response_condition:
                self._pending_reply_ids.add(request_id)
            try:
                sent_request_id = self._send_packet(
                    process.stdin,
                    "request",
                    operation,
                    {"arguments": arguments},
                    message_id=request_id,
                )
                if sent_request_id != request_id:
                    with self._response_condition:
                        self._pending_reply_ids.discard(request_id)
                        self._pending_reply_ids.add(sent_request_id)
                    request_id = sent_request_id
                try:
                    if response_timeout is None:
                        response = self._read_response(process, request_id)
                    else:
                        response = self._read_response(
                            process,
                            request_id,
                            timeout=response_timeout,
                            packet_name=operation,
                        )
                except TimeoutError as error:
                    if response_timeout is None:
                        raise
                    with suppress(Exception):
                        self.abort()
                    raise CEDTSTimeoutError(operation, response_timeout) from error
            finally:
                with self._response_condition:
                    self._pending_reply_ids.discard(request_id)
        if not response.get("ok", False):
            raise _worker_exception(
                response.get("error_type"),
                response.get(
                    "error", string("backends.worker_proxy.backend_worker_failed")
                ),
            )
        return response.get("value")

    def cancel_active_request(
        self,
        request_id: Optional[str] = None,
        *,
        wait_for_ack: bool = True,
    ) -> bool:
        """Cancel the currently active stream by its originating request ID.

        Args:
            request_id: Expected active request ID, or ``None`` to target the current stream.
            wait_for_ack: Whether to wait for the worker's ``cancel_ack`` packet.

        Returns:
            bool: Whether the worker accepted the cancellation request.
        """
        self._ensure_cancellation_state()
        with self._active_request_lock:
            active_request_id = self._active_request_id
            if active_request_id is None or (
                request_id is not None and request_id != active_request_id
            ):
                return False
            target_request_id = active_request_id
            state = self._request_cancellation_states.get(target_request_id)
            if state is None:
                state = self._terminal_cancellation_states.get(target_request_id)
            if state is None:
                state = _RequestCancellationState(target_request_id)
                self._request_cancellation_states[target_request_id] = state
            if state.terminal:
                return False
            cancel_event = state.cancel_ack_event
            if state.cancel_packet_id is None or cancel_event is None:
                cancel_event = threading.Event()
                cancel_packet_id = uuid4().hex
                state.cancel_packet_id = cancel_packet_id
                state.cancel_ack_event = cancel_event
                state.cancel_ack_result = None
                state.cancel_sent = True
                should_send = True
            else:
                cancel_packet_id = state.cancel_packet_id
                should_send = False
                if not state.cancel_sent:
                    state.cancel_sent = True
                    should_send = True
            self._sync_active_cancellation_fields(state)

        if cancel_packet_id is None:
            return False
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            return False
        if should_send:
            with self._active_request_lock:
                if state.terminal:
                    return False
            try:
                sent_cancel_packet_id = self._send_packet(
                    process.stdin,
                    "cancel",
                    "cancel",
                    {"target_message_id": target_request_id},
                    message_id=cancel_packet_id,
                )
                if sent_cancel_packet_id != cancel_packet_id:
                    with self._active_request_lock:
                        state.cancel_packet_id = sent_cancel_packet_id
                        self._sync_active_cancellation_fields(state)
            except (BrokenPipeError, OSError, CEDTSError):
                with self._active_request_lock:
                    state.cancel_sent = False
                    self._sync_active_cancellation_fields(state)
                return False
        if not wait_for_ack:
            return True
        if cancel_event is None or not cancel_event.wait(_CANCEL_ACK_TIMEOUT_SECONDS):
            return False
        with self._active_request_lock:
            return state.cancel_ack_result is True

    def _stream_request(
        self,
        operation: str,
        **arguments: WorkerValue,
    ) -> Iterator[WorkerValue]:
        """Send one streaming request and yield worker values."""
        self._ensure_response_state()
        process = self._process
        if (
            process is None
            or process.poll() is not None
            or getattr(self, "_closing", False)
        ):
            raise RuntimeError(
                string(
                    "backends.worker_not_running",
                    backend=self._manifest.backend_id,
                )
            )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError(string("backends.worker_streams_unavailable"))
        self._ensure_cancellation_state()
        with self._protocol_lock:
            if not hasattr(self, "_stream_active"):
                self._stream_active = threading.Event()
            self._stream_active.set()
            _emit_log(
                self._log_callback,
                string(
                    "backends.worker_proxy.stream_request_sent",
                    operation=operation,
                    arguments=tuple(arguments),
                ),
                loglevel="debug",
            )
            request_id = uuid4().hex
            with self._response_condition:
                self._pending_reply_ids.add(request_id)
            try:
                sent_request_id = self._send_packet(
                    process.stdin,
                    "request",
                    operation,
                    {"arguments": arguments},
                    message_id=request_id,
                )
            except Exception:
                with self._response_condition:
                    self._pending_reply_ids.discard(request_id)
                raise
            if sent_request_id != request_id:
                with self._response_condition:
                    self._pending_reply_ids.discard(request_id)
                    self._pending_reply_ids.add(sent_request_id)
                request_id = sent_request_id
            with self._active_request_lock:
                self._active_request_id = request_id
                self._request_cancellation_states[request_id] = (
                    _RequestCancellationState(request_id=request_id)
                )
                self._cancel_target = request_id
                self._cancel_packet_id = None
                self._cancel_ack_event = None
                self._cancel_ack_result = None
                self._cancel_sent = False
            completed = False
            stream_frame_count = 0
            try:
                while True:
                    try:
                        response = self._read_stream_frame(process, request_id)
                    except Exception:
                        with self._active_request_lock:
                            self._mark_request_terminal_locked(request_id)
                        completed = True
                        raise
                    if response.get("done", False):
                        _emit_log(
                            self._log_callback,
                            string(
                                "backends.worker_proxy.stream_completed",
                                frames=stream_frame_count,
                            ),
                            "info",
                            loglevel="debug",
                        )
                        with self._active_request_lock:
                            self._mark_request_terminal_locked(request_id)
                        completed = True
                        return
                    if response.get("stream", False):
                        stream_frame_count += 1
                        if stream_frame_count <= 3:
                            _emit_log(
                                self._log_callback,
                                string(
                                    "backends.worker_proxy.stream_frame_received",
                                    frame=stream_frame_count,
                                ),
                                "info",
                                loglevel="debug",
                            )
                        yield response.get("value")
            finally:
                if not completed:
                    _emit_log(
                        self._log_callback,
                        string(
                            "backends.worker_proxy.draining_incomplete_stream",
                            operation=operation,
                            frames=stream_frame_count,
                        ),
                        loglevel="debug",
                    )
                    self.cancel_active_request(request_id, wait_for_ack=False)
                    drained = self._drain_stream(process, request_id)
                    if not drained and process.poll() is None:
                        self._terminate_process(process)
                    with self._active_request_lock:
                        self._mark_request_terminal_locked(request_id)
                self._stream_active.clear()
                with self._active_request_lock:
                    if self._active_request_id == request_id:
                        self._active_request_id = None
                        self._cancel_target = None
                        self._cancel_packet_id = None
                        self._cancel_ack_event = None
                        self._cancel_ack_result = None
                        self._cancel_sent = False
                    state = self._terminal_cancellation_states.get(request_id)
                    if (
                        state is not None
                        and state.cancel_ack_event is not None
                        and state.cancel_ack_event.is_set()
                    ):
                        self._terminal_cancellation_states.pop(request_id, None)
                with self._response_condition:
                    self._pending_reply_ids.discard(request_id)
                    self._clear_response_queue_locked(request_id)

    def _read_stream_frame(
        self,
        process: subprocess.Popen[bytes],
        reply_to: str,
    ) -> WorkerResponse:
        """Read one streaming frame and raise on worker-reported failures."""
        response = self._read_response(
            process,
            reply_to,
            packet_name="generate_stream",
        )
        if not response.get("ok", False) and not response.get("cancelled", False):
            raise _worker_exception(
                response.get("error_type"),
                response.get(
                    "error", string("backends.worker_proxy.backend_worker_failed")
                ),
            )
        return response

    def _drain_stream(self, process: subprocess.Popen[bytes], reply_to: str) -> bool:
        """Consume remaining stream frames within a bounded cancellation window."""
        assert process.stdout is not None
        deadline = time.monotonic() + _STREAM_DRAIN_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                response = self._read_response(
                    process,
                    reply_to,
                    timeout=remaining,
                    packet_name="generate_stream",
                )
            except (EOFError, TimeoutError, CEDTSError, OSError, ValueError):
                return False
            if response.get("done", False) or not response.get("ok", True):
                return True

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
        value = self._request(
            "load_model",
            response_timeout=_BACKEND_MODEL_LOAD_TIMEOUT_SECONDS,
            model_id=model_id,
            **kwargs,
        )
        return RemoteModelHandle(cast(int, value))

    def preload_models(self) -> None:
        """Ensure backend models are available inside the worker environment."""
        self._request(
            "preload_models",
            response_timeout=_BACKEND_MODEL_OPERATION_TIMEOUT_SECONDS,
        )

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Unload worker-side models and clear the local handle.

        Args:
            release_cuda_cache: Whether the worker should synchronize CUDA and release cached accelerator blocks.
        """
        process = getattr(self, "_process", None)
        if (
            getattr(self, "_closed", False)
            or getattr(self, "_closing", False)
            or process is None
            or process.poll() is not None
        ):
            self.model = None
            return
        self._request("unload_model", release_cuda_cache=release_cuda_cache)
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

    def _await_shutdown_ack(
        self,
        process: subprocess.Popen[bytes],
        shutdown_id: str,
    ) -> WorkerResponse:
        """Read a shutdown acknowledgement without blocking escalation forever."""
        response = self._read_response(
            process,
            shutdown_id,
            timeout=_SHUTDOWN_ACK_TIMEOUT_SECONDS,
            packet_name="shutdown",
        )
        return response

    @staticmethod
    def _shutdown_ack_failure(response: WorkerResponse) -> Optional[BackendError]:
        """Return an error when the worker rejects the shutdown request."""
        if response.get("ok") is True:
            return None
        value = response.get("value")
        reason = response.get("error")
        if not isinstance(reason, str) and isinstance(value, dict):
            active_job_result = value.get("active_job_result")
            if isinstance(active_job_result, str):
                reason = active_job_result
        if not isinstance(reason, str):
            reason = string("backends.worker_proxy.backend_worker_failed")
        return BackendError(
            string(
                "backends.worker_proxy.shutdown_acknowledgement_failed",
                reason=reason,
            )
        )

    def _close_process_streams(
        self, process: Optional[subprocess.Popen[bytes]]
    ) -> None:
        """Close all control, binary, and diagnostic streams owned by a worker."""
        self._stop_packet_reader()
        try:
            process_streams = (
                getattr(process, "stdin", None),
                getattr(process, "stdout", None),
                getattr(process, "stderr", None),
            )
            owned_streams = (
                getattr(self, "_binary_input", None),
                getattr(self, "_binary_output", None),
            )
            for stream in process_streams + owned_streams:
                if stream is not None:
                    with suppress(OSError, ValueError):
                        stream.close()
        finally:
            self._binary_input = None
            self._binary_output = None

            stderr_thread = getattr(self, "_stderr_thread", None)
            self._stderr_thread = None
            if (
                stderr_thread is not None
                and stderr_thread is not threading.current_thread()
            ):
                stderr_thread.join(timeout=_WORKER_THREAD_JOIN_TIMEOUT_SECONDS)
            reader_thread = getattr(self, "_reader_thread", None)
            self._reader_thread = None
            if (
                reader_thread is not None
                and reader_thread is not threading.current_thread()
            ):
                reader_thread.join(timeout=_WORKER_THREAD_JOIN_TIMEOUT_SECONDS)

    def _terminate_process(self, process: subprocess.Popen[bytes]) -> None:
        """Escalate a worker that did not complete its graceful shutdown."""
        self._stop_packet_reader()
        with suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1)

    def close(self) -> None:
        """Gracefully stop the worker, escalating only after a shutdown timeout."""
        self._ensure_response_state()
        with self._close_lock:
            if self._closed:
                self._close_process_streams(self._process)
                self._clear_received_message_ids()
                self._clear_runtime_state()
                return
            self._closing = True
            process = self._process
            shutdown_failure: Optional[Exception] = None
            try:
                if process is None:
                    return
                _emit_log(
                    self._log_callback,
                    string(
                        "backends.worker_proxy.closing_worker",
                        backend=self._manifest.backend_id,
                        pid=process.pid,
                    ),
                    loglevel="debug",
                )

                if process.poll() is None and process.stdin is not None:
                    self.cancel_active_request(wait_for_ack=False)
                    shutdown_id: Optional[str] = None
                    try:
                        shutdown_id = self._send_packet(
                            process.stdin,
                            "shutdown",
                            "shutdown",
                            {"active_job_policy": "cancel"},
                        )
                    except (BrokenPipeError, OSError, CEDTSError):
                        with self._response_condition:
                            if shutdown_id is not None:
                                self._pending_reply_ids.discard(shutdown_id)
                        shutdown_id = None
                    if shutdown_id is not None:
                        try:
                            shutdown_response = self._await_shutdown_ack(
                                process, shutdown_id
                            )
                            shutdown_failure = self._shutdown_ack_failure(
                                shutdown_response
                            )
                        except (
                            BrokenPipeError,
                            OSError,
                            TimeoutError,
                            CEDTSError,
                        ) as error:
                            shutdown_failure = error
                        with self._response_condition:
                            self._pending_reply_ids.discard(shutdown_id)
                            self._clear_response_queue_locked(shutdown_id)

                if process.poll() is None:
                    try:
                        process.wait(timeout=_SHUTDOWN_ACK_TIMEOUT_SECONDS)
                    except subprocess.TimeoutExpired:
                        self._terminate_process(process)
                self._stop_packet_reader()
            finally:
                self._process = None
                self._closed = True
                self._close_process_streams(process)
                self._clear_received_message_ids()
                self._clear_runtime_state()
            if shutdown_failure is not None:
                raise shutdown_failure

    def abort(self) -> None:
        """Terminate the worker without waiting for an active generation."""
        self._closing = True
        self._stop_packet_reader()
        # Wake every CEDTS consumer before terminating the process.  A request
        # or event waiter may otherwise remain blocked on its condition while
        # the worker is stuck inside a backend-specific operation.
        self._clear_runtime_state()
        with self._close_lock:
            if self._closed and self._process is None:
                self._close_process_streams(None)
                self._clear_received_message_ids()
                return
            process = self._process
            try:
                if process is not None and process.poll() is None:
                    self._terminate_process(process)
            finally:
                self._process = None
                self._closed = True
                self._close_process_streams(process)
                self._clear_received_message_ids()
                self._clear_runtime_state()


class RemoteVCBackendProxy(CeluneVCBackend):
    """Expose a voice-conversion worker through the VC backend interface."""

    def __init__(
        self,
        manifest: BackendManifest,
        log: LogCallback,
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

    def unload_model(self, release_cuda_cache: bool = True) -> None:
        """Release voice-conversion assets in the worker.

        Args:
            release_cuda_cache: Whether the worker should synchronize CUDA and release cached accelerator blocks.
        """
        self._worker.unload_model(release_cuda_cache=release_cuda_cache)

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

    def abort(self) -> None:
        """Terminate the voice-conversion worker without waiting for conversion."""
        self._worker.abort()
