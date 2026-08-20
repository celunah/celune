# SPDX-License-Identifier: MIT
"""Worker entrypoint for running one backend in its private environment."""

import os
import sys
import json
import argparse
import threading
import traceback
from contextlib import suppress
from collections import OrderedDict
from collections.abc import Mapping, Callable
from typing import IO, Optional, cast

from ..i18n import string
from ..typing.aliases import LogLevel
from ..typing.common import JSONSerializable
from ..dataclasses.pipeline import VoiceConversionRequest
from .environment import BackendManifest, backend_manifest
from ..typing.backends import (
    BackendModel,
    BackendArguments,
    BackendDescription,
    _BackendRuntime,
)
from ..typing.worker import (
    WorkerValue,
    WorkerMessage,
    WorkerRequest,
    WorkerResponse,
    WorkerPayloadDescriptor,
)
from .worker_protocol import (
    CEDTS_VERSION,
    WORKER_CAPABILITIES,
    DEFAULT_CEDTS_LIMITS,
    SUPPORTED_OPERATIONS,
    CEDTSLimits,
    WorkerProtocolError,
    build_packet,
    send_message,
    send_payloads,
    decode_message,
    encode_message,
    receive_message,
    receive_payloads,
    limits_from_capabilities,
)

_WORKER_STDERR = sys.stderr
# Retain recent packet IDs to reject replayed packets without growing state for
# the lifetime of a long-running worker. IDs outside this replay window may be
# reused by a peer after the window has elapsed.
_MESSAGE_ID_REPLAY_WINDOW = 4096
_CALL_ARGUMENT_FIELDS = {
    "resolve_generation_language": frozenset({"method", "lang"}),
    "should_reload_for_language": frozenset({"method", "lang"}),
    "convert_live": frozenset({"method", "request"}),
    "stop_live": frozenset({"method"}),
}
_CANCELLABLE_OPERATIONS = frozenset({"generate_stream"})
_REQUEST_FIELDS = frozenset({"operation", "arguments"})
_MAX_BACKEND_ARGUMENT_FIELDS = 128
_MAX_ARGUMENT_NAME_LENGTH = 128
_SHUTDOWN_FINISH_TIMEOUT_SECONDS = 5.0
_SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 5.0


def _worker_protocol_error(key: str, **kwargs: str) -> WorkerProtocolError:
    """Create a localized worker protocol error."""
    return WorkerProtocolError(string(f"backends.worker_runtime.{key}", **kwargs))


def _remember_message_id(
    received_message_ids: OrderedDict[str, None], message_id: str
) -> bool:
    """Remember one packet ID and return whether it is outside the replay window."""
    if message_id in received_message_ids:
        return False
    received_message_ids[message_id] = None
    received_message_ids.move_to_end(message_id)
    if len(received_message_ids) > _MESSAGE_ID_REPLAY_WINDOW:
        received_message_ids.popitem(last=False)
    return True


def _mini_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved Mini backend constructor on demand."""
    from .tts.mini import Mini

    return cast(Callable[..., _BackendRuntime], Mini)


def _qwen3_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved Qwen3 backend constructor on demand."""
    from .tts.qwen3 import Qwen3

    return cast(Callable[..., _BackendRuntime], Qwen3)


def _dotstts_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved DotsTTS backend constructor on demand."""
    from .tts.dotstts import DotsTtsMF

    return cast(Callable[..., _BackendRuntime], DotsTtsMF)


def _voxcpm2_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved VoxCPM2 backend constructor on demand."""
    from .tts.voxcpm2 import VoxCPM2

    return cast(Callable[..., _BackendRuntime], VoxCPM2)


def _gpt_sovits_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved GPT-SoVITS backend constructor on demand."""
    from .tts.gpt_sovits import GPTSoVITS

    return cast(Callable[..., _BackendRuntime], GPTSoVITS)


def _seed_vc_constructor() -> Callable[..., _BackendRuntime]:
    """Load the approved Seed-VC backend constructor on demand."""
    from .vc.seedvc import CeluneSeedVCBackend

    return cast(Callable[..., _BackendRuntime], CeluneSeedVCBackend)


_BACKEND_REGISTRY: Mapping[
    str, tuple[str, Callable[[], Callable[..., _BackendRuntime]]]
] = {
    "mini": ("tts", _mini_constructor),
    "qwen3": ("tts", _qwen3_constructor),
    "dotstts": ("tts", _dotstts_constructor),
    "voxcpm2": ("tts", _voxcpm2_constructor),
    "gpt-sovits": ("tts", _gpt_sovits_constructor),
    "seed-vc": ("vc", _seed_vc_constructor),
}


def _parse_args() -> argparse.Namespace:
    """Parse the private worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-kwargs", default="{}")
    parser.add_argument("--binary-input-fd", type=int)
    parser.add_argument("--binary-output-fd", type=int)
    parser.add_argument("--binary-input-handle", type=int)
    parser.add_argument("--binary-output-handle", type=int)
    args = parser.parse_args()
    fd_mode = args.binary_input_fd is not None and args.binary_output_fd is not None
    handle_mode = (
        args.binary_input_handle is not None and args.binary_output_handle is not None
    )
    if fd_mode == handle_mode:
        parser.error("exactly one complete binary channel descriptor pair is required")
    return args


def _open_binary_streams(args: argparse.Namespace) -> tuple[IO[bytes], IO[bytes]]:
    """Open the worker's binary channels from file descriptors or Windows handles."""
    binary_input_fd: Optional[int] = None
    binary_output_fd: Optional[int] = None
    binary_input: Optional[IO[bytes]] = None
    binary_output: Optional[IO[bytes]] = None
    try:
        if args.binary_input_handle is not None:
            import msvcrt  # pylint: disable=E0401

            binary_flags = getattr(os, "O_BINARY", 0)
            open_osfhandle = cast(
                Callable[[int, int], int],
                msvcrt.open_osfhandle,  # type: ignore[missing-attribute]
            )
            binary_input_fd = open_osfhandle(
                args.binary_input_handle,
                os.O_RDONLY | binary_flags,
            )
            binary_output_fd = open_osfhandle(
                args.binary_output_handle,
                os.O_WRONLY | binary_flags,
            )
        else:
            binary_input_fd = args.binary_input_fd
            binary_output_fd = args.binary_output_fd
        binary_input = os.fdopen(binary_input_fd, "rb", buffering=0)
        binary_input_fd = None
        binary_output = os.fdopen(binary_output_fd, "wb", buffering=0)
        binary_output_fd = None
    except Exception:
        for stream in (binary_input, binary_output):
            if stream is not None:
                with suppress(OSError, ValueError):
                    stream.close()
        for descriptor in (binary_input_fd, binary_output_fd):
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)
        raise
    assert binary_input is not None
    assert binary_output is not None
    return binary_input, binary_output


def _detach_protocol_stream() -> IO[bytes]:
    """Reserve stdout for CEDTS packets and redirect backend output to stderr."""
    stdout_fd = sys.stdout.fileno()
    protocol_fd = os.dup(stdout_fd)
    try:
        os.dup2(sys.stderr.fileno(), stdout_fd)
    except Exception:
        os.close(protocol_fd)
        raise
    return os.fdopen(protocol_fd, "wb", buffering=0)


def _load_backend(
    manifest: BackendManifest,
    log: Callable[[str, str], None],
    fatal: Callable[[], None],
    kwargs: BackendArguments,
) -> _BackendRuntime:
    """Construct one backend from the worker's approved static registry."""
    try:
        expected_kind, constructor_loader = _BACKEND_REGISTRY[manifest.backend_id]
    except KeyError as error:
        raise WorkerProtocolError(
            string(
                "celune.unknown_backend",
                backend=manifest.backend_id,
                available=", ".join(_BACKEND_REGISTRY),
            )
        ) from error
    if manifest.kind != expected_kind:
        raise _worker_protocol_error("backend_worker_backend_kind_is_invalid")

    constructor = constructor_loader()
    if expected_kind == "vc":
        return constructor(log=log)
    return constructor(log=log, fatal=fatal, **kwargs)


def _describe(backend: _BackendRuntime) -> BackendDescription:
    """Return the backend metadata required by the core proxy."""
    return cast(
        BackendDescription,
        {
            "name": getattr(backend, "name", "unknown"),
            "chunk_rate": getattr(backend, "chunk_rate", 0.0),
            "supported_languages": list(getattr(backend, "supported_languages", ())),
            "voice_models": getattr(backend, "voice_models", None),
            "default_voice": getattr(backend, "default_voice", None),
            "model_name": getattr(backend, "model_name", None),
            "voices": getattr(backend, "voices", []),
            "clone_model_id": getattr(backend, "clone_model_id", None),
            "uses_voice_bundles": getattr(backend, "uses_voice_bundles", False),
            "max_new_tokens": getattr(backend, "max_new_tokens", 512),
            "is_fake": getattr(backend, "is_fake", False),
        },
    )


def _worker_log(
    message: str,
    severity: str = "info",
    *,
    loglevel: LogLevel = "info",
) -> None:
    """Write backend logs away from the binary protocol stream."""
    prefix = f"[{severity}]" if loglevel == "info" else f"[{loglevel}][{severity}]"
    print(f"{prefix} {message}", file=_WORKER_STDERR, flush=True)


def _error_response(error: Exception) -> WorkerResponse:
    """Build a response with the worker type name as inert diagnostic data."""
    traceback.print_exc(file=_WORKER_STDERR)
    error_type = type(error)
    return {
        "ok": False,
        "error": str(error),
        "error_type": f"{error_type.__module__}.{error_type.__qualname__}",
    }


def _release_worker_models(models: dict[int, BackendModel]) -> None:
    """Best-effort close every runtime held by the worker model table."""
    for model in tuple(models.values()):
        try:
            close = getattr(model, "close", None)
        except Exception:
            close = None
        if callable(close):
            with suppress(Exception):
                close()
            continue
        try:
            unload = getattr(model, "unload", None)
        except Exception:
            unload = None
        if callable(unload):
            with suppress(Exception):
                unload()
    models.clear()


def _stream_value_summary(value: WorkerValue) -> str:
    """Summarize one streamed audio value without importing backend packages."""
    audio = value[0] if isinstance(value, tuple) and value else value
    audio_type = f"{type(audio).__module__}.{type(audio).__name__}"
    shape = getattr(audio, "shape", None)
    size = getattr(audio, "size", None)
    numel = getattr(audio, "numel", None)
    if callable(numel):
        sample_count = cast(int, numel())
    elif isinstance(size, int):
        sample_count = size
    else:
        try:
            sample_count = len(audio)  # type: ignore[arg-type]
        except TypeError:
            sample_count = -1

    nonzero: Optional[bool] = None
    any_method = getattr(audio, "any", None)
    if callable(any_method):
        try:
            nonzero = bool(any_method())
        except (RuntimeError, TypeError, ValueError):
            pass
    return (
        f"type={audio_type} shape={shape!s} dtype={getattr(audio, 'dtype', None)!s} "
        f"samples={sample_count} nonzero={nonzero!s}"
    )


def _backend_method_map(
    backend: _BackendRuntime,
) -> dict[str, Callable[..., WorkerValue]]:
    """Return only the explicitly supported backend callback methods."""
    method_map: dict[str, Callable[..., WorkerValue]] = {}
    for method_name in _CALL_ARGUMENT_FIELDS:
        method = getattr(backend, method_name, None)
        if callable(method):
            method_map[method_name] = cast(Callable[..., WorkerValue], method)
    return method_map


def _validate_request_arguments(
    operation: object,
    arguments: object,
    backend: Optional[_BackendRuntime] = None,
) -> dict[str, WorkerValue]:
    """Validate one request's operation, argument object, and method allowlist."""
    if not isinstance(operation, str) or operation not in SUPPORTED_OPERATIONS:
        raise _worker_protocol_error("backend_worker_operation_is_unknown")
    if not isinstance(arguments, dict):
        raise _worker_protocol_error(
            "backend_worker_request_arguments_are_not_an_object"
        )
    if not all(isinstance(key, str) for key in arguments):
        raise _worker_protocol_error("backend_worker_request_argument_name_is_invalid")
    checked = cast(dict[str, WorkerValue], dict(arguments))

    def require_fields(
        required: frozenset[str],
        optional: frozenset[str] = frozenset(),
    ) -> None:
        allowed = required | optional
        if not required.issubset(checked) or not set(checked).issubset(allowed):
            raise _worker_protocol_error("backend_worker_request_arguments_are_invalid")

    if operation in {"describe", "preload_models"}:
        require_fields(frozenset())
    elif operation == "model_is_available_locally":
        require_fields(frozenset({"model"}), frozenset({"lang"}))
        if not isinstance(checked["model"], str) or (
            "lang" in checked
            and checked["lang"] is not None
            and not isinstance(checked["lang"], str)
        ):
            raise _worker_protocol_error("backend_worker_model_arguments_are_invalid")
    elif operation == "load_model":
        if "model_id" not in checked or not isinstance(checked["model_id"], str):
            raise _worker_protocol_error("backend_worker_model_id_is_invalid")
        _validate_backend_arguments(checked, {"model_id"})
    elif operation == "unload_model":
        require_fields(frozenset(), frozenset({"release_cuda_cache"}))
        if "release_cuda_cache" in checked and not isinstance(
            checked["release_cuda_cache"], bool
        ):
            raise _worker_protocol_error("backend_worker_unload_arguments_are_invalid")
    elif operation == "convert":
        require_fields(frozenset({"request"}))
        if not isinstance(checked["request"], VoiceConversionRequest):
            raise _worker_protocol_error("backend_worker_conversion_request_is_invalid")
    elif operation == "generate_stream":
        if not isinstance(checked.get("model_id"), int) or isinstance(
            checked.get("model_id"), bool
        ):
            raise _worker_protocol_error("backend_worker_model_handle_is_invalid")
        _validate_backend_arguments(checked, {"model_id"})
    elif operation == "call":
        method_name = checked.get("method")
        if not isinstance(method_name, str) or method_name not in _CALL_ARGUMENT_FIELDS:
            raise _worker_protocol_error("backend_worker_method_is_not_allowed")
        allowed = _CALL_ARGUMENT_FIELDS[method_name]
        if not set(checked).issubset(allowed):
            raise _worker_protocol_error("backend_worker_method_arguments_are_invalid")
        required = {
            "resolve_generation_language": {"method", "lang"},
            "should_reload_for_language": {"method", "lang"},
            "convert_live": {"method", "request"},
            "stop_live": {"method"},
        }[method_name]
        if not required.issubset(checked):
            raise _worker_protocol_error("backend_worker_method_arguments_are_invalid")
        if (
            method_name != "stop_live"
            and "lang" in checked
            and checked["lang"] is not None
            and not isinstance(checked["lang"], str)
        ):
            raise _worker_protocol_error("backend_worker_language_argument_is_invalid")
        if method_name == "convert_live" and not isinstance(
            checked.get("request"), VoiceConversionRequest
        ):
            raise _worker_protocol_error(
                "backend_worker_live_conversion_request_is_invalid"
            )
        if backend is not None and method_name not in _backend_method_map(backend):
            raise _worker_protocol_error("backend_worker_method_is_unavailable")
    for name, value in checked.items():
        if name == "request" and (
            operation == "convert"
            or (operation == "call" and checked.get("method") == "convert_live")
        ):
            continue
        _validate_json_argument(value)
    return checked


def _validate_json_argument(value: object, depth: int = 0) -> None:
    """Reject non-JSON values in operation arguments after payload decoding."""
    if depth > 64:
        raise _worker_protocol_error(
            "backend_worker_request_arguments_are_nested_too_deeply"
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_argument(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _worker_protocol_error(
                    "backend_worker_request_argument_name_is_invalid"
                )
            _validate_json_argument(item, depth + 1)
        return
    raise _worker_protocol_error("backend_worker_request_argument_type_is_invalid")


def _validate_backend_arguments(
    arguments: dict[str, WorkerValue],
    reserved: set[str],
) -> None:
    """Validate bounded backend-specific keyword arguments after decoding."""
    if len(arguments) - len(reserved) > _MAX_BACKEND_ARGUMENT_FIELDS:
        raise _worker_protocol_error(
            "backend_worker_backend_arguments_contain_too_many_fields"
        )
    for name, value in arguments.items():
        if name in reserved:
            continue
        if not name or len(name) > _MAX_ARGUMENT_NAME_LENGTH:
            raise _worker_protocol_error("backend_worker_argument_name_is_invalid")
        _validate_json_argument(value)


def _run_request(
    backend: _BackendRuntime,
    request: WorkerRequest,
    models: dict[int, BackendModel],
    next_model_id: int,
    protocol_stream: IO[bytes],
    binary_stream: Optional[IO[bytes]] = None,
    reply_to: Optional[str] = None,
    cancellation_event: Optional[threading.Event] = None,
    send_lock: Optional[threading.Lock] = None,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> tuple[WorkerResponse, int]:
    """Run one backend request and return its response and next model ID."""
    if not isinstance(request, dict) or not set(request).issubset(_REQUEST_FIELDS):
        raise _worker_protocol_error("backend_worker_request_object_is_invalid")
    operation = request.get("operation")
    arguments = _validate_request_arguments(
        operation,
        request.get("arguments", {}),
        backend,
    )
    _worker_log(
        string(
            "backends.worker_runtime.request_received",
            operation=operation,
            arguments=tuple(arguments),
        ),
        loglevel="debug",
    )
    if operation == "describe":
        return {"ok": True, "value": _describe(backend)}, next_model_id
    if operation == "model_is_available_locally":
        value = backend.model_is_available_locally(**cast(BackendArguments, arguments))
        return {"ok": True, "value": value}, next_model_id
    if operation == "preload_models":
        backend.preload_models()
        return {"ok": True, "value": None}, next_model_id
    if operation == "load_model":
        model = backend.load_model(**cast(BackendArguments, arguments))
        backend.model = model
        model_id = next_model_id
        models[model_id] = model
        return {"ok": True, "value": model_id}, next_model_id + 1
    if operation == "unload_model":
        try:
            backend.unload_model(
                release_cuda_cache=bool(arguments.get("release_cuda_cache", True))
            )
        finally:
            _release_worker_models(models)
        return {"ok": True, "value": None}, next_model_id
    if operation == "convert":
        return {
            "ok": True,
            "value": backend.convert(
                cast(VoiceConversionRequest, arguments["request"])
            ),
        }, next_model_id
    if operation == "call":
        method_name = cast(str, arguments["method"])
        method = _backend_method_map(backend).get(method_name)
        if method is None:
            raise _worker_protocol_error("backend_worker_method_is_unavailable")
        call_arguments = dict(arguments)
        del call_arguments["method"]
        return {
            "ok": True,
            "value": method(**call_arguments),
        }, next_model_id
    if operation == "generate_stream":
        model_id = cast(int, arguments["model_id"])
        model = models.get(model_id)
        if model is None:
            raise ValueError(string("backends.worker_model_missing", model_id=model_id))
        stream_arguments = dict(arguments)
        del stream_arguments["model_id"]
        if cancellation_event is not None and cancellation_event.is_set():
            return {"ok": False, "cancelled": True, "done": True}, next_model_id
        generator = backend.generate_stream(
            model,
            **cast(BackendArguments, stream_arguments),
        )
        stream_frame_count = 0
        for chunk in generator:
            if cancellation_event is not None and cancellation_event.is_set():
                close = getattr(generator, "close", None)
                if callable(close):
                    with suppress(Exception):
                        close()
                return {"ok": False, "cancelled": True, "done": True}, next_model_id
            stream_frame_count += 1
            if stream_frame_count <= 3:
                _worker_log(
                    string(
                        "backends.worker_runtime.stream_frame_emitted",
                        frame=stream_frame_count,
                        summary=_stream_value_summary(chunk),
                    ),
                    loglevel="debug",
                )
            _send_message(
                protocol_stream,
                binary_stream,
                build_packet(
                    "response",
                    cast(str, operation),
                    {"ok": True, "stream": True, "value": chunk},
                    reply_to=reply_to,
                ),
                send_lock,
                limits=limits,
            )
        if cancellation_event is not None and cancellation_event.is_set():
            return {"ok": False, "cancelled": True, "done": True}, next_model_id
        _worker_log(
            string(
                "backends.worker_runtime.stream_completed",
                frames=stream_frame_count,
            ),
            loglevel="debug",
        )
        return {"ok": True, "done": True}, next_model_id
    raise ValueError(string("backends.worker_unknown_operation", operation=operation))


def _send_message(
    protocol_stream: IO[bytes],
    binary_stream: Optional[IO[bytes]],
    message: WorkerMessage,
    send_lock: Optional[threading.Lock] = None,
    *,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Encode and send one worker control message and its binary payloads."""

    def send() -> None:
        """Write one complete control-and-binary message."""
        control, payloads = encode_message(message, limits=limits)
        send_message(protocol_stream, control, limits=limits)
        if binary_stream is None:
            if payloads:
                raise RuntimeError(
                    string(
                        "backends.worker_runtime.worker_binary_output_stream_is_unavailable"
                    )
                )
            return
        send_payloads(binary_stream, payloads, limits=limits)

    if send_lock is None:
        send()
    else:
        with send_lock:
            send()


def _send_error(
    protocol_stream: IO[bytes],
    binary_stream: IO[bytes],
    operation: str,
    error: Exception,
    *,
    reply_to: Optional[str] = None,
    send_lock: Optional[threading.Lock] = None,
    limits: CEDTSLimits = DEFAULT_CEDTS_LIMITS,
) -> None:
    """Send one structured CEDTS error packet."""
    _send_message(
        protocol_stream,
        binary_stream,
        build_packet(
            "error",
            operation,
            cast(dict[str, WorkerValue], _error_response(error)),
            reply_to=reply_to,
        ),
        send_lock,
        limits=limits,
    )


def _negotiate_hello(control: Mapping[str, object]) -> dict[str, JSONSerializable]:
    """Validate a core hello and return the worker's negotiated capabilities."""
    if control.get("cedts_version") != CEDTS_VERSION:
        raise _worker_protocol_error("unsupported_cedts_packet_version")
    if control.get("kind") != "hello" or control.get("operation") != "handshake":
        raise _worker_protocol_error("worker_expected_a_cedts_hello_packet")
    data = control.get("data")
    if not isinstance(data, dict):
        raise _worker_protocol_error("cedts_hello_data_is_invalid")
    versions = data.get("versions")
    if not isinstance(versions, list) or CEDTS_VERSION not in versions:
        raise _worker_protocol_error("no_compatible_cedts_version_was_offered")
    offered = data.get("capabilities")
    if not isinstance(offered, dict):
        raise _worker_protocol_error("cedts_capabilities_are_invalid")
    negotiated: dict[str, JSONSerializable] = {}
    for name in ("supported_operations", "supported_media_types"):
        offered_values = offered.get(name)
        worker_values = WORKER_CAPABILITIES[name]
        if not isinstance(offered_values, list) or not isinstance(worker_values, list):
            raise _worker_protocol_error("cedts_capability_is_invalid", name=name)
        negotiated[name] = [value for value in worker_values if value in offered_values]
    if "describe" not in cast(list[str], negotiated["supported_operations"]):
        raise _worker_protocol_error("no_compatible_backend_description_capability")
    for name in (
        "max_control_frame_size",
        "max_binary_frame_size",
        "max_aggregate_payload_size",
        "max_payload_descriptors",
        "max_json_depth",
        "max_string_length",
        "max_collection_entries",
    ):
        offered_limit = offered.get(name, getattr(DEFAULT_CEDTS_LIMITS, name))
        worker_limit = WORKER_CAPABILITIES[name]
        if not isinstance(offered_limit, int) or not isinstance(worker_limit, int):
            raise _worker_protocol_error("cedts_capability_is_invalid", name=name)
        negotiated[name] = min(offered_limit, worker_limit)
    for name in ("streaming", "cancellation", "callback"):
        offered_value = offered.get(name)
        worker_value = WORKER_CAPABILITIES[name]
        if not isinstance(offered_value, bool) or not isinstance(worker_value, bool):
            raise _worker_protocol_error("cedts_capability_is_invalid", name=name)
        negotiated[name] = offered_value and worker_value
    required = data.get("required_capabilities", {})
    if not isinstance(required, dict):
        raise _worker_protocol_error("cedts_required_capabilities_are_invalid")
    for name, value in required.items():
        if negotiated.get(name) != value:
            raise _worker_protocol_error("incompatible_cedts_capability", name=name)
    return negotiated


def main() -> int:
    """Run the backend request loop until the core closes the pipe."""
    args = _parse_args()
    protocol_stream = _detach_protocol_stream()
    binary_input, binary_output = _open_binary_streams(args)
    manifest = backend_manifest(args.backend)
    backend_kwargs = cast(BackendArguments, json.loads(args.backend_kwargs))
    send_lock = threading.Lock()
    effective_limits = DEFAULT_CEDTS_LIMITS

    def _worker_fatal() -> None:
        """Notify the proxy that the backend entered its fatal state."""
        _send_message(
            protocol_stream,
            binary_output,
            build_packet(
                "event",
                "fatal",
                {"fatal": True},
            ),
            send_lock,
            limits=effective_limits,
        )

    def _receive_packet() -> tuple[WorkerMessage, dict[str, WorkerValue]]:
        """Receive one CEDTS packet and its validated binary payloads."""
        control = receive_message(sys.stdin.buffer, limits=effective_limits)
        descriptors = control.get("payloads", [])
        if not isinstance(descriptors, list):
            raise _worker_protocol_error("worker_payload_descriptors_are_invalid")
        typed_descriptors = cast(list[WorkerPayloadDescriptor], descriptors)
        payloads = receive_payloads(
            binary_input,
            typed_descriptors,
            limits=effective_limits,
        )
        return control, decode_message(
            control,
            payloads,
            limits=effective_limits,
        )

    hello: WorkerMessage = {}
    try:
        hello, _ = _receive_packet()
        hello_id = cast(str, hello["message_id"])
        negotiated_capabilities = _negotiate_hello(hello)
        effective_limits = limits_from_capabilities(negotiated_capabilities)
    except (EOFError, BrokenPipeError):
        return 0
    except Exception as error:
        _send_error(
            protocol_stream,
            binary_output,
            "handshake",
            error,
            reply_to=(
                cast(str, hello.get("message_id"))
                if isinstance(hello.get("message_id"), str)
                else None
            ),
            send_lock=send_lock,
        )
        return 1

    _send_message(
        protocol_stream,
        binary_output,
        build_packet(
            "hello_ack",
            "handshake",
            {
                "cedts_version": CEDTS_VERSION,
                "capabilities": negotiated_capabilities,
            },
            reply_to=hello_id,
        ),
        send_lock,
        limits=effective_limits,
    )
    try:
        backend = _load_backend(manifest, _worker_log, _worker_fatal, backend_kwargs)
    except Exception as error:
        _send_error(
            protocol_stream,
            binary_output,
            "handshake",
            error,
            reply_to=hello_id,
            send_lock=send_lock,
            limits=effective_limits,
        )
        return 1
    _send_message(
        protocol_stream,
        binary_output,
        build_packet(
            "ready",
            "ready",
            {"capabilities": negotiated_capabilities},
            reply_to=hello_id,
        ),
        send_lock,
        limits=effective_limits,
    )
    models: dict[int, BackendModel] = {}
    next_model_id = 1
    received_message_ids: OrderedDict[str, None] = OrderedDict()
    active_request_lock = threading.Lock()
    active_request_id: Optional[str] = None
    active_request_operation: Optional[str] = None
    active_cancel_event: Optional[threading.Event] = None
    active_request_terminal = False
    active_thread: Optional[threading.Thread] = None
    shutting_down = False

    def _close_worker_streams() -> None:
        """Close both CEDTS channels after the worker has sent its final packet."""
        for stream in (
            protocol_stream,
            binary_input,
            binary_output,
            sys.stdin.buffer,
        ):
            with suppress(OSError, ValueError):
                stream.close()

    def _run_active_request(
        request: WorkerRequest,
        request_id: str,
        cancel_event: threading.Event,
    ) -> None:
        """Run one backend operation while the control loop handles packets."""
        nonlocal active_request_id, active_request_operation
        nonlocal active_cancel_event, active_request_terminal
        nonlocal active_thread, next_model_id
        response_kind = "response"
        try:
            response, updated_model_id = _run_request(
                backend,
                request,
                models,
                next_model_id,
                protocol_stream,
                binary_output,
                reply_to=request_id,
                cancellation_event=cancel_event,
                send_lock=send_lock,
                limits=effective_limits,
            )
            next_model_id = updated_model_id
        except Exception as error:
            response = _error_response(error)
            response_kind = "error"
            _worker_log(
                string(
                    "backends.worker_runtime.response_failed",
                    operation=request.get("operation"),
                    error=type(error).__name__,
                ),
                "error",
            )
        else:
            _worker_log(
                string(
                    "backends.worker_runtime.response_completed",
                    operation=request.get("operation"),
                    ok=response.get("ok", False),
                ),
                loglevel="debug",
            )
        with active_request_lock:
            cancellation_accepted = (
                active_request_id == request_id and cancel_event.is_set()
            )
            if active_request_id == request_id:
                active_request_terminal = True
        if (
            cancellation_accepted
            and request.get("operation") == "generate_stream"
            and response_kind == "response"
            and response.get("ok") is True
            and response.get("done") is True
        ):
            response = {"ok": False, "cancelled": True, "done": True}
        try:
            _send_message(
                protocol_stream,
                binary_output,
                build_packet(
                    response_kind,
                    cast(str, request.get("operation")),
                    cast(dict[str, WorkerValue], response),
                    reply_to=request_id,
                ),
                send_lock,
                limits=effective_limits,
            )
        finally:
            with active_request_lock:
                if active_request_id == request_id:
                    active_request_id = None
                    active_request_operation = None
                    active_cancel_event = None
                    active_request_terminal = False
                    active_thread = None

    while True:
        try:
            control, decoded = _receive_packet()
        except (EOFError, BrokenPipeError):
            return 0
        except Exception as error:
            print(str(error), file=_WORKER_STDERR)
            return 1
        packet_kind = control.get("kind")
        packet_operation = control.get("operation")
        packet_id = control.get("message_id")
        if control.get("cedts_version") != CEDTS_VERSION:
            _send_error(
                protocol_stream,
                binary_output,
                "protocol",
                _worker_protocol_error("unsupported_cedts_packet_version"),
                reply_to=cast(Optional[str], packet_id),
                send_lock=send_lock,
                limits=effective_limits,
            )
            continue
        if not isinstance(packet_id, str) or not _remember_message_id(
            received_message_ids, packet_id
        ):
            _send_error(
                protocol_stream,
                binary_output,
                "protocol",
                _worker_protocol_error("duplicate_or_invalid_request_message_id"),
                reply_to=cast(Optional[str], packet_id),
                send_lock=send_lock,
                limits=effective_limits,
            )
            continue
        if packet_kind == "cancel":
            data = decoded.get("data")
            cancel_data = (
                cast(dict[str, WorkerValue], data) if isinstance(data, dict) else {}
            )
            target_value = cancel_data.get("target_message_id")
            target_request_id = target_value if isinstance(target_value, str) else None
            with active_request_lock:
                accepted = (
                    isinstance(target_request_id, str)
                    and target_request_id == active_request_id
                    and active_request_operation in _CANCELLABLE_OPERATIONS
                    and not active_request_terminal
                    and active_cancel_event is not None
                )
                if accepted and active_cancel_event is not None:
                    active_cancel_event.set()
            _send_message(
                protocol_stream,
                binary_output,
                build_packet(
                    "cancel_ack",
                    "cancel",
                    {
                        "ok": accepted,
                        "cancelled": accepted,
                        "target_message_id": target_request_id,
                    },
                    reply_to=packet_id,
                ),
                send_lock,
                limits=effective_limits,
            )
            continue
        if packet_kind == "shutdown" or packet_operation == "shutdown":
            shutdown_data = decoded.get("data")
            shutdown_policy = (
                shutdown_data.get("active_job_policy")
                if isinstance(shutdown_data, dict)
                else None
            )
            if shutdown_policy not in {"cancel", "finish"}:
                _send_error(
                    protocol_stream,
                    binary_output,
                    "shutdown",
                    _worker_protocol_error("shutdown_active_job_policy_is_invalid"),
                    reply_to=cast(Optional[str], packet_id),
                    send_lock=send_lock,
                    limits=effective_limits,
                )
                continue
            shutting_down = True
            with active_request_lock:
                active_job = active_request_id is not None
                running_thread = active_thread
                shutdown_cancel_event = active_cancel_event
                shutdown_can_cancel = (
                    active_request_operation in _CANCELLABLE_OPERATIONS
                )
                if (
                    shutdown_policy == "cancel"
                    and shutdown_can_cancel
                    and shutdown_cancel_event is not None
                ):
                    shutdown_cancel_event.set()
            if running_thread is not None and active_job:
                if shutdown_policy == "finish":
                    running_thread.join(timeout=_SHUTDOWN_FINISH_TIMEOUT_SECONDS)
                    if (
                        running_thread.is_alive()
                        and shutdown_can_cancel
                        and shutdown_cancel_event is not None
                    ):
                        shutdown_cancel_event.set()
                        running_thread.join(timeout=_SHUTDOWN_FINISH_TIMEOUT_SECONDS)
                else:
                    running_thread.join(timeout=_SHUTDOWN_CANCEL_TIMEOUT_SECONDS)
            active_job_completed = (
                not active_job
                or running_thread is None
                or not running_thread.is_alive()
            )
            active_job_cancelled = (
                active_job
                and shutdown_can_cancel
                and shutdown_cancel_event is not None
                and shutdown_cancel_event.is_set()
            )
            if not active_job:
                active_job_result = "none"
            elif not active_job_completed:
                active_job_result = "timed_out"
            elif active_job_cancelled:
                active_job_result = "cancelled"
            else:
                active_job_result = "finished"
            shutdown_error: Optional[Exception] = None
            if active_job_completed:
                try:
                    backend.unload_model()
                except Exception as error:
                    shutdown_error = error
                    _worker_log(
                        string(
                            "backends.worker_runtime.shutdown_cleanup_failed",
                            error=type(error).__name__,
                        ),
                        "error",
                    )
                _release_worker_models(models)
            try:
                _send_message(
                    protocol_stream,
                    binary_output,
                    build_packet(
                        "shutdown_ack",
                        "shutdown",
                        {
                            "ok": shutdown_error is None and active_job_completed,
                            "value": cast(
                                dict[str, WorkerValue],
                                {
                                    "active_job_policy": shutdown_policy,
                                    "active_job_result": active_job_result,
                                    "active_job_cancelled": active_job_cancelled,
                                    "active_job_completed": active_job_completed,
                                },
                            ),
                        },
                        reply_to=packet_id,
                    ),
                    send_lock,
                    limits=effective_limits,
                )
            finally:
                received_message_ids.clear()
                _close_worker_streams()
            return 0
        if packet_kind != "request" or not isinstance(packet_id, str):
            _send_error(
                protocol_stream,
                binary_output,
                "protocol",
                _worker_protocol_error("worker_expected_a_request_packet"),
                reply_to=cast(Optional[str], packet_id),
                send_lock=send_lock,
                limits=effective_limits,
            )
            continue
        data = decoded.get("data")
        request_data = cast(dict[str, WorkerValue], data)
        if (
            not isinstance(data, dict)
            or set(data) != {"arguments"}
            or not isinstance(request_data.get("arguments"), dict)
        ):
            _send_error(
                protocol_stream,
                binary_output,
                "protocol",
                _worker_protocol_error("worker_request_data_is_invalid"),
                reply_to=packet_id,
                send_lock=send_lock,
                limits=effective_limits,
            )
            continue
        request = cast(
            WorkerRequest,
            {
                "operation": packet_operation,
                "arguments": request_data["arguments"],
            },
        )
        with active_request_lock:
            request_is_active = active_request_id is not None
        if shutting_down or request_is_active:
            _send_error(
                protocol_stream,
                binary_output,
                cast(str, packet_operation),
                _worker_protocol_error("worker_already_has_an_active_request"),
                reply_to=packet_id,
                send_lock=send_lock,
                limits=effective_limits,
            )
            continue
        cancel_event = threading.Event()
        request_thread = threading.Thread(
            target=_run_active_request,
            args=(request, packet_id, cancel_event),
            daemon=True,
        )
        with active_request_lock:
            active_request_id = packet_id
            active_request_operation = cast(str, packet_operation)
            active_cancel_event = cancel_event
            active_request_terminal = False
            active_thread = request_thread
        request_thread.start()
        continue


if __name__ == "__main__":
    raise SystemExit(main())
