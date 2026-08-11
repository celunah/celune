# SPDX-License-Identifier: MIT
"""Worker entrypoint for running one backend in its private environment."""

import argparse
import importlib
import json
import sys
import traceback
from collections.abc import Callable
from contextlib import suppress
from typing import IO, Optional, cast

from .environment import BackendManifest, backend_manifest
from .worker_protocol import receive_message, send_message
from ..dataclasses.pipeline import VoiceConversionRequest
from ..typing.backends import (
    BackendArguments,
    BackendDescription,
    BackendModel,
    _BackendRuntime,
)
from ..typing.worker import WorkerMessage, WorkerRequest, WorkerResponse, WorkerValue

_WORKER_STDERR = sys.stderr


def _parse_args() -> argparse.Namespace:
    """Parse the private worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-kwargs", default="{}")
    return parser.parse_args()


def _load_backend(
    manifest: BackendManifest,
    log: Callable[[str, str], None],
    fatal: Callable[[], None],
    kwargs: BackendArguments,
) -> _BackendRuntime:
    """Import and construct the backend described by one manifest."""
    module = importlib.import_module(manifest.backend_module)
    backend_class = getattr(module, manifest.backend_class)
    if manifest.kind == "vc":
        return cast(_BackendRuntime, backend_class(log=log))
    return cast(_BackendRuntime, backend_class(log=log, fatal=fatal, **kwargs))


def _describe(backend: _BackendRuntime) -> BackendDescription:
    """Return the backend metadata required by the core proxy."""
    return cast(
        BackendDescription,
        {
            "name": getattr(backend, "name", "unknown"),
            "chunk_rate": getattr(backend, "chunk_rate", 0.0),
            "supported_languages": getattr(backend, "supported_languages", ()),
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


def _worker_log(message: str, severity: str = "info") -> None:
    """Write backend logs away from the binary protocol stream."""
    print(f"[{severity}] {message}", file=_WORKER_STDERR, flush=True)


def _error_response(error: Exception) -> WorkerResponse:
    """Build a protocol response that preserves the worker exception type."""
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


def _run_request(
    backend: _BackendRuntime,
    request: WorkerRequest,
    models: dict[int, BackendModel],
    next_model_id: int,
    protocol_stream: IO[bytes],
) -> tuple[WorkerResponse, int]:
    """Run one backend request and return its response and next model ID."""
    operation = request.get("operation")
    arguments = request.get("arguments", {})
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
        setattr(backend, "model", model)
        model_id = next_model_id
        models[model_id] = model
        return {"ok": True, "value": model_id}, next_model_id + 1
    if operation == "unload_model":
        try:
            backend.unload_model()
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
        method_name = cast(str, arguments.pop("method"))
        method = getattr(backend, method_name)
        return {
            "ok": True,
            "value": cast(WorkerValue, method(**arguments)),
        }, next_model_id
    if operation == "generate_stream":
        model_id = cast(int, arguments.pop("model_id"))
        model = models.get(model_id)
        if model is None:
            raise ValueError(f"backend worker has no loaded model ID: {model_id}")
        generator = backend.generate_stream(
            model,
            **cast(BackendArguments, arguments),
        )
        stream_frame_count = 0
        for chunk in generator:
            stream_frame_count += 1
            if stream_frame_count <= 3:
                _worker_log(
                    f"[STREAM] worker emitted frame={stream_frame_count} "
                    f"{_stream_value_summary(chunk)}"
                )
            send_message(protocol_stream, {"ok": True, "stream": True, "value": chunk})
        _worker_log(f"[STREAM] worker completed frames={stream_frame_count}")
        return {"ok": True, "done": True}, next_model_id
    raise ValueError(f"unknown backend worker operation: {operation}")


def main() -> int:
    """Run the backend request loop until the core closes the pipe."""
    args = _parse_args()
    protocol_stream = sys.stdout.buffer
    manifest = backend_manifest(args.backend)
    backend_kwargs = cast(BackendArguments, json.loads(args.backend_kwargs))

    def _worker_fatal() -> None:
        """Notify the proxy that the backend entered its fatal state."""
        send_message(protocol_stream, cast(WorkerMessage, {"ok": True, "fatal": True}))

    backend = _load_backend(manifest, _worker_log, _worker_fatal, backend_kwargs)
    models: dict[int, BackendModel] = {}
    next_model_id = 1
    while True:
        try:
            request = cast(WorkerRequest, receive_message(sys.stdin.buffer))
        except (EOFError, BrokenPipeError):
            return 0
        except Exception as error:
            print(str(error), file=_WORKER_STDERR)
            return 1
        if request.get("operation") == "shutdown":
            with suppress(Exception):
                backend.unload_model()
            _release_worker_models(models)
            return 0
        try:
            response, next_model_id = _run_request(
                backend,
                request,
                models,
                next_model_id,
                protocol_stream,
            )
        except Exception as error:
            response = _error_response(error)
        send_message(protocol_stream, cast(WorkerMessage, response))


if __name__ == "__main__":
    raise SystemExit(main())
