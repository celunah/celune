# SPDX-License-Identifier: MIT
"""Worker entrypoint for running one backend in its private environment."""

import argparse
import importlib
import json
import sys
from collections.abc import Iterator
from typing import Protocol, cast

from .environment import BackendManifest, backend_manifest
from .worker_protocol import receive_message, send_message


class _BackendRuntime(Protocol):
    """Runtime method surface used by the generic worker loop."""

    def model_is_available_locally(self, **kwargs: object) -> object:
        """Return whether a model is available."""

    def preload_models(self) -> None:
        """Preload backend models."""

    def load_model(self, **kwargs: object) -> object:
        """Load one backend model."""

    def unload_model(self) -> None:
        """Unload backend models."""

    def generate_stream(self, model: object, **kwargs: object) -> Iterator[object]:
        """Generate streamed backend audio."""

    def convert(self, request: object) -> object:
        """Convert one voice-conversion request."""


def _parse_args() -> argparse.Namespace:
    """Parse the private worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-kwargs", default="{}")
    return parser.parse_args()


def _load_backend(
    manifest: BackendManifest,
    log: object,
    kwargs: dict[str, object],
) -> object:
    """Import and construct the backend described by one manifest."""
    module = importlib.import_module(manifest.backend_module)
    backend_class = getattr(module, manifest.backend_class)
    if manifest.kind == "vc":
        return backend_class(log=log)
    return backend_class(log=log, **kwargs)


def _describe(backend: object) -> dict[str, object]:
    """Return the backend metadata required by the core proxy."""
    return {
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
    }


def _worker_log(message: str, severity: str = "info") -> None:
    """Write backend logs away from the binary protocol stream."""
    print(f"[{severity}] {message}", file=sys.stderr, flush=True)


def _run_request(
    backend: _BackendRuntime,
    request: dict[str, object],
    models: dict[int, object],
    next_model_id: int,
) -> tuple[dict[str, object], int]:
    """Run one backend request and return its response and next model ID."""
    operation = request.get("operation")
    arguments = cast(dict[str, object], request.get("arguments", {}))
    if operation == "describe":
        return {"ok": True, "value": _describe(backend)}, next_model_id
    if operation == "model_is_available_locally":
        value = backend.model_is_available_locally(**arguments)
        return {"ok": True, "value": value}, next_model_id
    if operation == "preload_models":
        backend.preload_models()
        return {"ok": True, "value": None}, next_model_id
    if operation == "load_model":
        model = backend.load_model(**arguments)
        model_id = next_model_id
        models[model_id] = model
        return {"ok": True, "value": model_id}, next_model_id + 1
    if operation == "unload_model":
        backend.unload_model()
        models.clear()
        return {"ok": True, "value": None}, next_model_id
    if operation == "convert":
        return {
            "ok": True,
            "value": backend.convert(arguments["request"]),
        }, next_model_id
    if operation == "call":
        method_name = cast(str, arguments.pop("method"))
        method = getattr(backend, method_name)
        return {"ok": True, "value": method(**arguments)}, next_model_id
    if operation == "generate_stream":
        model_id = cast(int, arguments.pop("model_id"))
        model = models.get(model_id)
        if model is None:
            raise ValueError(f"backend worker has no loaded model ID: {model_id}")
        generator = backend.generate_stream(model, **arguments)
        for chunk in generator:
            send_message(
                sys.stdout.buffer, {"ok": True, "stream": True, "value": chunk}
            )
        return {"ok": True, "done": True}, next_model_id
    raise ValueError(f"unknown backend worker operation: {operation}")


def main() -> int:
    """Run the backend request loop until the core closes the pipe."""
    args = _parse_args()
    manifest = backend_manifest(args.backend)
    backend_kwargs = cast(dict[str, object], json.loads(args.backend_kwargs))
    backend = cast(
        _BackendRuntime, _load_backend(manifest, _worker_log, backend_kwargs)
    )
    models: dict[int, object] = {}
    next_model_id = 1
    while True:
        try:
            request = cast(dict[str, object], receive_message(sys.stdin.buffer))
        except (EOFError, BrokenPipeError):
            return 0
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 1
        if request.get("operation") == "shutdown":
            return 0
        try:
            response, next_model_id = _run_request(
                backend, request, models, next_model_id
            )
        except Exception as error:
            response = {"ok": False, "error": str(error)}
        send_message(sys.stdout.buffer, response)


if __name__ == "__main__":
    raise SystemExit(main())
