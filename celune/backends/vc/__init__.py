# SPDX-License-Identifier: Apache-2.0
"""Celune voice-conversion backend initialization manager."""

from collections.abc import Callable
from importlib import import_module
from typing import Optional, Union

from ..environment import BACKEND_MANIFESTS, BackendManifest, backend_manifest
from .base import CeluneVCBackend

__all__ = [
    "BACKEND_MANIFESTS",
    "BackendManifest",
    "VC_BACKENDS",
    "CeluneVCBackend",
    "backend_manifest",
    "resolve_vc_backend",
]

VC_BACKENDS = {
    "passthrough": (
        "celune.backends.vc.passthrough",
        "CelunePassthroughVCBackend",
    ),
    "seed-vc": (
        "celune.backends.vc.seedvc",
        "CeluneSeedVCBackend",
    ),
}


def _default_log(_msg: str, _severity: str = "info") -> None:
    """Default log signature for type checking."""


def resolve_vc_backend(
    backend_name: Union[str, type[CeluneVCBackend], CeluneVCBackend],
    log: Optional[Callable[[str, str], None]] = None,
    isolated: bool = False,
) -> CeluneVCBackend:
    """Resolve a voice-conversion backend specification into an instance.

    Args:
        backend_name: A backend name, backend class, or backend instance.
        log: Optional log callback to expose during backend construction.
        isolated: Whether to construct a worker-backed backend in its private environment.

    Returns:
        CeluneVCBackend: The resolved voice-conversion backend instance.

    Raises:
        ValueError: The named backend is unknown.
        TypeError: The backend specification is not a supported type.
    """
    log = log or _default_log

    if isinstance(backend_name, CeluneVCBackend):
        return backend_name

    if isinstance(backend_name, type) and issubclass(backend_name, CeluneVCBackend):
        return backend_name(log=log)

    if isinstance(backend_name, str):
        key = backend_name.strip().lower()

        try:
            module_name, class_name = VC_BACKENDS[key]
        except KeyError as e:
            raise ValueError(
                "unknown voice-conversion backend: "
                f"'{backend_name}' (available: {', '.join(VC_BACKENDS.keys())})"
            ) from e

        if isolated and key in BACKEND_MANIFESTS:
            from ..remote import RemoteVCBackendProxy

            return RemoteVCBackendProxy(BACKEND_MANIFESTS[key], log=log)

        module = import_module(module_name)
        backend_cls = getattr(module, class_name)
        return backend_cls(log=log)

    raise TypeError(
        "'backend_name' must be a voice-conversion backend instance, "
        "voice-conversion backend type, or backend name string"
    )
