# SPDX-License-Identifier: Apache-2.0
"""Celune voice-conversion backend initialization manager."""

from typing import Union, Optional

from .base import CeluneVCBackend
from ...typing.aliases import LogCallback, LogLevel
from ..environment import BACKEND_MANIFESTS, BackendManifest, backend_manifest

__all__ = [
    "BACKEND_MANIFESTS",
    "VC_BACKENDS",
    "BackendManifest",
    "CeluneVCBackend",
    "backend_manifest",
    "resolve_vc_backend",
]

VC_BACKENDS = {
    "seed-vc": (
        "celune.backends.vc.seedvc",
        "CeluneSeedVCBackend",
    ),
}


def _default_log(
    msg: str,
    severity: str = "info",
    *,
    loglevel: LogLevel = "info",
) -> None:
    """Default log signature for type checking."""
    del msg, severity, loglevel


def resolve_vc_backend(
    backend_name: Union[str, type[CeluneVCBackend], CeluneVCBackend],
    log: Optional[LogCallback] = None,
) -> CeluneVCBackend:
    """Resolve a voice-conversion backend specification into an instance.

    Args:
        backend_name: A backend name, backend class, or backend instance.
        log: Optional log callback to expose during backend construction.

    Returns:
        CeluneVCBackend: The resolved voice-conversion backend instance.

    Raises:
        ValueError: The named backend is unknown.
        TypeError: The backend specification is not a supported type.
    """
    if log is None:
        log = _default_log

    if isinstance(backend_name, CeluneVCBackend):
        return backend_name

    if isinstance(backend_name, type) and issubclass(backend_name, CeluneVCBackend):
        return backend_name(log=log)

    if isinstance(backend_name, str):
        key = backend_name.strip().lower()

        if key not in VC_BACKENDS:
            raise ValueError(
                "unknown voice-conversion backend: "
                f"'{backend_name}' (available: {', '.join(VC_BACKENDS.keys())})"
            )

        if key not in BACKEND_MANIFESTS:
            raise ValueError(f"unknown voice-conversion backend: '{backend_name}'")

        from ...cedts.remote import RemoteVCBackendProxy

        return RemoteVCBackendProxy(BACKEND_MANIFESTS[key], log=log)

    raise TypeError(
        "'backend_name' must be a voice-conversion backend instance, "
        "voice-conversion backend type, or backend name string"
    )
