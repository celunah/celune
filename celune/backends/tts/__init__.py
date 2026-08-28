# SPDX-License-Identifier: Apache-2.0
"""Celune backend initialization manager."""

from typing import Union, Optional
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

from ...i18n import string
from .base import CeluneBackend
from ...typing.backends import BackendModel
from ..environment import BACKEND_MANIFESTS, BackendManifest, backend_manifest

__all__ = [
    "BACKENDS",
    "BACKEND_MANIFESTS",
    "BackendManifest",
    "BackendModel",
    "CeluneBackend",
    "backend_manifest",
    "get_version",
    "resolve_backend",
]

BACKENDS = {
    "mini": ("celune.backends.tts.mini", "Mini"),
    "qwen3": ("celune.backends.tts.qwen3", "Qwen3"),
    "dotstts": ("celune.backends.tts.dotstts", "DotsTtsMF"),
    "voxcpm2": ("celune.backends.tts.voxcpm2", "VoxCPM2"),
    "gpt-sovits": ("celune.backends.tts.gpt_sovits", "GPTSoVITS"),
}


def _default_log(_msg: str, _severity: str = "info") -> None:
    """Default log signature for type checking."""


def get_version(package: str) -> str:
    """Get an installed package version.

    Args:
        package: The package name to resolve through import metadata.

    Returns:
        str: The installed package version, or ``"unknown"`` when the package cannot be found.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def resolve_backend(
    backend_name: Union[str, type[CeluneBackend], CeluneBackend],
    log: Optional[Callable[[str, str], None]] = None,
    fatal: Optional[Callable[[], None]] = None,
    **backend_kwargs,
) -> CeluneBackend:
    """Resolve a backend specification into a backend instance.

    Args:
        backend_name: A backend name, backend class, or backend instance.
        log: Optional log callback to expose during backend construction.
        fatal: Optional fatal callback to expose during backend construction.
        backend_kwargs: Backend-specific constructor options.

    Returns:
        CeluneBackend: The resolved backend instance.

    Raises:
        ValueError: The named backend is unknown.
        TypeError: The backend specification is not a supported type.
    """
    log = log or _default_log

    if isinstance(backend_name, CeluneBackend):
        backend_name.bind_fatal(fatal)
        return backend_name

    if isinstance(backend_name, type) and issubclass(backend_name, CeluneBackend):
        return backend_name(log=log, fatal=fatal, **backend_kwargs)

    if isinstance(backend_name, str):
        key = backend_name.strip().lower()

        if key not in BACKENDS:
            raise ValueError(
                string(
                    "celune.unknown_backend",
                    backend=backend_name,
                    available=", ".join(BACKENDS.keys()),
                )
            )

        if key not in BACKEND_MANIFESTS:
            raise ValueError(
                string(
                    "celune.unknown_backend",
                    backend=backend_name,
                    available=", ".join(BACKENDS.keys()),
                )
            )

        from ...cedts.remote import RemoteBackendProxy

        return RemoteBackendProxy(
            BACKEND_MANIFESTS[key],
            log=log,
            fatal=fatal,
            **backend_kwargs,
        )

    raise TypeError(string("celune.invalid_backend_type"))
