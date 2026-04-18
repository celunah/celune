"""Celune backend initialization manager."""

from typing import Callable, Union
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

from .base import CeluneBackend

BACKENDS = {
    "qwen3": ("celune.backends.qwen3", "Qwen3"),
    "voxcpm2": ("celune.backends.voxcpm2", "VoxCPM2"),
}


def get_version(package: str) -> str:
    """Get an installed package version.

    Args:
        package: The package name to resolve through import metadata.

    Returns:
        str: The installed package version, or ``"unknown"`` when the package
            cannot be found.
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def resolve_backend(
    backend_name: Union[str, type[CeluneBackend], CeluneBackend],
    log: Callable[[str, str], None] | None = None,
) -> CeluneBackend:
    """Resolve a backend specification into a backend instance.

    Args:
        backend_name: A backend name, backend class, or backend instance.
        log: Optional log callback to expose during backend construction.

    Returns:
        CeluneBackend: The resolved backend instance.
    """
    log = log or (lambda msg, severity="info": None)

    if isinstance(backend_name, CeluneBackend):
        return backend_name

    if isinstance(backend_name, type) and issubclass(backend_name, CeluneBackend):
        backend_name(log=log)

    if isinstance(backend_name, str):
        key = backend_name.strip().lower()

        try:
            module_name, class_name = BACKENDS[key]
        except KeyError as e:
            raise ValueError(
                f"unknown backend: '{backend_name}' (available: {', '.join(BACKENDS.keys())})"
            ) from e

        module = import_module(module_name)
        backend_cls = getattr(module, class_name)
        try:
            return backend_cls(log=log)
        except TypeError:
            return backend_cls()

    raise TypeError(
        "'backend_name' must be a backend instance, backend type, or backend name string"
    )
