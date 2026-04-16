"""Celune backend initialization manager."""

from .base import CeluneBackend
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

BACKENDS = {
    "qwen3": ("celune.backends.qwen3", "Qwen3"),
    "voxcpm2": ("celune.backends.voxcpm2", "VoxCPM2"),
}

def get_version(package) -> str:
    """Get a specified package version."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def resolve_backend(
    backend_name: str | type[CeluneBackend] | CeluneBackend,
) -> CeluneBackend:
    if isinstance(backend_name, CeluneBackend):
        return backend_name

    if isinstance(backend_name, type) and issubclass(backend_name, CeluneBackend):
        return backend_name()

    if isinstance(backend_name, str):
        key = backend_name.strip().lower()

        try:
            module_name, class_name = BACKENDS[key]
        except KeyError:
            raise ValueError(f"unknown backend: '{backend_name}' (available: {', '.join(BACKENDS.keys())})")

        module = import_module(module_name)
        backend_cls = getattr(module, class_name)
        return backend_cls()

    raise TypeError(
        "'backend_name' must be a backend instance, backend type, or backend name string"
    )