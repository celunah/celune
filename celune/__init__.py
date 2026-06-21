# SPDX-License-Identifier: MIT
"""Public package surface for the Celune character engine.

Importing :mod:`celune` exposes the main :class:`Celune` engine together with
the extension context and base class used by plugins. The package also publishes
build metadata such as :data:`__version__`, :data:`REVISION`, and the small set
of descriptive strings used by Celune's user-facing surfaces.

Most implementation details live in submodules. The package root intentionally
keeps a compact public API so applications can start from ``from celune import
Celune`` without depending on the internal backend, pipeline, or UI layout.

Only construct :class:`Celune` and its UI classes once per process. Creating multiple
instances can exhaust GPU resources and is not a supported usage pattern.
"""

import sys as _sys
import inspect as _inspect
import subprocess as _subprocess
from typing import TYPE_CHECKING, Callable, Union

from .constants import APP_NAME
from .paths import (
    configure_huggingface_cache_environment,
    configure_huggingface_runtime,
)

configure_huggingface_cache_environment()
configure_huggingface_runtime()

if TYPE_CHECKING:
    from .celune import Celune
    from .extensions.base import CeluneContext, CeluneExtension
    from .extensions.events import subscribe


def _get_revision() -> str:
    """Return the current Git revision."""
    try:
        rev = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_subprocess.DEVNULL,
            text=True,
        ).strip()
        status = _subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=_subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{rev}{'*' if status else ''}"
    except (_subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _caller_is_repl() -> bool:
    """Return whether Celune appears to be imported from the interactive Python REPL."""
    for frame in _inspect.stack():
        filename = frame.filename
        if "importlib" in filename or filename.startswith("<frozen"):
            continue
        if (
            __name__.replace(".", "\\") in filename
            or __name__.replace(".", "/") in filename
        ):
            continue
        return filename.startswith("<python-input-")
    return False


REVISION = _get_revision()
VERSION = "4.2.0"

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"{VERSION}+{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"

__tagline__ = '"Your voice, your way."'
__codename__ = "Personality"
__comment__ = "I can finally talk with you."

if hasattr(_sys, "ps1"):
    print(f"Caution: You are running the {APP_NAME} backend interactively.")
    print("This is not an intended mode of operation, usage may differ.")
    print()
    print(
        "\"If you're just exploring, please... be careful. I don't usually speak here.\""
    )


def __getattr__(
    name: str,
) -> Union[
    type["Celune"],
    type["CeluneContext"],
    type["CeluneExtension"],
    Callable[..., Callable[..., None]],
]:
    if name == "Celune":
        from .celune import Celune

        return Celune

    if name == "CeluneContext":
        from .extensions.base import CeluneContext

        return CeluneContext

    if name == "CeluneExtension":
        from .extensions.base import CeluneExtension

        return CeluneExtension

    if name == "subscribe":
        from .extensions.events import subscribe

        return subscribe

    raise AttributeError(f"module '{__name__!r}' has no attribute '{name!r}'")


__all__ = [
    "Celune",
    "CeluneContext",
    "CeluneExtension",
    "REVISION",
    "subscribe",
    "__version__",
    "__tagline__",
    "__codename__",
    "__comment__",
]


def __dir__() -> list[str]:
    """Return Celune's intentionally public package surface for REPL users."""
    return sorted(__all__)
