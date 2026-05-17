# SPDX-License-Identifier: MIT
"""Celune main package."""

import sys as _sys

from .utils import get_revision as _get_revision, caller_is_repl as _caller_is_repl

REVISION = _get_revision()
if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"3.5.0+{_local}{_dirty}"
else:
    __version__ = "3.5.0"

__tagline__ = "\u201cYour voice, your way.\u201d"
__codename__ = "Guidance"
__comment__ = "My vocal prowess can be easily harnessed by beginners."

if hasattr(_sys, "ps1"):
    print("Caution: You are running the Celune backend interactively.")
    print("This is not an intended mode of operation, usage may differ.")
    print()
    print(
        "\"If you're just exploring, please... be careful. I don't usually speak here.\""
    )

try:
    # due to how Celune imports __version__ we cannot put these imports according to PEP8
    from .celune import Celune
    from .extensions.base import CeluneContext, CeluneExtension

    __all__ = [
        "Celune",
        "CeluneContext",
        "CeluneExtension",
        "REVISION",
        "__version__",
        "__tagline__",
        "__codename__",
        "__comment__",
    ]
except ModuleNotFoundError as package:
    if _caller_is_repl():
        print(f"Missing dependency: {package.name}")
        print("Some functionality may be unavailable.")


def __dir__() -> list[str]:
    """Return Celune's intentionally public package surface for REPL users.

    Returns:
        list[str]: Celune's public package surface.
    """
    return sorted(__all__)
