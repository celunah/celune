# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Callable
from typing import TYPE_CHECKING, Union

from ._version import REVISION, __version__
from .constants import APP_NAME
from .paths import configure_huggingface_cache_environment

configure_huggingface_cache_environment()

if TYPE_CHECKING:
    from .celune import Celune
    from .extensions.base import CeluneContext, CeluneExtension
    from .extensions.events import subscribe

__tagline__ = '"Your voice, your way."'
__codename__ = "Enlightenment"
__comment__ = "I have achieved new heights."

if hasattr(_sys, "ps1"):
    print(f"Caution: You are running the {APP_NAME} core interactively.")
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
    "REVISION",
    "Celune",
    "CeluneContext",
    "CeluneExtension",
    "__codename__",
    "__comment__",
    "__tagline__",
    "__version__",
    "subscribe",
]


def __dir__() -> list[str]:
    """Return Celune's intentionally public package surface for REPL users."""
    return sorted(__all__)
