# SPDX-License-Identifier: Apache-2.0
"""Celune UI package with lazy imports for early startup."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import CeluneUI
    from .headless import CeluneHeadlessUI
    from .terminal import SelectMenu, LogRedirect, SelectMenuOption, SelectMenuWidget

__all__ = [
    "CeluneHeadlessUI",
    "CeluneUI",
    "LogRedirect",
    "SelectMenu",
    "SelectMenuOption",
    "SelectMenuWidget",
]


def __getattr__(name: str) -> type:
    """Load one UI surface only when a caller requests it."""
    if name == "CeluneUI":
        from .app import CeluneUI

        return CeluneUI
    if name == "CeluneHeadlessUI":
        from .headless import CeluneHeadlessUI

        return CeluneHeadlessUI
    if name in {"LogRedirect", "SelectMenu", "SelectMenuOption", "SelectMenuWidget"}:
        from . import terminal

        return getattr(terminal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
