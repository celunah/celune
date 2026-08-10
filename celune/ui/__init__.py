# SPDX-License-Identifier: Apache-2.0
"""Celune UI package with lazy imports for early startup."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import CeluneUI
    from .headless import CeluneHeadlessUI
    from .protocols import CeluneBaseUI, CeluneHeadlessBaseUI, CeluneTextualUI
    from .terminal import LogRedirect, SelectMenu

__all__ = [
    "CeluneBaseUI",
    "CeluneHeadlessBaseUI",
    "CeluneHeadlessUI",
    "CeluneTextualUI",
    "CeluneUI",
    "LogRedirect",
    "SelectMenu",
]


def __getattr__(name: str) -> type:
    """Load one UI surface only when a caller requests it."""
    if name == "CeluneUI":
        from .app import CeluneUI

        return CeluneUI
    if name == "CeluneHeadlessUI":
        from .headless import CeluneHeadlessUI

        return CeluneHeadlessUI
    if name in {"CeluneBaseUI", "CeluneHeadlessBaseUI", "CeluneTextualUI"}:
        from . import protocols

        return getattr(protocols, name)
    if name in {"LogRedirect", "SelectMenu"}:
        from . import terminal

        return getattr(terminal, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
