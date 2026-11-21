# SPDX-License-Identifier: Apache-2.0
"""Public package surface for Celune's user interfaces.

The package exposes the main Textual application, :class:`CeluneUI`, a
lightweight terminal interface, :class:`CeluneHeadlessUI`, and shared terminal
helpers for selection menus and log forwarding. The Textual app presents
runtime status and logs, handles user interaction and commands, and coordinates
audio workflows such as speech input and voice conversion while the core engine
owns Celune's application behavior and inference.

Implementation is split across focused modules for the app, runtime updates,
audio capture, interaction, commands, and debugging. Public names are loaded
lazily so importing :mod:`celune.ui` does not eagerly import an interface.
"""

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
