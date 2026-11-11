# SPDX-License-Identifier: Apache-2.0
"""Shared button behavior for Celune's Textual interface."""

from textual.content import ContentText
from textual.widgets import Button as TextualButton


def _discard_unused(value: object) -> None:
    """Use Celune's unused-value marker without eager core imports."""
    from ..utils import discard

    discard(value)


class CeluneButton(TextualButton):
    """Keep button layout current when a reactive label changes."""

    def watch_label(self, old_label: ContentText, label: ContentText) -> None:
        """Recalculate the button width when its reactive label changes."""
        _discard_unused(old_label)
        _discard_unused(label)
        if self.is_mounted:
            self.refresh(layout=True)
