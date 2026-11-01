# SPDX-License-Identifier: Apache-2.0
"""Shared button behavior for Celune's Textual interface."""

from textual.content import ContentText
from textual.widgets import Button as TextualButton


class CeluneButton(TextualButton):
    """Keep button layout current when a reactive label changes."""

    def watch_label(self, old_label: ContentText, label: ContentText) -> None:
        """Recalculate the button width when its reactive label changes."""
        del old_label, label
        if self.is_mounted:
            self.refresh(layout=True)
