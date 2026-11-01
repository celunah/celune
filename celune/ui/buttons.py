# SPDX-License-Identifier: Apache-2.0
"""Shared button behavior for Celune's Textual interface."""

from rich.cells import cell_len
from textual.content import Content, ContentText
from textual.geometry import Size
from textual.widgets import Button as TextualButton


class CeluneButton(TextualButton):
    """Size the button to its rendered label without interior padding."""

    def on_mount(self) -> None:
        """Remove Textual's default horizontal line padding."""
        self.styles.line_pad = 0

    def watch_label(self, old_label: ContentText, label: ContentText) -> None:
        """Recalculate the button width when its reactive label changes."""
        del old_label, label
        if self.is_mounted:
            self.refresh(layout=True)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        """Return the rendered label width without extra horizontal cells."""
        del container, viewport
        assert isinstance(self.label, Content)
        lines = self.label.plain.splitlines()
        if not lines:
            return 0
        return max(cell_len(line) for line in lines)
