"""Textual theme assets."""

from ..constants import SEVERITY_COLORS

CELUNE_CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #logs {
        height: 1fr;
        border: round $primary;
        color: $primary;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1;
    }

    /* give scrollbar colors only to the elements that will have a scrollbar */
    #logs, #input {
        scrollbar-color: $accent;
        scrollbar-color-hover: $secondary;
        scrollbar-color-active: $primary;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        background: $background;
    }

    #logs:focus {
        border: round $primary;
        background: transparent;
    }

    #input {
        min-height: 3;
        height: 3;
        width: 1fr;
        border: round $primary;
    }

    #style {
        width: 14;
        height: 3;
        border: round $primary;
        margin-right: 1;
        text-align: center;
        background: $background;
    }

    #input:focus {
        border: round $foreground;
        background: $background;
        background-tint: transparent;
    }

    #input .text-area--cursor-line {
        background: transparent;
    }

    #style:focus, #style:hover, #style.-active, #input:hover {
        border: round $foreground;
        background: $background;
        background-tint: transparent;
        tint: transparent;
    }

    #logs, #input {
        margin-left: 1;
        margin-right: 1;
    }

    #bottom {
        height: 1;
        background: $background;
        margin-left: 1;
        margin-right: 1;
        margin-bottom: 1;
        color: $primary;
    }

    #status {
        height: 1;
        width: 1fr;
        text-align: left;
        color: $primary;
    }

    #resources {
        height: 1;
        width: 1fr;
        text-align: right;
        color: $primary;
    }

    #header-container {
        height: 1;
        width: 1fr;
        layout: horizontal;
        align: center middle;
        margin-bottom: 1;
        margin-top: 1;
    }

    #header {
        width: auto;
        content-align: center middle;
        color: $primary;
        text-style: bold;
        padding: 0 2;
    }

    .line {
        width: 1fr;
        height: 1;
        border-top: solid $primary;
        margin: 0 2;  /* when zero two works, arno would be proud */
    }

    #controls {
        height: auto;
    }

"""


def severity_color(theme_name: str, severity: str = "info") -> str:
    """Return the configured color for a UI severity."""
    palette = SEVERITY_COLORS.get(theme_name, SEVERITY_COLORS["celune"])
    return palette.get(severity, palette["info"])
