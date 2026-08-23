"""Lightweight default themes used before the full runtime is loaded."""

from textual.theme import Theme


def default_theme_family() -> tuple[Theme, Theme]:
    """Return Celune's default dark and light themes.

    The UI uses these themes for its first loading frame. The full color
    module later replaces them with pack-derived themes when runtime
    dependencies are available.

    Returns:
        tuple[Theme, Theme]: The default dark and light Celune themes.
    """
    return (
        Theme(
            name="celune",
            primary="#cebaff",
            secondary="#a595ce",
            accent="#7c7099",
            foreground="#deceff",
            background="#1d1826",
            surface="#1d1826",
            warning="#f0e68c",
            error="#f07178",
            dark=True,
        ),
        Theme(
            name="celune_light",
            primary="#3a304c",
            secondary="#2c2439",
            accent="#1d1826",
            foreground="#574872",
            background="#deceff",
            surface="#deceff",
            warning="#6b5e00",
            error="#7a1f24",
            dark=False,
        ),
    )
