# SPDX-License-Identifier: MIT
"""Color utility helpers."""

import random

from textual.theme import Theme


def random_hex() -> str:
    """Return a random six-digit hex color."""
    hex_numbers = []
    for _ in range(6):
        hex_numbers.append(random.choice("123456789abcdef"))

    return f"#{''.join(hex_numbers)}"


# dark theme
THEME = Theme(
    name="celune",
    primary="#cebaff",  # Celune primary
    secondary="#a595cc",  # Celune secondary
    accent="#7c7099",  # Celune tertiary
    foreground="#e2ceff",  # Celune highlight
    background="#1d1826",  # Celune background
    surface="#1d1826",  # same as background
    warning="#f0e68c",  # Celune warning
    error="#f07178",  # Celune error
    dark=True,
)

# light theme (you serious?)
THEME_LIGHT = Theme(
    name="celune_light",
    primary="#33293f",  # Celune light primary
    secondary="#281732",  # Celune light secondary
    accent="#1e1126",  # Celune light tertiary
    foreground="#473d53",  # Celune light highlight
    background="#ece8ff",  # Celune light background
    surface="#ece8ff",  # same as background
    warning="#6b5e00",  # Celune light warning
    error="#7a1f24",  # Celune light error
    dark=False,
)

# lol funny theme
THEME_APRIL_FOOLS = Theme(
    name="celune_april_fools",
    primary=random_hex(),
    secondary=random_hex(),
    accent=random_hex(),
    foreground=random_hex(),
    background=random_hex(),
    surface=random_hex(),
    warning=random_hex(),
    error=random_hex(),
    dark=False,
)

# Celune theme data was moved here from the UI file
SEVERITY_COLORS = {
    "celune": {
        "info": "#cebaff",
        "warning": "#f0e68c",
        "error": "#f07178",
    },
    "celune_light": {
        "info": "#33293f",
        "warning": "#6b5e00",
        "error": "#7a1f24",
    },
    "celune_april_fools": {
        "info": random_hex(),
        "warning": random_hex(),
        "error": random_hex(),
    },
}
