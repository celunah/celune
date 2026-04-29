"""Shared Celune constants."""

import signal
import datetime
from typing import TypeVar

from textual.theme import Theme

# NEW AND IMPROVED! NOW WITH BETTER CASING & PUNCTUATION!
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v1.3"

# I use this to know when the next moon comes.
REFERENCE_NEW_MOON = datetime.datetime(2000, 1, 6, 18, 14, tzinfo=datetime.timezone.utc)

# exit codes
EXIT_SUCCESS = 0
EXIT_PENDING_UPDATE = 0
EXIT_FAILURE = 1
EXIT_NO_ANSI = 2
EXIT_ALREADY_RUNNING = 3
EXIT_MISSING_DEPENDENCIES = 4
EXIT_CELINE_DAY = 103

# SIGTSTP is not defined on Windows systems
SIGTSTP = getattr(signal, "SIGTSTP", None)

# generic type
T = TypeVar("T")

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
}

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
