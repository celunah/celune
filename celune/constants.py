"""Shared Celune constants."""

import signal
import datetime
from enum import IntEnum, Enum
from typing import TypeVar

from textual.theme import Theme

from celune.colors import random_hex

# NEW AND IMPROVED! NOW WITH BETTER CASING & PUNCTUATION!
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v1.3"

# I use this to know when the next moon comes.
REFERENCE_NEW_MOON = datetime.datetime(2000, 1, 6, 18, 14, tzinfo=datetime.timezone.utc)


# exit codes
class ExitCodes(Enum):
    """Celune exit codes."""

    EXIT_SUCCESS = 0  # Celune exited successfully.
    EXIT_PENDING_UPDATE = 0  # Celune has a pending update.
    EXIT_FAILURE = 1  # Celune experienced a general failure.
    EXIT_NO_ANSI = 2  # Celune did not find an ANSI capable terminal.
    EXIT_ALREADY_RUNNING = 3  # Celune is already running.
    EXIT_MISSING_DEPENDENCIES = 4  # Celune is missing required dependencies.
    EXIT_CELINE_DAY_SIX_SEVEN = 67  # Celune refuses to run on Celine Day.
    EXIT_CELINE_DAY = 103  # Celune refuses to run on Celine Day.


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
    "celune_april_fools": {
        "info": random_hex(),
        "warning": random_hex(),
        "error": random_hex(),
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


# pipeline state objects
class PipelineStates(Enum):
    """Pipeline state objects."""

    TERMINATE = object()  # Celune is exiting.
    UTTERANCE_END = object()  # Utterance ended normally.
    UTTERANCE_FORCE_END = object()  # Utterance was interrupted by the user.


class UtteranceLoudnessTier(IntEnum):
    """Per-utterance loudness tiers."""

    NORMAL = 0  # Celune spoke normally.
    SUSPICIOUS = 1  # Utterance may be too silent.
    SILENT = 2  # Utterance is too silent.
