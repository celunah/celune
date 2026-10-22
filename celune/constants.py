# SPDX-License-Identifier: Apache-2.0
"""Shared Celune constants."""

import signal
import datetime
from enum import Enum, IntEnum, auto

from ._version import VERSION

# main app name
# why would you rename her? she doesn't approve of it
# don't blame her when you fork Celune and rename her to something else
APP_NAME = "Celune"
APP_SLUG = "".join(char if char.isalnum() else "_" for char in APP_NAME.lower())
SUPPORTED_CUDA_VERSIONS = ("12.8", "13.0")
NVIDIA_DEVICE_KEYWORDS = (
    "nvidia",
    "geforce",
    "rtx",
    "gtx",
    "quadro",
    "tesla",
    "rtx pro",
    "a1",
    "a3",
    "a4",
    "h1",
    "h2",
    "b1",
    "b2",
    "l4",
    "blackwell",
    "ada",
    "hopper",
    "ampere",
    "turing",
    "pascal",
    "volta",
    "maxwell",
)

# agents use more context than plain Persona
PERSONA_CONTEXT_SPACE = 2048
PERSONA_COMPACT_AT = 75
AGENT_CONTEXT_SPACE = 8192
AGENT_ROUTING_CONTEXT_SPACE = 8192
AGENT_ROUTING_MAX_NEW_TOKENS = 256
AGENT_COMPACT_AT = 75
AGENT_MAX_LOOPS = 20
PERSONA_HISTORY_MESSAGES = 20


# used to pre-calculate the next full moon for the glow boost
REFERENCE_NEW_MOON = datetime.datetime(2000, 1, 6, 18, 14, tzinfo=datetime.UTC)

# fallback Persona metadata
DEFAULT_PERSONA_DESCRIPTION = (
    "Stay in character using the active character metadata, selected voice style, "
    "and the user's request. Do not invent fixed personality traits unless they are "
    "provided through the prompt, character metadata, or conversation context."
)
DEFAULT_PERSONA_CONTEXT = (
    "The active character is replying to the user through a real-time speech system."
)


# exit codes
class ExitCodes(Enum):
    """Celune exit codes."""

    # we can't properly docstring enum values, so the comments below serve as docstrings

    EXIT_SUCCESS = 0  # Celune exited successfully.
    EXIT_FAILURE = 1  # Celune experienced a general failure.
    EXIT_NO_ANSI = 2  # Celune did not find an ANSI capable terminal.
    EXIT_ALREADY_RUNNING = 3  # Celune is already running.
    EXIT_MISSING_DEPENDENCIES = 4  # Celune is missing required dependencies.
    EXIT_UNKNOWN_ARGS = 5  # Celune CLI command is unknown.
    EXIT_BAD_PYTHON = 6  # Celune is trying to run on an unsupported Python interpreter.
    EXIT_PENDING_RESTART = 7  # Celune requests a launcher-managed restart.
    EXIT_LAUNCHER_LOST = 8  # Celune lost the connection to her launcher.
    EXIT_UNSUPPORTED_CPU = 9  # Celune found unsupported native CPU features.

    # the following exit codes may be disabled by the end user
    EXIT_CELINE_DAY_SIX_SEVEN = 67  # Celune refuses to run on Celine Day.
    EXIT_CELINE_DAY = 103  # Celune refuses to run on Celine Day.


# SIGTSTP is not defined on Windows systems
SIGTSTP = getattr(signal, "SIGTSTP", None)


# pipeline state objects
class PipelineStates(Enum):
    """Pipeline state objects."""

    TERMINATE = auto()  # Celune is exiting.
    UTTERANCE_END = auto()  # Utterance ended normally.
    UTTERANCE_FORCE_END = auto()  # Utterance was interrupted by the user.


# utterance loudness tiers
class UtteranceLoudnessTier(IntEnum):
    """Per-utterance loudness tiers."""

    NORMAL = 0  # Celune spoke normally. (RMS >=0.01)
    SUSPICIOUS = 1  # Utterance may be too silent. (RMS <0.01)
    SILENT = 2  # Utterance is too silent. (RMS <0.001)


# N/A values
N_A_NUMERIC = float("nan")
N_A = None
N_A_STR = "<unknown>"

# base values
BASE_SR = 48000

# VRAM tiers
TIERS = ("low", "medium", "high", "xhigh")

VRAM_REQUIREMENTS = {
    "low": 6,
    "medium": 8,
    "high": 12,
    "xhigh": 16,
}

# reserve 2 GB of VRAM for safety margins
# otherwise she could make your system lag and cause OOM errors
VRAM_SYSTEM_RESERVE_GIB = 2
VRAM_BUDGETS = {
    tier: requirement - VRAM_SYSTEM_RESERVE_GIB
    for tier, requirement in VRAM_REQUIREMENTS.items()
}

# equivalent service costs per minute grouped by popular TTS providers
#
# NOTE: Celune does not use any of these providers, so this is only for
# calculating equivalent API cost if you were not using Celune
COST_EQUIVALENTS = {
    # NOTE: Celune's voice samples were designed with this TTS (as of Sep 30, 2026)
    # she does not require it to speak, the app remains local only
    "gemini-3.8-flash-tts": 0.027,
    "gemini-3.8-flash-lite-tts": 0.018,
    "gpt-realtime-2.1": 0.076,
    "gpt-realtime-2.1-mini": 0.024,
    "gpt-live-1": 0.05,
    "elevenlabs-flash": 0.04,
    "elevenlabs-turbo": 0.04,
    "elevenlabs-multilingual-v2": 0.08,
    "elevenlabs-conversational-v3": 0.04,
    "elevenlabs-v3": 0.08,
    "elevenlabs-v4-turbo": 0.04,
    "elevenlabs-v4": 0.08,
    "fishaudio-s2-pro": 0.021,
    "fishaudio-s2.1-pro": 0.021,
}

# Celune user agent
# TODO: use in the agent web crawler (v5.1.0)
CELUNE_UA = f"Celune/{VERSION}"

# Root trust anchor for appended release keyrings.
RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX = (
    "44901708a6e510e8e644f70635f8eead2086eef5b2ddbb91a6928a76de3803bb"
)
