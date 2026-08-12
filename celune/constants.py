# SPDX-License-Identifier: MIT
"""Shared Celune constants."""

import datetime
import itertools
import signal
from enum import Enum, IntEnum, auto
from typing import Optional

from ._version import VERSION
from .i18n import string
from .typing.common import JSON, JSONSerializable  # noqa: F401  # pylint: disable=W0611

# main app name
# why would you rename her? she doesn't approve of it
# don't blame her when you fork Celune and rename her to something else
APP_NAME = "Celune"
APP_SLUG = "".join(char if char.isalnum() else "_" for char in APP_NAME.lower())

# CeluneNorm v2.0 inherits v1.3's feature set but at an extended context length
# so Celune can process your normalized text more efficiently at either
# 1024 or 2048 tokens of available max context length
#
# uncomment the normalizer you wish to use here
# NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v2.0-ctx1024"
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v2.0-ctx2048"

# this embedding model is used to extract a voice embedding vector out of the target utterance,
# and analyze the voice automatically based on any given embeddings from your CEVOICE/CECHAR pack
VOICE_EMBEDDING_MODEL = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
VOICE_EMBEDDING_MODEL_REVISION = "7577f61c42737fc8064bba773e2a18602df92803"
# this embedding model is used to retrieve long-term Persona memories semantically when available
PERSONA_MEMORY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# this model is used to infer conversation emotion and derive Persona's target response mood
PERSONA_EMOTION_MODEL = "lunahr/emotispace-128"

# this model is loaded by Celune, and used to control the persona
PERSONA_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PERSONA_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PERSONA_HISTORY_MESSAGES = 20

REMOTE_CODE_MODEL_REVISIONS = {
    VOICE_EMBEDDING_MODEL: VOICE_EMBEDDING_MODEL_REVISION,
    PERSONA_MODEL_ID: PERSONA_MODEL_REVISION,
}


def remote_code_model_revision(model_id: str) -> Optional[str]:
    """Return the pinned revision for a remote-code model when Celune knows one.

    Args:
        model_id: Hugging Face model ID to resolve.

    Returns:
        Optional[str]: The pinned commit revision, or ``None`` when unknown.
    """
    return REMOTE_CODE_MODEL_REVISIONS.get(model_id)


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
    EXIT_PENDING_UPDATE = 7  # Celune has a pending update.
    EXIT_LAUNCHER_LOST = 8  # Celune lost the connection to her launcher.

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

CRASH_LINES = itertools.cycle(
    [
        string("osc.crash_1", app_name=APP_NAME),
        string("osc.crash_2", app_name=APP_NAME),
        string("osc.crash_3"),
        string("osc.crash_4"),
    ]
)

# equivalent service costs per minute grouped by popular TTS providers
#
# please note Celune does not use any of these providers, so this is only for
# calculating equivalent API cost had you not been using Celune
COST_EQUIVALENTS = {
    "gemini-flash-tts": 0.015,
    "gemini-pro-tts": 0.03,
    "openai-realtime": 0.096,
    "openai-realtime-mini": 0.03,
    "elevenlabs-flash": 0.05,
    "elevenlabs-turbo": 0.05,
    "elevenlabs-multilingual-v2": 0.1,
    "elevenlabs-multilingual-v3": 0.1,
    "fishaudio-s2-pro": 0.021,
    "fishaudio-s2.1-pro": 0.021,
}

# temporary Celune user agent, will be used properly in v5
CELUNE_UA = f"Celune/{VERSION}"
