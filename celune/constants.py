# SPDX-License-Identifier: Apache-2.0
"""Shared Celune constants."""

import signal
import datetime
from enum import Enum, IntEnum, auto
from typing import Literal, Optional, TypedDict

from ._version import VERSION

# main app name
# why would you rename her? she doesn't approve of it
# don't blame her when you fork Celune and rename her to something else
APP_NAME = "Celune"
APP_SLUG = "".join(char if char.isalnum() else "_" for char in APP_NAME.lower())
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

# Persona receives a smaller ordinary conversation context while agent tasks
# reserve the larger context needed for planning and tool-result history.
PERSONA_CONTEXT_SPACE = 8192
AGENT_CONTEXT_SPACE = 32768
AGENT_CONTEXT_COMPACTION_RATIO = 0.75
AGENT_CONTEXT_COMPACTION_THRESHOLD = int(
    AGENT_CONTEXT_SPACE * AGENT_CONTEXT_COMPACTION_RATIO
)
AGENT_MAX_ITERATIONS = 20


# These models are available to Persona, with exact revisions for each variant.
class PersonaModelRevisions(TypedDict):
    """Pinned Hugging Face revisions for one Persona model family."""

    official: str
    abliterated: str


class PersonaModelDefinition(TypedDict):
    """Registry entry for one official and abliterated Persona model family."""

    model: str
    organization: str
    tier: Literal["standard", "smart"]
    revisions: PersonaModelRevisions


PERSONA_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PERSONA_MODELS: tuple[PersonaModelDefinition, ...] = (
    {
        "model": "Qwen3-VL-4B-Instruct",
        "organization": "Qwen",
        "tier": "standard",
        "revisions": {
            "official": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "abliterated": "ce72a7c22aacb493fb94478de3bfbe834c61844a",
        },
    },
    {
        "model": "Qwen3-VL-8B-Instruct",
        "organization": "Qwen",
        "tier": "smart",
        "revisions": {
            "official": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "abliterated": "b47a0690b22eaf1d9a63874d967a03781c90f9cf",
        },
    },
    {
        "model": "Qwen3-VL-8B-Thinking",
        "organization": "Qwen",
        "tier": "smart",
        "revisions": {
            "official": "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
            "abliterated": "34bbf0d131d799ef233b2c20b074fbc9a0179ead",
        },
    },
    {
        "model": "Qwen3.5-4B",
        "organization": "Qwen",
        "tier": "standard",
        "revisions": {
            "official": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "abliterated": "5581467dfd52bf338c782006a6cdce05c42594be",
        },
    },
    {
        "model": "Qwen3.5-9B",
        "organization": "Qwen",
        "tier": "smart",
        "revisions": {
            "official": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "abliterated": "05b9e7c9b978ba29bdb8f50a49c30e4b91183339",
        },
    },
    {
        "model": "gemma-4-E2B-it",
        "organization": "google",
        "tier": "standard",
        "revisions": {
            "official": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
            "abliterated": "3d1e3d50d7a04585ce4ded197b2fd7a90c04647c",
        },
    },
    {
        "model": "gemma-4-E4B-it",
        "organization": "google",
        "tier": "smart",
        "revisions": {
            "official": "ee0ef6023621cff504d758262d4e04895a5af4a2",
            "abliterated": "03ce1f3a982b544afb03878ce80e7f042bcdc172",
        },
    },
)
PERSONA_HISTORY_MESSAGES = 20


def _persona_model_id(
    definition: PersonaModelDefinition,
    variant: Literal["official", "abliterated"],
) -> str:
    """Build a full Hugging Face repository ID from a registry entry."""
    if variant == "official":
        return f"{definition['organization']}/{definition['model']}"
    return f"huihui-ai/Huihui-{definition['model']}-abliterated"


PERSONA_MODEL_REVISIONS = {
    **{
        _persona_model_id(definition, "official"): definition["revisions"]["official"]
        for definition in PERSONA_MODELS
    },
    **{
        _persona_model_id(definition, "abliterated"): definition["revisions"][
            "abliterated"
        ]
        for definition in PERSONA_MODELS
    },
}

REMOTE_CODE_MODEL_REVISIONS = {
    VOICE_EMBEDDING_MODEL: VOICE_EMBEDDING_MODEL_REVISION,
    **PERSONA_MODEL_REVISIONS,
}


def remote_code_model_revision(model_id: str) -> Optional[str]:
    """Return the pinned revision for a remote-code model when Celune knows one.

    Args:
        model_id: Hugging Face model ID to resolve.

    Returns:
        Optional[str]: The pinned commit revision, or ``None`` when unknown.
    """
    return REMOTE_CODE_MODEL_REVISIONS.get(model_id)


def persona_model_tier(model_id: str) -> Optional[Literal["standard", "smart"]]:
    """Return the configured hardware tier for a Persona model ID.

    Args:
        model_id: Full Hugging Face model repository ID.

    Returns:
        Optional[Literal["standard", "smart"]]: The model tier, or ``None`` when
        the model is not in the Persona registry.
    """
    for definition in PERSONA_MODELS:
        if model_id in {
            _persona_model_id(definition, "official"),
            _persona_model_id(definition, "abliterated"),
        }:
            return definition["tier"]
    return None


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
