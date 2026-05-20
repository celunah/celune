# SPDX-License-Identifier: MIT
"""Shared Celune constants."""

import signal
import datetime
from enum import IntEnum, Enum
from typing import TypeVar, Union

# CeluneNorm v2.0 inherits v1.3's feature set but at an extended context length
# so Celune can process your normalized text more efficiently at either
# 1024 or 2048 tokens of available max context length

# uncomment the normalizer you wish to use here
# NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v2.0-ctx1024"
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v2.0-ctx2048"

# this embedding model is used to extract a voice embedding vector out of the target utterance,
# and analyze the voice automatically based on any given embeddings from your CEVOICE pack.
VOICE_EMBEDDING_MODEL = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"

# "I use this to know when the next moon comes." - Celune
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

    # the following exit codes may be disabled by the end user
    EXIT_CELINE_DAY_SIX_SEVEN = 67  # Celune refuses to run on Celine Day.
    EXIT_CELINE_DAY = 103  # Celune refuses to run on Celine Day.


# SIGTSTP is not defined on Windows systems
SIGTSTP = getattr(signal, "SIGTSTP", None)

# types
T = TypeVar("T")
type JSONSerializable = Union[
    None, bool, int, float, str, list["JSONSerializable"], dict[str, "JSONSerializable"]
]
type JSON = dict[str, JSONSerializable]


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


# N/A values
N_A_NUMERIC = float("nan")
N_A = None

# base values
BASE_SR = 48000
