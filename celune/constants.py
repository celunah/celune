"""Shared Celune constants."""

import signal
import datetime
from enum import IntEnum, Enum
from typing import TypeVar

import torch

# CeluneNorm v1.3 includes the most important changes, so that Celune can speak optimally.
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v1.3"

# voice embedding model
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
    EXIT_CELINE_DAY_SIX_SEVEN = 67  # Celune refuses to run on Celine Day.
    EXIT_CELINE_DAY = 103  # Celune refuses to run on Celine Day.


# SIGTSTP is not defined on Windows systems
SIGTSTP = getattr(signal, "SIGTSTP", None)

# generic type
T = TypeVar("T")


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


class PipelineActions(Enum):
    """Pipeline actions known to Celune."""

    SPEAK = "speak"  # called by celune.Celune.say()
    SFX = "sfx"  # called by celune.Celune.play()
    READINESS_SIGNAL = "readiness signal"  # called by celune.Celune.load() and celune.Celune.change_voice()
    VOICE_CHANGE = "voice change"  # called by celune.Celune.set_voice()


# N/A values
N_A_NUMERIC = float("nan")  # numeric N/A
N_A_TENSOR = torch.empty(0, 0)  # tensor N/A
N_A = None  # general N/A
