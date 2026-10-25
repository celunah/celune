# SPDX-License-Identifier: Apache-2.0
"""Operation and backend-mode type aliases."""

from enum import Enum
from typing import Literal

type BackendMode = Literal["normal", "ui_test", "agent_test"]
type OperationMode = Literal["performance", "converse", "agent"]

OPERATION_MODES: tuple[OperationMode, ...] = ("performance", "converse", "agent")


class OperationState(str, Enum):
    """Low-level engine state independent of ordinary runtime activity."""

    NORMAL = "normal"
    STOPPED = "stopped"
    SEMI_FATAL = "semi_fatal"
    FATAL = "fatal"
