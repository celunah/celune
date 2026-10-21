# SPDX-License-Identifier: Apache-2.0
"""Operation and backend-mode type aliases."""

from typing import Literal

type BackendMode = Literal["normal", "ui_test", "agent_test"]
type OperationMode = Literal["performance", "converse", "agent"]

OPERATION_MODES: tuple[OperationMode, ...] = ("performance", "converse", "agent")
