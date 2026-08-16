# SPDX-License-Identifier: MIT
"""Operation and backend-mode type aliases."""

from typing import Literal

type BackendMode = Literal["normal", "ui_test", "agent_test"]
type OperationMode = Literal["speak", "converse", "agent"]
