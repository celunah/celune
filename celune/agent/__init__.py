# SPDX-License-Identifier: MIT
"""Contracts for Celune's future local-only agent mode."""

from collections.abc import Mapping

from .contracts import (
    AgentContext,
    AgentOutput,
    AgentRequest,
    AgentResponseCallback,
    AgentSession,
    AgentTool,
    ToolCall,
    ToolResult,
)
from ..modes import mode_allows_agents, resolve_operation_mode
from ..typing.common import JSONSerializable
from .runtime import AgentRuntime


def agent_mode_enabled(
    config: Mapping[str, JSONSerializable],
) -> bool:
    """Return whether the global config selects the agent operation mode."""
    return mode_allows_agents(resolve_operation_mode(config))


__all__ = [
    "AgentContext",
    "AgentOutput",
    "AgentRequest",
    "AgentResponseCallback",
    "AgentRuntime",
    "AgentSession",
    "AgentTool",
    "ToolCall",
    "ToolResult",
    "agent_mode_enabled",
]
