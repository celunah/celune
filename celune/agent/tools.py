# SPDX-License-Identifier: MIT
"""Small, explicitly allowlisted tools owned by the Celune agent runtime."""

from __future__ import annotations

from collections.abc import Mapping

from ..typing.agent import (
    AgentContext,
    AgentTool,
    AgentToolBehavior,
    AgentToolDangerLevel,
    AgentToolExecutionStatus,
    AgentToolSchema,
    ToolCall,
    ToolExecutionResult,
)
from ..typing.common import JSON


class AgentStatusTool:
    """Read the current task state without changing local application state."""

    name = "read_agent_status"
    description = "Read the current Celune agent task status."

    def execute(self, call: ToolCall, context: AgentContext) -> ToolExecutionResult:
        """Return typed state for the task that owns this tool call."""
        task = context.task
        if call["arguments"]:
            return {
                "tool_call_id": call["id"],
                "output": None,
                "error": "read_agent_status does not accept arguments",
                "tool_id": self.name,
                "status": AgentToolExecutionStatus.FAILED,
            }
        if task is None:
            return {
                "tool_call_id": call["id"],
                "output": None,
                "error": "agent task context is unavailable",
                "tool_id": self.name,
                "status": AgentToolExecutionStatus.FAILED,
            }
        output: JSON = {
            "task_id": task.task_id,
            "session_id": task.session_id,
            "state": task.state.value,
            "iterations": task.iterations,
            "generated_tokens": task.generated_tokens,
        }
        return {
            "tool_call_id": call["id"],
            "output": output,
            "error": None,
            "tool_id": self.name,
            "status": AgentToolExecutionStatus.SUCCEEDED,
        }


def production_agent_tools() -> tuple[AgentTool, ...]:
    """Return the local tools that Celune exposes to the production agent."""
    return (AgentStatusTool(),)


def production_agent_tool_schemas() -> Mapping[str, AgentToolSchema]:
    """Return schemas for the explicitly allowlisted production tools."""
    return {
        AgentStatusTool.name: AgentToolSchema(
            tool_id=AgentStatusTool.name,
            display_name="Read agent status",
            description=AgentStatusTool.description,
            behavior=AgentToolBehavior.READ_ONLY,
            danger=AgentToolDangerLevel.LOW,
            approval_required=False,
            available=True,
        )
    }


def agent_test_tools() -> tuple[AgentTool, ...]:
    """Return the single read-only tool permitted by agent test mode."""
    return (AgentStatusTool(),)


def agent_test_tool_schemas() -> Mapping[str, AgentToolSchema]:
    """Return schemas for the read-only agent test tool allowlist."""
    return {AgentStatusTool.name: production_agent_tool_schemas()[AgentStatusTool.name]}
