# SPDX-License-Identifier: MIT
"""Placeholder lifecycle contract for Celune's future agent runtime."""

from __future__ import annotations

from typing import Optional, Sequence

from ..typing.agent import (
    AgentContext,
    AgentOutput,
    AgentRequest,
    AgentResponseCallback,
    AgentTool,
    ToolCall,
    ToolResult,
)


class AgentRuntime:
    """Unimplemented local-only agent orchestration contract."""

    def __init__(self, tools: Sequence[AgentTool] = ()) -> None:
        """Create the future agent runtime around local tools."""
        self.tools = tuple(tools)

    def create_context(self, request: AgentRequest) -> AgentContext:
        """Build the capability-aware context for one agent request."""
        raise NotImplementedError("agent context creation is not implemented")

    def plan(self, context: AgentContext) -> AgentOutput:
        """Select whether the next step is a response or a local tool call."""
        raise NotImplementedError("agent planning is not implemented")

    def select_tool(
        self,
        context: AgentContext,
        output: AgentOutput,
    ) -> Optional[ToolCall]:
        """Validate and select a tool call from one planning step."""
        raise NotImplementedError("agent tool selection is not implemented")

    def execute_tool(
        self,
        context: AgentContext,
        call: ToolCall,
    ) -> ToolResult:
        """Dispatch one local tool call."""
        raise NotImplementedError("agent tool execution is not implemented")

    def handle_tool_result(
        self,
        context: AgentContext,
        result: ToolResult,
    ) -> AgentOutput:
        """Convert a tool result into the next externally visible step."""
        raise NotImplementedError("agent tool-result handling is not implemented")

    def respond(self, context: AgentContext) -> AgentOutput:
        """Generate one non-tool agent response."""
        raise NotImplementedError("agent response generation is not implemented")

    def run(
        self,
        request: AgentRequest,
        callback: Optional[AgentResponseCallback] = None,
    ) -> AgentOutput:
        """Run the future agent loop and emit steps through an optional callback."""
        raise NotImplementedError("agent execution is not implemented")

    def pause(self, session_id: str) -> None:
        """Pause a future agent session."""
        raise NotImplementedError("agent pause is not implemented")

    def resume(self, session_id: str) -> None:
        """Resume a future agent session."""
        raise NotImplementedError("agent resume is not implemented")

    def cancel(self, session_id: str) -> None:
        """Cancel a future agent session."""
        raise NotImplementedError("agent cancellation is not implemented")
