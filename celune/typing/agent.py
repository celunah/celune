# SPDX-License-Identifier: Apache-2.0
"""Types for Celune's future local-only agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, Optional, Protocol, TypedDict, Union

from ..persona.capabilities import PersonaCapabilities
from .common import JSON, JSONSerializable
from .modes import OperationMode


class NeedleToolParameterSpec(TypedDict, total=False):
    """Rich parameter descriptor accepted by Needle tool definitions."""

    type: str
    description: NotRequired[str]
    required: NotRequired[bool]


type NeedleToolParameter = Union[str, NeedleToolParameterSpec]


class NeedleToolDefinition(TypedDict):
    """Tool definition supplied to Needle's function-call selector."""

    name: str
    parameters: dict[str, NeedleToolParameter]
    description: NotRequired[str]


class NeedleToolCall(TypedDict):
    """Tool invocation returned by Needle."""

    name: str
    arguments: dict[str, JSONSerializable]


type NeedleToolCatalog = list[NeedleToolDefinition]
type NeedleToolSelection = list[NeedleToolCall]


class ToolCall(TypedDict):
    """A requested invocation of one registered local tool."""

    id: str
    name: str
    arguments: dict[str, JSONSerializable]


class ToolResult(TypedDict):
    """The result or failure returned by one local tool."""

    tool_call_id: str
    output: Optional[JSONSerializable]
    error: Optional[str]


class AgentOutput(TypedDict):
    """One externally observable step produced by the agent runtime."""

    tool_call: Optional[ToolCall]
    response: Optional[str]
    end: bool
    paused: bool


@dataclass(frozen=True)
class AgentSession:
    """Stable identity and lifecycle state for one agent conversation."""

    session_id: str
    paused: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class AgentRequest:
    """Input supplied to a future agent run."""

    request: str
    history: tuple[JSON, ...] = ()
    session: AgentSession = field(
        default_factory=lambda: AgentSession(session_id="default")
    )


@dataclass(frozen=True)
class AgentContext:
    """Context available to planning, tool use, and response callbacks."""

    request: AgentRequest
    mode: OperationMode
    persona_capabilities: PersonaCapabilities


class AgentResponseCallback(Protocol):
    """Callback receiving each agent response or tool-call step."""

    def __call__(self, output: AgentOutput) -> None:
        """Receive one agent output step."""
        raise NotImplementedError("protocol not defined")


class AgentTool(Protocol):
    """Contract for a future local-only agent tool."""

    name: str
    description: str

    def execute(self, call: ToolCall, context: AgentContext) -> ToolResult:
        """Execute one validated local tool call."""
        raise NotImplementedError("protocol not defined")
