# SPDX-License-Identifier: MIT
"""Contracts for Celune's future local-only agent mode."""

from collections.abc import Mapping

from ..typing.agent import (
    AgentAbortReason,
    AgentApprovalDecision,
    AgentApprovalRequest,
    AgentApprovalResponse,
    AgentCancellationReason,
    AgentChoiceOption,
    AgentChoiceRequest,
    AgentChoiceResponse,
    AgentClassificationResult,
    AgentContext,
    AgentFailureReason,
    AgentInputClassification,
    AgentInterruption,
    AgentInterruptionKind,
    AgentOutput,
    AgentRequest,
    AgentResponseCallback,
    AgentRoute,
    AgentSession,
    AgentSessionState,
    AgentTask,
    AgentTaskConfig,
    AgentTaskState,
    AgentTool,
    AgentToolArgumentSchema,
    AgentToolBehavior,
    AgentToolDangerLevel,
    AgentToolExecutionStatus,
    AgentToolSchema,
    AgentToolValueType,
    NeedleToolCall,
    NeedleToolCatalog,
    NeedleToolDefinition,
    NeedleToolParameter,
    NeedleToolParameterSpec,
    NeedleToolSelection,
    ToolCall,
    ToolExecutionResult,
    ToolResult,
    ValidatedToolCall,
)
from ..modes import mode_allows_agents, resolve_operation_mode
from ..typing.common import JSONSerializable
from .needle import (
    NEEDLE_MODEL_ID,
    NeedleHandler,
    NeedleTokenizer,
    convert_needle_safetensors,
)
from .needle_model import NeedleConfig, NeedleModel
from .runtime import AgentRuntime
from .routing import AgentInputRouter


def agent_mode_enabled(
    config: Mapping[str, JSONSerializable],
) -> bool:
    """Return whether the global config selects the agent operation mode."""
    return mode_allows_agents(resolve_operation_mode(config))


__all__ = [
    "NEEDLE_MODEL_ID",
    "AgentAbortReason",
    "AgentApprovalDecision",
    "AgentApprovalRequest",
    "AgentApprovalResponse",
    "AgentCancellationReason",
    "AgentChoiceOption",
    "AgentChoiceRequest",
    "AgentChoiceResponse",
    "AgentClassificationResult",
    "AgentContext",
    "AgentFailureReason",
    "AgentInputClassification",
    "AgentInputRouter",
    "AgentInterruption",
    "AgentInterruptionKind",
    "AgentOutput",
    "AgentRequest",
    "AgentResponseCallback",
    "AgentRoute",
    "AgentRuntime",
    "AgentSession",
    "AgentSessionState",
    "AgentTask",
    "AgentTaskConfig",
    "AgentTaskState",
    "AgentTool",
    "AgentToolArgumentSchema",
    "AgentToolBehavior",
    "AgentToolDangerLevel",
    "AgentToolExecutionStatus",
    "AgentToolSchema",
    "AgentToolValueType",
    "NeedleConfig",
    "NeedleHandler",
    "NeedleModel",
    "NeedleTokenizer",
    "NeedleToolCall",
    "NeedleToolCatalog",
    "NeedleToolDefinition",
    "NeedleToolParameter",
    "NeedleToolParameterSpec",
    "NeedleToolSelection",
    "ToolCall",
    "ToolExecutionResult",
    "ToolResult",
    "ValidatedToolCall",
    "agent_mode_enabled",
    "convert_needle_safetensors",
]
