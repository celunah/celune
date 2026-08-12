# SPDX-License-Identifier: Apache-2.0
"""Types for Celune's future local-only agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NotRequired, Optional, Protocol, TypedDict, Union, cast

from ..constants import (
    AGENT_CONTEXT_COMPACTION_THRESHOLD,
    AGENT_CONTEXT_SPACE,
    AGENT_MAX_ITERATIONS,
)
from ..persona.capabilities import PersonaCapabilities
from .common import JSON, JSONSerializable
from .locks import ComponentBusyResult
from .modes import OperationMode


class NeedleToolParameterSpec(TypedDict, total=False):
    """Rich parameter descriptor accepted by Needle tool definitions."""

    type: str
    description: NotRequired[str]
    required: NotRequired[bool]
    item_type: NotRequired[str]


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


class AgentSessionState(str, Enum):
    """Lifecycle state of an agent session independent of one task."""

    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


class AgentTaskState(str, Enum):
    """Lifecycle states supported by one agent task."""

    QUEUED = "queued"
    IDLE = "idle"
    CLASSIFYING = "classifying"
    WORKING = "working"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_CHOICE = "awaiting_choice"
    EXECUTING_TOOL = "executing_tool"
    RESPONDING = "responding"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


class AgentToolValueType(str, Enum):
    """JSON-compatible value types accepted by a tool argument."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class AgentToolBehavior(str, Enum):
    """Whether a tool only observes state or can change it."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"


class AgentToolDangerLevel(str, Enum):
    """Risk classification used by future approval policy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentCancellationReason(str, Enum):
    """Reasons an active agent task can be cancelled."""

    USER_REQUEST = "user_request"
    SESSION_CANCELLED = "session_cancelled"
    RUNTIME_SHUTDOWN = "runtime_shutdown"


class AgentInterruptionKind(str, Enum):
    """User-originated interruption modes that keep a task continuable."""

    USER_INTERRUPT = "user_interrupt"
    USER_STEERING = "user_steering"


class AgentAbortReason(str, Enum):
    """Reasons a task stops without being successfully completed."""

    MAX_ITERATIONS = "max_iterations"
    MAX_GENERATED_TOKENS = "max_generated_tokens"
    CONTEXT_LIMIT = "context_limit"
    STUCK_TASK = "stuck_task"


class AgentFailureReason(str, Enum):
    """Reasons a task fails while processing a request."""

    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    INVALID_TOOL_CALL = "invalid_tool_call"
    NO_TOOLS_FOUND = "no_tools_found"
    NO_AVAILABLE_TOOLS = "no_available_tools"
    APPROVAL_DENIED = "approval_denied"
    PERMISSION_DENIED = "permission_denied"
    CHOICE_UNAVAILABLE = "choice_unavailable"
    INTERNAL_ERROR = "internal_error"


class AgentInputClassification(str, Enum):
    """Top-level classification for one user input."""

    CONVERSATION = "conversation"
    TASK = "task"


class AgentRoute(str, Enum):
    """Action selected for one classified input."""

    CONVERSATION = "conversation"
    TASK = "task"
    CLARIFICATION = "clarification"
    TASK_INPUT = "task_input"
    APPROVAL_RESPONSE = "approval_response"
    CHOICE_RESPONSE = "choice_response"
    CANCELLATION = "cancellation"
    INTERRUPTION = "interruption"


class AgentClassificationFailureKind(str, Enum):
    """Failure categories returned when semantic input classification is unavailable."""

    PERSONA_UNAVAILABLE = "persona_unavailable"
    BUSY = "busy"
    TRANSPORT = "transport"
    EMPTY_OUTPUT = "empty_output"
    MALFORMED_OUTPUT = "malformed_output"
    INVALID_SCHEMA = "invalid_schema"


@dataclass(frozen=True)
class AgentClassificationFailure:
    """Typed diagnostic preserved when the classifier cannot produce a result."""

    kind: AgentClassificationFailureKind
    detail: str

    def __post_init__(self) -> None:
        """Require a concrete classifier failure detail."""
        if not self.detail.strip():
            raise ValueError("agent classification failure detail must not be empty")

    def to_json(self) -> JSON:
        """Serialize the classifier failure for diagnostics and event metadata."""
        return {"kind": self.kind.value, "detail": self.detail}


class AgentApprovalDecision(str, Enum):
    """Decisions a caller can make for an approval request."""

    APPROVED = "approved"
    DENIED = "denied"


class AgentPermissionDecision(str, Enum):
    """Policy outcomes for one validated tool call."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class AgentPermissionReason(str, Enum):
    """Reasons that make a policy decision explainable to callers."""

    SAFE_READ_ONLY = "safe_read_only"
    MUTATING_TOOL = "mutating_tool"
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"
    DANGEROUS_TOOL = "dangerous_tool"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    TOOL_UNAVAILABLE = "tool_unavailable"
    TOOL_DISALLOWED = "tool_disallowed"
    LEGACY_TOOL = "legacy_tool"


class AgentToolExecutionStatus(str, Enum):
    """Terminal status values for one tool execution result."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCall(TypedDict):
    """A requested invocation of one registered local tool."""

    id: str
    name: str
    arguments: dict[str, JSONSerializable]


class AgentPermissionMetadata(TypedDict):
    """JSON-compatible permission metadata attached to execution results."""

    decision: str
    reason: str
    task_id: str
    tool_call_id: str
    tool_id: str
    approval_decision: Optional[str]


class ToolResult(TypedDict):
    """The result or failure returned by one local tool."""

    tool_call_id: str
    output: Optional[JSONSerializable]
    error: Optional[str]
    end_task: NotRequired[bool]
    permission: NotRequired[AgentPermissionMetadata]


class ValidatedToolCall(ToolCall):
    """A tool call that passed schema and availability validation."""

    tool_id: str
    behavior: AgentToolBehavior
    danger: AgentToolDangerLevel
    approval_required: bool


class ToolExecutionResult(ToolResult):
    """A tool result carrying status, completion intent, and the resolved tool ID."""

    tool_id: str
    status: AgentToolExecutionStatus


@dataclass(frozen=True)
class AgentTerminalOutcome:
    """Typed terminal outcome exposed to the character response boundary."""

    state: AgentTaskState
    failure_reason: Optional[AgentFailureReason] = None
    abort_reason: Optional[AgentAbortReason] = None
    cancellation_reason: Optional[AgentCancellationReason] = None
    detail: Optional[str] = None
    metadata: Optional[JSON] = None

    def __post_init__(self) -> None:
        """Require terminal state and reason fields to agree."""
        terminal_states = {
            AgentTaskState.COMPLETED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.ABORTED,
        }
        if self.state not in terminal_states:
            raise ValueError("agent terminal outcomes require a terminal task state")
        if self.state == AgentTaskState.FAILED and self.failure_reason is None:
            raise ValueError("failed agent outcomes require a failure reason")
        if self.state != AgentTaskState.FAILED and self.failure_reason is not None:
            raise ValueError("only failed agent outcomes can have a failure reason")
        if self.state == AgentTaskState.ABORTED and self.abort_reason is None:
            raise ValueError("aborted agent outcomes require an abort reason")
        if self.state != AgentTaskState.ABORTED and self.abort_reason is not None:
            raise ValueError("only aborted agent outcomes can have an abort reason")
        if self.state == AgentTaskState.CANCELLED and self.cancellation_reason is None:
            raise ValueError("cancelled agent outcomes require a cancellation reason")
        if (
            self.state != AgentTaskState.CANCELLED
            and self.cancellation_reason is not None
        ):
            raise ValueError(
                "only cancelled agent outcomes can have a cancellation reason"
            )
        if self.detail is not None and not self.detail.strip():
            raise ValueError("agent terminal outcome detail must not be empty")

    def to_json(self) -> JSON:
        """Serialize the terminal outcome for prompts, logs, and diagnostics."""
        return {
            "state": self.state.value,
            "failure_reason": (
                self.failure_reason.value if self.failure_reason is not None else None
            ),
            "abort_reason": (
                self.abort_reason.value if self.abort_reason is not None else None
            ),
            "cancellation_reason": (
                self.cancellation_reason.value
                if self.cancellation_reason is not None
                else None
            ),
            "detail": self.detail,
            "metadata": self.metadata,
        }


class AgentOutput(TypedDict):
    """One externally observable step produced by the agent runtime."""

    tool_call: Optional[ToolCall]
    response: Optional[str]
    end: bool
    paused: bool
    busy: NotRequired[ComponentBusyResult]
    terminal: NotRequired[AgentTerminalOutcome]


@dataclass(frozen=True)
class AgentSession:
    """Stable identity and lifecycle state for one agent conversation."""

    session_id: str
    paused: bool = False
    cancelled: bool = False
    state: AgentSessionState = AgentSessionState.IDLE
    task_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate stable session identity and compatible legacy flags."""
        if not self.session_id.strip():
            raise ValueError("agent session_id must not be empty")
        if self.task_id is not None and not self.task_id.strip():
            raise ValueError("agent task_id must not be empty")
        if self.paused and self.cancelled:
            raise ValueError("agent session cannot be paused and cancelled")

    def to_json(self) -> JSON:
        """Serialize the stable session contract into JSON-compatible data."""
        return {
            "session_id": self.session_id,
            "paused": self.paused,
            "cancelled": self.cancelled,
            "state": self.state.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class AgentRequest:
    """Input supplied to a future agent run."""

    request: str
    history: tuple[JSON, ...] = ()
    session: AgentSession = field(
        default_factory=lambda: AgentSession(session_id="default")
    )

    def __post_init__(self) -> None:
        """Reject empty agent requests while preserving conversation history."""
        if not self.request.strip():
            raise ValueError("agent request must not be empty")

    def to_json(self) -> JSON:
        """Serialize the request and its session into JSON-compatible data."""
        return {
            "request": self.request,
            "history": cast(JSONSerializable, list(self.history)),
            "session": self.session.to_json(),
        }


@dataclass(frozen=True)
class AgentClassificationResult:
    """Typed conversation-first classification and routing result."""

    classification: AgentInputClassification
    confidence: float
    task_request: Optional[AgentRequest] = None
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None
    reason: Optional[str] = None
    routing_metadata: Optional[JSON] = None
    route: AgentRoute = AgentRoute.CONVERSATION
    approval_decision: Optional[AgentApprovalDecision] = None
    choice_id: Optional[str] = None
    choice_freeform: Optional[str] = None
    interruption_kind: Optional[AgentInterruptionKind] = None
    failure: Optional[AgentClassificationFailure] = None
    intent: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate classification confidence and route-specific fields."""
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("agent classification confidence must be between 0 and 1")
        if self.intent is not None and not self.intent.strip():
            raise ValueError("agent classification intent must not be empty")
        if self.requires_clarification and not (
            self.clarification_prompt and self.clarification_prompt.strip()
        ):
            raise ValueError("agent clarification results require a non-empty prompt")
        if self.route == AgentRoute.TASK and (
            self.classification != AgentInputClassification.TASK
            or self.task_request is None
            or self.requires_clarification
        ):
            raise ValueError("task routes require an unambiguous task request")
        if self.failure is not None and self.route == AgentRoute.TASK:
            raise ValueError("classifier failures cannot create task routes")
        if self.route == AgentRoute.CONVERSATION and (
            self.classification != AgentInputClassification.CONVERSATION
            or self.requires_clarification
        ):
            raise ValueError("conversation routes require ordinary conversation input")
        if self.route == AgentRoute.CLARIFICATION and not self.requires_clarification:
            raise ValueError("clarification routes require clarification")
        if self.route == AgentRoute.APPROVAL_RESPONSE and (
            self.approval_decision is None
        ):
            raise ValueError("approval routes require an approval decision")
        if self.route == AgentRoute.CHOICE_RESPONSE and (
            self.choice_id is None and self.choice_freeform is None
        ):
            raise ValueError("choice routes require a choice value")
        if self.route == AgentRoute.INTERRUPTION and self.interruption_kind is None:
            raise ValueError("interruption routes require an interruption kind")
        if (
            self.route
            in {
                AgentRoute.TASK_INPUT,
                AgentRoute.APPROVAL_RESPONSE,
                AgentRoute.CHOICE_RESPONSE,
                AgentRoute.CANCELLATION,
                AgentRoute.INTERRUPTION,
            }
            and self.classification != AgentInputClassification.TASK
        ):
            raise ValueError("active-task routes require task classification")

    @property
    def clarification_required(self) -> bool:
        """Return the compatibility spelling for clarification requirement."""
        return self.requires_clarification

    def to_json(self) -> JSON:
        """Serialize the classification without exposing it in user-facing text."""
        return {
            "classification": self.classification.value,
            "confidence": self.confidence,
            "intent": self.intent,
            "task_request": (
                self.task_request.to_json() if self.task_request is not None else None
            ),
            "requires_clarification": self.requires_clarification,
            "clarification_prompt": self.clarification_prompt,
            "reason": self.reason,
            "routing_metadata": self.routing_metadata,
            "route": self.route.value,
            "approval_decision": (
                self.approval_decision.value
                if self.approval_decision is not None
                else None
            ),
            "choice_id": self.choice_id,
            "choice_freeform": self.choice_freeform,
            "interruption_kind": (
                self.interruption_kind.value
                if self.interruption_kind is not None
                else None
            ),
            "failure": self.failure.to_json() if self.failure is not None else None,
        }


@dataclass(frozen=True)
class AgentContext:
    """Context available to planning, tool use, and response callbacks."""

    request: AgentRequest
    mode: OperationMode
    persona_capabilities: PersonaCapabilities
    task: Optional[AgentTask] = None
    last_tool_result: Optional[ToolResult] = None
    classification_failure: Optional[AgentClassificationFailure] = None


@dataclass(frozen=True)
class AgentPermissionEvaluation:
    """Typed result of evaluating whether one tool call may execute."""

    task_id: str
    tool_call_id: str
    tool_id: str
    decision: AgentPermissionDecision
    reason: AgentPermissionReason
    approval_decision: Optional[AgentApprovalDecision] = None
    metadata: Optional[JSON] = None

    def __post_init__(self) -> None:
        """Validate the identifiers and policy enums in one evaluation."""
        if not self.task_id.strip():
            raise ValueError("agent permission task_id must not be empty")
        if not self.tool_call_id.strip() or not self.tool_id.strip():
            raise ValueError("agent permission tool identifiers must not be empty")
        if not isinstance(self.decision, AgentPermissionDecision):
            raise TypeError("agent permission decision must use its typed enum")
        if not isinstance(self.reason, AgentPermissionReason):
            raise TypeError("agent permission reason must use its typed enum")
        if self.approval_decision is not None and not isinstance(
            self.approval_decision, AgentApprovalDecision
        ):
            raise TypeError("agent approval decision must use its typed enum")

    def to_metadata(self) -> AgentPermissionMetadata:
        """Return the compact JSON metadata carried by execution results."""
        return {
            "decision": self.decision.value,
            "reason": self.reason.value,
            "task_id": self.task_id,
            "tool_call_id": self.tool_call_id,
            "tool_id": self.tool_id,
            "approval_decision": (
                self.approval_decision.value
                if self.approval_decision is not None
                else None
            ),
        }

    def to_json(self) -> JSON:
        """Serialize the policy decision and optional policy metadata."""
        payload = cast(JSON, dict(self.to_metadata()))
        payload["metadata"] = self.metadata
        return payload


@dataclass(frozen=True)
class AgentTaskConfig:
    """Limits and thresholds applied to one future agent task."""

    max_iterations: int = AGENT_MAX_ITERATIONS
    max_generated_tokens: Optional[int] = None
    context_compaction_threshold: int = AGENT_CONTEXT_COMPACTION_THRESHOLD
    stuck_task_threshold: int = 3
    context_space: int = AGENT_CONTEXT_SPACE

    def __post_init__(self) -> None:
        """Validate positive task limits and detection thresholds."""
        for name, value in (
            ("max_iterations", self.max_iterations),
            ("context_compaction_threshold", self.context_compaction_threshold),
            ("stuck_task_threshold", self.stuck_task_threshold),
            ("context_space", self.context_space),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"agent {name} must be a positive integer")
        if self.max_generated_tokens is not None and (
            isinstance(self.max_generated_tokens, bool)
            or not isinstance(self.max_generated_tokens, int)
            or self.max_generated_tokens <= 0
        ):
            raise ValueError(
                "agent max_generated_tokens must be a positive integer or None"
            )

    def to_json(self) -> JSON:
        """Serialize task limits into JSON-compatible data."""
        return {
            "max_iterations": self.max_iterations,
            "max_generated_tokens": self.max_generated_tokens,
            "context_compaction_threshold": self.context_compaction_threshold,
            "stuck_task_threshold": self.stuck_task_threshold,
            "context_space": self.context_space,
        }


@dataclass(frozen=True)
class AgentToolArgumentSchema:
    """Typed schema for one registered tool argument."""

    name: str
    value_type: AgentToolValueType
    description: str = ""
    required: bool = True
    item_type: Optional[AgentToolValueType] = None

    def __post_init__(self) -> None:
        """Validate argument names and array item-type declarations."""
        if not self.name.strip():
            raise ValueError("agent tool argument name must not be empty")
        if self.value_type == AgentToolValueType.ARRAY:
            if self.item_type is None:
                raise ValueError("array agent tool arguments require item_type")
        elif self.item_type is not None:
            raise ValueError("item_type is only valid for array agent arguments")

    def to_json(self) -> JSON:
        """Serialize the argument schema into JSON-compatible data."""
        payload: JSON = {
            "name": self.name,
            "type": self.value_type.value,
            "description": self.description,
            "required": self.required,
        }
        if self.item_type is not None:
            payload["item_type"] = self.item_type.value
        return payload


@dataclass(frozen=True)
class AgentToolSchema:
    """Permission-aware schema advertised for one local agent tool."""

    tool_id: str
    display_name: str
    description: str
    arguments: tuple[AgentToolArgumentSchema, ...] = ()
    behavior: AgentToolBehavior = AgentToolBehavior.READ_ONLY
    danger: AgentToolDangerLevel = AgentToolDangerLevel.LOW
    approval_required: bool = False
    available: bool = True

    def __post_init__(self) -> None:
        """Validate tool identity and uniqueness of its argument schema."""
        if not self.tool_id.strip():
            raise ValueError("agent tool_id must not be empty")
        if not self.display_name.strip():
            raise ValueError("agent tool display_name must not be empty")
        names = tuple(argument.name for argument in self.arguments)
        if len(names) != len(set(names)):
            raise ValueError("agent tool argument names must be unique")

    def to_json(self) -> JSON:
        """Serialize the complete tool schema into JSON-compatible data."""
        return {
            "tool_id": self.tool_id,
            "display_name": self.display_name,
            "description": self.description,
            "arguments": [argument.to_json() for argument in self.arguments],
            "behavior": self.behavior.value,
            "danger": self.danger.value,
            "approval_required": self.approval_required,
            "available": self.available,
        }


@dataclass(frozen=True)
class AgentInterruption:
    """A continuable user interruption or steering instruction."""

    kind: AgentInterruptionKind
    instruction: Optional[str] = None

    def __post_init__(self) -> None:
        """Require steering interruptions to carry replacement guidance."""
        if self.kind == AgentInterruptionKind.USER_STEERING and not (
            self.instruction and self.instruction.strip()
        ):
            raise ValueError("user steering requires a non-empty instruction")

    def to_json(self) -> JSON:
        """Serialize the interruption into JSON-compatible data."""
        return {
            "kind": self.kind.value,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class AgentApprovalRequest:
    """Request for permission to execute one validated tool call."""

    request_id: str
    task_id: str
    tool_call: ValidatedToolCall
    prompt: str
    permission: Optional[AgentPermissionEvaluation] = None

    def __post_init__(self) -> None:
        """Validate approval request identity and prompt text."""
        if not self.request_id.strip() or not self.task_id.strip():
            raise ValueError("agent approval request IDs must not be empty")
        if not self.prompt.strip():
            raise ValueError("agent approval prompt must not be empty")

    def to_json(self) -> JSON:
        """Serialize the approval request and validated tool call."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "tool_call": {
                "id": self.tool_call["id"],
                "name": self.tool_call["name"],
                "arguments": self.tool_call["arguments"],
                "tool_id": self.tool_call["tool_id"],
                "behavior": self.tool_call["behavior"].value,
                "danger": self.tool_call["danger"].value,
                "approval_required": self.tool_call["approval_required"],
            },
            "prompt": self.prompt,
            "permission": (
                self.permission.to_json() if self.permission is not None else None
            ),
        }


@dataclass(frozen=True)
class AgentChoiceOption:
    """One selectable option in a user choice request."""

    choice_id: str
    label: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate choice identity and display text."""
        if not self.choice_id.strip() or not self.label.strip():
            raise ValueError("agent choice IDs and labels must not be empty")


@dataclass(frozen=True)
class AgentChoiceRequest:
    """Request for a user choice while pausing task iteration."""

    request_id: str
    task_id: str
    prompt: str
    options: tuple[AgentChoiceOption, ...]
    allow_freeform: bool = False

    def __post_init__(self) -> None:
        """Require unique, non-empty choice options."""
        if not self.request_id.strip() or not self.task_id.strip():
            raise ValueError("agent choice request IDs must not be empty")
        if not self.prompt.strip():
            raise ValueError("agent choice prompt must not be empty")
        if not self.options and not self.allow_freeform:
            raise ValueError("agent choice requests require options or freeform input")
        choice_ids = tuple(option.choice_id for option in self.options)
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("agent choice IDs must be unique")

    def to_json(self) -> JSON:
        """Serialize the choice request and its options."""
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "options": [
                {
                    "choice_id": option.choice_id,
                    "label": option.label,
                    "description": option.description,
                }
                for option in self.options
            ],
            "allow_freeform": self.allow_freeform,
        }


@dataclass(frozen=True)
class AgentApprovalResponse:
    """User response to an approval request."""

    request_id: str
    decision: AgentApprovalDecision

    def __post_init__(self) -> None:
        """Validate the request identity being answered."""
        if not self.request_id.strip():
            raise ValueError("agent approval response request_id must not be empty")

    def to_json(self) -> JSON:
        """Serialize the approval response into JSON-compatible data."""
        return {
            "request_id": self.request_id,
            "decision": self.decision.value,
        }


@dataclass(frozen=True)
class AgentChoiceResponse:
    """User response to a choice request, optionally carrying steering text."""

    request_id: str
    choice_id: Optional[str] = None
    freeform: Optional[str] = None

    def __post_init__(self) -> None:
        """Require either a selected option or non-empty freeform input."""
        if not self.request_id.strip():
            raise ValueError("agent choice response request_id must not be empty")
        has_choice = self.choice_id is not None and self.choice_id.strip()
        has_freeform = self.freeform is not None and self.freeform.strip()
        if bool(has_choice) == bool(has_freeform):
            raise ValueError("agent choice response requires exactly one answer")

    def to_json(self) -> JSON:
        """Serialize the selected choice or steering text."""
        return {
            "request_id": self.request_id,
            "choice_id": self.choice_id,
            "freeform": self.freeform,
        }


_ALLOWED_AGENT_TASK_TRANSITIONS: dict[AgentTaskState, frozenset[AgentTaskState]] = {
    AgentTaskState.QUEUED: frozenset(
        {
            AgentTaskState.IDLE,
            AgentTaskState.PLANNING,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.IDLE: frozenset(
        {
            AgentTaskState.CLASSIFYING,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.CLASSIFYING: frozenset(
        {
            AgentTaskState.WORKING,
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.WORKING: frozenset(
        {
            AgentTaskState.PLANNING,
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.EXECUTING_TOOL,
            AgentTaskState.RESPONDING,
            AgentTaskState.COMPLETED,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.PLANNING: frozenset(
        {
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.EXECUTING_TOOL,
            AgentTaskState.RESPONDING,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.AWAITING_APPROVAL: frozenset(
        {
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.CANCELLED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.AWAITING_CHOICE: frozenset(
        {
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.CANCELLED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.EXECUTING_TOOL: frozenset(
        {
            AgentTaskState.PLANNING,
            AgentTaskState.RESPONDING,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.RESPONDING: frozenset(
        {
            AgentTaskState.PLANNING,
            AgentTaskState.COMPLETED,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.PAUSED: frozenset(
        {
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
            AgentTaskState.INTERRUPTED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.INTERRUPTED: frozenset(
        {
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
            AgentTaskState.PAUSED,
            AgentTaskState.CANCELLED,
            AgentTaskState.CANCELLING,
            AgentTaskState.ABORTED,
        }
    ),
    AgentTaskState.CANCELLING: frozenset({AgentTaskState.CANCELLED}),
    AgentTaskState.COMPLETED: frozenset(),
    AgentTaskState.FAILED: frozenset(),
    AgentTaskState.CANCELLED: frozenset(),
    AgentTaskState.ABORTED: frozenset(),
}


@dataclass
class AgentTask:
    """Mutable contract state for one agent task before runtime execution exists."""

    task_id: str
    session_id: str
    request: AgentRequest
    config: AgentTaskConfig = field(default_factory=AgentTaskConfig)
    state: AgentTaskState = AgentTaskState.QUEUED
    iterations: int = 0
    generated_tokens: int = 0
    context_tokens: int = 0
    stalled_iterations: int = 0
    generation: int = 0
    cancellation_reason: Optional[AgentCancellationReason] = None
    abort_reason: Optional[AgentAbortReason] = None
    failure_reason: Optional[AgentFailureReason] = None
    failure_detail: Optional[str] = None
    interruption: Optional[AgentInterruption] = None
    completion_metadata: Optional[JSON] = None
    permission_decision: Optional[AgentPermissionEvaluation] = None

    def __post_init__(self) -> None:
        """Validate task identity, counters, and terminal-reason consistency."""
        if not self.task_id.strip() or not self.session_id.strip():
            raise ValueError("agent task and session IDs must not be empty")
        for name, value in (
            ("iterations", self.iterations),
            ("generated_tokens", self.generated_tokens),
            ("context_tokens", self.context_tokens),
            ("stalled_iterations", self.stalled_iterations),
            ("generation", self.generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"agent {name} must be a non-negative integer")
        if self.iterations > self.config.max_iterations:
            raise ValueError("agent iterations cannot exceed max_iterations")
        if (
            self.config.max_generated_tokens is not None
            and self.generated_tokens > self.config.max_generated_tokens
        ):
            raise ValueError(
                "agent generated_tokens cannot exceed max_generated_tokens"
            )
        if self.stalled_iterations > self.config.stuck_task_threshold:
            raise ValueError("agent stalled_iterations exceeds stuck_task_threshold")
        if self.state == AgentTaskState.CANCELLED and self.cancellation_reason is None:
            raise ValueError("cancelled agent tasks require a cancellation reason")
        if (
            self.state != AgentTaskState.CANCELLED
            and self.cancellation_reason is not None
        ):
            raise ValueError(
                "only cancelled agent tasks can have a cancellation reason"
            )
        if self.state == AgentTaskState.ABORTED and self.abort_reason is None:
            raise ValueError("aborted agent tasks require an abort reason")
        if self.state != AgentTaskState.ABORTED and self.abort_reason is not None:
            raise ValueError("only aborted agent tasks can have an abort reason")
        if self.state == AgentTaskState.FAILED and self.failure_reason is None:
            raise ValueError("failed agent tasks require a failure reason")
        if self.state != AgentTaskState.FAILED and (
            self.failure_reason is not None or self.failure_detail is not None
        ):
            raise ValueError("only failed agent tasks can have failure details")
        if (
            self.state != AgentTaskState.COMPLETED
            and self.completion_metadata is not None
        ):
            raise ValueError("only completed agent tasks can have completion metadata")

    @property
    def is_terminal(self) -> bool:
        """Return whether the task can no longer continue."""
        return self.state in {
            AgentTaskState.COMPLETED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.ABORTED,
        }

    @property
    def needs_context_compaction(self) -> bool:
        """Return whether context has reached the configured compaction threshold."""
        return self.context_tokens >= self.config.context_compaction_threshold

    def transition(self, state: AgentTaskState) -> None:
        """Move to a valid next state without changing iteration accounting."""
        if state == self.state:
            return
        allowed = _ALLOWED_AGENT_TASK_TRANSITIONS[self.state]
        if state not in allowed:
            raise ValueError(
                f"invalid agent task transition: '{self.state.value}' -> '{state.value}'"
            )
        self.state = state

    def consume_iteration(self) -> bool:
        """Consume one planning iteration, aborting when the limit is reached."""
        if self.is_terminal:
            raise ValueError("cannot consume an iteration for a terminal agent task")
        if self.iterations >= self.config.max_iterations:
            self.abort(AgentAbortReason.MAX_ITERATIONS)
            return False
        self.iterations += 1
        return True

    def add_generated_tokens(self, count: int) -> bool:
        """Account for generated tokens and abort when the configured limit is exceeded."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("agent generated token count must be non-negative")
        if self.is_terminal:
            raise ValueError("cannot add tokens to a terminal agent task")
        if (
            self.config.max_generated_tokens is not None
            and self.generated_tokens + count > self.config.max_generated_tokens
        ):
            self.abort(AgentAbortReason.MAX_GENERATED_TOKENS)
            return False
        self.generated_tokens += count
        return True

    def update_context_tokens(self, count: int) -> None:
        """Update context usage without performing compaction itself."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("agent context token count must be non-negative")
        self.context_tokens = count

    def record_progress(self, progressed: bool) -> bool:
        """Track progress and abort after repeated non-progressing iterations."""
        if self.is_terminal:
            raise ValueError("cannot record progress for a terminal agent task")
        if progressed:
            self.stalled_iterations = 0
            return True
        self.stalled_iterations += 1
        if self.stalled_iterations >= self.config.stuck_task_threshold:
            self.abort(AgentAbortReason.STUCK_TASK)
            return False
        return True

    def interrupt(self, interruption: AgentInterruption) -> None:
        """Pause active work for user interruption or steering."""
        self.transition(AgentTaskState.INTERRUPTED)
        self.interruption = interruption

    def pause(self) -> None:
        """Pause a task while preserving its iteration and token counters."""
        self.transition(AgentTaskState.PAUSED)

    def resume(self, state: AgentTaskState = AgentTaskState.PLANNING) -> None:
        """Resume a paused or interrupted task at a validated lifecycle state."""
        if self.state not in {AgentTaskState.PAUSED, AgentTaskState.INTERRUPTED}:
            raise ValueError("only paused or interrupted agent tasks can resume")
        if state not in {
            AgentTaskState.PLANNING,
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
        }:
            raise ValueError("agent tasks can only resume into active lifecycle states")
        self.transition(state)
        self.interruption = None

    def complete(self, metadata: Optional[JSON] = None) -> None:
        """Mark a responding task as successfully completed."""
        if self.is_terminal:
            raise ValueError("cannot complete a terminal agent task")
        self.transition(AgentTaskState.COMPLETED)
        self.completion_metadata = metadata

    def fail(self, reason: AgentFailureReason, detail: Optional[str] = None) -> None:
        """Mark an active task as failed with a typed reason."""
        if self.is_terminal:
            raise ValueError("cannot fail a terminal agent task")
        self.transition(AgentTaskState.FAILED)
        self.failure_reason = reason
        self.failure_detail = detail.strip() if detail else None

    def cancel(self, reason: AgentCancellationReason) -> None:
        """Cancel any non-terminal task with a typed cancellation reason."""
        if self.is_terminal:
            raise ValueError("cannot cancel a terminal agent task")
        self.transition(AgentTaskState.CANCELLED)
        self.cancellation_reason = reason

    def abort(self, reason: AgentAbortReason) -> None:
        """Abort any non-terminal task with a typed safety or limit reason."""
        if self.is_terminal:
            return
        self.transition(AgentTaskState.ABORTED)
        self.abort_reason = reason

    def to_json(self) -> JSON:
        """Serialize task state and accounting into JSON-compatible data."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "request": self.request.to_json(),
            "config": self.config.to_json(),
            "state": self.state.value,
            "iterations": self.iterations,
            "generated_tokens": self.generated_tokens,
            "context_tokens": self.context_tokens,
            "stalled_iterations": self.stalled_iterations,
            "generation": self.generation,
            "cancellation_reason": (
                self.cancellation_reason.value
                if self.cancellation_reason is not None
                else None
            ),
            "abort_reason": (
                self.abort_reason.value if self.abort_reason is not None else None
            ),
            "failure_reason": (
                self.failure_reason.value if self.failure_reason is not None else None
            ),
            "failure_detail": self.failure_detail,
            "completion_metadata": self.completion_metadata,
            "permission_decision": (
                self.permission_decision.to_json()
                if self.permission_decision is not None
                else None
            ),
            "interruption": (
                self.interruption.to_json() if self.interruption is not None else None
            ),
        }


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


class AgentPlanner(Protocol):
    """Dependency boundary for one planner decision cycle."""

    def __call__(self, context: AgentContext, /) -> AgentOutput:
        """Produce one typed planning output."""
        raise NotImplementedError("protocol not defined")


class AgentToolSelector(Protocol):
    """Dependency boundary for selecting at most one tool call."""

    def __call__(
        self,
        context: AgentContext,
        output: AgentOutput,
        /,
    ) -> Optional[ToolCall]:
        """Select one typed tool call or no tool call."""
        raise NotImplementedError("protocol not defined")


class AgentToolExecutor(Protocol):
    """Dependency boundary for executing one selected tool call."""

    def __call__(self, context: AgentContext, call: ToolCall, /) -> ToolResult:
        """Return the typed result of one tool execution."""
        raise NotImplementedError("protocol not defined")


class AgentPermissionPolicy(Protocol):
    """Dependency boundary for deterministic tool permission evaluation."""

    def __call__(
        self,
        task: AgentTask,
        call: ValidatedToolCall,
        /,
    ) -> AgentPermissionEvaluation:
        """Return the policy decision for one validated tool call."""
        raise NotImplementedError("protocol not defined")


class AgentToolResultHandler(Protocol):
    """Dependency boundary for interpreting one typed tool result."""

    def __call__(
        self,
        context: AgentContext,
        result: ToolResult,
        /,
    ) -> AgentOutput:
        """Produce the next externally visible output after a tool result."""
        raise NotImplementedError("protocol not defined")


class AgentResponder(Protocol):
    """Dependency boundary for producing a final response."""

    def __call__(self, context: AgentContext, /) -> AgentOutput:
        """Produce one typed response output."""
        raise NotImplementedError("protocol not defined")


class AgentContextCompactor(Protocol):
    """Dependency boundary for future model-specific context compaction."""

    def __call__(self, context: AgentContext, /) -> AgentContext:
        """Return the compacted context without choosing a model policy."""
        raise NotImplementedError("protocol not defined")


class AgentTokenCounter(Protocol):
    """Dependency boundary for counting generated response tokens."""

    def __call__(self, text: str, /) -> int:
        """Return the generated-token count for one response string."""
        raise NotImplementedError("protocol not defined")
