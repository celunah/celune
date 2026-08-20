# SPDX-License-Identifier: MIT
"""Typed lifecycle ownership for Celune's future local-only agent runtime."""

from __future__ import annotations

from uuid import uuid4
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional, cast

from ..i18n import string
from ..typing.aliases import LogLevel
from ..typing.modes import OperationMode
from ..extensions.events import EventDispatcher
from ..typing.common import JSON, JSONSerializable
from ..typing.events import EventName, EventPayload
from ..persona.capabilities import PersonaCapabilities
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentBusyResult,
    ComponentLockRequirement,
)
from ..dataclasses.events import (
    AgentTaskFinishedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)
from ..typing.agent import (
    ToolCall,
    AgentTask,
    AgentTool,
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentPlanner,
    AgentRequest,
    AgentSession,
    AgentResponder,
    AgentTaskState,
    AgentTaskConfig,
    AgentToolSchema,
    AgentAbortReason,
    AgentInterruption,
    AgentSessionState,
    AgentTokenCounter,
    AgentToolBehavior,
    AgentToolExecutor,
    AgentToolSelector,
    ValidatedToolCall,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentChoiceResponse,
    ToolExecutionResult,
    AgentApprovalRequest,
    AgentTerminalOutcome,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentContextCompactor,
    AgentInterruptionKind,
    AgentPermissionPolicy,
    AgentPermissionReason,
    AgentResponseCallback,
    AgentToolResultHandler,
    AgentCancellationReason,
    AgentPermissionDecision,
    AgentToolExecutionStatus,
    AgentPermissionEvaluation,
    AgentClassificationFailure,
)

if TYPE_CHECKING:
    from ..locks import ComponentLockLease, ComponentLockManager
    from ..celune import Celune


def _default_token_counter(text: str) -> int:
    """Count response words as a dependency-free token estimate."""
    return len(text.split())


def _is_json_value(value: JSONSerializable) -> bool:
    """Return whether a runtime value is compatible with Celune JSON metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


def _empty_output(
    response: Optional[str] = None,
    *,
    end: bool = True,
    paused: bool = False,
    terminal: Optional[AgentTerminalOutcome] = None,
) -> AgentOutput:
    """Build the stable output shape required by the agent contract."""
    output: AgentOutput = {
        "tool_call": None,
        "response": response,
        "end": end,
        "paused": paused,
    }
    if terminal is not None:
        output["terminal"] = terminal
    return output


def _paused_output() -> AgentOutput:
    """Build an output that preserves a waiting or interrupted task."""
    return _empty_output(end=False, paused=True)


def _busy_output(busy: ComponentBusyResult) -> AgentOutput:
    """Build a paused output that carries typed component-conflict metadata."""
    return {
        **_paused_output(),
        "busy": busy,
    }


class DefaultAgentPermissionPolicy:
    """Apply deterministic approval rules to validated local tool calls."""

    def __init__(
        self,
        *,
        approval_available: bool = True,
        disallowed_tool_ids: Sequence[str] = (),
    ) -> None:
        """Configure approval availability and an explicit deny list."""
        if not isinstance(approval_available, bool):
            raise TypeError("agent approval_available must be a boolean")
        normalized = tuple(tool_id.strip() for tool_id in disallowed_tool_ids)
        if any(not tool_id for tool_id in normalized):
            raise ValueError("agent disallowed tool IDs must not be empty")
        self.approval_available = approval_available
        self.disallowed_tool_ids = frozenset(normalized)

    def __call__(
        self,
        task: AgentTask,
        call: ValidatedToolCall,
        /,
    ) -> AgentPermissionEvaluation:
        """Return the default policy decision for one validated tool call."""
        if call["tool_id"] in self.disallowed_tool_ids:
            return self._evaluation(
                task,
                call,
                AgentPermissionDecision.DENY,
                AgentPermissionReason.TOOL_DISALLOWED,
            )
        if call["danger"] == AgentToolDangerLevel.HIGH:
            if not self.approval_available:
                return self._evaluation(
                    task,
                    call,
                    AgentPermissionDecision.DENY,
                    AgentPermissionReason.APPROVAL_UNAVAILABLE,
                )
            return self._evaluation(
                task,
                call,
                AgentPermissionDecision.REQUIRE_APPROVAL,
                AgentPermissionReason.DANGEROUS_TOOL,
            )
        if call["behavior"] == AgentToolBehavior.MUTATING:
            if not self.approval_available:
                return self._evaluation(
                    task,
                    call,
                    AgentPermissionDecision.DENY,
                    AgentPermissionReason.APPROVAL_UNAVAILABLE,
                )
            return self._evaluation(
                task,
                call,
                AgentPermissionDecision.REQUIRE_APPROVAL,
                AgentPermissionReason.MUTATING_TOOL,
            )
        if call["approval_required"]:
            if not self.approval_available:
                return self._evaluation(
                    task,
                    call,
                    AgentPermissionDecision.DENY,
                    AgentPermissionReason.APPROVAL_UNAVAILABLE,
                )
            return self._evaluation(
                task,
                call,
                AgentPermissionDecision.REQUIRE_APPROVAL,
                AgentPermissionReason.EXPLICIT_APPROVAL_REQUIRED,
            )
        reason = (
            AgentPermissionReason.SAFE_READ_ONLY
            if call["behavior"] == AgentToolBehavior.READ_ONLY
            else AgentPermissionReason.LEGACY_TOOL
        )
        return self._evaluation(task, call, AgentPermissionDecision.ALLOW, reason)

    @staticmethod
    def _evaluation(
        task: AgentTask,
        call: ValidatedToolCall,
        decision: AgentPermissionDecision,
        reason: AgentPermissionReason,
    ) -> AgentPermissionEvaluation:
        """Build one policy evaluation with the current task and call IDs."""
        return AgentPermissionEvaluation(
            task_id=task.task_id,
            tool_call_id=call["id"],
            tool_id=call["tool_id"],
            decision=decision,
            reason=reason,
        )


class AgentRuntime:
    """Own bounded agent task orchestration and lifecycle state."""

    def __init__(
        self,
        tools: Sequence[AgentTool] = (),
        *,
        event_dispatcher: Optional[EventDispatcher] = None,
        celune: Optional[Celune] = None,
        mode: OperationMode = "agent",
        persona_capabilities: Optional[PersonaCapabilities] = None,
        planner: Optional[AgentPlanner] = None,
        tool_selector: Optional[AgentToolSelector] = None,
        tool_executor: Optional[AgentToolExecutor] = None,
        tool_result_handler: Optional[AgentToolResultHandler] = None,
        responder: Optional[AgentResponder] = None,
        compactor: Optional[AgentContextCompactor] = None,
        token_counter: Optional[AgentTokenCounter] = None,
        tool_schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        permission_policy: Optional[AgentPermissionPolicy] = None,
    ) -> None:
        """Create a lifecycle owner around local tools and an existing event bus."""
        self.tools = tuple(tools)
        self._event_dispatcher = event_dispatcher
        self._celune = celune
        self._mode = mode
        self._persona_capabilities = persona_capabilities or PersonaCapabilities()
        self._planner = planner
        self._tool_selector = tool_selector
        self._tool_executor = tool_executor
        self._tool_result_handler = tool_result_handler
        self._responder = responder
        self._compactor = compactor
        self._token_counter = token_counter or _default_token_counter
        self._tool_schemas = self._index_tool_schemas(tool_schemas)
        self._permission_policy = permission_policy or DefaultAgentPermissionPolicy()
        self._tasks: dict[str, AgentTask] = {}
        self._contexts: dict[str, AgentContext] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._pending_approvals: dict[str, AgentApprovalRequest] = {}
        self._pending_choices: dict[str, AgentChoiceRequest] = {}
        self._suspension_origins: dict[str, AgentTaskState] = {}
        self._pending_tool_calls: dict[str, ToolCall] = {}
        self._last_tool_calls: dict[str, ToolCall] = {}
        self._terminal_events: set[str] = set()
        self._component_locks: Optional[ComponentLockManager] = getattr(
            celune,
            "component_locks",
            None,
        )
        self._last_busy: Optional[ComponentBusyResult] = None

    def _log(self, message: str, *, loglevel: LogLevel = "debug") -> None:
        """Forward agent diagnostics through Celune's configured log gate."""
        if self._celune is None:
            return
        try:
            log = self._celune.log
        except AttributeError:
            return
        log(message, loglevel=loglevel)

    @property
    def last_busy(self) -> Optional[ComponentBusyResult]:
        """Return the latest typed component conflict, if one occurred."""
        return self._last_busy

    @staticmethod
    def _index_tool_schemas(
        schemas: Optional[Mapping[str, AgentToolSchema]],
    ) -> dict[str, AgentToolSchema]:
        """Index existing tool schemas by both registered names and tool IDs."""
        indexed: dict[str, AgentToolSchema] = {}
        for key, schema in (schemas or {}).items():
            if not key.strip():
                raise ValueError("agent tool schema keys must not be empty")
            for schema_key in (key, schema.tool_id):
                previous = indexed.get(schema_key)
                if previous is not None and previous != schema:
                    raise ValueError(
                        f"agent tool schemas collide for key '{schema_key}'"
                    )
                indexed[schema_key] = schema
        return indexed

    def create_context(
        self,
        request: AgentRequest,
        task: Optional[AgentTask] = None,
        classification_failure: Optional[AgentClassificationFailure] = None,
    ) -> AgentContext:
        """Build a context that keeps the request and optional task together."""
        return AgentContext(
            request=request,
            mode=self._mode,
            persona_capabilities=self._persona_capabilities,
            task=task,
            classification_failure=classification_failure,
        )

    def create_task(
        self,
        request: AgentRequest,
        config: Optional[AgentTaskConfig] = None,
        *,
        task_id: Optional[str] = None,
    ) -> AgentTask:
        """Create an idle task for one explicit agent action request."""
        session_id = request.session.session_id
        if request.session.paused or request.session.cancelled:
            raise ValueError(
                "cannot create an agent task from a paused or cancelled session"
            )
        current = self._session_task(session_id)
        if current is not None and not current.is_terminal:
            raise ValueError(f"agent session '{session_id}' already has an active task")

        resolved_task_id = task_id or uuid4().hex
        if not resolved_task_id.strip():
            raise ValueError("agent task_id must not be empty")
        if resolved_task_id in self._tasks:
            raise ValueError(f"agent task '{resolved_task_id}' already exists")

        task = AgentTask(
            task_id=resolved_task_id,
            session_id=session_id,
            request=request,
            config=config if config is not None else AgentTaskConfig(),
        )
        self._tasks[task.task_id] = task
        self._contexts[task.task_id] = self.create_context(request, task)
        self._sessions[session_id] = AgentSession(
            session_id=session_id,
            state=AgentSessionState.IDLE,
            task_id=task.task_id,
        )
        self._transition(task, AgentTaskState.IDLE)
        self._log(
            f"[AGENT] task_created task={task.task_id} session={task.session_id}",
            loglevel="verbose",
        )
        return task

    def get_task(self, task_id: str) -> AgentTask:
        """Return a known task or reject an unknown lifecycle identifier."""
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown agent task '{task_id}'")
        return task

    def get_context(self, task_id: str) -> AgentContext:
        """Return the stable context associated with a known task."""
        self.get_task(task_id)
        return self._contexts[task_id]

    def get_session(self, session_id: str) -> AgentSession:
        """Return the current explicit and legacy-compatible session state."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"unknown agent session '{session_id}'")
        return session

    def get_active_task(self, session_id: str) -> Optional[AgentTask]:
        """Return the non-terminal task currently associated with a session."""
        task = self._session_task(session_id)
        if task is None or task.is_terminal:
            return None
        return task

    def get_pending_approval(
        self,
        task_id: str,
    ) -> Optional[AgentApprovalRequest]:
        """Return the pending approval request for a task, when one exists."""
        self.get_task(task_id)
        return self._pending_approvals.get(task_id)

    def get_pending_choice(self, task_id: str) -> Optional[AgentChoiceRequest]:
        """Return the pending choice request for a task, when one exists."""
        self.get_task(task_id)
        return self._pending_choices.get(task_id)

    def start_task(self, task_id: str) -> AgentTask:
        """Move an idle task into the explicit classification phase."""
        task = self.get_task(task_id)
        if task.state != AgentTaskState.IDLE:
            raise ValueError("only idle agent tasks can start classification")
        self._log(
            f"[AGENT] classify_start task={task.task_id}",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.CLASSIFYING)
        return task

    def classify_task(self, task_id: str) -> AgentTask:
        """Complete the lifecycle classification boundary without heuristics."""
        task = self.get_task(task_id)
        if task.state != AgentTaskState.CLASSIFYING:
            raise ValueError("only classifying agent tasks can begin work")
        self._log(
            f"[AGENT] classify_complete task={task.task_id}",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.WORKING)
        return task

    def request_approval(
        self,
        task_id: str,
        request: AgentApprovalRequest,
    ) -> AgentTask:
        """Pause a task for approval without consuming an iteration."""
        task = self.get_task(task_id)
        self._validate_request_task(task, request.task_id)
        if task.state not in {
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
        }:
            raise ValueError(
                "agent approval can only pause classifying or working tasks"
            )
        if request.permission is None:
            validated, schema = self._validated_permission_call(request.tool_call)
            if schema is not None and not schema.available:
                return self.fail_task(
                    task_id,
                    AgentFailureReason.PERMISSION_DENIED,
                    AgentPermissionReason.TOOL_UNAVAILABLE.value,
                )
            permission = self._permission_policy(task, validated)
            self._validate_permission_evaluation(permission, task, validated)
            if permission.decision == AgentPermissionDecision.DENY:
                return self.fail_task(
                    task_id,
                    AgentFailureReason.PERMISSION_DENIED,
                    permission.reason.value,
                )
            if permission.decision == AgentPermissionDecision.ALLOW:
                permission = replace(
                    permission,
                    decision=AgentPermissionDecision.REQUIRE_APPROVAL,
                    reason=AgentPermissionReason.EXPLICIT_APPROVAL_REQUIRED,
                )
            request = replace(
                request,
                tool_call=validated,
                permission=permission,
            )
        else:
            self._validate_permission_evaluation(
                request.permission,
                task,
                request.tool_call,
            )
        task.permission_decision = request.permission
        self._pending_approvals[task_id] = request
        self._pending_tool_calls[task_id] = request.tool_call
        self._log(
            f"[AGENT] approval_requested task={task.task_id} "
            f"tool={request.tool_call['name']} request={request.request_id}",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.AWAITING_APPROVAL)
        self._emit(
            "agent_approval_requested",
            AgentApprovalRequestedEvent(
                celune=cast("Celune", self._celune),
                task_id=task.task_id,
                session_id=task.session_id,
                request=request,
            ),
        )
        return task

    def respond_to_approval(
        self,
        task_id: str,
        response: AgentApprovalResponse,
    ) -> AgentTask:
        """Resume or fail a task after validating its pending approval response."""
        task = self.get_task(task_id)
        pending = self._pending_approvals.get(task_id)
        if task.state != AgentTaskState.AWAITING_APPROVAL or pending is None:
            raise ValueError("agent task is not awaiting approval")
        if pending.request_id != response.request_id:
            raise ValueError(
                "agent approval response does not match the pending request"
            )
        self._pending_approvals.pop(task_id)
        if pending.permission is not None:
            task.permission_decision = replace(
                pending.permission,
                approval_decision=response.decision,
            )
        if response.decision == AgentApprovalDecision.DENIED:
            self._pending_tool_calls.pop(task_id, None)
            self._log(
                f"[AGENT] approval_response task={task.task_id} decision=denied",
                loglevel="verbose",
            )
            reason = (
                AgentFailureReason.PERMISSION_DENIED
                if pending.permission is not None
                else AgentFailureReason.APPROVAL_DENIED
            )
            return self.fail_task(task_id, reason)
        self._log(
            f"[AGENT] approval_response task={task.task_id} decision=approved",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.WORKING)
        return task

    def request_choice(
        self,
        task_id: str,
        request: AgentChoiceRequest,
    ) -> AgentTask:
        """Pause a task for a user choice without consuming an iteration."""
        task = self.get_task(task_id)
        self._validate_request_task(task, request.task_id)
        if task.state not in {
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
        }:
            raise ValueError("agent choice can only pause classifying or working tasks")
        self._pending_choices[task_id] = request
        self._log(
            f"[AGENT] choice_requested task={task.task_id} "
            f"request={request.request_id} options={len(request.options)}",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.AWAITING_CHOICE)
        self._emit(
            "agent_choice_requested",
            AgentChoiceRequestedEvent(
                celune=cast("Celune", self._celune),
                task_id=task.task_id,
                session_id=task.session_id,
                request=request,
            ),
        )
        return task

    def respond_to_choice(
        self,
        task_id: str,
        response: AgentChoiceResponse,
    ) -> AgentTask:
        """Resume or fail a task after validating its pending choice response."""
        task = self.get_task(task_id)
        pending = self._pending_choices.get(task_id)
        if task.state != AgentTaskState.AWAITING_CHOICE or pending is None:
            raise ValueError("agent task is not awaiting a choice")
        if pending.request_id != response.request_id:
            raise ValueError("agent choice response does not match the pending request")
        valid_choice = response.choice_id is not None and any(
            option.choice_id == response.choice_id for option in pending.options
        )
        valid_freeform = response.freeform is not None and pending.allow_freeform
        self._pending_choices.pop(task_id)
        if not valid_choice and not valid_freeform:
            self._log(
                f"[AGENT] choice_response task={task.task_id} decision=invalid",
                loglevel="verbose",
            )
            return self.fail_task(task_id, AgentFailureReason.CHOICE_UNAVAILABLE)
        self._log(
            f"[AGENT] choice_response task={task.task_id} decision=accepted",
            loglevel="verbose",
        )
        self._transition(task, AgentTaskState.WORKING)
        return task

    def complete_task(
        self,
        task_id: str,
        metadata: Optional[JSON] = None,
    ) -> AgentTask:
        """Complete a working task and emit one terminal lifecycle event."""
        task = self.get_task(task_id)
        old_state = task.state
        self._log(
            f"[AGENT] complete task={task.task_id} state={old_state.value}",
            loglevel="verbose",
        )
        task.complete(metadata)
        self._after_transition(task, old_state)
        self._clear_task_state(task.task_id)
        self._emit_terminal(task)
        return task

    def fail_task(
        self,
        task_id: str,
        reason: AgentFailureReason,
        detail: Optional[str] = None,
    ) -> AgentTask:
        """Fail an active task and preserve its typed reason and detail."""
        task = self.get_task(task_id)
        old_state = task.state
        self._log(
            f"[AGENT] fail task={task.task_id} state={old_state.value} "
            f"reason={reason.value}",
            loglevel="verbose",
        )
        if detail:
            self._log(
                f"[AGENT] failure_detail task={task.task_id} detail={detail}",
                loglevel="debug",
            )
        task.fail(reason, detail)
        self._after_transition(task, old_state)
        self._clear_task_state(task.task_id)
        self._emit_terminal(task)
        return task

    def abort_task(
        self,
        task_id: str,
        reason: AgentAbortReason,
    ) -> AgentTask:
        """Abort an active task and preserve its typed abort reason."""
        task = self.get_task(task_id)
        old_state = task.state
        self._log(
            f"[AGENT] abort task={task.task_id} state={old_state.value} "
            f"reason={reason.value}",
            loglevel="verbose",
        )
        task.abort(reason)
        self._after_transition(task, old_state)
        self._clear_task_state(task.task_id)
        self._emit_terminal(task)
        return task

    def pause(self, session_id: str) -> None:
        """Pause the current task for a session while preserving its origin state."""
        task = self._require_session_task(session_id)
        if task.is_terminal or task.state == AgentTaskState.CANCELLING:
            raise ValueError("cannot pause a terminal or cancelling agent task")
        self._invalidate_generation(task)
        self._suspension_origins[task.task_id] = task.state
        old_state = task.state
        task.pause()
        self._after_transition(task, old_state)

    def resume(self, session_id: str) -> None:
        """Resume a paused or interrupted task at its prior lifecycle boundary."""
        task = self._require_session_task(session_id)
        if task.state not in {AgentTaskState.PAUSED, AgentTaskState.INTERRUPTED}:
            raise ValueError("only paused or interrupted agent tasks can resume")
        self._invalidate_generation(task)
        old_state = task.state
        origin = self._suspension_origins.pop(task.task_id, AgentTaskState.WORKING)
        target = origin
        if target in {
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
        } and not (
            task.task_id in self._pending_approvals
            or task.task_id in self._pending_choices
        ):
            target = AgentTaskState.PLANNING
        if target not in {
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.PLANNING,
            AgentTaskState.WORKING,
            AgentTaskState.CLASSIFYING,
        }:
            target = AgentTaskState.WORKING
        task.transition(target)
        task.interruption = None
        self._after_transition(task, old_state)

    def interrupt_task(
        self,
        task_id: str,
        interruption: AgentInterruption,
    ) -> AgentTask:
        """Interrupt or steer a task while keeping it continuable."""
        task = self.get_task(task_id)
        if task.is_terminal or task.state == AgentTaskState.CANCELLING:
            raise ValueError("cannot interrupt a terminal or cancelling agent task")
        self._invalidate_generation(task)
        self._suspension_origins[task.task_id] = task.state
        old_state = task.state
        self._invalidate_pending_interaction(task_id)
        task.interrupt(interruption)
        self._append_interruption_history(task, interruption)
        self._after_transition(task, old_state)
        return task

    def steer_task(
        self,
        task_id: str,
        interruption: AgentInterruption,
    ) -> AgentTask:
        """Apply steering to an existing task and resume at a planning boundary."""
        if interruption.kind != AgentInterruptionKind.USER_STEERING:
            raise ValueError("agent steering requires a user steering interruption")
        task = self.get_task(task_id)
        if task.state == AgentTaskState.IDLE:
            self.start_task(task_id)
            self.classify_task(task_id)
        self.interrupt_task(task_id, interruption)
        old_state = task.state
        task.transition(AgentTaskState.PLANNING)
        self._after_transition(task, old_state)
        return task

    def cancel_task(
        self,
        task_id: str,
        reason: AgentCancellationReason = AgentCancellationReason.USER_REQUEST,
    ) -> AgentTask:
        """Cancel any active or waiting task without leaving cancelling state behind."""
        task = self.get_task(task_id)
        if task.is_terminal:
            raise ValueError("cannot cancel a terminal agent task")
        self._log(
            f"[AGENT] cancel task={task.task_id} state={task.state.value} "
            f"reason={reason.value}",
            loglevel="verbose",
        )
        self._invalidate_generation(task)
        if task.state != AgentTaskState.CANCELLING:
            self._transition(task, AgentTaskState.CANCELLING)
        cancellation_error: Optional[Exception] = None
        try:
            task.cancel(reason)
        except Exception as exc:
            cancellation_error = exc
        if task.state == AgentTaskState.CANCELLING:
            old_state = task.state
            task.transition(AgentTaskState.CANCELLED)
            task.cancellation_reason = reason
            self._after_transition(task, old_state)
        else:
            self._after_transition(task, AgentTaskState.CANCELLING)
        self._clear_task_state(task_id)
        self._emit_terminal(task)
        if cancellation_error is not None:
            raise cancellation_error
        return task

    def pause_task(self, task_id: str) -> AgentTask:
        """Pause a task by its task identifier."""
        task = self.get_task(task_id)
        self.pause(task.session_id)
        return task

    def transition_task(
        self,
        task_id: str,
        state: AgentTaskState,
    ) -> AgentTask:
        """Apply one non-terminal validated transition for future lifecycle phases."""
        if state in {
            AgentTaskState.COMPLETED,
            AgentTaskState.FAILED,
            AgentTaskState.CANCELLED,
            AgentTaskState.ABORTED,
        }:
            raise ValueError(
                "use the typed terminal lifecycle method for terminal states"
            )
        task = self.get_task(task_id)
        self._transition(task, state)
        return task

    def run(
        self,
        request: AgentRequest,
        callback: Optional[AgentResponseCallback] = None,
    ) -> AgentOutput:
        """Run one bounded agent task through planning and typed tool execution.

        An existing non-terminal task for ``request.session.session_id`` is always
        reused.  When no task exists, this compatibility entry point creates one;
        routed Phase 3 requests therefore never create a second task.
        """
        task = self._session_task(request.session.session_id)
        if task is None:
            task = self.create_task(request)
        if task.is_terminal:
            return self._terminal_output(task, callback)

        self._log(
            f"[AGENT] run_start task={task.task_id} state={task.state.value} "
            f"generation={task.generation}",
            loglevel="verbose",
        )

        generation = task.generation
        lease: Optional[ComponentLockLease] = None
        if self._component_locks is not None:
            acquisition, lease = self._component_locks.try_acquire_lease(
                (ComponentLockRequirement(ComponentLockName.AGENT),),
                ComponentLockOwner(
                    operation_id=f"agent:{task.task_id}:{generation}",
                    task_id=task.task_id,
                    session_id=task.session_id,
                    generation_id=generation,
                ),
            )
            if not acquisition.acquired:
                busy = acquisition.busy
                assert busy is not None
                self._last_busy = busy
                self._log(
                    f"[AGENT] busy task={task.task_id} "
                    f"components={','.join(component.value for component in busy.components)}",
                    loglevel="verbose",
                )
                return _busy_output(busy)
            self._last_busy = None
        failure_reason = AgentFailureReason.MODEL_ERROR
        try:
            last_output: Optional[AgentOutput] = None
            while not task.is_terminal:
                if not self._run_is_current(task, generation):
                    return self._interrupted_output(task, callback)
                if task.state in {
                    AgentTaskState.AWAITING_APPROVAL,
                    AgentTaskState.AWAITING_CHOICE,
                    AgentTaskState.PAUSED,
                    AgentTaskState.INTERRUPTED,
                }:
                    return _paused_output()

                if task.state == AgentTaskState.IDLE:
                    self.start_task(task.task_id)
                if task.state == AgentTaskState.CLASSIFYING:
                    self.classify_task(task.task_id)
                if task.state == AgentTaskState.WORKING:
                    self._transition(task, AgentTaskState.PLANNING)
                if task.state != AgentTaskState.PLANNING:
                    raise ValueError(
                        "agent task cannot enter planning from its current state"
                    )

                if not self._compact_if_needed(task):
                    return self._interrupted_output(task, callback)
                if task.is_terminal:
                    return self._terminal_output(task, callback)

                pending_call = self._pending_tool_calls.pop(task.task_id, None)
                call = pending_call
                if call is None:
                    if task.iterations >= task.config.max_iterations:
                        self._log(
                            f"[AGENT] max_iterations task={task.task_id} "
                            f"iterations={task.iterations}",
                            loglevel="verbose",
                        )
                        self.abort_task(task.task_id, AgentAbortReason.MAX_ITERATIONS)
                        return self._terminal_output(task, callback)
                    self._log(
                        f"[AGENT] plan_start task={task.task_id} "
                        f"iteration={task.iterations + 1}",
                        loglevel="verbose",
                    )
                    output = self._validate_output(self._invoke_plan(task))
                    self._log_output(task, "plan", output)
                    if not self._run_is_current(task, generation):
                        return self._interrupted_output(task, callback)
                    if not self._publish_output(task, output, callback, generation):
                        return self._interrupted_output(task, callback)
                    if task.is_terminal:
                        return self._terminal_output(task, callback)
                    if task.state in {
                        AgentTaskState.AWAITING_APPROVAL,
                        AgentTaskState.AWAITING_CHOICE,
                        AgentTaskState.PAUSED,
                        AgentTaskState.INTERRUPTED,
                    }:
                        return _paused_output()
                    if output["paused"]:
                        self._transition(task, AgentTaskState.PAUSED)
                        return output

                    failure_reason = AgentFailureReason.INVALID_TOOL_CALL
                    selected = (
                        None
                        if output["tool_call"] is None and output["end"]
                        else self._select_or_fail_for_missing_tools(task, output)
                    )
                    if not self._run_is_current(task, generation):
                        return self._interrupted_output(task, callback)
                    call = self._validate_tool_call(cast(JSONSerializable, selected))
                    if call is not None:
                        self._log(
                            f"[AGENT] tool_selected task={task.task_id} "
                            f"tool={call['name']} call={call['id']}",
                            loglevel="verbose",
                        )
                    if task.state in {
                        AgentTaskState.AWAITING_APPROVAL,
                        AgentTaskState.AWAITING_CHOICE,
                        AgentTaskState.PAUSED,
                        AgentTaskState.INTERRUPTED,
                    }:
                        if call is not None:
                            self._pending_tool_calls[task.task_id] = call
                        return _paused_output()
                    if output["tool_call"] is not None and call is None:
                        raise ValueError(
                            "agent selector did not return the planned tool call"
                        )
                    if call is None:
                        if output["end"]:
                            return self._complete_with_output(task, output, callback)
                        self.fail_task(
                            task.task_id,
                            AgentFailureReason.NO_AVAILABLE_TOOLS,
                            "no registered available tool matched the action intent",
                        )
                        return self._terminal_output(task, callback)
                    self._record_action(task, call)
                    if task.is_terminal:
                        return self._terminal_output(task, callback)
                    if task.state in {
                        AgentTaskState.AWAITING_APPROVAL,
                        AgentTaskState.AWAITING_CHOICE,
                        AgentTaskState.PAUSED,
                        AgentTaskState.INTERRUPTED,
                    }:
                        self._pending_tool_calls[task.task_id] = call
                        return _paused_output()

                    failure_reason = AgentFailureReason.INTERNAL_ERROR
                    call = self._authorize_tool(task, call)
                    if call is None:
                        if task.is_terminal:
                            return self._terminal_output(task, callback)
                        return _paused_output()

                failure_reason = AgentFailureReason.TOOL_ERROR
                self._transition(task, AgentTaskState.EXECUTING_TOOL)
                self._log(
                    f"[AGENT] tool_execute task={task.task_id} "
                    f"tool={call['name']} call={call['id']}",
                    loglevel="verbose",
                )
                raw_result = self._invoke_execute(task, call)
                if not self._run_is_current(task, generation):
                    self._record_stale_tool_result(task, raw_result)
                    return self._interrupted_output(task, callback)
                result = self._normalize_tool_result(
                    call,
                    raw_result,
                    task.permission_decision,
                )
                self._log(
                    f"[AGENT] tool_result task={task.task_id} tool={call['name']} "
                    f"status={result['status'].value}",
                    loglevel="verbose",
                )
                if task.is_terminal:
                    return self._terminal_output(task, callback)
                self._transition(task, AgentTaskState.PLANNING)
                self._contexts[task.task_id] = replace(
                    self.get_context(task.task_id),
                    last_tool_result=result,
                )
                if result["status"] != AgentToolExecutionStatus.SUCCEEDED:
                    self.fail_task(
                        task.task_id,
                        AgentFailureReason.TOOL_ERROR,
                        result.get("error") or "agent tool execution failed",
                    )
                    return self._terminal_output(task, callback)
                if result.get("end_task", False):
                    return self._complete_with_output(
                        task,
                        _empty_output(),
                        callback,
                    )
                handled = self._validate_output(
                    self._invoke_handle_result(task, result)
                )
                self._log_output(task, "tool_result_handled", handled)
                if not self._run_is_current(task, generation):
                    return self._interrupted_output(task, callback)
                if not self._publish_output(task, handled, callback, generation):
                    return self._interrupted_output(task, callback)
                if task.is_terminal:
                    return self._terminal_output(task, callback)
                if handled["paused"]:
                    if task.state == AgentTaskState.PLANNING:
                        self._transition(task, AgentTaskState.PAUSED)
                    return handled
                if handled["end"]:
                    return self._complete_with_output(task, handled, callback)
                if not self._consume_iteration(task):
                    return self._terminal_output(task, callback)
                last_output = handled

            return self._terminal_output(task, callback, last_output)
        except Exception as exc:
            if not self._run_is_current(task, generation):
                return self._interrupted_output(task, callback)
            if task.is_terminal:
                return self._terminal_output(task, callback)
            self._log(
                f"[AGENT] run_error task={task.task_id} "
                f"reason={failure_reason.value} error={exc}",
                loglevel="debug",
            )
            self.fail_task(task.task_id, failure_reason, str(exc))
            return self._terminal_output(task, callback)
        finally:
            if lease is not None:
                lease.release()
            self._log(
                f"[AGENT] run_end task={task.task_id} state={task.state.value} "
                f"iterations={task.iterations}",
                loglevel="verbose",
            )

    def _authorize_tool(
        self,
        task: AgentTask,
        call: ToolCall,
    ) -> Optional[ToolCall]:
        """Evaluate one selected call before it crosses the executor boundary."""
        validated, schema = self._validated_permission_call(call)
        if schema is not None and not schema.available:
            evaluation = AgentPermissionEvaluation(
                task_id=task.task_id,
                tool_call_id=validated["id"],
                tool_id=validated["tool_id"],
                decision=AgentPermissionDecision.DENY,
                reason=AgentPermissionReason.TOOL_UNAVAILABLE,
            )
        else:
            evaluation = self._permission_policy(task, validated)
            self._validate_permission_evaluation(evaluation, task, validated)
        task.permission_decision = evaluation
        if evaluation.decision == AgentPermissionDecision.DENY:
            self._log(
                f"[AGENT] permission task={task.task_id} tool={validated['name']} "
                f"decision={evaluation.decision.value} reason={evaluation.reason.value}",
                loglevel="verbose",
            )
            self.fail_task(
                task.task_id,
                AgentFailureReason.PERMISSION_DENIED,
                evaluation.reason.value,
            )
            return None
        if evaluation.decision == AgentPermissionDecision.REQUIRE_APPROVAL:
            if not isinstance(validated, dict):
                raise ValueError("agent permission policy returned an invalid call")
            request = AgentApprovalRequest(
                request_id=f"approval-{uuid4().hex}",
                task_id=task.task_id,
                tool_call=validated,
                prompt=string("agent.approval_prompt", tool_name=validated["name"]),
                permission=evaluation,
            )
            self._log(
                f"[AGENT] permission task={task.task_id} tool={validated['name']} "
                f"decision={evaluation.decision.value} reason={evaluation.reason.value}",
                loglevel="verbose",
            )
            self.request_approval(task.task_id, request)
            return None
        self._log(
            f"[AGENT] permission task={task.task_id} tool={validated['name']} "
            f"decision={evaluation.decision.value}",
            loglevel="verbose",
        )
        return validated if schema is not None or "tool_id" in call else call

    def _validated_permission_call(
        self,
        call: ToolCall,
    ) -> tuple[ValidatedToolCall, Optional[AgentToolSchema]]:
        """Resolve schema metadata without allowing a selector to override it."""
        schema = self._tool_schemas.get(call["name"])
        if schema is None and "tool_id" in call:
            schema = self._tool_schemas.get(cast(ValidatedToolCall, call)["tool_id"])
        if schema is not None:
            if (
                "tool_id" in call
                and cast(ValidatedToolCall, call)["tool_id"] != schema.tool_id
            ):
                raise ValueError("agent tool call metadata does not match its schema")
            return (
                {
                    "id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "tool_id": schema.tool_id,
                    "behavior": schema.behavior,
                    "danger": schema.danger,
                    "approval_required": schema.approval_required,
                },
                schema,
            )
        if "tool_id" in call:
            return cast(ValidatedToolCall, call), None
        return (
            {
                "id": call["id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "tool_id": call["name"],
                "behavior": AgentToolBehavior.READ_ONLY,
                "danger": AgentToolDangerLevel.LOW,
                "approval_required": False,
            },
            None,
        )

    @staticmethod
    def _validate_permission_evaluation(
        evaluation: AgentPermissionEvaluation,
        task: AgentTask,
        call: ValidatedToolCall,
    ) -> None:
        """Reject policy results that do not describe the evaluated task and call."""
        if not isinstance(evaluation, AgentPermissionEvaluation):
            raise TypeError("agent permission policy returned an invalid evaluation")
        if (
            evaluation.task_id != task.task_id
            or evaluation.tool_call_id != call["id"]
            or evaluation.tool_id != call["tool_id"]
        ):
            raise ValueError("agent permission policy returned mismatched metadata")

    @staticmethod
    def _permission_metadata(
        evaluation: Optional[AgentPermissionEvaluation],
    ) -> Optional[dict[str, JSONSerializable]]:
        """Return JSON-compatible permission metadata for a typed tool result."""
        if evaluation is None:
            return None
        return cast(dict[str, JSONSerializable], evaluation.to_metadata())

    def _invoke_plan(self, task: AgentTask) -> AgentOutput:
        """Invoke the injected planner or the overridable runtime method."""
        context = self.get_context(task.task_id)
        return (
            self._planner(context) if self._planner is not None else self.plan(context)
        )

    def _invoke_select_tool(
        self,
        task: AgentTask,
        output: AgentOutput,
    ) -> Optional[ToolCall]:
        """Invoke the injected selector or the overridable runtime method."""
        context = self.get_context(task.task_id)
        if self._tool_selector is not None:
            return self._tool_selector(context, output)
        return self.select_tool(context, output)

    def _select_or_fail_for_missing_tools(
        self,
        task: AgentTask,
        output: AgentOutput,
    ) -> Optional[ToolCall]:
        """Select a tool or preserve a typed catalog failure on the task."""
        if output.get("tool_call") is None and not self.tools:
            self.fail_task(
                task.task_id,
                AgentFailureReason.NO_TOOLS_FOUND,
                "the agent tool catalog is empty",
            )
            return None

        if output.get("tool_call") is None:
            available_tools = [
                tool
                for tool in self.tools
                if self._tool_schemas.get(tool.name, None) is None
                or self._tool_schemas[tool.name].available
            ]
            if not available_tools:
                self.fail_task(
                    task.task_id,
                    AgentFailureReason.NO_AVAILABLE_TOOLS,
                    "the registered agent tools are unavailable",
                )
                return None
        return self._invoke_select_tool(task, output)

    def _invoke_execute(self, task: AgentTask, call: ToolCall) -> ToolResult:
        """Invoke the injected executor or the overridable runtime method."""
        context = self.get_context(task.task_id)
        if self._tool_executor is not None:
            return self._tool_executor(context, call)
        return self.execute_tool(context, call)

    def _invoke_handle_result(
        self,
        task: AgentTask,
        result: ToolResult,
    ) -> AgentOutput:
        """Invoke the injected result handler or the overridable runtime method."""
        context = self.get_context(task.task_id)
        if self._tool_result_handler is not None:
            return self._tool_result_handler(context, result)
        return self.handle_tool_result(context, result)

    def _invoke_respond(self, task: AgentTask) -> AgentOutput:
        """Invoke the injected responder or the overridable runtime method."""
        context = self.get_context(task.task_id)
        return (
            self._responder(context)
            if self._responder is not None
            else self.respond(context)
        )

    def _log_output(
        self,
        task: AgentTask,
        source: str,
        output: AgentOutput,
    ) -> None:
        """Log the shape of an agent output without logging prompt content."""
        call = output["tool_call"]
        tool_name = call["name"] if call is not None else "none"
        response_length = len(output["response"]) if output["response"] else 0
        self._log(
            f"[AGENT] {source}_result task={task.task_id} tool={tool_name} "
            f"response_length={response_length} end={output['end']} "
            f"paused={output['paused']}",
            loglevel="verbose",
        )

    def _compact_if_needed(self, task: AgentTask) -> bool:
        """Run the injected compactor when the typed threshold is reached."""
        if not task.needs_context_compaction:
            return True
        if self._compactor is None:
            self.abort_task(task.task_id, AgentAbortReason.CONTEXT_LIMIT)
            return False
        try:
            compacted = self._compactor(self.get_context(task.task_id))
        except Exception as exc:
            self.fail_task(task.task_id, AgentFailureReason.INTERNAL_ERROR, str(exc))
            return False
        if not isinstance(compacted, AgentContext) or compacted.task is not task:
            self.fail_task(
                task.task_id,
                AgentFailureReason.INTERNAL_ERROR,
                "agent compactor returned an invalid context",
            )
            return False
        self._contexts[task.task_id] = compacted
        if task.needs_context_compaction:
            self.abort_task(task.task_id, AgentAbortReason.CONTEXT_LIMIT)
            return False
        return True

    def _consume_iteration(self, task: AgentTask) -> bool:
        """Account one completed decision cycle and publish limit termination."""
        old_state = task.state
        consumed = task.consume_iteration()
        if consumed:
            return True
        self._after_direct_abort(task, old_state)
        return False

    def _record_action(self, task: AgentTask, call: ToolCall) -> None:
        """Detect repeated identical calls before execution begins."""
        previous = self._last_tool_calls.get(task.task_id)
        progressed = previous is None or previous != call
        self._last_tool_calls[task.task_id] = call
        old_state = task.state
        if task.record_progress(progressed):
            return
        self._after_direct_abort(task, old_state)

    def _publish_output(
        self,
        task: AgentTask,
        output: AgentOutput,
        callback: Optional[AgentResponseCallback],
        generation: int,
    ) -> bool:
        """Account generated response tokens and notify the caller safely."""
        if not self._run_is_current(task, generation):
            return False
        response = output["response"]
        if response is not None:
            count = self._token_counter(response)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "agent token counter must return a non-negative integer"
                )
            old_state = task.state
            if not task.add_generated_tokens(count):
                self._after_direct_abort(task, old_state)
                return False
        if not self._run_is_current(task, generation):
            return False
        if callback is not None:
            try:
                callback(output)
            except Exception as exc:
                self.fail_task(
                    task.task_id, AgentFailureReason.INTERNAL_ERROR, str(exc)
                )
                return False
        return not task.is_terminal

    def _interrupted_output(
        self,
        task: AgentTask,
        callback: Optional[AgentResponseCallback],
    ) -> AgentOutput:
        """Return a pause for stale work or the proper output for a terminal task."""
        if task.is_terminal:
            return self._terminal_output(task, callback)
        return _paused_output()

    @staticmethod
    def _validate_output(value: AgentOutput) -> AgentOutput:
        """Validate and normalize one planner, handler, or responder output."""
        if not isinstance(value, dict):
            raise TypeError("agent dependency returned a non-object output")
        response = value.get("response")
        end = value.get("end")
        paused = value.get("paused")
        busy = value.get("busy")
        terminal = value.get("terminal")
        if response is not None and not isinstance(response, str):
            raise TypeError("agent output response must be text or null")
        if not isinstance(end, bool) or not isinstance(paused, bool):
            raise TypeError("agent output end and paused fields must be boolean")
        if busy is not None and not isinstance(busy, ComponentBusyResult):
            raise TypeError(
                "agent output busy metadata must be typed component metadata"
            )
        if busy is not None and not paused:
            raise ValueError("agent busy output must be paused")
        if terminal is not None and not isinstance(terminal, AgentTerminalOutcome):
            raise TypeError("agent terminal metadata must be typed terminal metadata")
        if terminal is not None and (
            not end
            or response is not None
            or paused
            or value.get("tool_call") is not None
        ):
            raise ValueError("agent terminal outputs must be empty and final")
        call = AgentRuntime._validate_tool_call(
            cast(JSONSerializable, value.get("tool_call"))
        )
        if call is not None and (end or paused):
            raise ValueError("agent tool outputs cannot also end or pause")
        if call is None and response is None and not end and not paused:
            raise ValueError(
                "agent output must contain a response, tool call, pause, or end"
            )
        normalized: AgentOutput = {
            "tool_call": call,
            "response": response,
            "end": end,
            "paused": paused,
        }
        if busy is not None:
            normalized["busy"] = busy
        if terminal is not None:
            normalized["terminal"] = terminal
        return normalized

    @staticmethod
    def _validate_tool_call(value: JSONSerializable) -> Optional[ToolCall]:
        """Validate one selector result without permitting multiple calls."""
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("agent selector returned a non-object tool call")
        call_id = value.get("id")
        name = value.get("name")
        arguments = value.get("arguments")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or not isinstance(name, str)
            or not name.strip()
            or not isinstance(arguments, dict)
            or not all(
                isinstance(key, str) and _is_json_value(argument)
                for key, argument in arguments.items()
            )
        ):
            raise ValueError("agent selector returned an invalid tool call")
        call: ToolCall = {
            "id": call_id,
            "name": name,
            "arguments": cast(dict[str, JSONSerializable], arguments),
        }
        metadata_names = {
            "tool_id",
            "behavior",
            "danger",
            "approval_required",
        }
        if not any(name in value for name in metadata_names):
            return call
        if not all(name in value for name in metadata_names):
            raise ValueError("agent selector returned incomplete tool metadata")
        tool_id = value.get("tool_id")
        behavior = value.get("behavior")
        danger = value.get("danger")
        approval_required = value.get("approval_required")
        if (
            not isinstance(tool_id, str)
            or not tool_id.strip()
            or not isinstance(behavior, str)
            or not isinstance(danger, str)
            or not isinstance(approval_required, bool)
        ):
            raise ValueError("agent selector returned invalid tool metadata")
        try:
            typed_call = cast(
                ValidatedToolCall,
                {
                    **call,
                    "tool_id": tool_id,
                    "behavior": AgentToolBehavior(behavior),
                    "danger": AgentToolDangerLevel(danger),
                    "approval_required": approval_required,
                },
            )
        except ValueError as exc:
            raise ValueError("agent selector returned invalid tool metadata") from exc
        return typed_call

    @staticmethod
    def _normalize_tool_result(
        call: ToolCall,
        value: ToolResult,
        permission: Optional[AgentPermissionEvaluation] = None,
    ) -> ToolExecutionResult:
        """Validate one executor result and fill its typed execution status."""
        if not isinstance(value, dict):
            raise TypeError("agent executor returned a non-object tool result")
        call_id = value.get("tool_call_id")
        output = value.get("output")
        error = value.get("error")
        end_task = value.get("end_task", False)
        if (
            not isinstance(call_id, str)
            or call_id != call["id"]
            or (output is not None and not _is_json_value(output))
            or (error is not None and not isinstance(error, str))
            or not isinstance(end_task, bool)
        ):
            raise ValueError("agent executor returned an invalid tool result")
        raw_status = value.get("status")
        raw_tool_id = value.get("tool_id")
        if raw_status is None:
            status = (
                AgentToolExecutionStatus.FAILED
                if error is not None
                else AgentToolExecutionStatus.SUCCEEDED
            )
            tool_id = call["name"]
        else:
            if not isinstance(raw_status, str):
                raise ValueError("agent tool result status must be text")
            try:
                status = AgentToolExecutionStatus(raw_status)
            except ValueError as exc:
                raise ValueError("agent tool result status is invalid") from exc
            if not isinstance(raw_tool_id, str) or not raw_tool_id.strip():
                raise ValueError("typed agent tool results require a tool_id")
            tool_id = raw_tool_id
        return {
            "tool_call_id": call_id,
            "output": output,
            "error": error,
            "tool_id": tool_id,
            "status": status,
            "end_task": end_task,
            **(
                {"permission": permission.to_metadata()}
                if permission is not None
                else {}
            ),
        }

    def _complete_with_output(
        self,
        task: AgentTask,
        output: AgentOutput,
        callback: Optional[AgentResponseCallback],
    ) -> AgentOutput:
        """Complete after an already-published final output."""
        if task.state == AgentTaskState.PLANNING:
            self._transition(task, AgentTaskState.RESPONDING)
        if task.state != AgentTaskState.RESPONDING:
            raise ValueError("agent task cannot complete from its current state")
        if not self._consume_iteration(task):
            return self._terminal_output(task, callback)
        metadata: JSON = {
            "iterations": task.iterations,
            "generated_tokens": task.generated_tokens,
        }
        result = cast(
            Optional[ToolExecutionResult],
            self.get_context(task.task_id).last_tool_result,
        )
        if isinstance(result, dict):
            result_metadata = cast(dict[str, JSONSerializable], dict(result))
            status = result_metadata.get("status")
            if isinstance(status, AgentToolExecutionStatus):
                result_metadata["status"] = status.value
            metadata["tool_result"] = result_metadata
        self.complete_task(
            task.task_id,
            metadata,
        )
        return output

    def _terminal_output(
        self,
        task: AgentTask,
        callback: Optional[AgentResponseCallback],
        output: Optional[AgentOutput] = None,
    ) -> AgentOutput:
        """Return and optionally publish one typed terminal outcome for a task."""
        if output is not None:
            return output
        terminal_outcome = AgentTerminalOutcome(
            state=task.state,
            failure_reason=task.failure_reason,
            abort_reason=task.abort_reason,
            cancellation_reason=task.cancellation_reason,
            detail=task.failure_detail,
            metadata=task.completion_metadata,
        )
        terminal = _empty_output(terminal=terminal_outcome)
        self._safe_callback(callback, terminal)
        return terminal

    @staticmethod
    def _safe_callback(
        callback: Optional[AgentResponseCallback],
        output: AgentOutput,
    ) -> None:
        """Deliver a terminal output without masking lifecycle cleanup."""
        if callback is None:
            return
        try:
            callback(output)
        except Exception:
            return

    def _after_direct_abort(self, task: AgentTask, old_state: AgentTaskState) -> None:
        """Publish lifecycle events after a task accounting method aborts directly."""
        self._after_transition(task, old_state)
        self._clear_task_state(task.task_id)
        self._emit_terminal(task)

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

    def cancel(
        self,
        session_id: str,
        reason: AgentCancellationReason = AgentCancellationReason.USER_REQUEST,
    ) -> None:
        """Cancel the current task for a session."""
        self.cancel_task(self._require_session_task(session_id).task_id, reason)

    def _session_task(self, session_id: str) -> Optional[AgentTask]:
        """Return the current task for a session, if one is registered."""
        session = self._sessions.get(session_id)
        if session is None or session.task_id is None:
            return None
        return self._tasks.get(session.task_id)

    def _require_session_task(self, session_id: str) -> AgentTask:
        """Return a session task or reject an unknown session."""
        task = self._session_task(session_id)
        if task is None:
            raise ValueError(f"unknown agent session '{session_id}'")
        return task

    @staticmethod
    def _validate_request_task(task: AgentTask, request_task_id: str) -> None:
        """Reject approval or choice requests belonging to another task."""
        if task.task_id != request_task_id:
            raise ValueError("agent request belongs to another task")

    def _transition(self, task: AgentTask, state: AgentTaskState) -> None:
        """Apply a Phase 1 transition and publish its typed state event."""
        old_state = task.state
        self._log(
            f"[AGENT] transition task={task.task_id} "
            f"old={old_state.value} new={state.value}",
            loglevel="debug",
        )
        task.transition(state)
        if old_state != state:
            self._after_transition(task, old_state)

    def _after_transition(self, task: AgentTask, old_state: AgentTaskState) -> None:
        """Synchronize session compatibility flags and emit one state change."""
        self._sync_session(task)
        self._emit(
            "agent_task_state_changed",
            AgentTaskStateChangedEvent(
                celune=cast("Celune", self._celune),
                task_id=task.task_id,
                session_id=task.session_id,
                old_state=old_state,
                new_state=task.state,
            ),
        )

    def _sync_session(self, task: AgentTask) -> None:
        """Keep explicit session state and legacy booleans aligned to task state."""
        state = task.state
        if state == AgentTaskState.IDLE:
            session_state = AgentSessionState.IDLE
            paused = False
            cancelled = False
        elif state in {
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
        }:
            session_state = AgentSessionState.PAUSED
            paused = True
            cancelled = False
        elif state == AgentTaskState.COMPLETED:
            session_state = AgentSessionState.COMPLETED
            paused = False
            cancelled = False
        elif state == AgentTaskState.FAILED:
            session_state = AgentSessionState.FAILED
            paused = False
            cancelled = False
        elif state == AgentTaskState.CANCELLED:
            session_state = AgentSessionState.CANCELLED
            paused = False
            cancelled = True
        elif state == AgentTaskState.ABORTED:
            session_state = AgentSessionState.ABORTED
            paused = False
            cancelled = False
        else:
            session_state = AgentSessionState.ACTIVE
            paused = False
            cancelled = False
        self._sessions[task.session_id] = replace(
            self._sessions[task.session_id],
            paused=paused,
            cancelled=cancelled,
            state=session_state,
            task_id=task.task_id,
        )

    def _emit_terminal(self, task: AgentTask) -> None:
        """Publish a terminal event once, preserving all available metadata."""
        if not task.is_terminal or task.task_id in self._terminal_events:
            return
        self._terminal_events.add(task.task_id)
        self._emit(
            "agent_task_finished",
            AgentTaskFinishedEvent(
                celune=cast("Celune", self._celune),
                task_id=task.task_id,
                session_id=task.session_id,
                state=task.state,
                abort_reason=task.abort_reason,
                failure_reason=task.failure_reason,
                cancellation_reason=task.cancellation_reason,
                completion_metadata=task.completion_metadata,
            ),
        )

    @staticmethod
    def _invalidate_generation(task: AgentTask) -> None:
        """Invalidate work already running against the task's prior generation."""
        task.generation += 1

    def _invalidate_pending_interaction(self, task_id: str) -> None:
        """Reject stale approval, choice, and selected-call responses."""
        self._pending_approvals.pop(task_id, None)
        self._pending_choices.pop(task_id, None)
        self._pending_tool_calls.pop(task_id, None)

    def _append_interruption_history(
        self,
        task: AgentTask,
        interruption: AgentInterruption,
    ) -> None:
        """Record interruption metadata in the task request and active context."""
        entry = cast(
            JSON,
            {
                "role": "user",
                "content": interruption.instruction or "",
                "agent_interruption": interruption.kind.value,
            },
        )
        request = task.request
        next_request = replace(
            request,
            request=(
                interruption.instruction
                if interruption.kind == AgentInterruptionKind.USER_STEERING
                and interruption.instruction is not None
                else request.request
            ),
            history=(*request.history, entry),
        )
        task.request = next_request
        self._contexts[task.task_id] = replace(
            self.get_context(task.task_id),
            request=next_request,
        )

    def _record_stale_tool_result(self, task: AgentTask, result: ToolResult) -> None:
        """Keep a late tool result as diagnostics without returning it to planning."""
        entry = cast(
            JSON,
            {
                "type": "stale_tool_result",
                "tool_result": cast(JSONSerializable, result),
            },
        )
        request = task.request
        next_request = replace(request, history=(*request.history, entry))
        task.request = next_request
        self._contexts[task.task_id] = replace(
            self.get_context(task.task_id),
            request=next_request,
        )

    @staticmethod
    def _run_is_current(task: AgentTask, generation: int) -> bool:
        """Return whether a worker may still publish work for this task generation."""
        return (
            task.generation == generation
            and not task.is_terminal
            and task.state
            not in {AgentTaskState.INTERRUPTED, AgentTaskState.CANCELLING}
        )

    def _clear_task_state(self, task_id: str) -> None:
        """Drop pending user responses and suspension bookkeeping for a terminal task."""
        self._pending_approvals.pop(task_id, None)
        self._pending_choices.pop(task_id, None)
        self._suspension_origins.pop(task_id, None)
        self._pending_tool_calls.pop(task_id, None)
        self._last_tool_calls.pop(task_id, None)

    def _emit(self, event_name: EventName, event: EventPayload) -> None:
        """Forward an event through the existing dispatcher when configured."""
        if self._event_dispatcher is None:
            return
        self._log(
            f"[AGENT] emit name={event_name} payload={type(event).__name__}",
            loglevel="debug",
        )
        try:
            self._event_dispatcher.emit(event_name, event)
        except Exception as exc:
            self._log(
                f"[AGENT] emit_error name={event_name} error={exc}",
                loglevel="debug",
            )
            return
        self._log(
            f"[AGENT] emit_return name={event_name}",
            loglevel="debug",
        )
