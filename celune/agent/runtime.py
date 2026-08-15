# SPDX-License-Identifier: MIT
"""Typed lifecycle ownership for Celune's future local-only agent runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, cast
from uuid import uuid4

from ..dataclasses.events import (
    AgentApprovalRequestedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskFinishedEvent,
    AgentTaskStateChangedEvent,
)
from ..extensions.events import EventDispatcher
from ..persona.capabilities import PersonaCapabilities
from ..typing.agent import (
    AgentAbortReason,
    AgentApprovalDecision,
    AgentApprovalRequest,
    AgentApprovalResponse,
    AgentCancellationReason,
    AgentChoiceRequest,
    AgentChoiceResponse,
    AgentContext,
    AgentFailureReason,
    AgentInterruption,
    AgentOutput,
    AgentRequest,
    AgentResponseCallback,
    AgentSession,
    AgentSessionState,
    AgentTask,
    AgentTaskConfig,
    AgentTaskState,
    AgentTool,
    ToolCall,
    ToolResult,
)
from ..typing.common import JSON
from ..typing.events import EventName, EventPayload
from ..typing.modes import OperationMode

if TYPE_CHECKING:
    from ..celune import Celune


class AgentRuntime:
    """Own agent task lifecycle state without running the future model loop."""

    def __init__(
        self,
        tools: Sequence[AgentTool] = (),
        *,
        event_dispatcher: Optional[EventDispatcher] = None,
        celune: Optional[Celune] = None,
        mode: OperationMode = "agent",
        persona_capabilities: Optional[PersonaCapabilities] = None,
    ) -> None:
        """Create a lifecycle owner around local tools and an existing event bus."""
        self.tools = tuple(tools)
        self._event_dispatcher = event_dispatcher
        self._celune = celune
        self._mode = mode
        self._persona_capabilities = persona_capabilities or PersonaCapabilities()
        self._tasks: dict[str, AgentTask] = {}
        self._contexts: dict[str, AgentContext] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._pending_approvals: dict[str, AgentApprovalRequest] = {}
        self._pending_choices: dict[str, AgentChoiceRequest] = {}
        self._suspension_origins: dict[str, AgentTaskState] = {}
        self._terminal_events: set[str] = set()

    def create_context(
        self,
        request: AgentRequest,
        task: Optional[AgentTask] = None,
    ) -> AgentContext:
        """Build a context that keeps the request and optional task together."""
        return AgentContext(
            request=request,
            mode=self._mode,
            persona_capabilities=self._persona_capabilities,
            task=task,
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
        self._transition(task, AgentTaskState.CLASSIFYING)
        return task

    def classify_task(self, task_id: str) -> AgentTask:
        """Complete the lifecycle classification boundary without heuristics."""
        task = self.get_task(task_id)
        if task.state != AgentTaskState.CLASSIFYING:
            raise ValueError("only classifying agent tasks can begin work")
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
        }:
            raise ValueError(
                "agent approval can only pause classifying or working tasks"
            )
        self._pending_approvals[task_id] = request
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
        if response.decision == AgentApprovalDecision.DENIED:
            return self.fail_task(task_id, AgentFailureReason.APPROVAL_DENIED)
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
        }:
            raise ValueError("agent choice can only pause classifying or working tasks")
        self._pending_choices[task_id] = request
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
            return self.fail_task(task_id, AgentFailureReason.CHOICE_UNAVAILABLE)
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
        self._suspension_origins[task.task_id] = task.state
        old_state = task.state
        task.pause()
        self._after_transition(task, old_state)

    def resume(self, session_id: str) -> None:
        """Resume a paused or interrupted task at its prior lifecycle boundary."""
        task = self._require_session_task(session_id)
        if task.state not in {AgentTaskState.PAUSED, AgentTaskState.INTERRUPTED}:
            raise ValueError("only paused or interrupted agent tasks can resume")
        old_state = task.state
        origin = self._suspension_origins.pop(task.task_id, AgentTaskState.WORKING)
        target = (
            origin
            if origin
            in {
                AgentTaskState.AWAITING_APPROVAL,
                AgentTaskState.AWAITING_CHOICE,
            }
            else AgentTaskState.WORKING
        )
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
        self._suspension_origins[task.task_id] = task.state
        old_state = task.state
        task.interrupt(interruption)
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
        """Run the future agent loop and emit steps through an optional callback."""
        raise NotImplementedError("agent execution is not implemented")

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

    def _clear_task_state(self, task_id: str) -> None:
        """Drop pending user responses and suspension bookkeeping for a terminal task."""
        self._pending_approvals.pop(task_id, None)
        self._pending_choices.pop(task_id, None)
        self._suspension_origins.pop(task_id, None)

    def _emit(self, event_name: EventName, event: EventPayload) -> None:
        """Forward an event through the existing dispatcher when configured."""
        if self._event_dispatcher is None:
            return
        try:
            self._event_dispatcher.emit(event_name, event)
        except Exception:
            return
