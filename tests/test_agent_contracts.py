# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Celune's additive agent contracts."""

from __future__ import annotations

from contextlib import nullcontext
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, cast

import pytest

from celune.extensions.events import EVENT_NAMES, EventDispatcher
from celune.dataclasses.events import (
    AgentTaskFinishedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)
from celune.agent.contracts import (
    AgentTask,
    AgentRequest,
    AgentSession,
    AgentTaskState,
    AgentTaskConfig,
    AgentToolSchema,
    AgentAbortReason,
    AgentChoiceOption,
    AgentInterruption,
    AgentSessionState,
    AgentToolBehavior,
    ValidatedToolCall,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentToolValueType,
    AgentChoiceResponse,
    ToolExecutionResult,
    AgentApprovalRequest,
    AgentTerminalOutcome,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentInterruptionKind,
    AgentCancellationReason,
    AgentToolArgumentSchema,
    AgentToolExecutionStatus,
)

if TYPE_CHECKING:
    from celune.celune import Celune


def _task(config: Optional[AgentTaskConfig] = None) -> AgentTask:
    """Build a valid task fixture for contract tests."""
    return AgentTask(
        task_id="task-1",
        session_id="session-1",
        request=AgentRequest(request="test request"),
        config=config or AgentTaskConfig(),
    )


def _validated_tool_call() -> ValidatedToolCall:
    """Build a schema-validated tool call fixture."""
    return {
        "id": "call-1",
        "name": "search",
        "arguments": {"query": "Celune"},
        "tool_id": "search",
        "behavior": AgentToolBehavior.READ_ONLY,
        "danger": AgentToolDangerLevel.LOW,
        "approval_required": False,
    }


class TestAgentContracts:
    """Verify construction and lifecycle rules without invoking the runtime."""

    def test_configuration_and_tool_schema_validate_and_serialize(self) -> None:
        """Validate task limits and complete typed tool schemas."""
        config = AgentTaskConfig(
            max_loops=4,
            max_tokens=128,
            context_size=256,
            compact_at=100,
            stuck_task_threshold=2,
        )
        schema = AgentToolSchema(
            tool_id="search",
            display_name="Search",
            description="Read local indexed content.",
            arguments=(
                AgentToolArgumentSchema(
                    name="query",
                    value_type=AgentToolValueType.STRING,
                ),
                AgentToolArgumentSchema(
                    name="limit",
                    value_type=AgentToolValueType.INTEGER,
                    required=False,
                ),
            ),
            behavior=AgentToolBehavior.READ_ONLY,
            danger=AgentToolDangerLevel.LOW,
            available=True,
        )

        assert config.to_json()["max_loops"] == 4
        assert schema.arguments[0].to_json()["type"] == "string"
        json.dumps(schema.to_json())

        default_config = AgentTaskConfig()
        assert default_config.max_loops == 20
        assert default_config.max_tokens is None
        assert default_config.context_size == 32768
        assert default_config.compact_at == 75
        assert default_config.context_compaction_threshold == 24576
        assert default_config.to_json()["max_tokens"] is None

        with pytest.raises(ValueError):
            AgentTaskConfig(max_loops=0)
        with pytest.raises(ValueError):
            AgentTaskConfig(max_tokens=True)
        with pytest.raises(ValueError):
            AgentTaskConfig(max_tokens=0)
        with pytest.raises(ValueError):
            AgentTask(
                task_id="task-1",
                session_id="session-1",
                request=AgentRequest(request="test request"),
                state=AgentTaskState.ABORTED,
            )
        with pytest.raises(ValueError):
            AgentToolArgumentSchema(
                name="items",
                value_type=AgentToolValueType.ARRAY,
            )
        with pytest.raises(ValueError):
            AgentToolSchema(
                tool_id="duplicate",
                display_name="Duplicate",
                description="Invalid schema",
                arguments=(
                    AgentToolArgumentSchema("value", AgentToolValueType.STRING),
                    AgentToolArgumentSchema("value", AgentToolValueType.STRING),
                ),
            )

    def test_requests_and_responses_validate(self) -> None:
        """Validate approval, choice, interruption, and execution DTOs."""
        approval = AgentApprovalRequest(
            request_id="approval-1",
            task_id="task-1",
            tool_call=_validated_tool_call(),
            prompt="Allow this read-only tool?",
        )
        choice = AgentChoiceRequest(
            request_id="choice-1",
            task_id="task-1",
            prompt="Choose a voice",
            options=(AgentChoiceOption("soft", "Soft"),),
        )

        assert approval.tool_call["tool_id"] == "search"
        assert choice.options[0].choice_id == "soft"
        approval_response = AgentApprovalResponse(
            "approval-1", AgentApprovalDecision.APPROVED
        )
        choice_response = AgentChoiceResponse("choice-1", choice_id="soft")
        AgentChoiceResponse("choice-1", freeform="Use the default voice")
        steering_response = AgentChoiceResponse("choice-1", freeform="Steer the task")
        assert approval.to_json()["prompt"] == "Allow this read-only tool?"
        assert approval_response.to_json()["decision"] == "approved"
        assert choice_response.to_json()["choice_id"] == "soft"
        assert steering_response.to_json()["freeform"] == "Steer the task"
        json.dumps(choice.to_json())
        execution_result: ToolExecutionResult = {
            "tool_call_id": "call-1",
            "output": "indexed result",
            "error": None,
            "tool_id": "search",
            "status": AgentToolExecutionStatus.SUCCEEDED,
        }
        assert execution_result["status"] == AgentToolExecutionStatus.SUCCEEDED
        terminal = AgentTerminalOutcome(
            state=AgentTaskState.ABORTED,
            abort_reason=AgentAbortReason.MAX_LOOPS,
            metadata={"iterations": 20},
        )
        assert terminal.to_json()["abort_reason"] == "max_loops"
        json.dumps(terminal.to_json())

        with pytest.raises(ValueError):
            AgentRequest(request=" ")
        with pytest.raises(ValueError):
            AgentInterruption(AgentInterruptionKind.USER_STEERING)
        with pytest.raises(ValueError):
            AgentChoiceResponse("choice-1")
        with pytest.raises(ValueError):
            AgentChoiceResponse("choice-1", choice_id="soft", freeform="both")
        with pytest.raises(ValueError):
            AgentApprovalResponse(" ", AgentApprovalDecision.DENIED)
        with pytest.raises(ValueError):
            AgentChoiceRequest(
                request_id="choice-1",
                task_id="task-1",
                prompt="Choose",
                options=(),
            )

    def test_valid_and_invalid_state_transitions(self) -> None:
        """Accept supported task transitions and reject terminal rewinds."""
        task = _task()
        task.transition(AgentTaskState.PLANNING)
        task.transition(AgentTaskState.AWAITING_APPROVAL)
        assert task.iterations == 0
        task.transition(AgentTaskState.PLANNING)
        task.transition(AgentTaskState.AWAITING_CHOICE)
        task.transition(AgentTaskState.PLANNING)
        task.transition(AgentTaskState.EXECUTING_TOOL)
        task.transition(AgentTaskState.RESPONDING)
        task.complete()
        assert task.state == AgentTaskState.COMPLETED

        with pytest.raises(ValueError):
            _task().transition(AgentTaskState.RESPONDING)
        with pytest.raises(ValueError):
            task.transition(AgentTaskState.PLANNING)
        with pytest.raises(ValueError):
            task.fail(AgentFailureReason.INTERNAL_ERROR)

        failed_task = _task()
        failed_task.transition(AgentTaskState.PLANNING)
        failed_task.fail(AgentFailureReason.MODEL_ERROR, "model unavailable")
        assert failed_task.state == AgentTaskState.FAILED
        assert failed_task.failure_reason == AgentFailureReason.MODEL_ERROR

    def test_approval_pause_does_not_consume_iteration_and_interruption_resumes(
        self,
    ) -> None:
        """Keep iteration accounting stable across approval and steering pauses."""
        task = _task()
        task.transition(AgentTaskState.PLANNING)
        assert task.consume_iteration()
        task.transition(AgentTaskState.AWAITING_APPROVAL)
        task.transition(AgentTaskState.PLANNING)
        assert task.iterations == 1
        task.interrupt(
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING,
                instruction="Use the shorter answer.",
            )
        )
        assert task.state == AgentTaskState.INTERRUPTED
        task.resume()
        assert task.state == AgentTaskState.PLANNING
        assert task.interruption is None

    def test_iteration_token_context_and_stuck_limits(self) -> None:
        """Abort tasks when iteration, token, or stuck thresholds are reached."""
        task = _task(
            AgentTaskConfig(
                max_loops=2,
                max_tokens=3,
                context_size=5,
                compact_at=100,
                stuck_task_threshold=2,
            )
        )
        task.transition(AgentTaskState.PLANNING)
        assert task.consume_iteration()
        assert task.consume_iteration()
        assert not task.consume_iteration()
        assert task.abort_reason == AgentAbortReason.MAX_LOOPS

        token_task = _task(AgentTaskConfig(max_tokens=3))
        token_task.transition(AgentTaskState.PLANNING)
        assert token_task.add_generated_tokens(3)
        assert not token_task.add_generated_tokens(1)

        unlimited_task = _task(AgentTaskConfig())
        assert unlimited_task.add_generated_tokens(100_000)
        assert token_task.abort_reason == AgentAbortReason.MAX_TOKENS

        context_task = _task(AgentTaskConfig(context_size=5, compact_at=100))
        context_task.update_context_tokens(5)
        assert context_task.needs_context_compaction

        stuck_task = _task(AgentTaskConfig(stuck_task_threshold=2))
        stuck_task.transition(AgentTaskState.PLANNING)
        assert stuck_task.record_progress(False)
        assert not stuck_task.record_progress(False)
        assert stuck_task.abort_reason == AgentAbortReason.STUCK_TASK

    def test_cancellation_is_valid_from_active_states_and_serializes(self) -> None:
        """Allow typed cancellation from each active task state."""
        for state in (
            AgentTaskState.QUEUED,
            AgentTaskState.PLANNING,
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.EXECUTING_TOOL,
            AgentTaskState.RESPONDING,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
        ):
            with nullcontext():
                task = _task()
                if state != AgentTaskState.QUEUED:
                    task.transition(AgentTaskState.PLANNING)
                if state in {
                    AgentTaskState.AWAITING_APPROVAL,
                    AgentTaskState.AWAITING_CHOICE,
                    AgentTaskState.EXECUTING_TOOL,
                    AgentTaskState.RESPONDING,
                }:
                    task.transition(state)
                elif state == AgentTaskState.PAUSED:
                    task.pause()
                elif state == AgentTaskState.INTERRUPTED:
                    task.interrupt(
                        AgentInterruption(AgentInterruptionKind.USER_INTERRUPT)
                    )
                task.cancel(AgentCancellationReason.USER_REQUEST)
                assert task.state == AgentTaskState.CANCELLED

        task = _task()
        task.cancel(AgentCancellationReason.RUNTIME_SHUTDOWN)
        assert task.to_json()["cancellation_reason"] == "runtime_shutdown"

        session = AgentSession(
            session_id="session-1",
            state=AgentSessionState.ACTIVE,
            task_id="task-1",
        )
        assert session.to_json()["state"] == "active"

    def test_lifecycle_events_use_existing_dispatcher(self) -> None:
        """Deliver typed lifecycle payloads through the existing event dispatcher."""
        assert "agent_task_finished" in EVENT_NAMES
        assert "agent_approval_requested" in EVENT_NAMES

        celune = cast("Celune", SimpleNamespace())
        state_event = AgentTaskStateChangedEvent(
            celune,
            "task-1",
            "session-1",
            AgentTaskState.PLANNING,
            AgentTaskState.RESPONDING,
        )
        approval_event = AgentApprovalRequestedEvent(
            celune,
            "task-1",
            "session-1",
            AgentApprovalRequest(
                "approval-1", "task-1", _validated_tool_call(), "Allow?"
            ),
        )
        choice_event = AgentChoiceRequestedEvent(
            celune,
            "task-1",
            "session-1",
            AgentChoiceRequest(
                "choice-1", "task-1", "Choose", (AgentChoiceOption("one", "One"),)
            ),
        )
        finished_event = AgentTaskFinishedEvent(
            celune,
            "task-1",
            "session-1",
            AgentTaskState.CANCELLED,
            cancellation_reason=AgentCancellationReason.USER_REQUEST,
        )
        received: list[str] = []
        dispatcher = EventDispatcher(log_warning=lambda _message, _level: None)
        dispatcher.subscribe(
            "agent_task_state_changed",
            lambda event: received.append(event.new_state.value),
        )
        dispatcher.subscribe(
            "agent_approval_requested",
            lambda event: received.append(event.request.request_id),
        )
        dispatcher.subscribe(
            "agent_choice_requested",
            lambda event: received.append(event.request.request_id),
        )
        dispatcher.subscribe(
            "agent_task_finished",
            lambda event: received.append(event.state.value),
        )
        dispatcher.emit("agent_task_state_changed", state_event)
        dispatcher.emit("agent_approval_requested", approval_event)
        dispatcher.emit("agent_choice_requested", choice_event)
        dispatcher.emit("agent_task_finished", finished_event)
        assert received == ["responding", "approval-1", "choice-1", "cancelled"]
