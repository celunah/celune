# SPDX-License-Identifier: MIT
"""Focused tests for Celune's additive agent contracts."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, cast

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


class AgentContractTests(unittest.TestCase):
    """Verify construction and lifecycle rules without invoking the runtime."""

    def test_configuration_and_tool_schema_validate_and_serialize(self) -> None:
        """Validate task limits and complete typed tool schemas."""
        config = AgentTaskConfig(
            max_iterations=4,
            max_generated_tokens=128,
            context_compaction_threshold=256,
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

        self.assertEqual(config.to_json()["max_iterations"], 4)
        self.assertEqual(schema.arguments[0].to_json()["type"], "string")
        json.dumps(schema.to_json())

        default_config = AgentTaskConfig()
        self.assertEqual(default_config.max_iterations, 20)
        self.assertIsNone(default_config.max_generated_tokens)
        self.assertEqual(default_config.context_space, 32768)
        self.assertEqual(default_config.context_compaction_threshold, 24576)
        self.assertIsNone(default_config.to_json()["max_generated_tokens"])

        with self.assertRaises(ValueError):
            AgentTaskConfig(max_iterations=0)
        with self.assertRaises(ValueError):
            AgentTaskConfig(max_generated_tokens=True)
        with self.assertRaises(ValueError):
            AgentTaskConfig(max_generated_tokens=0)
        with self.assertRaises(ValueError):
            AgentTask(
                task_id="task-1",
                session_id="session-1",
                request=AgentRequest(request="test request"),
                state=AgentTaskState.ABORTED,
            )
        with self.assertRaises(ValueError):
            AgentToolArgumentSchema(
                name="items",
                value_type=AgentToolValueType.ARRAY,
            )
        with self.assertRaises(ValueError):
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

        self.assertEqual(approval.tool_call["tool_id"], "search")
        self.assertEqual(choice.options[0].choice_id, "soft")
        approval_response = AgentApprovalResponse(
            "approval-1", AgentApprovalDecision.APPROVED
        )
        choice_response = AgentChoiceResponse("choice-1", choice_id="soft")
        AgentChoiceResponse("choice-1", freeform="Use the default voice")
        steering_response = AgentChoiceResponse("choice-1", freeform="Steer the task")
        self.assertEqual(approval.to_json()["prompt"], "Allow this read-only tool?")
        self.assertEqual(approval_response.to_json()["decision"], "approved")
        self.assertEqual(choice_response.to_json()["choice_id"], "soft")
        self.assertEqual(steering_response.to_json()["freeform"], "Steer the task")
        json.dumps(choice.to_json())
        execution_result: ToolExecutionResult = {
            "tool_call_id": "call-1",
            "output": "indexed result",
            "error": None,
            "tool_id": "search",
            "status": AgentToolExecutionStatus.SUCCEEDED,
        }
        self.assertEqual(execution_result["status"], AgentToolExecutionStatus.SUCCEEDED)
        terminal = AgentTerminalOutcome(
            state=AgentTaskState.ABORTED,
            abort_reason=AgentAbortReason.MAX_ITERATIONS,
            metadata={"iterations": 20},
        )
        self.assertEqual(terminal.to_json()["abort_reason"], "max_iterations")
        json.dumps(terminal.to_json())

        with self.assertRaises(ValueError):
            AgentRequest(request=" ")
        with self.assertRaises(ValueError):
            AgentInterruption(AgentInterruptionKind.USER_STEERING)
        with self.assertRaises(ValueError):
            AgentChoiceResponse("choice-1")
        with self.assertRaises(ValueError):
            AgentChoiceResponse("choice-1", choice_id="soft", freeform="both")
        with self.assertRaises(ValueError):
            AgentApprovalResponse(" ", AgentApprovalDecision.DENIED)
        with self.assertRaises(ValueError):
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
        self.assertEqual(task.iterations, 0)
        task.transition(AgentTaskState.PLANNING)
        task.transition(AgentTaskState.AWAITING_CHOICE)
        task.transition(AgentTaskState.PLANNING)
        task.transition(AgentTaskState.EXECUTING_TOOL)
        task.transition(AgentTaskState.RESPONDING)
        task.complete()
        self.assertEqual(task.state, AgentTaskState.COMPLETED)

        with self.assertRaises(ValueError):
            _task().transition(AgentTaskState.RESPONDING)
        with self.assertRaises(ValueError):
            task.transition(AgentTaskState.PLANNING)
        with self.assertRaises(ValueError):
            task.fail(AgentFailureReason.INTERNAL_ERROR)

        failed_task = _task()
        failed_task.transition(AgentTaskState.PLANNING)
        failed_task.fail(AgentFailureReason.MODEL_ERROR, "model unavailable")
        self.assertEqual(failed_task.state, AgentTaskState.FAILED)
        self.assertEqual(failed_task.failure_reason, AgentFailureReason.MODEL_ERROR)

    def test_approval_pause_does_not_consume_iteration_and_interruption_resumes(
        self,
    ) -> None:
        """Keep iteration accounting stable across approval and steering pauses."""
        task = _task()
        task.transition(AgentTaskState.PLANNING)
        self.assertTrue(task.consume_iteration())
        task.transition(AgentTaskState.AWAITING_APPROVAL)
        task.transition(AgentTaskState.PLANNING)
        self.assertEqual(task.iterations, 1)
        task.interrupt(
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING,
                instruction="Use the shorter answer.",
            )
        )
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)
        task.resume()
        self.assertEqual(task.state, AgentTaskState.PLANNING)
        self.assertIsNone(task.interruption)

    def test_iteration_token_context_and_stuck_limits(self) -> None:
        """Abort tasks when iteration, token, or stuck thresholds are reached."""
        task = _task(
            AgentTaskConfig(
                max_iterations=2,
                max_generated_tokens=3,
                context_compaction_threshold=5,
                stuck_task_threshold=2,
            )
        )
        task.transition(AgentTaskState.PLANNING)
        self.assertTrue(task.consume_iteration())
        self.assertTrue(task.consume_iteration())
        self.assertFalse(task.consume_iteration())
        self.assertEqual(task.abort_reason, AgentAbortReason.MAX_ITERATIONS)

        token_task = _task(AgentTaskConfig(max_generated_tokens=3))
        token_task.transition(AgentTaskState.PLANNING)
        self.assertTrue(token_task.add_generated_tokens(3))
        self.assertFalse(token_task.add_generated_tokens(1))

        unlimited_task = _task(AgentTaskConfig())
        self.assertTrue(unlimited_task.add_generated_tokens(100_000))
        self.assertEqual(token_task.abort_reason, AgentAbortReason.MAX_GENERATED_TOKENS)

        context_task = _task(AgentTaskConfig(context_compaction_threshold=5))
        context_task.update_context_tokens(5)
        self.assertTrue(context_task.needs_context_compaction)

        stuck_task = _task(AgentTaskConfig(stuck_task_threshold=2))
        stuck_task.transition(AgentTaskState.PLANNING)
        self.assertTrue(stuck_task.record_progress(False))
        self.assertFalse(stuck_task.record_progress(False))
        self.assertEqual(stuck_task.abort_reason, AgentAbortReason.STUCK_TASK)

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
            with self.subTest(state=state):
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
                self.assertEqual(task.state, AgentTaskState.CANCELLED)

        task = _task()
        task.cancel(AgentCancellationReason.RUNTIME_SHUTDOWN)
        self.assertEqual(task.to_json()["cancellation_reason"], "runtime_shutdown")

        session = AgentSession(
            session_id="session-1",
            state=AgentSessionState.ACTIVE,
            task_id="task-1",
        )
        self.assertEqual(session.to_json()["state"], "active")

    def test_lifecycle_events_use_existing_dispatcher(self) -> None:
        """Deliver typed lifecycle payloads through the existing event dispatcher."""
        self.assertIn("agent_task_finished", EVENT_NAMES)
        self.assertIn("agent_approval_requested", EVENT_NAMES)

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
        self.assertEqual(
            received, ["responding", "approval-1", "choice-1", "cancelled"]
        )


if __name__ == "__main__":
    unittest.main()
