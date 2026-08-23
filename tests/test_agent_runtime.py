# SPDX-License-Identifier: Apache-2.0
"""Focused lifecycle tests for Celune's Phase 2 agent runtime."""

from __future__ import annotations

from contextlib import nullcontext
import threading
from types import SimpleNamespace
from unittest import mock
from typing import TYPE_CHECKING, cast

import pytest

from celune.extensions.events import EventDispatcher
from celune.dataclasses.events import (
    AgentTaskFinishedEvent,
    AgentChoiceRequestedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)
from celune.agent import (
    ToolCall,
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentRequest,
    AgentRuntime,
    AgentSession,
    AgentTaskState,
    AgentAbortReason,
    AgentChoiceOption,
    AgentInterruption,
    AgentSessionState,
    AgentToolBehavior,
    ValidatedToolCall,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentChoiceResponse,
    AgentApprovalRequest,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentInterruptionKind,
    AgentCancellationReason,
)

if TYPE_CHECKING:
    from celune.celune import Celune


def _validated_call() -> ValidatedToolCall:
    """Build one valid approval fixture without executing a tool."""
    return {
        "id": "call-1",
        "name": "read_status",
        "arguments": {},
        "tool_id": "read_status",
        "behavior": AgentToolBehavior.READ_ONLY,
        "danger": AgentToolDangerLevel.LOW,
        "approval_required": True,
    }


class TestAgentRuntimeLifecycle:  # pylint: disable=attribute-defined-outside-init
    """Verify lifecycle ownership without invoking a model or local tool."""

    def setup_method(self) -> None:
        """Create a fresh runtime and event recorder for each test."""
        self.event_names: list[str] = []
        self.state_events: list[AgentTaskStateChangedEvent] = []
        self.finished_events: list[AgentTaskFinishedEvent] = []
        self.approval_events: list[AgentApprovalRequestedEvent] = []
        self.choice_events: list[AgentChoiceRequestedEvent] = []
        self.agent_log = mock.Mock()
        self.dispatcher = EventDispatcher(
            log_warning=lambda _message, _severity: None,
        )
        self.dispatcher.subscribe("agent_task_state_changed", self._record_state_event)
        self.dispatcher.subscribe(
            "agent_approval_requested", self._record_approval_event
        )
        self.dispatcher.subscribe("agent_choice_requested", self._record_choice_event)
        self.dispatcher.subscribe("agent_task_finished", self._record_finished_event)
        self.runtime = AgentRuntime(
            event_dispatcher=self.dispatcher,
            celune=cast("Celune", SimpleNamespace(log=self.agent_log)),
        )

    def _record_state_event(self, event: AgentTaskStateChangedEvent) -> None:
        self.event_names.append("state")
        self.state_events.append(event)

    def _record_approval_event(self, event: AgentApprovalRequestedEvent) -> None:
        self.event_names.append("approval")
        self.approval_events.append(event)

    def _record_choice_event(self, event: AgentChoiceRequestedEvent) -> None:
        self.event_names.append("choice")
        self.choice_events.append(event)

    def _record_finished_event(self, event: AgentTaskFinishedEvent) -> None:
        self.event_names.append("finished")
        self.finished_events.append(event)

    def _request(self, session_id: str = "session-1") -> AgentRequest:
        """Build an explicit action request for one test session."""
        return AgentRequest(
            request="Perform the requested action.",
            session=AgentSession(session_id=session_id),
        )

    def _working_task(self):
        """Create and advance one task to the working boundary."""
        task = self.runtime.create_task(self._request(), task_id="task-1")
        self.runtime.start_task(task.task_id)
        self.runtime.classify_task(task.task_id)
        return task

    def test_task_creation_preserves_explicit_request_and_context(self) -> None:
        """Create an idle task and bind its context to the same task instance."""
        request = self._request()
        task = self.runtime.create_task(request, task_id="task-1")

        assert task.state == AgentTaskState.IDLE
        assert self.runtime.get_context(task.task_id).task is task
        assert self.runtime.get_context(task.task_id).request is request
        session = self.runtime.get_session(request.session.session_id)
        assert session.state == AgentSessionState.IDLE
        assert not session.paused
        assert not session.cancelled
        assert session.task_id == task.task_id
        assert self.state_events[0].old_state == AgentTaskState.QUEUED
        assert self.state_events[0].new_state == AgentTaskState.IDLE

    def test_agent_diagnostics_use_the_core_log_gate(self) -> None:
        """Forward agent lifecycle diagnostics to Celune's configured logger."""
        task = self.runtime.create_task(self._request(), task_id="task-1")
        self.runtime.start_task(task.task_id)

        messages = [call.args[0] for call in self.agent_log.call_args_list]
        assert any(message.startswith("[AGENT] task_created") for message in messages)
        transition_call = next(
            call
            for call in self.agent_log.call_args_list
            if call.args[0].startswith("[AGENT] transition")
        )
        assert transition_call.kwargs["loglevel"] == "debug"

    def test_valid_lifecycle_transitions_and_terminal_metadata(self) -> None:
        """Cover classification, work, pauses, interruption, and completion."""
        task = self.runtime.create_task(self._request(), task_id="task-1")
        self.runtime.start_task(task.task_id)
        self.runtime.classify_task(task.task_id)
        self.runtime.pause(task.session_id)
        self.runtime.resume(task.session_id)
        self.runtime.interrupt_task(
            task.task_id,
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING, "Use concise output."
            ),
        )
        self.runtime.resume(task.session_id)
        self.runtime.complete_task(task.task_id, {"source": "explicit_action"})

        assert task.state == AgentTaskState.COMPLETED
        assert task.completion_metadata == {"source": "explicit_action"}
        session = self.runtime.get_session(task.session_id)
        assert session.state == AgentSessionState.COMPLETED
        assert not session.paused
        assert not session.cancelled
        assert len(self.finished_events) == 1
        assert self.event_names == ["state"] * 8 + ["finished"]
        assert [(event.old_state, event.new_state) for event in self.state_events] == [
            (AgentTaskState.QUEUED, AgentTaskState.IDLE),
            (AgentTaskState.IDLE, AgentTaskState.CLASSIFYING),
            (AgentTaskState.CLASSIFYING, AgentTaskState.WORKING),
            (AgentTaskState.WORKING, AgentTaskState.PAUSED),
            (AgentTaskState.PAUSED, AgentTaskState.WORKING),
            (AgentTaskState.WORKING, AgentTaskState.INTERRUPTED),
            (AgentTaskState.INTERRUPTED, AgentTaskState.WORKING),
            (AgentTaskState.WORKING, AgentTaskState.COMPLETED),
        ]
        assert self.finished_events[0].completion_metadata == {
            "source": "explicit_action"
        }

    def test_approval_and_choice_pauses_preserve_context_and_iterations(self) -> None:
        """Keep context and iteration accounting stable across user responses."""
        task = self._working_task()
        context = self.runtime.get_context(task.task_id)
        approval = AgentApprovalRequest(
            request_id="approval-1",
            task_id=task.task_id,
            tool_call=_validated_call(),
            prompt="Allow the read-only status check?",
        )
        self.runtime.request_approval(task.task_id, approval)
        assert task.state == AgentTaskState.AWAITING_APPROVAL
        assert self.runtime.get_context(task.task_id) is context
        assert task.iterations == 0
        self.runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse("approval-1", AgentApprovalDecision.APPROVED),
        )

        choice = AgentChoiceRequest(
            request_id="choice-1",
            task_id=task.task_id,
            prompt="Choose the output style.",
            options=(AgentChoiceOption("brief", "Brief"),),
        )
        self.runtime.request_choice(task.task_id, choice)
        assert task.state == AgentTaskState.AWAITING_CHOICE
        assert self.runtime.get_context(task.task_id) is context
        self.runtime.respond_to_choice(
            task.task_id,
            AgentChoiceResponse("choice-1", choice_id="brief"),
        )
        assert task.state == AgentTaskState.WORKING
        assert task.iterations == 0
        assert len(self.approval_events) == 1
        assert len(self.choice_events) == 1
        assert self.event_names[-6:] == [
            "state",
            "approval",
            "state",
            "state",
            "choice",
            "state",
        ]

    def test_steering_resumes_at_planning_and_invalidates_waiting_requests(
        self,
    ) -> None:
        """Steering preserves task progress and rejects stale approval responses."""
        task = self._working_task()
        task.iterations = 1
        approval = AgentApprovalRequest(
            request_id="approval-1",
            task_id=task.task_id,
            tool_call=_validated_call(),
            prompt="Allow?",
        )
        self.runtime.request_approval(task.task_id, approval)
        generation = task.generation

        self.runtime.steer_task(
            task.task_id,
            AgentInterruption(
                AgentInterruptionKind.USER_STEERING,
                "Use the safer read-only path.",
            ),
        )

        assert task.state == AgentTaskState.PLANNING
        assert task.generation > generation
        assert task.iterations == 1
        assert self.runtime.get_pending_approval(task.task_id) is None
        assert task.request.request == "Use the safer read-only path."
        assert "Use the safer read-only path." in [
            entry.get("content") for entry in task.request.history
        ]
        with pytest.raises(ValueError):
            self.runtime.respond_to_approval(
                task.task_id,
                AgentApprovalResponse("approval-1", AgentApprovalDecision.APPROVED),
            )

    def test_interruption_clears_approval_and_choice_without_progress(self) -> None:
        """Interrupt waiting interactions without consuming work or accepting stale answers."""
        for waiting_state in (
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
        ):
            with nullcontext():
                runtime = AgentRuntime()
                task = runtime.create_task(
                    self._request(session_id=f"interrupt-{waiting_state.value}"),
                    task_id=f"interrupt-{waiting_state.value}",
                )
                runtime.start_task(task.task_id)
                runtime.classify_task(task.task_id)
                if waiting_state == AgentTaskState.AWAITING_APPROVAL:
                    runtime.request_approval(
                        task.task_id,
                        AgentApprovalRequest(
                            "approval-1", task.task_id, _validated_call(), "Allow?"
                        ),
                    )
                else:
                    runtime.request_choice(
                        task.task_id,
                        AgentChoiceRequest(
                            "choice-1",
                            task.task_id,
                            "Choose",
                            (AgentChoiceOption("one", "One"),),
                        ),
                    )

                runtime.interrupt_task(
                    task.task_id,
                    AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
                )

                assert task.state == AgentTaskState.INTERRUPTED
                assert task.iterations == 0
                assert runtime.get_pending_approval(task.task_id) is None
                assert runtime.get_pending_choice(task.task_id) is None
                session = runtime.get_session(task.session_id)
                assert session.state == AgentSessionState.PAUSED
                assert session.paused

    def test_stale_planner_output_after_interruption_is_discarded(self) -> None:
        """An interrupted planner cannot publish its old response."""
        started = threading.Event()
        release = threading.Event()
        outputs: list[AgentOutput] = []

        def planner(_context: AgentContext) -> AgentOutput:
            """Block one planner response until the test interrupts it."""
            started.set()
            release.wait(timeout=2)
            return {
                "tool_call": None,
                "response": "stale planner response",
                "end": True,
                "paused": False,
            }

        def record_output(output: AgentOutput) -> None:
            """Record only outputs accepted by the active task generation."""
            outputs.append(output)

        runtime = AgentRuntime(planner=planner)
        task = runtime.create_task(self._request(), task_id="stale-planner")
        worker = threading.Thread(
            target=lambda: runtime.run(task.request, record_output),
            daemon=True,
        )
        worker.start()
        assert started.wait(timeout=2)

        runtime.interrupt_task(
            task.task_id,
            AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
        )
        release.set()
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert task.state == AgentTaskState.INTERRUPTED
        assert not outputs
        assert len(runtime._terminal_events) == 0

    def test_stale_non_cooperative_tool_result_is_diagnostic_only(self) -> None:
        """A late tool result cannot advance an interrupted task iteration."""
        started = threading.Event()
        release = threading.Event()
        call: ToolCall = {
            "id": "call-1",
            "name": "read_status",
            "arguments": {},
        }

        def planner(_context: AgentContext) -> AgentOutput:
            """Return one read-only call for the non-cooperative tool fixture."""
            return {
                "tool_call": call,
                "response": "Reading status.",
                "end": False,
                "paused": False,
            }

        def execute(_context: AgentContext, _call: ToolCall) -> ToolResult:
            """Block the fixture tool until the task has been interrupted."""
            started.set()
            release.wait(timeout=2)
            return {
                "tool_call_id": "call-1",
                "output": {"status": "late"},
                "error": None,
            }

        runtime = AgentRuntime(
            planner=planner,
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=execute,
        )
        task = runtime.create_task(self._request(), task_id="stale-tool")
        worker = threading.Thread(target=lambda: runtime.run(task.request), daemon=True)
        worker.start()
        assert started.wait(timeout=2)

        runtime.interrupt_task(
            task.task_id,
            AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
        )
        release.set()
        worker.join(timeout=2)

        assert not worker.is_alive()
        assert task.state == AgentTaskState.INTERRUPTED
        assert runtime.get_context(task.task_id).last_tool_result is None
        assert any(
            entry.get("type") == "stale_tool_result" for entry in task.request.history
        )

    def test_invalid_transitions_and_response_ids_are_rejected(self) -> None:
        """Reject lifecycle calls that do not match the Phase 1 transition table."""
        task = self.runtime.create_task(self._request(), task_id="task-1")
        with pytest.raises(ValueError):
            self.runtime.classify_task(task.task_id)
        self.runtime.start_task(task.task_id)
        with pytest.raises(ValueError):
            self.runtime.start_task(task.task_id)
        with pytest.raises(ValueError):
            self.runtime.complete_task(task.task_id)

        with pytest.raises(ValueError):
            self.runtime.request_approval(
                task.task_id,
                AgentApprovalRequest(
                    "approval-1", "other-task", _validated_call(), "Allow?"
                ),
            )

    def test_cancellation_is_valid_from_each_active_and_waiting_state(self) -> None:
        """Cancel idle, active, paused, interrupted, approval, and choice states."""
        for state in (
            AgentTaskState.IDLE,
            AgentTaskState.CLASSIFYING,
            AgentTaskState.WORKING,
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
            AgentTaskState.PAUSED,
            AgentTaskState.INTERRUPTED,
        ):
            with nullcontext():
                runtime = AgentRuntime()
                task = runtime.create_task(
                    self._request(session_id=f"session-{state.value}"),
                    task_id=f"task-{state.value}",
                )
                if state in {
                    AgentTaskState.CLASSIFYING,
                    AgentTaskState.WORKING,
                }:
                    runtime.start_task(task.task_id)
                if state == AgentTaskState.WORKING:
                    runtime.classify_task(task.task_id)
                if state == AgentTaskState.AWAITING_APPROVAL:
                    runtime.start_task(task.task_id)
                    runtime.classify_task(task.task_id)
                    runtime.request_approval(
                        task.task_id,
                        AgentApprovalRequest(
                            "approval-1", task.task_id, _validated_call(), "Allow?"
                        ),
                    )
                if state == AgentTaskState.AWAITING_CHOICE:
                    runtime.start_task(task.task_id)
                    runtime.classify_task(task.task_id)
                    runtime.request_choice(
                        task.task_id,
                        AgentChoiceRequest(
                            "choice-1",
                            task.task_id,
                            "Choose",
                            (AgentChoiceOption("one", "One"),),
                        ),
                    )
                if state == AgentTaskState.PAUSED:
                    runtime.start_task(task.task_id)
                    runtime.classify_task(task.task_id)
                    runtime.pause(task.session_id)
                if state == AgentTaskState.INTERRUPTED:
                    runtime.start_task(task.task_id)
                    runtime.classify_task(task.task_id)
                    runtime.interrupt_task(
                        task.task_id,
                        AgentInterruption(AgentInterruptionKind.USER_INTERRUPT),
                    )

                runtime.cancel_task(task.task_id)
                assert task.state == AgentTaskState.CANCELLED

    def test_cancellation_during_approval_and_choice_clears_waiting_state(self) -> None:
        """Cancel approval and choice pauses without leaving pending requests."""
        for waiting_state in (
            AgentTaskState.AWAITING_APPROVAL,
            AgentTaskState.AWAITING_CHOICE,
        ):
            with nullcontext():
                runtime = AgentRuntime()
                task = runtime.create_task(
                    self._request(session_id=f"waiting-{waiting_state.value}"),
                    task_id=f"waiting-{waiting_state.value}",
                )
                runtime.start_task(task.task_id)
                runtime.classify_task(task.task_id)
                if waiting_state == AgentTaskState.AWAITING_APPROVAL:
                    runtime.request_approval(
                        task.task_id,
                        AgentApprovalRequest(
                            "approval-1", task.task_id, _validated_call(), "Allow?"
                        ),
                    )
                else:
                    runtime.request_choice(
                        task.task_id,
                        AgentChoiceRequest(
                            "choice-1",
                            task.task_id,
                            "Choose",
                            (AgentChoiceOption("one", "One"),),
                        ),
                    )
                runtime.cancel_task(task.task_id)
                assert task.state == AgentTaskState.CANCELLED
                with pytest.raises(ValueError):
                    if waiting_state == AgentTaskState.AWAITING_APPROVAL:
                        runtime.respond_to_approval(
                            task.task_id,
                            AgentApprovalResponse(
                                "approval-1", AgentApprovalDecision.APPROVED
                            ),
                        )
                    else:
                        runtime.respond_to_choice(
                            task.task_id,
                            AgentChoiceResponse("choice-1", choice_id="one"),
                        )

    def test_failure_abort_and_terminal_events_are_exactly_once(self) -> None:
        """Preserve failure and abort reasons and never duplicate terminal events."""
        failed = self._working_task()
        self.runtime.fail_task(
            failed.task_id, AgentFailureReason.MODEL_ERROR, "offline"
        )
        assert failed.failure_reason == AgentFailureReason.MODEL_ERROR
        assert failed.failure_detail == "offline"
        with pytest.raises(ValueError):
            self.runtime.fail_task(failed.task_id, AgentFailureReason.INTERNAL_ERROR)

        aborted = self.runtime.create_task(
            self._request("abort-session"), task_id="abort"
        )
        self.runtime.start_task(aborted.task_id)
        self.runtime.classify_task(aborted.task_id)
        self.runtime.abort_task(aborted.task_id, AgentAbortReason.STUCK_TASK)
        assert aborted.abort_reason == AgentAbortReason.STUCK_TASK
        assert aborted.state == AgentTaskState.ABORTED
        assert len(self.finished_events) == 2

    def test_cancellation_cleanup_finishes_after_lifecycle_exception(self) -> None:
        """Finalize cancellation even when the task cancellation handler raises."""
        task = self._working_task()
        with (
            mock.patch.object(
                task, "cancel", side_effect=RuntimeError("cancel failed")
            ),
            pytest.raises(RuntimeError),
        ):
            self.runtime.cancel_task(task.task_id)
        assert task.state == AgentTaskState.CANCELLED
        assert [
            (event.old_state, event.new_state) for event in self.state_events[-2:]
        ] == [
            (AgentTaskState.WORKING, AgentTaskState.CANCELLING),
            (AgentTaskState.CANCELLING, AgentTaskState.CANCELLED),
        ]
        assert (
            self.runtime.get_session(task.session_id).state
            == AgentSessionState.CANCELLED
        )
        assert self.event_names.count("finished") == 1

    def test_legacy_session_flags_follow_explicit_task_state(self) -> None:
        """Keep paused and cancelled compatibility fields synchronized."""
        task = self._working_task()
        self.runtime.pause(task.session_id)
        paused = self.runtime.get_session(task.session_id)
        assert paused.state == AgentSessionState.PAUSED
        assert paused.paused
        assert not paused.cancelled
        self.runtime.resume(task.session_id)
        assert (
            self.runtime.get_session(task.session_id).state == AgentSessionState.ACTIVE
        )
        self.runtime.cancel_task(
            task.task_id, AgentCancellationReason.SESSION_CANCELLED
        )
        cancelled = self.runtime.get_session(task.session_id)
        assert cancelled.state == AgentSessionState.CANCELLED
        assert not cancelled.paused
        assert cancelled.cancelled
