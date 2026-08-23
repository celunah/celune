# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the bounded Phase 4 agent execution loop."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, cast

from celune.extensions.events import EventDispatcher
from celune.agent import (
    ToolCall,
    ToolResult,
    AgentOutput,
    AgentRequest,
    AgentRuntime,
    AgentSession,
    AgentTaskState,
    AgentTaskConfig,
    AgentAbortReason,
    AgentChoiceOption,
    AgentToolBehavior,
    ValidatedToolCall,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentChoiceResponse,
    ToolExecutionResult,
    AgentApprovalRequest,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentToolExecutionStatus,
)


def _request(session_id: str = "session-1") -> AgentRequest:
    """Build one explicit action request for loop tests."""
    return AgentRequest(
        request="Perform the requested action.",
        session=AgentSession(session_id=session_id),
    )


def _output(
    *,
    response: Optional[str] = None,
    tool_call: Optional[ToolCall] = None,
    end: bool = False,
    paused: bool = False,
) -> AgentOutput:
    """Build one complete typed planner output."""
    return {
        "tool_call": tool_call,
        "response": response,
        "end": end,
        "paused": paused,
    }


def _call(call_id: str = "call-1") -> ToolCall:
    """Build one single-call action for a fake selector."""
    return {
        "id": call_id,
        "name": "read_status",
        "arguments": {"name": "process"},
    }


def _result(call: ToolCall, *, error: Optional[str] = None) -> ToolExecutionResult:
    """Build one typed executor result."""
    return {
        "tool_call_id": call["id"],
        "output": None if error is not None else {"running": True},
        "error": error,
        "tool_id": call["name"],
        "status": (
            AgentToolExecutionStatus.FAILED
            if error is not None
            else AgentToolExecutionStatus.SUCCEEDED
        ),
    }


def _validated_call(call: ToolCall) -> ValidatedToolCall:
    """Add the existing Phase 1 validation metadata to one call."""
    return {
        "id": call["id"],
        "name": call["name"],
        "arguments": call["arguments"],
        "tool_id": call["name"],
        "behavior": AgentToolBehavior.READ_ONLY,
        "danger": AgentToolDangerLevel.LOW,
        "approval_required": True,
    }


class TestAgentLoop:
    """Verify bounded orchestration with replaceable dependencies."""

    def test_one_step_completion_consumes_phase_three_task(self) -> None:
        """Reuse a routed idle task and complete it without creating another task."""
        request = _request()
        runtime = AgentRuntime(
            planner=lambda _context: _output(response="Done.", end=True),
        )
        task = runtime.create_task(request, task_id="task-1")

        result = runtime.run(request)

        assert runtime.get_task("task-1") is task
        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 1
        assert result["response"] == "Done."

    def test_multi_step_result_returns_to_planner(self) -> None:
        """Pass a typed tool result into the next planner context."""
        plans = [
            _output(response="Checking.", tool_call=_call()),
            _output(response="It is running.", end=True),
        ]
        planner_contexts = []
        handled_results: list[ToolResult] = []
        call = _call()

        def planner(context):
            planner_contexts.append(context)
            return plans.pop(0)

        def selector(_context, output):
            return output["tool_call"]

        def execute(_context, selected):
            assert selected == call
            return _result(call)

        def handle(context, result):
            handled_results.append(result)
            assert context.last_tool_result is result
            return _output(response="Result received.")

        runtime = AgentRuntime(
            planner=planner,
            tool_selector=selector,
            tool_executor=execute,
            tool_result_handler=handle,
        )

        task = runtime.create_task(_request(), task_id="task-1")
        result = runtime.run(task.request)

        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 2
        assert len(handled_results) == 1
        assert planner_contexts[0].last_tool_result is None
        assert planner_contexts[1].last_tool_result == handled_results[0]
        assert result["response"] == "It is running."

    def test_iteration_limit_aborts_before_an_additional_decision(self) -> None:
        """Stop before planning a cycle beyond the configured iteration limit."""
        next_call = 0

        def planner(_context):
            nonlocal next_call
            next_call += 1
            return _output(tool_call=_call(f"call-{next_call}"))

        runtime = AgentRuntime(
            planner=planner,
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, call: _result(call),
            tool_result_handler=lambda _context, _result: _output(
                response="Continuing."
            ),
        )
        task = runtime.create_task(
            _request(),
            AgentTaskConfig(max_iterations=2, stuck_task_threshold=20),
            task_id="task-1",
        )

        runtime.run(task.request)

        assert task.state == AgentTaskState.ABORTED
        assert task.abort_reason == AgentAbortReason.MAX_ITERATIONS
        assert task.iterations == 2
        assert next_call == 2

    def test_generated_token_limit_is_independent_of_iterations(self) -> None:
        """Abort on generated-token accounting without consuming an iteration."""
        runtime = AgentRuntime(
            planner=lambda _context: _output(response="too many words", end=True),
            token_counter=lambda _text: 4,
        )
        task = runtime.create_task(
            _request(),
            AgentTaskConfig(max_generated_tokens=3),
            task_id="task-1",
        )

        result = runtime.run(task.request)

        assert task.state == AgentTaskState.ABORTED
        assert task.abort_reason == AgentAbortReason.MAX_GENERATED_TOKENS
        assert task.iterations == 0
        assert result["end"] is True

    def test_compaction_callback_runs_at_threshold(self) -> None:
        """Signal compaction through the injected dependency before planning."""
        compacted = []

        def compact(context):
            compacted.append(context)
            assert context.task is not None
            context.task.update_context_tokens(0)
            return context

        runtime = AgentRuntime(
            compactor=compact,
            planner=lambda _context: _output(response="Done.", end=True),
        )
        task = runtime.create_task(
            _request(),
            AgentTaskConfig(context_compaction_threshold=4),
            task_id="task-1",
        )
        task.update_context_tokens(4)

        runtime.run(task.request)

        assert len(compacted) == 1
        assert task.state == AgentTaskState.COMPLETED
        assert task.context_tokens == 0

    def test_approval_pause_does_not_consume_iteration_or_lose_call(self) -> None:
        """Resume an approved pending call without repeating the planner cycle."""
        planner_calls = 0
        call = _call()

        def planner(context):
            nonlocal planner_calls
            planner_calls += 1
            if planner_calls == 1:
                assert context.task is not None
                runtime.request_approval(
                    context.task.task_id,
                    AgentApprovalRequest(
                        "approval-1",
                        context.task.task_id,
                        _validated_call(call),
                        "Allow the status check?",
                    ),
                )
                return _output(paused=True)
            return _output(response="Approved and complete.", end=True)

        runtime = AgentRuntime(
            planner=planner,
            tool_executor=lambda _context, selected: _result(selected),
            tool_result_handler=lambda _context, _result: _output(
                response="Complete.", end=True
            ),
        )
        task = runtime.create_task(_request(), task_id="task-1")

        paused = runtime.run(task.request)
        assert paused["paused"] is True
        assert task.state == AgentTaskState.AWAITING_APPROVAL
        assert task.iterations == 0
        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse("approval-1", AgentApprovalDecision.APPROVED),
        )
        runtime.run(task.request)

        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 1
        assert planner_calls == 1

    def test_selector_approval_pause_preserves_selected_call(self) -> None:
        """Preserve a selected call when approval is requested by the selector."""
        call = _call()
        executions = 0

        def selector(context, output):
            assert context.task is not None
            runtime.request_approval(
                context.task.task_id,
                AgentApprovalRequest(
                    "approval-1",
                    context.task.task_id,
                    _validated_call(call),
                    "Allow the status check?",
                ),
            )
            return output["tool_call"]

        def execute(_context, selected):
            nonlocal executions
            executions += 1
            return _result(selected)

        runtime = AgentRuntime(
            planner=lambda _context: _output(tool_call=call),
            tool_selector=selector,
            tool_executor=execute,
            tool_result_handler=lambda _context, _result: _output(
                response="Complete.", end=True
            ),
        )
        task = runtime.create_task(_request(), task_id="task-1")

        paused = runtime.run(task.request)

        assert paused["paused"]
        assert task.state == AgentTaskState.AWAITING_APPROVAL
        assert task.iterations == 0
        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse("approval-1", AgentApprovalDecision.APPROVED),
        )
        runtime.run(task.request)

        assert task.state == AgentTaskState.COMPLETED
        assert executions == 1

    def test_choice_pause_does_not_consume_iteration(self) -> None:
        """Resume a choice pause with the same task context and accounting."""
        planner_calls = 0

        def planner(context):
            nonlocal planner_calls
            planner_calls += 1
            if planner_calls == 1:
                assert context.task is not None
                runtime.request_choice(
                    context.task.task_id,
                    AgentChoiceRequest(
                        "choice-1",
                        context.task.task_id,
                        "Choose the output style.",
                        (AgentChoiceOption("brief", "Brief"),),
                    ),
                )
                return _output(paused=True)
            return _output(response="Brief result.", end=True)

        runtime = AgentRuntime(planner=planner)
        task = runtime.create_task(_request(), task_id="task-1")

        runtime.run(task.request)
        assert task.state == AgentTaskState.AWAITING_CHOICE
        assert task.iterations == 0
        runtime.respond_to_choice(
            task.task_id,
            AgentChoiceResponse("choice-1", choice_id="brief"),
        )
        runtime.run(task.request)

        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 1
        assert planner_calls == 2

    def test_dependency_cancellation_leaves_terminal_task(self) -> None:
        """Handle cancellation during planning, selection, and execution."""

        def make_runtime(phase: str) -> AgentRuntime:
            """Build one runtime whose selected dependency cancels the task."""
            runtime: AgentRuntime

            def planner(context):
                if phase == "planning":
                    assert context.task is not None
                    runtime.cancel_task(context.task.task_id)
                return _output(tool_call=_call())

            def selector(context, output):
                if phase == "selection":
                    assert context.task is not None
                    runtime.cancel_task(context.task.task_id)
                return output["tool_call"]

            def execute(context, call):
                if phase == "execution":
                    assert context.task is not None
                    runtime.cancel_task(context.task.task_id)
                return _result(call)

            runtime = AgentRuntime(
                planner=planner,
                tool_selector=selector,
                tool_executor=execute,
                tool_result_handler=lambda _context, _result: _output(
                    response="done", end=True
                ),
            )
            return runtime

        for phase in ("planning", "selection", "execution"):
            with nullcontext():
                runtime = make_runtime(phase)
                task = runtime.create_task(
                    _request(session_id=f"session-{phase}"),
                    task_id=f"task-{phase}",
                )

                runtime.run(task.request)

                assert task.state == AgentTaskState.CANCELLED
                assert runtime.get_session(task.session_id).cancelled

    def test_planner_and_tool_failures_are_typed(self) -> None:
        """Convert planner and executor exceptions into terminal failure reasons."""
        planner_runtime = AgentRuntime(
            planner=lambda _context: (_ for _ in ()).throw(RuntimeError("planner")),
        )
        planner_task = planner_runtime.create_task(_request(), task_id="planner")
        planner_runtime.run(planner_task.request)
        assert planner_task.state == AgentTaskState.FAILED
        assert planner_task.failure_reason == AgentFailureReason.MODEL_ERROR

        tool_runtime = AgentRuntime(
            planner=lambda _context: _output(tool_call=_call()),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, _call: (_ for _ in ()).throw(
                RuntimeError("tool")
            ),
        )
        tool_task = tool_runtime.create_task(_request("tool-session"), task_id="tool")
        tool_runtime.run(tool_task.request)
        assert tool_task.state == AgentTaskState.FAILED
        assert tool_task.failure_reason == AgentFailureReason.TOOL_ERROR

    def test_empty_tool_catalog_is_reported_as_a_typed_failure(self) -> None:
        """Distinguish an empty tool catalog from a model or execution failure."""
        runtime = AgentRuntime(
            planner=lambda _context: _output(response="Perform the action."),
        )
        task = runtime.create_task(_request("empty-tools"), task_id="empty-tools")

        result = runtime.run(task.request)

        assert task.state == AgentTaskState.FAILED
        assert task.failure_reason == AgentFailureReason.NO_TOOLS_FOUND
        terminal = result.get("terminal")
        assert terminal is not None
        assert terminal.failure_reason == AgentFailureReason.NO_TOOLS_FOUND

    def test_successful_terminal_tool_completes_without_result_handler(self) -> None:
        """Treat a successful terminal tool result as the task result itself."""
        call = _call()
        handled: list[ToolResult] = []
        outputs: list[AgentOutput] = []

        def record_result(_context, result: ToolResult) -> AgentOutput:
            """Record a result if the terminal result guard regresses."""
            handled.append(result)
            return _output(response="unexpected")

        def record_output(output: AgentOutput) -> None:
            """Record the externally visible runtime outputs."""
            outputs.append(output)

        def execute_terminal(_context, selected: ToolCall) -> ToolExecutionResult:
            """Return one successfully executed terminal tool result."""
            return {
                "tool_call_id": selected["id"],
                "output": {"spoken": True},
                "error": None,
                "tool_id": selected["name"],
                "status": AgentToolExecutionStatus.SUCCEEDED,
                "end_task": True,
            }

        runtime = AgentRuntime(
            planner=lambda _context: _output(
                response="I will check that.",
                tool_call=call,
            ),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=execute_terminal,
            tool_result_handler=record_result,
        )
        task = runtime.create_task(_request(), task_id="terminal-tool")

        result = runtime.run(task.request, record_output)

        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 1
        assert result["response"] is None
        assert not handled
        assert task.completion_metadata is not None
        tool_result = task.completion_metadata["tool_result"]
        assert isinstance(tool_result, dict)
        assert tool_result["status"] == "succeeded"
        assert tool_result["end_task"] is True
        assert [output["response"] for output in outputs] == ["I will check that."]

    def test_failed_typed_tool_result_fails_without_claiming_success(self) -> None:
        """Convert a typed executor failure into the runtime failure path."""
        call = _call()
        handled: list[ToolResult] = []

        def record_result(_context, result: ToolResult) -> AgentOutput:
            """Record a result if failed tools incorrectly reach the responder."""
            handled.append(result)
            return _output(response="unexpected")

        def execute_failed(_context, selected: ToolCall) -> ToolExecutionResult:
            """Return one structured executor failure."""
            return {
                "tool_call_id": selected["id"],
                "output": None,
                "error": "speech could not be queued",
                "tool_id": selected["name"],
                "status": AgentToolExecutionStatus.FAILED,
            }

        runtime = AgentRuntime(
            planner=lambda _context: _output(tool_call=call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=execute_failed,
            tool_result_handler=record_result,
        )
        task = runtime.create_task(_request("failed-tool"), task_id="failed-tool")

        runtime.run(task.request)

        assert task.state == AgentTaskState.FAILED
        assert task.failure_reason == AgentFailureReason.TOOL_ERROR
        assert task.failure_detail == "speech could not be queued"
        assert not handled

    def test_malformed_outputs_and_repeated_actions_abort_safely(self) -> None:
        """Reject malformed dependencies and terminate repeated identical actions."""
        malformed = AgentRuntime(
            planner=lambda _context: cast(AgentOutput, {"end": True}),
        )
        malformed_task = malformed.create_task(_request(), task_id="malformed")
        malformed.run(malformed_task.request)
        assert malformed_task.state == AgentTaskState.FAILED

        invalid_selector = AgentRuntime(
            planner=lambda _context: _output(tool_call=_call()),
            tool_selector=lambda _context, _output: cast(
                ToolCall,
                {"id": "bad", "name": "bad", "arguments": []},
            ),
        )
        invalid_task = invalid_selector.create_task(
            _request("invalid-session"), task_id="invalid"
        )
        invalid_selector.run(invalid_task.request)
        assert invalid_task.state == AgentTaskState.FAILED
        assert invalid_task.failure_reason == AgentFailureReason.INVALID_TOOL_CALL

        callbacks: list[AgentOutput] = []
        stuck = AgentRuntime(
            planner=lambda _context: _output(tool_call=_call()),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, call: _result(call),
            tool_result_handler=lambda _context, _result: _output(response="again"),
        )
        stuck_task = stuck.create_task(
            _request("stuck-session"),
            AgentTaskConfig(stuck_task_threshold=2, max_iterations=10),
            task_id="stuck",
        )

        def record_output(output: AgentOutput) -> None:
            """Record callback output with the public callback contract."""
            callbacks.append(output)

        final = stuck.run(stuck_task.request, record_output)
        assert stuck_task.state == AgentTaskState.ABORTED
        assert stuck_task.abort_reason == AgentAbortReason.STUCK_TASK
        assert final["response"] is None
        terminal = final.get("terminal")
        assert terminal is not None
        assert terminal.abort_reason == AgentAbortReason.STUCK_TASK
        assert callbacks[-1].get("terminal") == terminal
        assert (
            sum(1 for callback in callbacks if callback.get("terminal") is not None)
            == 1
        )

    def test_event_order_and_terminal_event_are_deterministic(self) -> None:
        """Publish lifecycle state changes before the single terminal event."""
        names: list[str] = []
        dispatcher = EventDispatcher(log_warning=lambda _message, _severity: None)
        dispatcher.subscribe(
            "agent_task_state_changed",
            lambda _event: names.append("state"),
        )
        dispatcher.subscribe(
            "agent_task_finished",
            lambda _event: names.append("finished"),
        )
        runtime = AgentRuntime(
            event_dispatcher=dispatcher,
            planner=lambda _context: _output(response="Done.", end=True),
        )
        task = runtime.create_task(_request("events"), task_id="events")

        runtime.run(task.request)
        runtime.run(task.request)

        assert names == ["state"] * 6 + ["finished"]
