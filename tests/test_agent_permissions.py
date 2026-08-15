# SPDX-License-Identifier: MIT
"""Focused tests for Phase 6 agent permission evaluation and approval gates."""

from __future__ import annotations

from typing import Optional
from unittest import TestCase

from celune.agent import (
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentOutput,
    AgentFailureReason,
    AgentPermissionDecision,
    AgentPermissionReason,
    AgentRequest,
    AgentRuntime,
    AgentSession,
    AgentTaskState,
    AgentToolBehavior,
    AgentToolDangerLevel,
    AgentToolSchema,
    DefaultAgentPermissionPolicy,
    ToolCall,
    ToolResult,
)


def _request(session_id: str = "permission-session") -> AgentRequest:
    """Build one explicit action request for permission tests."""
    return AgentRequest(
        request="Perform the requested action.",
        session=AgentSession(session_id=session_id),
    )


def _call(name: str = "read_status") -> ToolCall:
    """Build one simple tool call for a deterministic planner."""
    return {"id": f"call-{name}", "name": name, "arguments": {}}


def _schema(
    tool_id: str,
    behavior: AgentToolBehavior,
    danger: AgentToolDangerLevel = AgentToolDangerLevel.LOW,
    *,
    approval_required: bool = False,
    available: bool = True,
) -> AgentToolSchema:
    """Build one permission-bearing tool schema."""
    return AgentToolSchema(
        tool_id=tool_id,
        display_name=tool_id.replace("_", " ").title(),
        description=f"Use {tool_id}.",
        behavior=behavior,
        danger=danger,
        approval_required=approval_required,
        available=available,
    )


def _result(call: ToolCall) -> ToolResult:
    """Build one successful executor result."""
    return {
        "tool_call_id": call["id"],
        "output": {"ok": True},
        "error": None,
    }


def _output(
    call: Optional[ToolCall] = None,
    *,
    response: Optional[str] = None,
    end: bool = False,
) -> AgentOutput:
    """Build one planner or result-handler output."""
    return {
        "tool_call": call,
        "response": response,
        "end": end,
        "paused": False,
    }


class AgentPermissionTests(TestCase):
    """Verify that policy decisions control the existing executor boundary."""

    def test_read_only_tool_executes_without_approval(self) -> None:
        """Allow an available safe tool and expose its permission metadata."""
        call = _call()
        results: list[ToolResult] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(call["name"], AgentToolBehavior.READ_ONLY)
            },
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: _result(selected),
            tool_result_handler=lambda _context, result: (
                results.append(result) or _output(response="Done.", end=True)
            ),
        )

        task = runtime.create_task(_request(), task_id="read-task")
        runtime.run(task.request)

        self.assertEqual(task.state, AgentTaskState.COMPLETED)
        self.assertIsNotNone(task.permission_decision)
        assert task.permission_decision is not None
        self.assertEqual(
            task.permission_decision.decision, AgentPermissionDecision.ALLOW
        )
        self.assertEqual(
            task.permission_decision.reason, AgentPermissionReason.SAFE_READ_ONLY
        )
        self.assertEqual(results[0]["permission"]["decision"], "allow")

    def test_mutating_tool_waits_without_calling_executor(self) -> None:
        """Pause before a state-changing tool and retain the exact pending call."""
        call = _call("delete_file")
        executions: list[ToolCall] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(call["name"], AgentToolBehavior.MUTATING)
            },
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: (
                executions.append(selected) or _result(selected)
            ),
            tool_result_handler=lambda _context, _result: _output(
                response="Done.", end=True
            ),
        )
        task = runtime.create_task(_request(), task_id="mutating-task")

        paused = runtime.run(task.request)

        approval = runtime.get_pending_approval(task.task_id)
        self.assertTrue(paused["paused"])
        self.assertEqual(task.state, AgentTaskState.AWAITING_APPROVAL)
        self.assertEqual(task.iterations, 0)
        self.assertEqual(executions, [])
        self.assertIsNotNone(approval)
        assert approval is not None
        self.assertEqual(approval.task_id, task.task_id)
        self.assertEqual(approval.tool_call["id"], call["id"])
        self.assertEqual(approval.tool_call["arguments"], call["arguments"])
        self.assertEqual(approval.tool_call["behavior"], AgentToolBehavior.MUTATING)
        self.assertEqual(approval.tool_call["danger"], AgentToolDangerLevel.LOW)
        self.assertIsNotNone(approval.permission)
        assert approval.permission is not None
        self.assertEqual(
            approval.permission.reason, AgentPermissionReason.MUTATING_TOOL
        )

        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(approval.request_id, AgentApprovalDecision.APPROVED),
        )
        runtime.run(task.request)

        self.assertEqual(task.state, AgentTaskState.COMPLETED)
        self.assertEqual(executions, [approval.tool_call])
        self.assertIsNotNone(task.permission_decision)
        assert task.permission_decision is not None
        self.assertEqual(
            task.permission_decision.approval_decision, AgentApprovalDecision.APPROVED
        )

    def test_dangerous_tool_denies_when_approval_is_unavailable(self) -> None:
        """Deny high-risk calls when no approval channel is configured."""
        call = _call("delete_account")
        executions: list[ToolCall] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(
                    call["name"],
                    AgentToolBehavior.MUTATING,
                    AgentToolDangerLevel.HIGH,
                )
            },
            permission_policy=DefaultAgentPermissionPolicy(approval_available=False),
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: (
                executions.append(selected) or _result(selected)
            ),
        )
        task = runtime.create_task(_request(), task_id="danger-task")

        runtime.run(task.request)

        self.assertEqual(task.state, AgentTaskState.FAILED)
        self.assertEqual(task.failure_reason, AgentFailureReason.PERMISSION_DENIED)
        self.assertIsNotNone(task.permission_decision)
        assert task.permission_decision is not None
        self.assertEqual(
            task.permission_decision.reason, AgentPermissionReason.APPROVAL_UNAVAILABLE
        )
        self.assertEqual(executions, [])
        self.assertIsNone(runtime.get_pending_approval(task.task_id))

    def test_explicitly_disallowed_tool_is_denied(self) -> None:
        """Reject a tool ID on the policy deny list before execution."""
        call = _call("read_status")
        executions: list[ToolCall] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(call["name"], AgentToolBehavior.READ_ONLY)
            },
            permission_policy=DefaultAgentPermissionPolicy(
                disallowed_tool_ids=(call["name"],)
            ),
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: (
                executions.append(selected) or _result(selected)
            ),
        )
        task = runtime.create_task(
            _request("disallowed-session"), task_id="disallowed-task"
        )

        runtime.run(task.request)

        self.assertEqual(task.failure_reason, AgentFailureReason.PERMISSION_DENIED)
        self.assertIsNotNone(task.permission_decision)
        assert task.permission_decision is not None
        self.assertEqual(
            task.permission_decision.reason, AgentPermissionReason.TOOL_DISALLOWED
        )
        self.assertEqual(executions, [])

    def test_denied_approval_never_reaches_executor(self) -> None:
        """Turn a denied policy approval into a typed terminal failure."""
        call = _call("send_message")
        executions: list[ToolCall] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(
                    call["name"],
                    AgentToolBehavior.MUTATING,
                    approval_required=True,
                )
            },
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: (
                executions.append(selected) or _result(selected)
            ),
        )
        task = runtime.create_task(_request(), task_id="denied-task")
        runtime.run(task.request)
        approval = runtime.get_pending_approval(task.task_id)
        assert approval is not None

        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(approval.request_id, AgentApprovalDecision.DENIED),
        )

        self.assertEqual(task.state, AgentTaskState.FAILED)
        self.assertEqual(task.failure_reason, AgentFailureReason.PERMISSION_DENIED)
        self.assertIsNotNone(task.permission_decision)
        assert task.permission_decision is not None
        self.assertEqual(
            task.permission_decision.approval_decision, AgentApprovalDecision.DENIED
        )
        self.assertEqual(executions, [])

    def test_duplicate_or_mismatched_approval_cannot_execute_twice(self) -> None:
        """Reject stale responses and execute one approved call exactly once."""
        call = _call("write_file")
        executions: list[ToolCall] = []
        runtime = AgentRuntime(
            tool_schemas={
                call["name"]: _schema(call["name"], AgentToolBehavior.MUTATING)
            },
            planner=lambda _context: _output(call),
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=lambda _context, selected: (
                executions.append(selected) or _result(selected)
            ),
            tool_result_handler=lambda _context, _result: _output(
                response="Written.", end=True
            ),
        )
        task = runtime.create_task(_request(), task_id="duplicate-task")
        other = runtime.create_task(
            _request("other-permission-session"), task_id="other-task"
        )
        runtime.run(task.request)
        approval = runtime.get_pending_approval(task.task_id)
        assert approval is not None

        with self.assertRaises(ValueError):
            runtime.respond_to_approval(
                other.task_id,
                AgentApprovalResponse(
                    approval.request_id, AgentApprovalDecision.APPROVED
                ),
            )
        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(approval.request_id, AgentApprovalDecision.APPROVED),
        )
        with self.assertRaises(ValueError):
            runtime.respond_to_approval(
                task.task_id,
                AgentApprovalResponse(
                    approval.request_id, AgentApprovalDecision.APPROVED
                ),
            )
        runtime.run(task.request)

        self.assertEqual(len(executions), 1)
        self.assertEqual(task.state, AgentTaskState.COMPLETED)

    def test_unavailable_tool_is_denied_and_cancellation_clears_approval(self) -> None:
        """Deny unavailable tools and cancel an independent pending approval cleanly."""
        unavailable_call = _call("offline_tool")
        unavailable = AgentRuntime(
            tool_schemas={
                unavailable_call["name"]: _schema(
                    unavailable_call["name"],
                    AgentToolBehavior.READ_ONLY,
                    available=False,
                )
            },
            planner=lambda _context: _output(unavailable_call),
            tool_selector=lambda _context, output: output["tool_call"],
        )
        unavailable_task = unavailable.create_task(
            _request("unavailable-session"), task_id="unavailable-task"
        )
        unavailable.run(unavailable_task.request)
        self.assertEqual(
            unavailable_task.failure_reason, AgentFailureReason.PERMISSION_DENIED
        )
        self.assertIsNotNone(unavailable_task.permission_decision)
        assert unavailable_task.permission_decision is not None
        self.assertEqual(
            unavailable_task.permission_decision.reason,
            AgentPermissionReason.TOOL_UNAVAILABLE,
        )

        pending_call = _call("mutate_later")
        pending = AgentRuntime(
            tool_schemas={
                pending_call["name"]: _schema(
                    pending_call["name"], AgentToolBehavior.MUTATING
                )
            },
            planner=lambda _context: _output(pending_call),
            tool_selector=lambda _context, output: output["tool_call"],
        )
        pending_task = pending.create_task(
            _request("cancel-session"), task_id="cancel-task"
        )
        pending.run(pending_task.request)
        self.assertIsNotNone(pending.get_pending_approval(pending_task.task_id))
        pending.cancel_task(pending_task.task_id)
        self.assertEqual(pending_task.state, AgentTaskState.CANCELLED)
        self.assertIsNone(pending.get_pending_approval(pending_task.task_id))


if __name__ == "__main__":
    import unittest

    unittest.main()
