# SPDX-License-Identifier: MIT
"""Focused tests for Celune's conversation-first agent routing boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest import TestCase, mock

from celune.agent import (
    AgentApprovalRequest,
    AgentChoiceOption,
    AgentChoiceRequest,
    AgentInputClassification,
    AgentInputRouter,
    AgentInterruptionKind,
    AgentRoute,
    AgentRuntime,
    AgentTaskState,
    AgentToolBehavior,
    AgentToolDangerLevel,
    ValidatedToolCall,
)

if TYPE_CHECKING:
    from celune.celune import Celune


def _call() -> ValidatedToolCall:
    """Build a typed approval fixture without executing it."""
    return {
        "id": "call-1",
        "name": "read_status",
        "arguments": {},
        "tool_id": "read_status",
        "behavior": AgentToolBehavior.READ_ONLY,
        "danger": AgentToolDangerLevel.LOW,
        "approval_required": True,
    }


class AgentRoutingTests(TestCase):
    """Verify routing keeps conversation and task input separate."""

    def setUp(self) -> None:
        self.engine = SimpleNamespace(persona_history=[])
        self.runtime = AgentRuntime()
        self.router = AgentInputRouter(cast("Celune", self.engine), self.runtime)

    def test_greetings_questions_and_explanations_stay_conversation(self) -> None:
        """Keep social conversation and information requests on Persona."""
        for text in (
            "Hello.",
            "How are you?",
            "What do you think about this?",
            "Explain this error.",
            "How do I set my voice?",
        ):
            with self.subTest(text=text):
                result = self.router.route(text)
                self.assertEqual(
                    result.classification, AgentInputClassification.CONVERSATION
                )
                self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_direct_action_creates_idle_task_without_execution(self) -> None:
        """Create a typed task while leaving the execution loop untouched."""
        with mock.patch.object(self.runtime, "execute_tool") as execute_tool:
            result = self.router.route("Check whether this process is running.")

        self.assertEqual(result.classification, AgentInputClassification.TASK)
        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertIsNotNone(result.task_request)
        task = self.runtime.get_active_task("default")
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.state, AgentTaskState.IDLE)
        execute_tool.assert_not_called()

    def test_ambiguous_request_asks_for_clarification_without_a_task(self) -> None:
        """Do not guess that an underspecified imperative is a tool request."""
        result = self.router.route("Please handle this")

        self.assertEqual(result.route, AgentRoute.CLARIFICATION)
        self.assertTrue(result.requires_clarification)
        self.assertIsNone(result.task_request)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_ambiguous_input_can_use_existing_persona_classifier(self) -> None:
        """Use the existing Persona request boundary only for unresolved input."""
        response = SimpleNamespace(
            raise_for_status=mock.Mock(),
            json=lambda: {
                "text": json.dumps(
                    {
                        "classification": "task",
                        "confidence": 0.91,
                        "task_request": "Handle the active file.",
                        "requires_clarification": False,
                        "reason": "explicit_target_from_context",
                    }
                )
            },
        )
        self.engine.vision = SimpleNamespace(post=mock.Mock(return_value=response))
        with mock.patch(
            "celune.agent.routing.build_agent_classification_request",
            return_value={"format": "celune_agent_classification"},
        ) as build_request:
            result = self.router.route("Please handle this", persona_ready=True)

        build_request.assert_called_once_with(self.engine, "Please handle this")
        self.assertEqual(result.route, AgentRoute.TASK)
        task_request = result.task_request
        self.assertIsNotNone(task_request)
        assert task_request is not None
        self.assertEqual(task_request.request, "Handle the active file.")

    def test_follow_up_without_active_task_is_classified_normally(self) -> None:
        """Treat a follow-up-looking phrase as conversation when no task exists."""
        result = self.router.route("Continue with the explanation.")

        self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_follow_up_with_active_task_becomes_task_input(self) -> None:
        """Steer the existing task instead of creating a second task."""
        created = self.router.route("Open this file and tell me what is wrong.")
        task = self.runtime.get_active_task("default")
        self.assertIsNotNone(task)
        assert task is not None

        result = self.router.route("Use a concise explanation.")

        self.assertEqual(created.route, AgentRoute.TASK)
        self.assertEqual(result.route, AgentRoute.TASK_INPUT)
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)
        interruption = task.interruption
        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertEqual(interruption.kind, AgentInterruptionKind.USER_STEERING)
        self.assertEqual(interruption.instruction, "Use a concise explanation.")
        active_task = self.runtime.get_active_task("default")
        self.assertIsNotNone(active_task)
        assert active_task is not None
        self.assertEqual(active_task.task_id, task.task_id)

    def test_approval_and_choice_answers_route_to_active_task(self) -> None:
        """Deliver pending approval and choice answers to the existing runtime."""
        task = self.runtime.create_task(
            self.router._make_request("Open this file and tell me what is wrong."),
            task_id="task-1",
        )
        self.runtime.start_task(task.task_id)
        self.runtime.classify_task(task.task_id)
        self.runtime.request_approval(
            task.task_id,
            AgentApprovalRequest("approval-1", task.task_id, _call(), "Allow?"),
        )

        approval_result = self.router.route("yes")
        self.assertEqual(approval_result.route, AgentRoute.APPROVAL_RESPONSE)
        self.assertEqual(task.state, AgentTaskState.WORKING)

        self.runtime.request_choice(
            task.task_id,
            AgentChoiceRequest(
                "choice-1",
                task.task_id,
                "Choose a format",
                (AgentChoiceOption("brief", "Brief"),),
            ),
        )
        choice_result = self.router.route("brief")
        self.assertEqual(choice_result.route, AgentRoute.CHOICE_RESPONSE)
        self.assertEqual(task.state, AgentTaskState.WORKING)

    def test_cancellation_and_interruption_route_to_active_task(self) -> None:
        """Route explicit cancellation and interruption without conversation."""
        task = self.runtime.create_task(
            self.router._make_request("Check whether this process is running."),
            task_id="task-1",
        )
        self.runtime.start_task(task.task_id)
        self.runtime.classify_task(task.task_id)

        interruption = self.router.route("hold on")
        self.assertEqual(interruption.route, AgentRoute.INTERRUPTION)
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)

        self.runtime.resume(task.session_id)
        cancellation = self.router.route("cancel it")
        self.assertEqual(cancellation.route, AgentRoute.CANCELLATION)
        self.assertEqual(task.state, AgentTaskState.CANCELLED)


if __name__ == "__main__":
    import unittest

    unittest.main()
