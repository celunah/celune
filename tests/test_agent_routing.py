# SPDX-License-Identifier: MIT
"""Focused tests for Celune's conversation-first agent routing boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import TestCase, mock
from typing import TYPE_CHECKING, cast

from celune.i18n import string
from celune.typing.common import JSONSerializable
from celune.agent import (
    AgentRoute,
    AgentRuntime,
    AgentTaskState,
    AgentInputRouter,
    AgentChoiceOption,
    AgentToolBehavior,
    ValidatedToolCall,
    AgentChoiceRequest,
    AgentFailureReason,
    AgentApprovalRequest,
    AgentToolDangerLevel,
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentInterruptionKind,
    AgentInputClassification,
    AgentClassificationFailureKind,
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
        self.engine = SimpleNamespace(persona_history=[], config={})
        self.runtime = AgentRuntime()
        self.router = AgentInputRouter(cast("Celune", self.engine), self.runtime)

    def _set_classifier(self, *payloads: dict[str, JSONSerializable]) -> None:
        """Install deterministic structured Persona routing responses."""
        responses = []
        for payload in payloads:
            responses.append(
                SimpleNamespace(
                    raise_for_status=mock.Mock(),
                    json=lambda payload=payload: {"text": json.dumps(payload)},
                )
            )
        self.engine.vision = SimpleNamespace(post=mock.Mock(side_effect=responses))

    def test_greetings_questions_and_explanations_stay_conversation(self) -> None:
        """Keep social conversation and information requests on Persona."""
        self._set_classifier({"classification": "conversation", "confidence": 0.98})
        for text in (
            "Hello.",
            "How are you?",
            "What do you think about this?",
            "Explain this error.",
            "How do I set my voice?",
        ):
            with self.subTest(text=text):
                result = self.router.route(text, persona_ready=True)
                self.assertEqual(
                    result.classification, AgentInputClassification.CONVERSATION
                )
                self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_classifier_failure_logging_is_generic_only_at_normal_level(self) -> None:
        """Keep diagnostics detailed only when verbose or debug logs are enabled."""
        self.engine.log = mock.Mock()
        self.engine.log_level = "info"
        self.engine.vision = SimpleNamespace(
            post=mock.Mock(side_effect=RuntimeError("transport detail"))
        )
        self.router.route("Delete the fixture.", persona_ready=True)
        self.assertEqual(
            self.engine.log.call_args.args[-2:],
            (string("agent.classifier_failed_summary"), "warning"),
        )

        self.engine.log.reset_mock()
        self.engine.log_level = "debug"
        self.router.route("Delete the fixture.", persona_ready=True)
        message = self.engine.log.call_args.args[0]
        self.assertIn("transport", message)
        self.assertEqual(self.engine.log.call_args.kwargs["loglevel"], "verbose")

    def test_active_classifier_failure_releases_task_for_conversation(self) -> None:
        """Release a broken active task so later input can return to Persona."""
        task = self.runtime.create_task(
            self.router._make_request("Check the current process."),
            task_id="task-1",
        )
        self.engine.vision = SimpleNamespace(
            post=mock.Mock(side_effect=RuntimeError("classifier unavailable"))
        )

        result = self.router.route("What do you think about this?", persona_ready=True)

        self.assertEqual(result.route, AgentRoute.CLARIFICATION)
        self.assertIsNotNone(result.failure)
        self.assertEqual(task.state, AgentTaskState.FAILED)
        self.assertEqual(task.failure_reason, AgentFailureReason.INTERNAL_ERROR)
        self.assertIsNone(self.runtime.get_active_task("default"))

        self._set_classifier({"classification": "conversation", "confidence": 0.95})
        follow_up = self.router.route("How are you?", persona_ready=True)
        self.assertEqual(follow_up.route, AgentRoute.CONVERSATION)

    def test_direct_action_creates_idle_task_without_execution(self) -> None:
        """Create a typed task while leaving the execution loop untouched."""
        self._set_classifier(
            {
                "classification": "task",
                "route": "task",
                "confidence": 0.96,
                "intent": "inspect_process",
                "task_request": "Check whether this process is running.",
            }
        )
        with mock.patch.object(self.runtime, "execute_tool") as execute_tool:
            result = self.router.route(
                "Could you verify whether this process is active?",
                persona_ready=True,
            )

        self.assertEqual(result.classification, AgentInputClassification.TASK)
        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertIsNotNone(result.task_request)
        task = self.runtime.get_active_task("default")
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.state, AgentTaskState.IDLE)
        execute_tool.assert_not_called()
        self.assertEqual(result.intent, "inspect_process")
        self.assertEqual(result.to_json()["intent"], "inspect_process")

    def test_valid_low_confidence_task_is_not_downgraded_to_conversation(
        self,
    ) -> None:
        """Honor an explicit task decision when its request is structurally valid."""
        self._set_classifier(
            {
                "classification": "task",
                "route": "task",
                "confidence": 0.51,
                "task_request": "Check the current working directory.",
                "requires_clarification": False,
            }
        )

        result = self.router.route(
            "Check the current working directory.",
            persona_ready=True,
        )

        self.assertEqual(result.classification, AgentInputClassification.TASK)
        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertIsNotNone(self.runtime.get_active_task("default"))

    def test_agent_test_backend_routes_even_when_operation_mode_is_conversation(
        self,
    ) -> None:
        """Let the restricted agent test backend reach semantic task routing."""
        self.engine.mode = "converse"
        self.engine.backend_mode = "agent_test"
        self._set_classifier(
            {
                "classification": "task",
                "route": "task",
                "confidence": 0.99,
                "task_request": "Tell me the current working directory.",
            }
        )

        result = self.router.route(
            "Tell me the current working directory.",
            persona_ready=True,
        )

        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertIsNotNone(self.runtime.get_active_task("default"))

    def test_classifier_unavailable_keeps_input_on_conversation_path(self) -> None:
        """Do not infer a task when the semantic classifier is unavailable."""
        result = self.router.route("Please handle this")

        self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertFalse(result.requires_clarification)
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertEqual(
            result.failure.kind,
            AgentClassificationFailureKind.PERSONA_UNAVAILABLE,
        )
        self.assertIsNone(result.task_request)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_classifier_failures_are_observable_and_fail_closed(self) -> None:
        """Reject malformed, empty, and failed Persona classifier responses."""
        cases = (
            (
                SimpleNamespace(
                    json=lambda: {"text": "not json"},
                ),
                AgentClassificationFailureKind.MALFORMED_OUTPUT,
            ),
            (
                SimpleNamespace(json=lambda: {"text": ""}),
                AgentClassificationFailureKind.EMPTY_OUTPUT,
            ),
        )
        for response, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                response.raise_for_status = mock.Mock()
                self.engine.vision = SimpleNamespace(
                    post=mock.Mock(side_effect=(response, response))
                )
                result = self.router.route("Delete the fixture.", persona_ready=True)
                self.assertEqual(result.route, AgentRoute.CONVERSATION)
                self.assertIsNotNone(result.failure)
                assert result.failure is not None
                self.assertEqual(result.failure.kind, expected_kind)
                self.assertIsNone(self.runtime.get_active_task("default"))

        self.engine.vision = SimpleNamespace(
            post=mock.Mock(side_effect=RuntimeError("Persona transport failed"))
        )
        result = self.router.route("Delete the fixture.", persona_ready=True)
        self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertEqual(
            result.failure.kind,
            AgentClassificationFailureKind.TRANSPORT,
        )
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_malformed_output_gets_one_repair_request_through_persona(self) -> None:
        """Retry one malformed VLM response without changing the routing boundary."""
        first = SimpleNamespace(
            raise_for_status=mock.Mock(),
            json=lambda: {"text": "I will take care of that."},
        )
        second = SimpleNamespace(
            raise_for_status=mock.Mock(),
            json=lambda: {
                "text": json.dumps(
                    {
                        "classification": "task",
                        "route": "task",
                        "confidence": 0.97,
                        "task_request": "Delete the fixture.",
                    }
                )
            },
        )
        self.engine.vision = SimpleNamespace(
            post=mock.Mock(side_effect=(first, second))
        )

        result = self.router.route("Delete the fixture.", persona_ready=True)

        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertEqual(self.engine.vision.post.call_count, 2)
        second_request = self.engine.vision.post.call_args_list[1].kwargs["json"]
        messages = second_request["messages"]
        self.assertIsInstance(messages, list)
        assert isinstance(messages, list)
        first_message = messages[0]
        self.assertIsInstance(first_message, dict)
        assert isinstance(first_message, dict)
        content = first_message.get("content")
        self.assertIsInstance(content, str)
        assert isinstance(content, str)
        self.assertIn("previous routing output was rejected", content)

    def test_incompatible_new_task_route_gets_one_schema_repair(self) -> None:
        """Repair a task classification that incorrectly uses a follow-up route."""
        first = SimpleNamespace(
            raise_for_status=mock.Mock(),
            json=lambda: {
                "text": json.dumps(
                    {
                        "classification": "task",
                        "route": "task_input",
                        "confidence": 0.98,
                    }
                )
            },
        )
        second = SimpleNamespace(
            raise_for_status=mock.Mock(),
            json=lambda: {
                "text": json.dumps(
                    {
                        "classification": "task",
                        "route": "task",
                        "confidence": 0.98,
                        "task_request": "Check the current working directory.",
                    }
                )
            },
        )
        self.engine.vision = SimpleNamespace(
            post=mock.Mock(side_effect=(first, second))
        )

        result = self.router.route(
            "Tell me the current working directory.",
            persona_ready=True,
        )

        self.assertEqual(result.route, AgentRoute.TASK)
        self.assertEqual(self.engine.vision.post.call_count, 2)
        second_request = self.engine.vision.post.call_args_list[1].kwargs["json"]
        system = second_request["system"]
        self.assertIsInstance(system, str)
        assert isinstance(system, str)
        self.assertIn("use classification task and route task exactly", system)

    def test_ambiguous_request_asks_for_clarification(self) -> None:
        """Do not guess that an underspecified request is a tool request."""
        self._set_classifier(
            {
                "classification": "conversation",
                "confidence": 0.42,
                "requires_clarification": True,
                "clarification_prompt": "What would you like me to handle?",
            }
        )
        result = self.router.route("Please take care of this", persona_ready=True)

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
        self._set_classifier({"classification": "conversation", "confidence": 0.95})
        result = self.router.route(
            "Could you continue explaining the previous point?",
            persona_ready=True,
        )

        self.assertEqual(result.route, AgentRoute.CONVERSATION)
        self.assertIsNone(self.runtime.get_active_task("default"))

    def test_follow_up_with_active_task_becomes_task_input(self) -> None:
        """Steer the existing task instead of creating a second task."""
        self._set_classifier(
            {
                "classification": "task",
                "route": "task",
                "confidence": 0.96,
            },
            {
                "classification": "task",
                "route": "task_input",
                "confidence": 0.94,
            },
        )
        created = self.router.route(
            "Could you inspect this file and explain the problem?",
            persona_ready=True,
        )
        task = self.runtime.get_active_task("default")
        self.assertIsNotNone(task)
        assert task is not None

        result = self.router.route("Keep the explanation concise.", persona_ready=True)

        self.assertEqual(created.route, AgentRoute.TASK)
        self.assertEqual(result.route, AgentRoute.TASK_INPUT)
        self.assertEqual(task.state, AgentTaskState.PLANNING)
        interruption = task.interruption
        self.assertIsNotNone(interruption)
        assert interruption is not None
        self.assertEqual(interruption.kind, AgentInterruptionKind.USER_STEERING)
        self.assertEqual(interruption.instruction, "Keep the explanation concise.")
        self.assertEqual(task.request.request, "Keep the explanation concise.")
        self.assertIn(
            "Keep the explanation concise.",
            [entry.get("content") for entry in task.request.history],
        )
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

        self._set_classifier(
            {
                "classification": "task",
                "route": "approval_response",
                "confidence": 0.99,
                "approval_decision": "approved",
            },
            {
                "classification": "task",
                "route": "choice_response",
                "confidence": 0.99,
                "choice_id": "brief",
            },
            {
                "classification": "task",
                "route": "task_input",
                "confidence": 0.92,
            },
            {
                "classification": "task",
                "route": "task_input",
                "confidence": 0.92,
            },
        )
        approval_result = self.router.route(
            "That is fine, proceed.", persona_ready=True
        )
        self.assertEqual(approval_result.route, AgentRoute.APPROVAL_RESPONSE)
        self.assertIsNotNone(approval_result.approval_decision)
        assert approval_result.approval_decision is not None
        self.assertEqual(approval_result.approval_decision.value, "approved")
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
        choice_result = self.router.route("Use the concise format.", persona_ready=True)
        self.assertEqual(choice_result.route, AgentRoute.CHOICE_RESPONSE)
        self.assertEqual(choice_result.choice_id, "brief")
        self.assertEqual(task.state, AgentTaskState.WORKING)

        self.runtime.request_approval(
            task.task_id,
            AgentApprovalRequest("approval-2", task.task_id, _call(), "Allow?"),
        )
        approval_steering = self.router.route(
            "Only use the read-only status tool.", persona_ready=True
        )
        self.assertEqual(approval_steering.route, AgentRoute.CLARIFICATION)
        self.assertEqual(task.state, AgentTaskState.AWAITING_APPROVAL)
        self.assertIsNotNone(self.runtime.get_pending_approval(task.task_id))

        self.runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(
                "approval-2",
                decision=AgentApprovalDecision.APPROVED,
            ),
        )
        self.runtime.request_choice(
            task.task_id,
            AgentChoiceRequest(
                "choice-2",
                task.task_id,
                "Choose a format",
                (AgentChoiceOption("brief", "Brief"),),
            ),
        )
        choice_steering = self.router.route(
            "Keep the response short and factual.", persona_ready=True
        )
        self.assertEqual(choice_steering.route, AgentRoute.CLARIFICATION)
        self.assertEqual(task.state, AgentTaskState.AWAITING_CHOICE)
        self.assertIsNotNone(self.runtime.get_pending_choice(task.task_id))

    def test_cancellation_and_interruption_route_to_active_task(self) -> None:
        """Route explicit cancellation and interruption without conversation."""
        task = self.runtime.create_task(
            self.router._make_request("Check whether this process is running."),
            task_id="task-1",
        )
        self.runtime.start_task(task.task_id)
        self.runtime.classify_task(task.task_id)

        self._set_classifier(
            {
                "classification": "task",
                "route": "interruption",
                "confidence": 0.98,
                "interruption_kind": "user_interrupt",
            },
            {
                "classification": "task",
                "route": "cancellation",
                "confidence": 0.98,
            },
        )
        interruption = self.router.route(
            "Please pause the current work.", persona_ready=True
        )
        self.assertEqual(interruption.route, AgentRoute.INTERRUPTION)
        self.assertEqual(task.state, AgentTaskState.INTERRUPTED)

        self.runtime.resume(task.session_id)
        cancellation = self.router.route(
            "I changed my mind; abandon this task.", persona_ready=True
        )
        self.assertEqual(cancellation.route, AgentRoute.CANCELLATION)
        self.assertEqual(task.state, AgentTaskState.CANCELLED)


if __name__ == "__main__":
    import unittest

    unittest.main()
