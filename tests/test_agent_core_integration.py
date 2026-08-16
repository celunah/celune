# SPDX-License-Identifier: Apache-2.0
"""Offline integration coverage for Celune's core-owned agent workflow."""

from __future__ import annotations

from typing import Optional, cast
from unittest import TestCase, mock

from celune.agent import (
    AgentContext,
    AgentInputRouter,
    AgentOutput,
    AgentRequest,
    AgentRuntime,
    AgentTaskState,
    AgentToolBehavior,
    AgentToolDangerLevel,
    AgentToolSchema,
    AgentToolExecutionStatus,
    ToolCall,
    ToolExecutionResult,
    ToolResult,
)
from celune.agent.tools import AgentStatusTool
from celune.celune import Celune
from celune.dataclasses.events import (
    AgentApprovalRequestedEvent,
    AgentTaskFinishedEvent,
    AgentTaskStateChangedEvent,
)
from celune.pipeline import deliver_persona_response as celune_deliver_persona_response
from celune.persona.impl import PersonaClient
from celune.typing.common import JSONSerializable
from celune.typing.persona import PersonaClientResponse

from .support import FakeBackend, FakeGlow


class _PersonaResponse:
    """Small response adapter for the in-process Persona boundary."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        """Match the response method used by the production bridge."""

    def json(self) -> dict[str, JSONSerializable]:
        """Return one common Persona response shape."""
        return {"response": self.text}


class _PersonaFixture:
    """Return deterministic action-intent and final Persona responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, JSONSerializable]] = []
        self.responses = [
            _PersonaResponse(
                '{"classification":"task","route":"task",'
                '"confidence":0.98,"task_request":"Check the current agent status."}'
            ),
            _PersonaResponse("Read the current agent status."),
            _PersonaResponse("The agent task completed successfully."),
        ]

    def post(self, json: dict[str, JSONSerializable]) -> _PersonaResponse:
        """Record one request and return the next deterministic response."""
        self.requests.append(json)
        return self.responses.pop(0)


class _RoutingFixture(PersonaClient):
    """Return only structured semantic routing responses for core tests."""

    def __init__(self, *payloads: str) -> None:
        super().__init__()
        self.payloads = list(payloads)

    def post(self, json: dict[str, JSONSerializable]) -> PersonaClientResponse:
        """Return the next classifier payload and ignore request details."""
        del json
        return PersonaClientResponse({"response": self.payloads.pop(0)})


class _NeedleFixture:
    """Select the single registered read-only status tool without a model."""

    def __init__(self) -> None:
        self.intents: list[str] = []

    def __call__(
        self,
        _context: AgentContext,
        output: AgentOutput,
    ) -> Optional[ToolCall]:
        """Return one deterministic call for the Persona action intent."""
        intent = output["response"]
        if intent is not None:
            self.intents.append(intent)
        return {
            "id": "needle-status-call",
            "name": "read_agent_status",
            "arguments": {},
        }


class AgentCoreIntegrationTests(TestCase):
    """Run a complete routed task through a real lightweight Celune core."""

    def _make_core(self) -> Celune:
        """Create the core with model, audio-device, and glow work replaced by fakes."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={"mode": "agent"},
                tts_backend=FakeBackend,
            )
        self.addCleanup(core.close)
        return core

    def test_core_routes_approves_and_completes_a_mutating_tool_workflow(self) -> None:
        """Route an explicit request through the core, pause, approve, and complete."""
        core = self._make_core()
        executions: list[ToolCall] = []
        event_names: list[str] = []

        def record_state(_event: AgentTaskStateChangedEvent) -> None:
            """Record lifecycle transitions emitted by the core event dispatcher."""
            event_names.append("state")

        def record_approval(_event: AgentApprovalRequestedEvent) -> None:
            """Record the approval pause emitted by the core event dispatcher."""
            event_names.append("approval")

        def record_finished(_event: AgentTaskFinishedEvent) -> None:
            """Record the terminal event emitted by the core event dispatcher."""
            event_names.append("finished")

        core._event_dispatcher.subscribe("agent_task_state_changed", record_state)
        core._event_dispatcher.subscribe("agent_approval_requested", record_approval)
        core._event_dispatcher.subscribe("agent_task_finished", record_finished)

        call: ToolCall = {
            "id": "call-delete-test-item",
            "name": "delete_test_item",
            "arguments": {"item": "fixture"},
        }
        schema = AgentToolSchema(
            tool_id="delete_test_item",
            display_name="Delete test item",
            description="Delete the isolated fixture item.",
            behavior=AgentToolBehavior.MUTATING,
            danger=AgentToolDangerLevel.LOW,
        )

        def planner(_context: AgentContext) -> AgentOutput:
            """Return the deterministic action intent for this integration test."""
            return {
                "tool_call": call,
                "response": "Delete the isolated fixture item.",
                "end": False,
                "paused": False,
            }

        def execute(_context: AgentContext, selected: ToolCall) -> ToolExecutionResult:
            """Record the validated call and return one successful fixture result."""
            executions.append(selected)
            return {
                "tool_call_id": selected["id"],
                "output": {"deleted": True, "item": selected["arguments"]["item"]},
                "error": None,
                "tool_id": selected["name"],
                "status": AgentToolExecutionStatus.SUCCEEDED,
            }

        def handle_result(
            _context: AgentContext,
            _result: ToolResult,
        ) -> AgentOutput:
            """Convert the structured fixture result into a terminal response."""
            return {
                "tool_call": None,
                "response": "Deleted: fixture.",
                "end": True,
                "paused": False,
            }

        runtime = AgentRuntime(
            event_dispatcher=core._event_dispatcher,
            celune=core,
            planner=planner,
            tool_selector=lambda _context, output: output["tool_call"],
            tool_executor=execute,
            tool_result_handler=handle_result,
            tool_schemas={schema.tool_id: schema},
        )
        core.agent_runtime = runtime
        core._agent_router = AgentInputRouter(core, runtime)
        core.vision = _RoutingFixture(
            '{"classification":"task","route":"task","confidence":0.98}',
            '{"classification":"task","route":"approval_response",'
            '"confidence":0.99,"approval_decision":"approved"}',
        )

        route = core.route_input(
            "Please remove the isolated fixture item.", persona_ready=True
        )
        self.assertEqual(route.route.value, "task")
        self.assertIsNotNone(route.task_request)
        assert route.task_request is not None

        core._run_agent_route(route)
        task = runtime.get_active_task("default")
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.state, AgentTaskState.AWAITING_APPROVAL)
        self.assertEqual(executions, [])

        approval = runtime.get_pending_approval(task.task_id)
        self.assertIsNotNone(approval)
        assert approval is not None

        approval_route = core.route_input(
            "That is approved; continue.", persona_ready=True
        )
        self.assertEqual(approval_route.route.value, "approval_response")
        core._run_agent_route(approval_route)

        self.assertEqual(task.state, AgentTaskState.COMPLETED)
        self.assertEqual(executions, [approval.tool_call])
        self.assertEqual(task.iterations, 1)
        self.assertIsNotNone(task.completion_metadata)
        self.assertEqual(runtime.get_active_task("default"), None)
        self.assertEqual(event_names.count("approval"), 1)
        self.assertEqual(event_names.count("finished"), 1)
        self.assertLess(event_names.index("approval"), event_names.index("finished"))

    def test_core_owns_production_path_from_persona_to_speech(self) -> None:
        """Run one safe task through core-owned Persona, Needle, and speech paths."""
        selector = _NeedleFixture()
        persona = _PersonaFixture()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={"mode": "agent"},
                tts_backend=FakeBackend,
                agent_tool_selector=selector,
            )
        self.addCleanup(core.close)
        speech = mock.patch("celune.pipeline.queue_speech", return_value=True)
        delivery = mock.patch(
            "celune.celune.deliver_persona_response",
            wraps=celune_deliver_persona_response,
        )
        speech_mock = speech.start()
        delivery_mock = delivery.start()
        self.addCleanup(speech.stop)
        self.addCleanup(delivery.stop)
        core.vision = cast(PersonaClient, persona)

        self.assertIsNotNone(core.agent_runtime._planner)
        self.assertIsNotNone(core.agent_runtime._tool_result_handler)
        self.assertEqual(
            tuple(tool.name for tool in core.agent_runtime.tools),
            ("read_agent_status",),
        )

        conversation = core.route_input("Hello.", persona_ready=False)
        self.assertEqual(conversation.route.value, "conversation")
        self.assertIsNone(core.agent_runtime.get_active_task("default"))
        self.assertEqual(persona.requests, [])

        route = core.route_input(
            "Could you verify the current agent status?", persona_ready=True
        )
        self.assertEqual(route.route.value, "task")
        self.assertIsNotNone(route.task_request)
        assert route.task_request is not None
        outputs: list[AgentOutput] = []
        original_run = core.agent_runtime.run

        def record_run(request: AgentRequest) -> AgentOutput:
            """Capture the runtime output while preserving the real call."""
            output = original_run(request)
            outputs.append(output)
            return output

        with (
            mock.patch.object(core.agent_runtime, "run", side_effect=record_run),
            mock.patch.object(
                AgentStatusTool,
                "execute",
                autospec=True,
                side_effect=AgentStatusTool.execute,
            ) as execute,
        ):
            core._run_agent_route(route)
        self.assertEqual(
            outputs[0]["response"], "The agent task completed successfully."
        )
        self.assertTrue(outputs[0]["end"])

        metadata = route.routing_metadata
        self.assertIsInstance(metadata, dict)
        task_id = metadata.get("task_id")
        self.assertIsInstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        self.assertEqual(task.state, AgentTaskState.COMPLETED)
        self.assertEqual(task.iterations, 1)
        self.assertEqual(selector.intents, ["Read the current agent status."])
        self.assertEqual(len(persona.requests), 3)
        second_system = persona.requests[2]["system"]
        self.assertIsInstance(second_system, str)
        self.assertIn("Last tool result", second_system)
        self.assertIn("succeeded", second_system)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(delivery_mock.call_count, 1)
        self.assertEqual(speech_mock.call_count, 1)
        self.assertEqual(
            speech_mock.call_args.kwargs["display_text"],
            "The agent task completed successfully.",
        )
        self.assertIsNone(core.agent_runtime.get_active_task("default"))

    def test_needle_loading_failure_is_recorded_and_task_fails_safely(self) -> None:
        """Expose a typed terminal failure when production Needle cannot load."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch("celune.pipeline.queue_speech", return_value=True),
            mock.patch(
                "celune.celune.NeedleToolSelector.from_pretrained",
                side_effect=RuntimeError("checkpoint unavailable"),
            ),
        ):
            core = Celune(config={"mode": "agent"}, tts_backend=FakeBackend)
            self.addCleanup(core.close)
            core.vision = _RoutingFixture(
                '{"classification":"task","route":"task","confidence":0.98}',
                "Check the current agent status.",
            )
            route = core.route_input(
                "Could you verify the current agent status?", persona_ready=True
            )
            assert route.task_request is not None
            core._run_agent_route(route)

        metadata = route.routing_metadata
        self.assertIsInstance(metadata, dict)
        task_id = metadata.get("task_id")
        self.assertIsInstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        self.assertEqual(task.state, AgentTaskState.FAILED)
        self.assertIsNotNone(task.failure_reason)
        assert task.failure_reason is not None
        self.assertEqual(task.failure_reason.value, "invalid_tool_call")
        self.assertIn("checkpoint unavailable", core.agent_needle_error or "")
        self.assertFalse(core.agent_needle_ready)
