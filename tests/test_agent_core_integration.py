# SPDX-License-Identifier: Apache-2.0
"""Offline integration coverage for Celune's core-owned agent workflow."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, cast
from unittest import mock
from collections.abc import Mapping, Sequence

from celune.celune import Celune
from celune.constants import PERSONA_DEFAULT_MODEL_ID
from celune.typing.aliases import LogLevel
from celune.typing.agent import NeedleToolCall, NeedleToolCatalog
from celune.persona.impl import PersonaClient
from celune.typing.common import JSONSerializable
from celune.typing.persona import PersonaClientResponse
from celune.agent.tools import AgentStatusTool, production_agent_tool_schemas
from celune.pipeline import deliver_persona_response as celune_deliver_persona_response
from celune.agent.needle.impl import (
    NeedleHandler,
    NeedleToolSelector,
)
from celune.dataclasses.events import (
    AgentTaskFinishedEvent,
    AgentTaskStateChangedEvent,
    AgentApprovalRequestedEvent,
)
from celune.agent import (
    ToolCall,
    AgentTool,
    AgentRoute,
    ToolResult,
    AgentOutput,
    AgentContext,
    AgentRequest,
    AgentRuntime,
    AgentSession,
    AgentTaskState,
    AgentToolSchema,
    AgentInputRouter,
    AgentToolBehavior,
    AgentToolSelector,
    AgentFailureReason,
    ToolExecutionResult,
    AgentToolDangerLevel,
    AgentResponseCallback,
    AgentCancellationReason,
    AgentInputClassification,
    AgentToolExecutionStatus,
    AgentClassificationResult,
    AgentClassificationFailure,
    AgentClassificationFailureKind,
)

from .support import CeluneTestCase, FakeGlow, FakeBackend


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


class _SpeakPersonaFixture:
    """Return routing, progress, and approval responses for a terminal speak task."""

    def __init__(self) -> None:
        self.requests: list[dict[str, JSONSerializable]] = []
        self.responses = [
            _PersonaResponse(
                '{"classification":"task","route":"task",'
                '"confidence":0.98,"task_request":"Say hello."}'
            ),
            _PersonaResponse("I will say hello."),
            _PersonaResponse(
                '{"classification":"task","route":"approval_response",'
                '"confidence":0.99,"approval_decision":"approved"}'
            ),
        ]

    def post(self, json: dict[str, JSONSerializable]) -> _PersonaResponse:
        """Return the next deterministic response through the Persona boundary."""
        self.requests.append(json)
        return self.responses.pop(0)


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


class _SpeakNeedleHandler:
    """Return one controlled speak selection through the real Needle adapter."""

    @staticmethod
    def catalog_for_tools(
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        available_only: bool = False,
    ) -> NeedleToolCatalog:
        """Build the same registered catalog used by the production selector."""
        return NeedleHandler.catalog_for_tools(
            tools,
            schemas=schemas,
            available_only=available_only,
        )

    def select_one_tool(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = 96,
    ) -> NeedleToolCall:
        """Select the registered speak tool without model or network work."""
        del query, tools, max_new_tokens
        return {"name": "speak", "arguments": {"text": "hello"}}

    def close(self) -> None:
        """Match the lifecycle method of a loaded Needle handler."""


class TestAgentCoreIntegration(CeluneTestCase):
    """Run a complete routed task through a real lightweight Celune core."""

    def _make_core(
        self,
        *,
        agent_tool_selector: Optional[AgentToolSelector] = None,
    ) -> Celune:
        """Create the core with model, audio-device, and glow work replaced by fakes."""
        cuda_patch = mock.patch(
            "celune.vram.torch.cuda.is_available", return_value=False
        )
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={"mode": "agent", "vram": "xhigh"},
                tts_backend=FakeBackend,
                agent_tool_selector=agent_tool_selector,
            )
        self.addCleanup(core.close)
        return core

    def test_production_catalog_contains_typed_offline_tools(self) -> None:
        """Expose the complete Appendix B catalog with permission metadata."""
        schemas = production_agent_tool_schemas()
        expected = {
            "read_agent_status",
            "query_status",
            "query_capabilities",
            "query_models",
            "query_locks",
            "query_audio_state",
            "query_agent_task",
            "run_health_check",
            "speak",
            "stop_speech",
            "pause_speech",
            "resume_speech",
            "set_voice",
            "set_voice_prompt",
            "set_playback_speed",
            "set_reverb",
            "clear_speech_queue",
            "set_character",
            "query_character",
            "set_conversation_mode",
            "set_agent_mode",
            "sleep",
            "wake",
            "remember",
            "recall",
            "forget",
            "clear_recent_context",
            "summarize_context",
            "pause_task",
            "resume_task",
            "cancel_task",
            "query_task",
            "query_task_history",
        }
        assert set(schemas) == expected
        for schema in schemas.values():
            assert schema.description
            assert schema.display_name
            if schema.behavior == AgentToolBehavior.MUTATING:
                assert schema.approval_required

    def test_local_management_registry_is_opt_in_and_warns(self) -> None:
        """Local-management tools appear only when explicitly enabled."""
        logs: list[str] = []

        def capture_log(
            msg: str,
            severity: str = "info",
            *,
            loglevel: LogLevel = "info",
        ) -> None:
            """Capture startup diagnostics for the ordering assertion."""
            del severity, loglevel
            logs.append(msg)

        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={
                    "mode": "agent",
                    "vram": "xhigh",
                    "agent": {"fs_tools": True},
                },
                tts_backend=FakeBackend,
                log_callback=capture_log,
            )
        self.addCleanup(core.close)
        assert "local_system_info" in core._agent_tool_schemas

        with (
            mock.patch.object(core, "load_available_voices", return_value=False),
            mock.patch.object(core, "fatal"),
        ):
            assert not core.load(skip_runtime_check=True)

        assert logs
        assert logs[0].startswith("Celune ")
        assert any("UNSANDBOXED AGENT" in message for message in logs)
        assert (
            next(
                (
                    index
                    for index, message in enumerate(logs)
                    if "UNSANDBOXED AGENT" in message
                )
            )
            > 0
        )
        assert "local_system_info" in production_agent_tool_schemas(
            include_local_management=True
        )

    def test_query_status_executes_against_the_real_core(self) -> None:
        """Execute a read-only catalog tool against the core-owned task context."""
        core = self._make_core()
        task = core.agent_runtime.create_task(AgentRequest("query status"))
        tool = next(tool for tool in core._agent_tools if tool.name == "query_status")
        result = cast(
            ToolExecutionResult,
            tool.execute(
                {"id": "status-call", "name": "query_status", "arguments": {}},
                core.agent_runtime.get_context(task.task_id),
            ),
        )
        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = result["output"]
        assert isinstance(output, dict)
        assert output["mode"] == "agent"
        core.agent_runtime.cancel_task(task.task_id)

    def test_query_models_reports_configured_persona_model_id(self) -> None:
        """Report the configured Persona model ID through the production tool."""
        core = self._make_core()
        core.config["persona"] = {"model_id": "fixture/persona-custom"}
        task = core.agent_runtime.create_task(AgentRequest("query models"))
        tool = next(tool for tool in core._agent_tools if tool.name == "query_models")

        result = cast(
            ToolExecutionResult,
            tool.execute(
                {"id": "models-call", "name": "query_models", "arguments": {}},
                core.agent_runtime.get_context(task.task_id),
            ),
        )

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = result["output"]
        assert isinstance(output, dict)
        assert output["persona_model"] == "fixture/persona-custom"
        core.agent_runtime.cancel_task(task.task_id)

    def test_query_models_uses_the_default_persona_model_without_custom_config(
        self,
    ) -> None:
        """Report the canonical default Persona model when none is configured."""
        core = self._make_core()
        task = core.agent_runtime.create_task(AgentRequest("query models"))
        tool = next(tool for tool in core._agent_tools if tool.name == "query_models")

        result = cast(
            ToolExecutionResult,
            tool.execute(
                {"id": "models-call", "name": "query_models", "arguments": {}},
                core.agent_runtime.get_context(task.task_id),
            ),
        )

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = result["output"]
        assert isinstance(output, dict)
        assert output["persona_model"] == PERSONA_DEFAULT_MODEL_ID
        core.agent_runtime.cancel_task(task.task_id)

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
        assert route.route.value == "task"
        assert route.task_request is not None

        with mock.patch("celune.celune.deliver_persona_response", return_value=True):
            core._run_agent_route(route)
        task = runtime.get_active_task("default")
        assert task is not None
        assert task.state == AgentTaskState.AWAITING_APPROVAL
        assert not executions

        approval = runtime.get_pending_approval(task.task_id)
        assert approval is not None

        approval_route = core.route_input(
            "That is approved; continue.", persona_ready=True
        )
        assert approval_route.route.value == "approval_response"
        with mock.patch("celune.celune.deliver_persona_response", return_value=True):
            core._run_agent_route(approval_route)

        assert task.state == AgentTaskState.COMPLETED
        assert executions == [approval.tool_call]
        assert task.iterations == 1
        assert task.completion_metadata is not None
        assert runtime.get_active_task("default") is None
        assert event_names.count("approval") == 1
        assert event_names.count("finished") == 1
        assert event_names.index("approval") < event_names.index("finished")

    def test_core_owns_production_path_from_persona_to_speech(self) -> None:
        """Run one safe task through core-owned Persona, Needle, and speech paths."""
        cuda_patch = mock.patch(
            "celune.vram.torch.cuda.is_available", return_value=False
        )
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        selector = _NeedleFixture()
        persona = _PersonaFixture()
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={"mode": "agent", "vram": "xhigh"},
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

        assert core.agent_runtime._planner is not None
        assert core.agent_runtime._tool_result_handler is not None
        assert {tool.name for tool in core.agent_runtime.tools} == set(
            production_agent_tool_schemas()
        )

        conversation = core.route_input("Hello.", persona_ready=False)
        assert conversation.route.value == "conversation"
        assert core.agent_runtime.get_active_task("default") is None
        assert not persona.requests

        route = core.route_input(
            "Could you verify the current agent status?", persona_ready=True
        )
        assert route.route.value == "task"
        assert route.task_request is not None
        outputs: list[AgentOutput] = []
        original_run = core.agent_runtime.run

        def record_run(
            request: AgentRequest,
            *,
            callback: Optional[AgentResponseCallback] = None,
        ) -> AgentOutput:
            """Capture the runtime output while preserving the real call."""
            output = original_run(request, callback=callback)
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
        assert outputs[0]["response"] == "The agent task completed successfully."
        assert outputs[0]["end"]

        metadata = route.routing_metadata
        assert isinstance(metadata, dict)
        task_id = metadata.get("task_id")
        assert isinstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        assert task.state == AgentTaskState.COMPLETED
        assert task.iterations == 1
        assert selector.intents == ["Read the current agent status."]
        assert len(persona.requests) == 3
        second_system = persona.requests[2]["system"]
        assert isinstance(second_system, str)
        assert "Last tool result" in second_system
        assert "succeeded" in second_system
        assert execute.call_count == 1
        assert delivery_mock.call_count == 2
        assert speech_mock.call_count == 2
        assert (
            speech_mock.call_args.kwargs["display_text"]
            == "The agent task completed successfully."
        )
        assert core.agent_runtime.get_active_task("default") is None

    def test_core_logs_one_typed_route_before_downstream_processing(self) -> None:
        """Expose semantic routing diagnostics only through the debug log gate."""
        core = self._make_core()
        logs: list[str] = []
        core.log_level = "debug"
        core.log_callback = lambda message, *_args, **_kwargs: logs.append(message)
        core.vision = _RoutingFixture(
            '{"classification":"task","route":"task",'
            '"confidence":0.98,"intent":"inspect_status"}'
        )

        route = core.route_input("Check the current agent status.", persona_ready=True)

        assert route.route.value == "task"
        assert (
            logs[-1]
            == "[ROUTE] request=Check the current agent status. type=agent intent=inspect_status confidence=98%"
        )

    def test_core_route_log_uses_say_for_classifier_fallback(self) -> None:
        """Mark an unavailable semantic route as a direct speech fallback."""
        core = self._make_core()
        logs: list[str] = []
        core.log_level = "debug"
        core.log_callback = lambda message, *_args, **_kwargs: logs.append(message)

        route = core.route_input("Please handle this", persona_ready=False)

        assert route.failure is not None
        assert logs[-1] == "[ROUTE] request=Please handle this type=say"

    def test_core_route_log_omits_agent_type_when_agent_mode_is_disabled(self) -> None:
        """Keep disabled agent mode on the ordinary Persona route."""
        core = self._make_core()
        logs: list[str] = []
        core.mode = "converse"
        core.log_level = "debug"
        core.log_callback = lambda message, *_args, **_kwargs: logs.append(message)

        route = core.route_input("Check the current agent status.", persona_ready=False)

        assert route.route.value == "conversation"
        assert (
            logs[-1]
            == "[ROUTE] request=Check the current agent status. type=persona confidence=100%"
        )

    def test_failed_agent_task_is_explained_by_active_persona(self) -> None:
        """Use the active Persona voice for a no-tool-match terminal failure."""
        core = self._make_core(agent_tool_selector=lambda _context, _output: None)
        persona = _PersonaFixture()
        persona.responses = [
            _PersonaResponse(
                '{"classification":"task","route":"task",'
                '"confidence":0.98,"task_request":"Inspect the unavailable thing."}'
            ),
            _PersonaResponse("Inspect the unavailable thing."),
            _PersonaResponse("I could not find a suitable way to do that just now."),
        ]
        core.vision = cast(PersonaClient, persona)
        delivered: list[str] = []

        def record_delivery(_engine: Celune, _request: str, response: str) -> bool:
            """Capture the character-generated terminal response."""
            delivered.append(response)
            return True

        route = core.route_input("Inspect the unavailable thing.", persona_ready=True)
        with mock.patch(
            "celune.celune.deliver_persona_response", side_effect=record_delivery
        ):
            assert core._run_agent_route(route)

        metadata = route.routing_metadata
        assert isinstance(metadata, dict)
        task_id = metadata.get("task_id")
        assert isinstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        assert task.state == AgentTaskState.FAILED
        assert task.failure_reason is not None
        assert task.failure_reason.value == "no_available_tools"
        assert delivered[-1:] == [
            "I could not find a suitable way to do that just now."
        ]
        failure_prompt = persona.requests[-1]["system"]
        assert isinstance(failure_prompt, str)
        assert "Failure response instruction" in failure_prompt
        assert "no_available_tools" in failure_prompt

    def test_classification_failure_is_explained_by_active_persona(self) -> None:
        """Use the active Persona voice when intent classification fails."""
        core = self._make_core()
        persona = _PersonaFixture()
        persona.responses = [
            _PersonaResponse("I cannot confidently route that request yet."),
        ]
        core.vision = cast(PersonaClient, persona)
        failure = AgentClassificationFailure(
            AgentClassificationFailureKind.MALFORMED_OUTPUT,
            "classifier returned malformed output",
        )
        route = AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.0,
            reason="classifier_failure",
            route=AgentRoute.CONVERSATION,
            failure=failure,
        )
        delivered: list[str] = []
        with mock.patch(
            "celune.celune.deliver_persona_response",
            side_effect=lambda _engine, _request, response: (
                delivered.append(response) or True
            ),
        ):
            assert core._speak_agent_classification_failure(
                "Please handle this ambiguous request.", failure, route
            )

        assert delivered == ["I cannot confidently route that request yet."]
        failure_prompt = persona.requests[-1]["system"]
        assert isinstance(failure_prompt, str)
        assert "Classification failure" in failure_prompt
        assert "malformed_output" in failure_prompt
        assert persona.requests[-1]["context_space"] == 8192

    def test_typed_permission_approval_tool_and_cancel_failures_use_persona(
        self,
    ) -> None:
        """Use character-generated explanations for common terminal outcomes."""
        core = self._make_core()
        persona = _PersonaFixture()
        responses = {
            "permission": "I could not do that because it is not permitted.",
            "approval": "You declined that action, so I left it undone.",
            "tool": "The selected tool failed before it could finish.",
            "cancel": "I stopped that task when you asked me to.",
        }
        persona.responses = [
            _PersonaResponse(response) for response in responses.values()
        ]
        core.vision = cast(PersonaClient, persona)
        delivered: list[str] = []

        def record_delivery(_engine: Celune, _request: str, response: str) -> bool:
            """Capture one character-generated terminal response."""
            delivered.append(response)
            return True

        with mock.patch(
            "celune.celune.deliver_persona_response", side_effect=record_delivery
        ):
            for label, reason in (
                ("permission", AgentFailureReason.PERMISSION_DENIED),
                ("approval", AgentFailureReason.APPROVAL_DENIED),
                ("tool", AgentFailureReason.TOOL_ERROR),
            ):
                with nullcontext():
                    task = core.agent_runtime.create_task(
                        AgentRequest(
                            f"Run the {label} case.",
                            session=AgentSession(session_id=f"{label}-session"),
                        ),
                        task_id=f"{label}-task",
                    )
                    core.agent_runtime.start_task(task.task_id)
                    core.agent_runtime.classify_task(task.task_id)
                    core.agent_runtime.fail_task(task.task_id, reason, label)
                    route = AgentClassificationResult(
                        classification=AgentInputClassification.TASK,
                        confidence=1.0,
                        route=AgentRoute.TASK_INPUT,
                        routing_metadata={"task_id": task.task_id},
                    )
                    assert core._run_agent_route(route)

            task = core.agent_runtime.create_task(
                AgentRequest(
                    "Run the cancel case.",
                    session=AgentSession(session_id="cancel-session"),
                ),
                task_id="cancel-task",
            )
            core.agent_runtime.start_task(task.task_id)
            core.agent_runtime.classify_task(task.task_id)
            core.agent_runtime.cancel_task(
                task.task_id,
                AgentCancellationReason.USER_REQUEST,
            )
            route = AgentClassificationResult(
                classification=AgentInputClassification.TASK,
                confidence=1.0,
                route=AgentRoute.TASK_INPUT,
                routing_metadata={"task_id": task.task_id},
            )
            assert core._run_agent_route(route)

        assert list(responses.values()) == delivered
        for request, label in zip(persona.requests, responses):
            prompt = request["system"]
            assert isinstance(prompt, str)
            assert "Failure response instruction" in prompt
            assert label in prompt

    def test_terminal_speak_tool_is_the_task_result(self) -> None:
        """Run the real registered speak tool without a duplicate final response."""
        core = self._make_core()
        selector = NeedleToolSelector(
            cast(NeedleHandler, _SpeakNeedleHandler()),
            core._agent_tools,
            schemas=production_agent_tool_schemas(),
        )
        core._agent_needle_selector = selector
        persona = _SpeakPersonaFixture()
        core.vision = cast(PersonaClient, persona)
        speech = mock.patch("celune.pipeline.queue_speech", return_value=True)
        delivery = mock.patch(
            "celune.celune.deliver_persona_response",
            wraps=celune_deliver_persona_response,
        )
        speech_mock = speech.start()
        delivery_mock = delivery.start()
        self.addCleanup(speech.stop)
        self.addCleanup(delivery.stop)

        route = core.route_input("Please say hello.", persona_ready=True)
        assert route.route.value == "task"
        core._run_agent_route(route)
        metadata = route.routing_metadata
        assert isinstance(metadata, dict)
        task_id = metadata.get("task_id")
        assert isinstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        assert task.state == AgentTaskState.AWAITING_APPROVAL

        approval_route = core.route_input("Approved.", persona_ready=True)
        assert approval_route.route.value == "approval_response"
        assert core._run_agent_route(approval_route)

        assert task.state == AgentTaskState.COMPLETED
        result = cast(
            Optional[ToolExecutionResult],
            core.agent_runtime.get_context(task_id).last_tool_result,
        )
        assert result is not None
        assert result["tool_id"] == "speak"
        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert result["end_task"] is True
        assert delivery_mock.call_count == 1
        assert speech_mock.call_count == 2
        assert [call.args[1] for call in speech_mock.call_args_list] == [
            "I will say hello.",
            "hello",
        ]
        assert len(persona.requests) == 3

    def test_needle_loading_failure_is_recorded_and_task_fails_safely(self) -> None:
        """Expose a typed terminal failure when production Needle cannot load."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
            mock.patch("celune.pipeline.queue_speech", return_value=True),
            mock.patch("celune.vram.torch.cuda.is_available", return_value=False),
            mock.patch(
                "celune.celune.NeedleToolSelector.from_pretrained",
                side_effect=RuntimeError("checkpoint unavailable"),
            ),
        ):
            core = Celune(
                config={"mode": "agent", "vram": "xhigh"},
                tts_backend=FakeBackend,
            )
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
        assert isinstance(metadata, dict)
        task_id = metadata.get("task_id")
        assert isinstance(task_id, str)
        task = core.agent_runtime.get_task(task_id)
        assert task.state == AgentTaskState.FAILED
        assert task.failure_reason is not None
        assert task.failure_reason.value == "invalid_tool_call"
        assert "checkpoint unavailable" in (core.agent_needle_error or "")
        assert not core.agent_needle_ready
