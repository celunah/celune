# SPDX-License-Identifier: MIT
"""Tests for the typed Needle selector adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, cast
from unittest import TestCase, mock

import torch

from celune.agent import (
    AgentApprovalDecision,
    AgentApprovalResponse,
    AgentContext,
    AgentFailureReason,
    AgentInterruption,
    AgentInterruptionKind,
    AgentRequest,
    AgentRuntime,
    AgentSession,
    AgentTool,
    AgentToolArgumentSchema,
    AgentToolBehavior,
    AgentToolDangerLevel,
    AgentToolExecutionStatus,
    AgentToolSchema,
    AgentToolValueType,
    NeedleHandler,
    NeedleSelectionError,
    NeedleToolParameterSpec,
    NeedleToolSelector,
    NeedleTokenizer,
    ValidatedToolCall,
)
from celune.agent.needle import _parse_single_selection
from celune.agent.needle_model import NeedleModel
from celune.persona.capabilities import PersonaCapabilities
from celune.typing.agent import (
    AgentOutput,
    NeedleToolCall,
    NeedleToolCatalog,
    ToolCall,
    ToolExecutionResult,
)
from celune.typing.common import JSONSerializable


def _tool(name: str) -> AgentTool:
    """Build a minimal registered tool double."""
    return cast(
        AgentTool,
        SimpleNamespace(name=name, description=f"{name} description"),
    )


def _schemas() -> dict[str, AgentToolSchema]:
    """Build schemas covering required, optional, and unavailable tools."""
    return {
        "SetTimer": AgentToolSchema(
            tool_id="set_timer",
            display_name="Set timer",
            description="Set a timer.",
            arguments=(
                AgentToolArgumentSchema(
                    "minutes",
                    AgentToolValueType.INTEGER,
                ),
                AgentToolArgumentSchema(
                    "label",
                    AgentToolValueType.STRING,
                    required=False,
                ),
            ),
            behavior=AgentToolBehavior.MUTATING,
            danger=AgentToolDangerLevel.MEDIUM,
            approval_required=True,
        ),
        "DisabledTool": AgentToolSchema(
            tool_id="disabled_tool",
            display_name="Disabled tool",
            description="Unavailable tool.",
            available=False,
        ),
    }


def _context() -> AgentContext:
    """Build one selector context without a running task."""
    return AgentContext(
        request=AgentRequest(
            request="Set the requested timer.",
            session=AgentSession(session_id="session-1"),
        ),
        mode="agent",
        persona_capabilities=PersonaCapabilities(),
    )


def _output(intent: Optional[str] = "Set a timer for five minutes") -> AgentOutput:
    """Build one planner output carrying the natural-language action intent."""
    return {
        "tool_call": None,
        "response": intent,
        "end": False,
        "paused": False,
    }


class _FakeNeedleHandler:
    """Deterministic handler double that records the adapter boundary."""

    def __init__(self, selection: NeedleToolCall) -> None:
        self.selection = selection
        self.query: Optional[str] = None
        self.catalog: NeedleToolCatalog = []

    def catalog_for_tools(
        self,
        tools: tuple[AgentTool, ...],
        *,
        schemas: dict[str, AgentToolSchema],
        available_only: bool,
    ) -> NeedleToolCatalog:
        """Record and build the catalog through the production helper."""
        self.catalog = NeedleHandler.catalog_for_tools(
            tools,
            schemas=schemas,
            available_only=available_only,
        )
        return self.catalog

    def select_one_tool(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int,
    ) -> NeedleToolCall:
        """Record the action intent and return a deterministic model result."""
        self.query = query
        self.catalog = tools
        if max_new_tokens <= 0:
            raise AssertionError("invalid test generation limit")
        return self.selection


class NeedleSelectorTests(TestCase):
    """Verify strict selection, schema validation, and metadata preservation."""

    def test_action_intent_and_available_schema_reach_needle(self) -> None:
        """Pass the planner intent and current available catalog to Needle."""
        tools = (_tool("SetTimer"), _tool("DisabledTool"))
        selection = _FakeNeedleHandler(
            {"name": "SetTimer", "arguments": {"minutes": 5}}
        )
        selector = NeedleToolSelector(
            cast(NeedleHandler, selection),
            tools,
            schemas=_schemas(),
        )

        result = selector(_context(), _output())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(selection.query, "Set a timer for five minutes")
        self.assertEqual([item["name"] for item in selection.catalog], ["SetTimer"])
        minutes = cast(
            NeedleToolParameterSpec,
            selection.catalog[0]["parameters"]["minutes"],
        )
        self.assertEqual(minutes["type"], "integer")
        self.assertEqual(result["name"], "SetTimer")
        validated = cast(ValidatedToolCall, result)
        self.assertEqual(validated["tool_id"], "set_timer")
        self.assertEqual(validated["behavior"], AgentToolBehavior.MUTATING)
        self.assertEqual(validated["danger"], AgentToolDangerLevel.MEDIUM)
        self.assertTrue(validated["approval_required"])

    def test_schema_mapping_can_be_keyed_by_tool_id(self) -> None:
        """Resolve a Phase 1 schema by its canonical tool identifier."""
        handler = _FakeNeedleHandler({"name": "SetTimer", "arguments": {"minutes": 5}})
        schema = _schemas()["SetTimer"]
        selector = NeedleToolSelector(
            cast(NeedleHandler, handler),
            (_tool("SetTimer"),),
            schemas={schema.tool_id: schema},
        )

        result = selector(_context(), _output())

        assert result is not None
        self.assertEqual(cast(ValidatedToolCall, result)["tool_id"], "set_timer")
        minutes = cast(
            NeedleToolParameterSpec,
            handler.catalog[0]["parameters"]["minutes"],
        )
        self.assertEqual(minutes["type"], "integer")

    def test_invalid_selection_shapes_and_arguments_are_rejected(self) -> None:
        """Reject unknown tools, unavailable tools, and schema-invalid arguments."""
        cases = (
            ("Unknown", {"minutes": 5}),
            ("DisabledTool", {}),
            ("SetTimer", {"minutes": "five"}),
            ("SetTimer", {"label": "missing minutes"}),
            ("SetTimer", {"minutes": 5, "unexpected": True}),
        )
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                handler = _FakeNeedleHandler(
                    {
                        "name": name,
                        "arguments": cast(
                            dict[str, JSONSerializable],
                            arguments,
                        ),
                    }
                )
                selector = NeedleToolSelector(
                    cast(NeedleHandler, handler),
                    (_tool("SetTimer"), _tool("DisabledTool")),
                    schemas=_schemas(),
                )
                with self.assertRaises(NeedleSelectionError):
                    selector(_context(), _output())

    def test_empty_intent_and_strict_json_shapes_are_rejected(self) -> None:
        """Reject an empty planner intent and malformed or multiple JSON calls."""
        handler = _FakeNeedleHandler({"name": "SetTimer", "arguments": {"minutes": 5}})
        selector = NeedleToolSelector(
            cast(NeedleHandler, handler),
            (_tool("SetTimer"),),
            schemas={"SetTimer": _schemas()["SetTimer"]},
        )
        with self.assertRaises(NeedleSelectionError):
            selector(_context(), _output(" "))
        with self.assertRaises(NeedleSelectionError):
            _parse_single_selection("not json", {})
        with self.assertRaises(NeedleSelectionError):
            _parse_single_selection(
                '[{"name":"one","arguments":{}},{"name":"two","arguments":{}}]',
                {},
            )

    def test_handler_uses_tokenizer_for_strict_single_call_selection(self) -> None:
        """Use the handler tokenizer and restore the canonical tool name."""

        class FakeTokenizer:
            """Minimal tokenizer double for the handler boundary."""

            def __init__(self) -> None:
                """Record tokenizer inputs for the handler assertion."""
                self.encoded: list[str] = []

            def encode(self, value: str) -> list[int]:
                """Return deterministic token IDs for one input string."""
                self.encoded.append(value)
                return [2, 3]

            def decode(self, _values: list[int]) -> str:
                """Return one deterministic JSON tool call."""
                return '{"name":"set_timer","arguments":{"minutes":5}}'

        class FakeModel:
            """Minimal model double for the handler boundary."""

            config = SimpleNamespace(max_seq_len=16)

            def generate(
                self,
                input_ids: torch.Tensor,
                max_new_tokens: int,
            ) -> torch.Tensor:
                """Return deterministic generated token IDs."""
                if max_new_tokens != 96 or input_ids.shape[0] != 1:
                    raise AssertionError("unexpected Needle generation request")
                return torch.tensor([[1, 2]])

        tokenizer = FakeTokenizer()
        handler = NeedleHandler(
            cast(NeedleModel, FakeModel()),
            cast(NeedleTokenizer, tokenizer),
            torch.device("cpu"),
        )
        result = handler.select_one_tool(
            "Set a timer.",
            [{"name": "SetTimer", "parameters": {}}],
        )

        self.assertEqual(result["name"], "SetTimer")
        self.assertEqual(len(tokenizer.encoded), 2)

    def test_from_pretrained_loads_through_the_verified_handler(self) -> None:
        """Build the adapter through the pinned handler preparation API."""
        fake_handler = cast(
            NeedleHandler, _FakeNeedleHandler({"name": "SetTimer", "arguments": {}})
        )
        with mock.patch.object(
            NeedleHandler,
            "from_pretrained",
            return_value=fake_handler,
        ) as load:
            selector = NeedleToolSelector.from_pretrained(
                (_tool("SetTimer"),),
                schemas={
                    "SetTimer": AgentToolSchema(
                        "set_timer", "Set timer", "Set a timer."
                    )
                },
                revision="revision-1",
            )

        self.assertIs(selector.handler, fake_handler)
        load.assert_called_once_with(
            model_id="Cactus-Compute/needle",
            device=None,
            cache_dir=None,
            revision="revision-1",
            source_filename="model.safetensors",
            pickle_converter=None,
        )

    def test_valid_call_reaches_executor_once_and_never_executes_directly(self) -> None:
        """Keep execution in AgentRuntime after one validated Needle selection."""
        handler = _FakeNeedleHandler({"name": "SetTimer", "arguments": {"minutes": 5}})
        selector = NeedleToolSelector(
            cast(NeedleHandler, handler),
            (_tool("SetTimer"),),
            schemas={"SetTimer": _schemas()["SetTimer"]},
        )
        calls: list[ValidatedToolCall] = []

        def planner(_context: AgentContext) -> AgentOutput:
            """Return one Persona action intent."""
            return {
                "tool_call": {"id": "planned", "name": "SetTimer", "arguments": {}},
                "response": "Set a timer for five minutes",
                "end": False,
                "paused": False,
            }

        def execute(_context: AgentContext, call: ToolCall) -> ToolExecutionResult:
            """Record the call at the existing executor boundary."""
            calls.append(cast(ValidatedToolCall, call))
            return {
                "tool_call_id": call["id"],
                "output": {"ok": True},
                "error": None,
                "tool_id": "set_timer",
                "status": AgentToolExecutionStatus.SUCCEEDED,
            }

        runtime = AgentRuntime(
            tools=(_tool("SetTimer"),),
            planner=planner,
            tool_selector=selector,
            tool_executor=execute,
            tool_result_handler=lambda _context, _result: {
                "tool_call": None,
                "response": "Timer set.",
                "end": True,
                "paused": False,
            },
        )
        task = runtime.create_task(
            AgentRequest(
                request="Set a timer.",
                session=AgentSession(session_id="session-1"),
            ),
            task_id="task-1",
        )

        result = runtime.run(task.request)

        self.assertEqual(task.failure_reason, None)
        self.assertEqual(task.state.value, "awaiting_approval")
        self.assertEqual(len(calls), 0)
        approval = runtime.get_pending_approval(task.task_id)
        self.assertIsNotNone(approval)
        assert approval is not None
        runtime.respond_to_approval(
            task.task_id,
            AgentApprovalResponse(approval.request_id, AgentApprovalDecision.APPROVED),
        )
        result = runtime.run(task.request)

        self.assertEqual(task.state.value, "completed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "SetTimer")
        self.assertEqual(calls[0]["tool_id"], "set_timer")
        self.assertTrue(calls[0]["approval_required"])
        self.assertEqual(result["response"], "Timer set.")

    def test_needle_failure_becomes_typed_runtime_failure(self) -> None:
        """Convert selector failures into the existing invalid-call reason."""

        class FailingHandler(_FakeNeedleHandler):
            """Handler double that fails during selection."""

            def select_one_tool(
                self,
                query: str,
                tools: NeedleToolCatalog,
                max_new_tokens: int,
            ) -> NeedleToolCall:
                """Raise the typed selection failure."""
                raise NeedleSelectionError("malformed output")

        selector = NeedleToolSelector(
            cast(
                NeedleHandler,
                FailingHandler({"name": "SetTimer", "arguments": {}}),
            ),
            (_tool("SetTimer"),),
            schemas={"SetTimer": _schemas()["SetTimer"]},
        )
        runtime = AgentRuntime(
            planner=lambda _context: {
                "tool_call": {"id": "planned", "name": "SetTimer", "arguments": {}},
                "response": "Set a timer.",
                "end": False,
                "paused": False,
            },
            tool_selector=selector,
        )
        task = runtime.create_task(
            AgentRequest(request="Set a timer."),
            task_id="task-1",
        )

        runtime.run(task.request)

        self.assertEqual(task.failure_reason, AgentFailureReason.INVALID_TOOL_CALL)
        self.assertEqual(task.state.value, "failed")

    def test_cancellation_and_interruption_during_selection_skip_execution(
        self,
    ) -> None:
        """Preserve Phase 4 cancellation and interruption semantics at Needle."""
        for action in ("cancel", "interrupt"):
            with self.subTest(action=action):
                runtime: AgentRuntime

                class InterruptingHandler(_FakeNeedleHandler):
                    """Handler double that changes the active task state."""

                    def __init__(self, action_name: str) -> None:
                        super().__init__(
                            {"name": "SetTimer", "arguments": {"minutes": 5}}
                        )
                        self.action_name = action_name
                        self.runtime: Optional[AgentRuntime] = None

                    def select_one_tool(
                        self,
                        query: str,
                        tools: NeedleToolCatalog,
                        max_new_tokens: int,
                    ) -> NeedleToolCall:
                        """Cancel or interrupt before returning a model call."""
                        assert self.runtime is not None
                        task = self.runtime.get_active_task("session-1")
                        assert task is not None
                        if self.action_name == "cancel":
                            self.runtime.cancel_task(task.task_id)
                        else:
                            self.runtime.interrupt_task(
                                task.task_id,
                                AgentInterruption(
                                    AgentInterruptionKind.USER_INTERRUPT,
                                ),
                            )
                        return {"name": "SetTimer", "arguments": {"minutes": 5}}

                interrupting_handler = InterruptingHandler(action)
                selector = NeedleToolSelector(
                    cast(NeedleHandler, interrupting_handler),
                    (_tool("SetTimer"),),
                    schemas={"SetTimer": _schemas()["SetTimer"]},
                )
                executions = 0

                def execute(
                    _context: AgentContext, _call: ToolCall
                ) -> ToolExecutionResult:
                    """Fail the test if cancellation reaches tool execution."""
                    nonlocal executions
                    executions += 1
                    return {
                        "tool_call_id": _call["id"],
                        "output": None,
                        "error": None,
                        "tool_id": "set_timer",
                        "status": AgentToolExecutionStatus.SUCCEEDED,
                    }

                runtime = AgentRuntime(
                    planner=lambda _context: {
                        "tool_call": {
                            "id": "planned",
                            "name": "SetTimer",
                            "arguments": {},
                        },
                        "response": "Set a timer.",
                        "end": False,
                        "paused": False,
                    },
                    tool_selector=selector,
                    tool_executor=execute,
                )
                interrupting_handler.runtime = runtime
                task = runtime.create_task(
                    AgentRequest(
                        request="Set a timer.",
                        session=AgentSession(session_id="session-1"),
                    ),
                    task_id="task-1",
                )

                result = runtime.run(task.request)

                self.assertEqual(executions, 0)
                if action == "cancel":
                    self.assertEqual(task.state.value, "cancelled")
                else:
                    self.assertTrue(result["paused"])
                    self.assertEqual(task.state.value, "interrupted")
