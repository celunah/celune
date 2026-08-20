# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the explicit Celune test-mode command hierarchy."""

from __future__ import annotations

import io
import contextlib
from typing import Optional, cast
from unittest import TestCase, mock
from collections.abc import Mapping, Sequence

from celune import entrypoint
from celune.i18n import string
from celune.celune import Celune
from celune.typing.common import JSON
from celune.config import config_log_level
from celune.test_mode import run_agent_test
from celune.persona.impl import PersonaClient
from celune.typing.persona import PersonaClientResponse
from celune.agent.needle import NeedleHandler, NeedleToolSelector
from celune.typing.agent import (
    AgentTool,
    AgentRoute,
    NeedleToolCall,
    AgentToolSchema,
    NeedleToolCatalog,
    AgentInputClassification,
    AgentClassificationResult,
    AgentClassificationFailure,
    AgentClassificationFailureKind,
)

from .support import FakeGlow, FakeBackend


class _TestPersonaClient:
    """Return deterministic model-shaped text for the isolated core test."""

    def __init__(self) -> None:
        self.request_count = 0
        self.requests: list[JSON] = []

    def post(self, json: JSON) -> PersonaClientResponse:
        """Return an action intent followed by a tool-result response."""
        self.requests.append(json)
        self.request_count += 1
        responses = (
            '{"classification":"task","route":"task","confidence":0.98}',
            "Read the current agent status.",
            "The current agent status was read successfully.",
        )
        response = responses[min(self.request_count - 1, len(responses) - 1)]
        return PersonaClientResponse({"text": response})


class _TestNeedleHandler:
    """Use the real Needle selector adapter with deterministic model output."""

    @staticmethod
    def catalog_for_tools(
        tools: Sequence[AgentTool],
        *,
        schemas: Optional[Mapping[str, AgentToolSchema]] = None,
        available_only: bool = False,
    ) -> NeedleToolCatalog:
        """Build the production catalog without loading model weights."""
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
        """Return the one safe tool selected by this controlled adapter."""
        del query, tools, max_new_tokens
        return {"name": "local_current_working_directory", "arguments": {}}

    def close(self) -> None:
        """Release the controlled adapter without external model resources."""


class TestCommandTests(TestCase):
    """Verify parent and child test command dispatch without starting Celune."""

    def test_parent_command_displays_available_modes(self) -> None:
        """The parent command lists the two supported explicit test modes."""
        with contextlib.redirect_stdout(io.StringIO()) as output:
            entrypoint.handle_test([], "celune")

        self.assertIn("Available test modes: ui, agent", output.getvalue())
        self.assertIn("Usage: celune test [ui|agent]", output.getvalue())

    def test_ui_command_dispatches_existing_ui_test_mode(self) -> None:
        """The UI child selects the existing fake-backend startup path."""
        with mock.patch.object(entrypoint, "start") as start:
            entrypoint.handle_test(["ui"], "celune")

        start.assert_called_once_with(testing=True, test_mode="ui")

    def test_agent_command_dispatches_agent_test_mode(self) -> None:
        """The agent child selects the explicit agent workflow path."""
        with mock.patch.object(entrypoint, "start") as start:
            entrypoint.handle_test(["agent"], "celune")

        start.assert_called_once_with(testing=True, test_mode="agent")

    def test_test_mode_accepts_verbose_log_level_override(self) -> None:
        """Forward the verbose override to the selected test runtime."""
        with mock.patch.object(entrypoint, "start") as start:
            entrypoint.handle_test(["agent", "--verbose"], "celune")

        start.assert_called_once_with(
            log_level="verbose",
            testing=True,
            test_mode="agent",
        )

    def test_test_mode_accepts_debug_log_level_override(self) -> None:
        """Forward the debug override to the selected test runtime."""
        with mock.patch.object(entrypoint, "start") as start:
            entrypoint.handle_test(["ui", "--log-level=debug"], "celune")

        start.assert_called_once_with(
            log_level="debug",
            testing=True,
            test_mode="ui",
        )

    def test_agent_test_config_resolves_configured_log_level(self) -> None:
        """Resolve the agent test log level from its loaded configuration."""
        self.assertEqual(
            config_log_level({"log_level": "verbose"}, env_name="__missing__"),
            "verbose",
        )


class TestFinishedLifecycleTests(TestCase):
    """Verify the stopped-but-alive boundary shared by explicit test modes."""

    def _make_core(self) -> Celune:
        """Create a lightweight core for test-mode lifecycle coverage."""
        with (
            mock.patch("celune.celune.AudioRGBGlow", FakeGlow),
            mock.patch("celune.celune.default_loader", return_value=None),
            mock.patch("celune.celune.persona_is_available", return_value=False),
        ):
            core = Celune(
                config={"mode": "agent"},
                tts_backend=FakeBackend,
                backend_mode="agent_test",
            )
        self.addCleanup(core.close)
        return core

    def test_success_is_recorded_once_and_stops_new_work(self) -> None:
        """A successful result leaves the core stopped without closing it."""
        core = self._make_core()
        with (
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "log") as log,
        ):
            result = core.finish_test_mode(
                "ui",
                True,
                task_state="none",
            )

        self.assertEqual(result["success"], True)
        self.assertEqual(core.cur_state, "stopped")
        self.assertTrue(core.test_finished)
        self.assertFalse(core._closed)
        self.assertFalse(core.exit_requested)
        self.assertFalse(
            any(
                args and "Test mode ui succeeded" in args[0]
                for args, _kwargs in log.call_args_list
            )
        )
        self.assertFalse(
            any(
                args and args[0] == string("pipeline.exiting")
                for args, _kwargs in log.call_args_list
            )
        )
        self.assertFalse(core.think("ignored"))
        self.assertFalse(core.say("ignored"))
        with self.assertRaises(RuntimeError):
            core.route_input("Check the current agent status.")
        self.assertIs(core.finish_test_mode("ui", False), result)

    def test_failure_is_recorded_and_explicit_close_remains_available(self) -> None:
        """Failures still stop the engine and allow the explicit shutdown path."""
        core = self._make_core()
        with mock.patch.object(core, "stop_live_audio"):
            result = core.finish_test_mode(
                "agent",
                False,
                task_state="failed",
                detail="controlled failure",
            )

        payload = result
        self.assertFalse(payload["success"])
        self.assertEqual(payload["detail"], "controlled failure")
        self.assertEqual(core.cur_state, "stopped")
        core.close()
        self.assertTrue(core._closed)

    def test_cleanup_exception_still_reaches_stopped_state(self) -> None:
        """A cleanup failure is recorded as a failed test without stranding the core."""
        core = self._make_core()
        with mock.patch.object(
            core,
            "stop_live_audio",
            side_effect=RuntimeError("microphone cleanup failed"),
        ):
            result = core.finish_test_mode("ui", True)

        self.assertFalse(result["success"])
        self.assertEqual(core.cur_state, "stopped")

    def test_agent_workflow_uses_the_real_core_runtime_boundaries(self) -> None:
        """The controlled agent task completes through routing and production tools."""
        core = self._make_core()
        self.assertEqual(core.backend_mode, "agent_test")
        self.assertEqual(
            tuple(tool.name for tool in core._agent_tools),
            ("local_current_working_directory",),
        )
        persona = _TestPersonaClient()
        core.vision = cast(PersonaClient, persona)
        core.persona_ready = True
        selector = NeedleToolSelector(
            cast(NeedleHandler, _TestNeedleHandler()),
            core._agent_tools,
            schemas=core._agent_tool_schemas,
        )
        with mock.patch.object(
            NeedleToolSelector,
            "from_pretrained",
            return_value=selector,
        ) as load_selector:
            result = run_agent_test(core)
        load_selector.assert_called_once()

        payload = result
        self.assertTrue(payload["success"])
        self.assertEqual(payload["mode"], "agent")
        self.assertEqual(payload["engine_state"], "stopped")
        self.assertEqual(payload["task_state"], "completed")
        self.assertEqual(
            persona.requests[0]["user"],
            "Check the current working directory and report the result.",
        )
        detail = payload["detail"]
        self.assertIsInstance(detail, str)
        assert isinstance(detail, str)
        self.assertIn("tool=local_current_working_directory", detail)
        self.assertIn("status=succeeded", detail)
        self.assertEqual(core.cur_state, "stopped")
        self.assertFalse(core.say("queued after test"))

    def test_agent_test_reports_no_task_detected(self) -> None:
        """Distinguish an ordinary conversation result from a test crash."""
        core = self._make_core()
        route = AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.98,
            route=AgentRoute.CONVERSATION,
        )
        with (
            mock.patch("celune.test_mode._wait_for_persona"),
            mock.patch("celune.test_mode._start_agent_test_pipeline"),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        self.assertFalse(result["success"])
        self.assertEqual(result["detail"], "no task detected")

    def test_agent_test_reports_classification_failure(self) -> None:
        """Preserve the typed classifier failure category in the test result."""
        core = self._make_core()
        route = AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.0,
            route=AgentRoute.CONVERSATION,
            failure=AgentClassificationFailure(
                AgentClassificationFailureKind.MALFORMED_OUTPUT,
                "invalid JSON",
            ),
        )
        with (
            mock.patch("celune.test_mode._wait_for_persona"),
            mock.patch("celune.test_mode._start_agent_test_pipeline"),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        self.assertFalse(result["success"])
        self.assertEqual(result["detail"], "classification failed: malformed_output")

    def test_agent_test_reports_task_detected_but_not_started(self) -> None:
        """Distinguish a task route without a runtime task identity."""
        core = self._make_core()
        route = AgentClassificationResult(
            classification=AgentInputClassification.TASK,
            confidence=0.9,
            task_request=core._agent_router._make_request(
                "Check the current working directory."
            ),
            route=AgentRoute.TASK,
            routing_metadata={},
        )
        with (
            mock.patch("celune.test_mode._wait_for_persona"),
            mock.patch("celune.test_mode._start_agent_test_pipeline"),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        self.assertFalse(result["success"])
        self.assertEqual(result["detail"], "task detected but not started")
