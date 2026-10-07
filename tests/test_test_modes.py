# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for the explicit Celune test-mode command hierarchy."""

from __future__ import annotations

import io
import threading
import contextlib
from types import SimpleNamespace
from typing import ClassVar, Optional, cast
from unittest import mock
from collections.abc import Mapping, Callable, Sequence

import pytest
from textual.widgets import TextArea

from celune import entrypoint
from celune.i18n import string
from celune.test import run_agent_test, _agent_test_succeeded
from celune.celune import Celune
from celune.config import config_log_level
from celune.ui.app import CeluneUI
from celune.agent.needle import NeedleHandler, NeedleToolSelector
from celune.persona.impl import PersonaClient
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
from celune.typing.common import JSON
from celune.typing.persona import PersonaClientResponse

from .support import FakeGlow, FakeBackend, CeluneTestCase


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

    def select_tools(
        self,
        query: str,
        tools: NeedleToolCatalog,
        max_new_tokens: int = 96,
    ) -> list[NeedleToolCall]:
        """Return the one safe tool selected by this controlled adapter."""
        del query, tools, max_new_tokens
        return [{"name": "local_current_working_directory", "arguments": {}}]

    def close(self) -> None:
        """Release the controlled adapter without external model resources."""


class TestCommandTests:
    """Verify parent and child test command dispatch without starting Celune."""

    def test_parent_command_displays_available_modes(self) -> None:
        """The parent command lists the two supported explicit test modes."""
        with contextlib.redirect_stdout(io.StringIO()) as output:
            entrypoint.handle_test([], "celune")

        assert "Available test modes: ui, agent" in output.getvalue()
        assert "Usage: celune test [ui|agent]" in output.getvalue()

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
        assert (
            config_log_level({"log_level": "verbose"}, env_name="__missing__")
            == "verbose"
        )

    def test_agent_test_exit_status_requires_a_successful_report(self) -> None:
        """Missing and failed diagnostic reports must map to a failure exit."""
        assert not _agent_test_succeeded(None)
        assert not _agent_test_succeeded({"success": False})
        assert _agent_test_succeeded({"success": True})

    def test_failed_agent_report_exits_after_the_test_ui_returns(self) -> None:
        """Return the standard failure code after the report UI has closed."""

        class TestCore:
            """Hold the diagnostic result from the UI completion callback."""

            test_result: Optional[JSON] = None

            def __init__(self, **_kwargs: object) -> None:
                self.test_result = None

        class TestUI:
            """Run the callback and record when the test UI has returned."""

            instances: ClassVar[list[object]] = []

            def __init__(
                self,
                *,
                startup_messages: list[str],
                test_completion_callback: Callable[
                    [TestCore, bool, Optional[str]], None
                ],
            ) -> None:
                del startup_messages
                self.test_completion_callback = test_completion_callback
                self.return_code = 0
                self.celune: Optional[TestCore] = None
                self.closed = False
                self.instances.append(self)

            @staticmethod
            def receive_startup_diagnostic(_message: str) -> None:
                """Accept launcher diagnostics without rendering them."""

            @staticmethod
            def prepare_theme() -> None:
                """Provide the theme setup boundary used by the real UI."""

            def run(self) -> None:
                """Invoke completion before marking the UI returned."""
                assert self.celune is not None
                self.test_completion_callback(self.celune, True, None)
                self.closed = True

        runtime = SimpleNamespace(
            Celune=TestCore,
            ExitCodes=entrypoint.EXIT_CODES,
        )

        def fail_report(core: TestCore, **_kwargs: object) -> JSON:
            """Record a failed result in the fake core after UI startup."""
            core.test_result = {"success": False}
            return core.test_result

        TestUI.instances.clear()
        with (
            mock.patch("celune.watchdog.start_watchdog"),
            mock.patch("celune.entrypoint._load_runtime", return_value=runtime),
            mock.patch("celune.entrypoint._load_core_runtime", return_value=runtime),
            mock.patch(
                "celune.entrypoint._load_test_runtime_config",
                return_value=({}, None),
            ),
            mock.patch("celune.entrypoint.migrate_legacy_app_data"),
            mock.patch("celune.entrypoint._print_startup_diagnostic"),
            mock.patch("celune.ui.CeluneUI", TestUI),
            mock.patch("celune.test.run_agent_test", side_effect=fail_report),
            pytest.raises(SystemExit) as exit_info,
        ):
            entrypoint.start(testing=True, test_mode="agent")

        assert exit_info.value.code == entrypoint.EXIT_CODES.EXIT_FAILURE.value
        assert isinstance(TestUI.instances[0], TestUI)
        assert TestUI.instances[0].closed


class TestFinishedLifecycleTests(CeluneTestCase):
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

        assert result["success"] is True
        assert core.cur_state == "stopped"
        assert core.test_finished
        assert not core._closed
        assert not core.exit_requested
        assert not (
            any(
                (
                    args and "Test mode ui succeeded" in args[0]
                    for args, _kwargs in log.call_args_list
                )
            )
        )
        assert not (
            any(
                (
                    args and args[0] == string("pipeline.exiting")
                    for args, _kwargs in log.call_args_list
                )
            )
        )
        assert not core.think("ignored")
        assert not core.say("ignored")
        with pytest.raises(RuntimeError):
            core.route_input("Check the current agent status.")
        assert core.finish_test_mode("ui", False) is result

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
        assert not payload["success"]
        assert payload["detail"] == "controlled failure"
        assert core.cur_state == "stopped"
        core.close()
        assert core._closed

    def test_agent_report_totals_include_failures_and_skips(self) -> None:
        """Summarize every check and make any failed check fail the report."""
        core = self._make_core()
        checks: list[JSON] = [
            {"name": "catalog", "status": "passed", "detail": "covered"},
            {"name": "optional", "status": "skipped", "detail": "disabled"},
            {"name": "runtime", "status": "failed", "detail": "controlled"},
        ]
        with (
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "log") as log,
        ):
            result = core.finish_test_mode("agent", True, checks=checks)

        assert result["success"] is False
        summary = cast(JSON, result["summary"])
        assert summary == {"passed": 1, "failed": 1, "skipped": 1}
        messages = [args[0] for args, _kwargs in log.call_args_list if args]
        assert (
            string(
                "test.agent_report_summary",
                passed=1,
                failed=1,
                skipped=1,
            )
            in messages
        )

    def test_cleanup_exception_still_reaches_stopped_state(self) -> None:
        """A cleanup failure is recorded as a failed test without stranding the core."""
        core = self._make_core()
        with mock.patch.object(
            core,
            "stop_live_audio",
            side_effect=RuntimeError("microphone cleanup failed"),
        ):
            result = core.finish_test_mode("ui", True)

        assert not result["success"]
        assert core.cur_state == "stopped"

    def test_agent_workflow_uses_the_real_core_runtime_boundaries(self) -> None:
        """The controlled agent task completes through routing and production tools."""
        core = self._make_core()
        assert core.backend_mode == "agent_test"
        assert tuple(tool.name for tool in core._agent_tools) == (
            "local_current_working_directory",
        )
        persona = _TestPersonaClient()
        core.vision = cast(PersonaClient, persona)
        core.persona_ready = True
        selector = NeedleToolSelector(
            cast(NeedleHandler, _TestNeedleHandler()),
            core._agent_tools,
            schemas=core._agent_tool_schemas,
        )
        with (
            mock.patch("celune.test.run_agent_feature_checks", return_value=[]),
            mock.patch.object(
                NeedleToolSelector,
                "from_pretrained",
                return_value=selector,
            ) as load_selector,
        ):
            result = run_agent_test(core)
        load_selector.assert_called_once()

        payload = result
        assert payload["success"]
        assert payload["mode"] == "agent"
        assert payload["engine_state"] == "stopped"
        assert payload["task_state"] == "completed"
        assert (
            persona.requests[0]["user"]
            == "Check the current working directory and report the result."
        )
        checks = cast(list[JSON], payload["checks"])
        task_check = next(check for check in checks if check["name"] == "live.task")
        task_detail = task_check.get("detail")
        assert isinstance(task_detail, str)
        assert "tool=local_current_working_directory" in task_detail
        assert "status=succeeded" in task_detail
        summary = cast(dict[str, int], payload["summary"])
        assert summary["failed"] == 0
        assert core.cur_state == "stopped"
        assert not core.say("queued after test")

    def test_agent_test_speaks_each_tool_result_before_user_prompts(self) -> None:
        """Keep tool announcements and spoken requests in the requested order."""
        core = self._make_core()
        sequence: list[str] = []

        def feature_checks(_engine, *, tool_result_callback=None):
            assert tool_result_callback is not None
            tool_result_callback("query_status", "passed", "result=ready")
            tool_result_callback("query_models", "passed", "result=loaded")
            return []

        def interaction_checks(_engine, request_user):
            assert request_user is not None
            assert request_user("Choice question") == "any answer"
            assert request_user("Approval question") == "any answer"
            return [
                {"name": "choice", "status": "passed", "detail": "processed"},
                {"name": "approval", "status": "passed", "detail": "processed"},
            ]

        def request_user(prompt: str) -> str:
            sequence.append(f"shown:{prompt}")
            return "any answer"

        with (
            mock.patch("celune.test._run_live_checks", return_value=([], None)),
            mock.patch(
                "celune.test.run_agent_feature_checks",
                side_effect=feature_checks,
            ),
            mock.patch(
                "celune.test.run_agent_interaction_checks",
                side_effect=interaction_checks,
            ),
            mock.patch(
                "celune.test._speak_agent_test_message",
                side_effect=lambda _engine, text, _timeout: sequence.append(
                    f"spoken:{text}"
                ),
            ),
            mock.patch.object(core, "stop_live_audio"),
        ):
            result = run_agent_test(core, request_user=request_user)

        assert result["success"]
        assert [entry.split(":", 1)[0] for entry in sequence] == [
            "spoken",
            "spoken",
            "spoken",
            "shown",
            "spoken",
            "shown",
        ]
        assert sequence[2] == "spoken:Choice question"
        assert sequence[3] == "shown:Choice question"
        assert sequence[4] == "spoken:Approval question"
        assert sequence[5] == "shown:Approval question"

    def test_agent_test_ui_response_channel_processes_typed_answer(self) -> None:
        """Unlock input only for a prompt and deliver one typed response."""
        ui = CeluneUI()
        self.addCleanup(setattr, CeluneUI, "_instance", None)
        ui.input_box = TextArea()
        ui.celune = cast(
            Celune,
            SimpleNamespace(backend_mode="agent_test", test_finished=False),
        )
        ui.safe_log = mock.Mock()
        ui.safe_status = mock.Mock()
        ready = threading.Event()

        def change_input_state(*, locked: bool) -> None:
            if not locked:
                ready.set()

        ui.change_input_state = change_input_state
        answers: list[Optional[str]] = []
        worker = threading.Thread(
            target=lambda: answers.append(
                ui.request_agent_test_response("Choose an option", timeout_seconds=2)
            ),
            daemon=True,
        )
        worker.start()
        assert ready.wait(timeout=1)
        ui.input_box.load_text("2")
        assert ui._submit_text(ui.input_box.text)
        worker.join(timeout=1)

        assert answers == ["2"]
        ui.safe_log.assert_called_once_with("Choose an option")
        ui.safe_status.assert_any_call(string("ui.agent_test_prompt_ready"))
        ui.safe_status.assert_any_call(string("ui.agent_test_answer_received"))

    def test_agent_test_reports_no_task_detected(self) -> None:
        """Distinguish an ordinary conversation result from a test crash."""
        core = self._make_core()
        route = AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=0.98,
            route=AgentRoute.CONVERSATION,
        )
        with (
            mock.patch("celune.test._wait_for_persona"),
            mock.patch("celune.test._start_agent_test_pipeline"),
            mock.patch("celune.test.run_agent_feature_checks", return_value=[]),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        assert not result["success"]
        checks = cast(list[JSON], result["checks"])
        routing = next(check for check in checks if check["name"] == "live.routing")
        assert routing["detail"] == "no task detected"

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
            mock.patch("celune.test._wait_for_persona"),
            mock.patch("celune.test._start_agent_test_pipeline"),
            mock.patch("celune.test.run_agent_feature_checks", return_value=[]),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        assert not result["success"]
        checks = cast(list[JSON], result["checks"])
        routing = next(check for check in checks if check["name"] == "live.routing")
        assert routing["detail"] == "classification failed: malformed_output"

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
            mock.patch("celune.test._wait_for_persona"),
            mock.patch("celune.test._start_agent_test_pipeline"),
            mock.patch("celune.test.run_agent_feature_checks", return_value=[]),
            mock.patch.object(core, "stop_live_audio"),
            mock.patch.object(core, "route_input", return_value=route),
        ):
            result = run_agent_test(core)

        assert not result["success"]
        checks = cast(list[JSON], result["checks"])
        routing = next(check for check in checks if check["name"] == "live.routing")
        assert routing["detail"] == "task detected but not started"
