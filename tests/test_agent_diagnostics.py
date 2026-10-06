# SPDX-License-Identifier: Apache-2.0
"""Verify the CLI agent diagnostic covers tools and isolates side effects."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

from celune.celune import Celune
from celune.agent.tools import production_agent_tools, production_agent_tool_schemas
from celune.typing.agent import AgentToolSchema, AgentToolBehavior
from celune.typing.common import JSON
from celune.agent.diagnostics import run_agent_feature_checks

from .support import FakeGlow, FakeBackend, CeluneTestCase


def _check_name(check: JSON) -> str:
    """Read one diagnostic name without trusting its JSON value type."""
    name = check.get("name")
    return name if isinstance(name, str) else ""


class TestAgentDiagnostics(CeluneTestCase):
    """Exercise the complete active feature matrix without model inference."""

    def _core(self) -> Celune:
        """Create a healthy fake core for direct production-handler checks."""
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
        core.loaded = True
        core.model_ready.set()
        core.persona_ready = True
        core.mode = "agent"
        self.addCleanup(core.close)
        return core

    def test_active_catalog_tools_are_checked_and_safe(self) -> None:
        """Run each active tool and ensure process operations stay stubbed."""
        core = self._core()
        with mock.patch.dict(os.environ, {"CELUNE_AGENT_FS_TOOLS": "true"}):
            checks = run_agent_feature_checks(core)

        expected_tools = set(
            production_agent_tool_schemas(include_local_management=True)
        )
        actual_tools = {
            _check_name(check)
            for check in checks
            if _check_name(check) in expected_tools
        }
        failed = [check for check in checks if check["status"] == "failed"]
        assert actual_tools == expected_tools
        assert not failed
        unavailable = {
            _check_name(check): check
            for check in checks
            if _check_name(check) in {"pause_speech", "resume_speech"}
        }
        assert set(unavailable) == {"pause_speech", "resume_speech"}
        assert all(check["status"] == "passed" for check in unavailable.values())
        assert all(check["detail"] == "status=failed" for check in unavailable.values())
        assert {
            "runtime.lifecycle.pause_resume",
            "runtime.lifecycle.interruption",
            "runtime.lifecycle.steering",
            "runtime.lifecycle.cancellation",
            "runtime.lifecycle.events",
            "runtime.validation",
            "runtime.permissions",
            "runtime.permission_denied",
            "runtime.permission_approval_unavailable",
            "runtime.process_launch_gate",
            "runtime.process_terminate_gate",
            "runtime.application_launch_gate",
            "runtime.application_close_gate",
            "runtime.approval.approved",
            "runtime.approval.denied",
            "runtime.choice",
            "runtime.context.compaction",
            "runtime.context.limit",
            "runtime.tokens.limit",
            "runtime.outcome.completed",
            "runtime.outcome.failed",
            "runtime.outcome.aborted",
            "runtime.outcome.cancelled",
        }.issubset({_check_name(check) for check in checks})

    def test_disabled_local_management_is_reported_as_skipped(self) -> None:
        """Omit optional local tools while retaining an explicit skip result."""
        core = self._core()
        with mock.patch.dict(os.environ, {"CELUNE_AGENT_FS_TOOLS": "false"}):
            checks = run_agent_feature_checks(core)

        names = {_check_name(check) for check in checks}
        assert "local_management" in names
        assert not any(
            name.startswith("local_") and name != "local_management" for name in names
        )
        skipped = next(
            check for check in checks if _check_name(check) == "local_management"
        )
        assert skipped["status"] == "skipped"
        assert not any(check["status"] == "failed" for check in checks)
        assert {
            "runtime.process_launch_gate",
            "runtime.process_terminate_gate",
            "runtime.application_launch_gate",
            "runtime.application_close_gate",
        }.issubset(
            {_check_name(check) for check in checks if check["status"] == "skipped"}
        )

    def test_one_runtime_failure_does_not_hide_later_checks(self) -> None:
        """Keep unrelated runtime and tool outcomes after a check fails."""
        core = self._core()
        with mock.patch(
            "celune.agent.diagnostics._check_lifecycle_interruption",
            side_effect=RuntimeError("controlled lifecycle failure"),
        ):
            checks = run_agent_feature_checks(core)

        results = {_check_name(check): check for check in checks}
        assert results["runtime.lifecycle.interruption"]["status"] == "failed"
        assert "runtime.lifecycle.steering" in results
        assert "runtime.outcome.completed" in results
        assert "query_status" in results

    def test_unmapped_production_tool_fails_coverage_and_its_own_check(self) -> None:
        """Report an unmapped catalog tool while continuing the tool pass."""
        core = self._core()
        tools = production_agent_tools(core)
        schemas = dict(production_agent_tool_schemas())
        unmapped_schema = AgentToolSchema(
            tool_id="unmapped_diagnostic",
            display_name="Unmapped diagnostic",
            description="Unmapped coverage probe.",
            behavior=AgentToolBehavior.READ_ONLY,
        )
        schemas[unmapped_schema.tool_id] = unmapped_schema
        unmapped_tool = SimpleNamespace(
            name=unmapped_schema.tool_id,
            description=unmapped_schema.description,
            execute=mock.Mock(),
        )
        with (
            mock.patch(
                "celune.agent.diagnostics.production_agent_tools",
                return_value=(*tools, unmapped_tool),
            ),
            mock.patch(
                "celune.agent.diagnostics.production_agent_tool_schemas",
                return_value=schemas,
            ),
        ):
            checks = run_agent_feature_checks(core)

        results = {_check_name(check): check for check in checks}
        assert results["catalog.coverage"]["status"] == "failed"
        assert results["unmapped_diagnostic"]["status"] == "failed"
        assert results["query_status"]["status"] in {"passed", "failed"}
        unmapped_tool.execute.assert_not_called()
