# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Celune's opt-in local-management agent registry."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest import TestCase, mock

from celune.agent.tools import (
    OfflineAgentTool,
    local_management_tool_schemas,
    local_management_tools,
)
from celune.celune import Celune
from celune.persona.capabilities import PersonaCapabilities
from celune.typing.agent import AgentContext, AgentRequest, AgentToolExecutionStatus
from celune.typing.modes import OperationMode


class LocalManagementToolTests(TestCase):
    """Verify typed local filesystem and diagnostic behavior without a shell."""

    def _context(self) -> AgentContext:
        """Build the smallest valid context accepted by a local tool."""
        return AgentContext(
            request=AgentRequest("inspect local state"),
            mode=cast(OperationMode, "agent"),
            persona_capabilities=PersonaCapabilities(),
        )

    def _tool(self, tool_id: str) -> OfflineAgentTool:
        """Return one named local-management tool from the canonical registry."""
        tool = next(
            tool
            for tool in local_management_tools(cast(Celune, object()))
            if tool.name == tool_id
        )
        self.assertIsInstance(tool, OfflineAgentTool)
        return tool

    def test_registry_schemas_are_typed_and_permission_aware(self) -> None:
        """Every local tool has identity, arguments, availability, and danger metadata."""
        schemas = local_management_tool_schemas()
        expected = {
            "local_current_working_directory",
            "local_list_directory",
            "local_file_metadata",
            "local_read_text",
            "local_write_text",
            "local_make_directory",
            "local_copy",
            "local_move",
            "local_delete",
            "local_list_processes",
            "local_inspect_process",
            "local_launch_process",
            "local_terminate_process",
            "local_system_info",
            "local_discover_application",
            "local_running_applications",
            "local_launch_application",
            "local_close_application",
        }
        self.assertEqual(set(schemas), expected)
        for schema in schemas.values():
            self.assertTrue(schema.tool_id)
            self.assertTrue(schema.display_name)
            self.assertTrue(schema.description)
            self.assertTrue(schema.available)
            self.assertEqual(schema.to_json()["tool_id"], schema.tool_id)
            if schema.behavior.value == "mutating":
                self.assertTrue(schema.approval_required)

    def test_current_working_directory_returns_the_resolved_process_directory(
        self,
    ) -> None:
        """The read-only diagnostic reports the exact resolved current directory."""
        directory = Path.cwd().resolve()
        with mock.patch("celune.agent.tools.Path.cwd", return_value=directory):
            result = self._tool("local_current_working_directory").execute(
                {
                    "id": "cwd",
                    "name": "local_current_working_directory",
                    "arguments": {},
                },
                self._context(),
            )
        self.assertEqual(result["status"], AgentToolExecutionStatus.SUCCEEDED)
        output = cast(dict[str, object], result["output"])
        self.assertEqual(output["path"], str(directory))

    def test_local_filesystem_operations_use_exact_absolute_paths(self) -> None:
        """Read and write operations report exact targets and reject ambiguous paths."""
        with self.subTest("read and write"):
            root = Path.cwd() / ".agent-local-management-test"
            root.mkdir(exist_ok=True)
            path = root / "sample.txt"

            def cleanup() -> None:
                """Remove the temporary file and directory after the subtest."""
                path.unlink(missing_ok=True)
                if root.exists():
                    root.rmdir()

            self.addCleanup(cleanup)
            write = self._tool("local_write_text").execute(
                {
                    "id": "write",
                    "name": "local_write_text",
                    "arguments": {"path": str(path), "text": "hello"},
                },
                self._context(),
            )
            self.assertEqual(write["status"], AgentToolExecutionStatus.SUCCEEDED)
            read = self._tool("local_read_text").execute(
                {
                    "id": "read",
                    "name": "local_read_text",
                    "arguments": {"path": str(path)},
                },
                self._context(),
            )
            self.assertEqual(read["status"], AgentToolExecutionStatus.SUCCEEDED)
            self.assertEqual(cast(dict[str, object], read["output"])["text"], "hello")

        invalid = self._tool("local_read_text").execute(
            {
                "id": "invalid",
                "name": "local_read_text",
                "arguments": {"path": "relative.txt"},
            },
            self._context(),
        )
        self.assertEqual(invalid["status"], AgentToolExecutionStatus.FAILED)
        self.assertEqual(
            cast(dict[str, object], invalid["output"])["result"], "invalid_path"
        )

    def test_local_system_info_is_a_safe_registered_operation(self) -> None:
        """The diagnostic operation returns bounded structured local state."""
        result = self._tool("local_system_info").execute(
            {"id": "system", "name": "local_system_info", "arguments": {}},
            self._context(),
        )
        self.assertEqual(result["status"], AgentToolExecutionStatus.SUCCEEDED)
        output = cast(dict[str, object], result["output"])
        self.assertEqual(output["result"], "success")
        self.assertIn("platform", output)
        self.assertIn("cuda", output)
