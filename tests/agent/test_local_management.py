# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Celune's opt-in local-management agent registry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, cast
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from contextlib import nullcontext

from celune.celune import Celune
from celune.agent.tools import (
    OfflineAgentTool,
    local_management_tools,
    local_management_tool_schemas,
)
from celune.typing.agent import AgentContext, AgentRequest, AgentToolExecutionStatus
from celune.typing.modes import OperationMode
from celune.persona.capabilities import PersonaCapabilities

from ..support import CeluneTestCase


class TestLocalManagementTools(CeluneTestCase):
    """Verify typed local filesystem and diagnostic behavior without a shell."""

    def _context(
        self,
        request: str = "inspect local state",
        original_request: Optional[str] = None,
    ) -> AgentContext:
        """Build the smallest valid context accepted by a local tool."""
        return AgentContext(
            request=AgentRequest(request, original_request=original_request),
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
        assert isinstance(tool, OfflineAgentTool)
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
        assert set(schemas) == expected
        for schema in schemas.values():
            assert schema.tool_id
            assert schema.display_name
            assert schema.description
            assert schema.available
            assert schema.to_json()["tool_id"] == schema.tool_id
            if schema.behavior.value == "mutating":
                assert schema.approval_required

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
        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = cast(dict[str, object], result["output"])
        assert output["path"] == str(directory)

    def test_local_reads_resolve_relative_paths_but_writes_require_absolute_paths(
        self,
    ) -> None:
        """Resolve relative read paths against CWD and keep mutations explicit."""
        with nullcontext():
            root = (Path.cwd() / ".agent-local-management-test").resolve()
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
            assert write["status"] == AgentToolExecutionStatus.SUCCEEDED
            read = self._tool("local_read_text").execute(
                {
                    "id": "read",
                    "name": "local_read_text",
                    "arguments": {"path": str(path)},
                },
                self._context(),
            )
            assert read["status"] == AgentToolExecutionStatus.SUCCEEDED
            assert cast(dict[str, object], read["output"])["text"] == "hello"

            with mock.patch("celune.agent.tools.Path.cwd", return_value=root):
                listing = self._tool("local_list_directory").execute(
                    {
                        "id": "list-relative",
                        "name": "local_list_directory",
                        "arguments": {"path": "."},
                    },
                    self._context(),
                )
                relative_read = self._tool("local_read_text").execute(
                    {
                        "id": "read-relative",
                        "name": "local_read_text",
                        "arguments": {"path": "sample.txt"},
                    },
                    self._context(),
                )
            assert listing["status"] == AgentToolExecutionStatus.SUCCEEDED
            assert cast(dict[str, object], listing["output"])["path"] == str(root)
            assert relative_read["status"] == AgentToolExecutionStatus.SUCCEEDED
            assert cast(dict[str, object], relative_read["output"])["text"] == "hello"

        invalid = self._tool("local_write_text").execute(
            {
                "id": "invalid",
                "name": "local_write_text",
                "arguments": {"path": "relative.txt", "text": "hello"},
            },
            self._context(),
        )
        assert invalid["status"] == AgentToolExecutionStatus.FAILED
        assert cast(dict[str, object], invalid["output"])["result"] == "invalid_path"

    def test_directory_listing_recovers_one_existing_path_from_the_request(
        self,
    ) -> None:
        """Use the user's exact directory when Needle corrupts its path argument."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory).resolve()
            selected_path = str(root).replace("\\", "").replace("/", "")
            result = self._tool("local_list_directory").execute(
                {
                    "id": "list-recovered-path",
                    "name": "local_list_directory",
                    "arguments": {"path": selected_path},
                },
                self._context(
                    "List the requested directory.",
                    f'List the directory "{root}".',
                ),
            )

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = cast(dict[str, object], result["output"])
        assert output["path"] == str(root)

    def test_directory_listing_uses_requested_path_over_an_existing_wrong_path(
        self,
    ) -> None:
        """Honor the user's unique directory even when Needle chose another one."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory).resolve()
            requested = root / "requested"
            selected = root / "selected"
            requested.mkdir()
            selected.mkdir()
            result = self._tool("local_list_directory").execute(
                {
                    "id": "list-exact-requested-path",
                    "name": "local_list_directory",
                    "arguments": {"path": str(selected)},
                },
                self._context(
                    "List the requested directory.",
                    f'List the directory "{requested}".',
                ),
            )

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = cast(dict[str, object], result["output"])
        assert output["path"] == str(requested)

    def test_make_directory_recovers_the_requested_target_path(self) -> None:
        """Use the exact requested target when Needle corrupts its parent path."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory).resolve()
            target = root / "created"
            result = self._tool("local_make_directory").execute(
                {
                    "id": "make-recovered-path",
                    "name": "local_make_directory",
                    "arguments": {"path": str(root / "missing-parent" / "created")},
                },
                self._context(
                    "Create the requested directory.",
                    f'Create the directory "{target}".',
                ),
            )
            exists = target.is_dir()

        assert exists
        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = cast(dict[str, object], result["output"])
        assert output["path"] == str(target)

    def test_directory_listing_does_not_choose_between_requested_directories(
        self,
    ) -> None:
        """Leave a missing target unchanged when the request names two directories."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            result = self._tool("local_list_directory").execute(
                {
                    "id": "list-ambiguous-path",
                    "name": "local_list_directory",
                    "arguments": {"path": str(root / "missing")},
                },
                self._context(f'Compare "{first}" with {second}.'),
            )

        assert result["status"] == AgentToolExecutionStatus.FAILED
        output = cast(dict[str, object], result["output"])
        assert output["result"] == "missing"

    def test_local_system_info_is_a_safe_registered_operation(self) -> None:
        """The diagnostic operation returns bounded structured local state."""
        result = self._tool("local_system_info").execute(
            {"id": "system", "name": "local_system_info", "arguments": {}},
            self._context(),
        )
        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        output = cast(dict[str, object], result["output"])
        assert output["result"] == "success"
        assert "platform" in output
        assert "cuda" in output
        assert "devices" in cast(dict[str, object], output["gpu"])

    def test_local_system_info_reports_detected_cuda_devices(self) -> None:
        """Include CUDA device identity and memory when PyTorch detects a GPU."""
        properties = SimpleNamespace(name="Fixture GPU", total_memory=8_000_000_000)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=1),
            mock.patch("torch.cuda.get_device_capability", return_value=(8, 6)),
            mock.patch("torch.cuda.get_device_properties", return_value=properties),
        ):
            result = self._tool("local_system_info").execute(
                {"id": "system", "name": "local_system_info", "arguments": {}},
                self._context(),
            )

        output = cast(dict[str, object], result["output"])
        gpu = cast(dict[str, object], output["gpu"])
        devices = cast(list[dict[str, object]], gpu["devices"])
        assert gpu["cuda_available"] is True
        assert devices == [
            {
                "index": 0,
                "name": "Fixture GPU",
                "total_memory": 8_000_000_000,
                "compute_capability": "8.6",
            }
        ]
