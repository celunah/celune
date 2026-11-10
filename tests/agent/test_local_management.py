# SPDX-License-Identifier: Apache-2.0
"""Focused tests for Celune's opt-in local-management agent registry."""

from __future__ import annotations

import os
import datetime
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
    recover_local_tool_call,
    infer_local_tool_arguments,
    local_management_tool_schemas,
)
from celune.typing.agent import (
    ToolCall,
    ToolResult,
    AgentContext,
    AgentRequest,
    AgentToolExecutionStatus,
)
from celune.typing.modes import OperationMode
from celune.persona.capabilities import PersonaCapabilities

from ..support import CeluneTestCase
from ..platform import WINDOWS_ONLY


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

    def test_termination_recovers_one_mismatched_pid_from_process_listing(self) -> None:
        """Use the sole listed process with the exact requested identity."""
        context = self._context("Close notepad.exe")
        context = AgentContext(
            request=context.request,
            mode=context.mode,
            persona_capabilities=context.persona_capabilities,
            last_tool_result=cast(
                ToolResult,
                {
                    "tool_call_id": "list-processes",
                    "tool_id": "local_list_processes",
                    "status": AgentToolExecutionStatus.SUCCEEDED,
                    "output": {
                        "processes": [{"pid": 4321, "name": "notepad.exe"}],
                        "truncated": False,
                    },
                },
            ),
        )
        call: ToolCall = {
            "id": "terminate",
            "name": "local_terminate_process",
            "arguments": {"pid": 1234, "expected_name": "notepad.exe"},
        }

        recovered = recover_local_tool_call(call, context)

        assert recovered["arguments"]["pid"] == 4321

    def test_termination_does_not_rewrite_an_explicit_pid(self) -> None:
        """Keep a user-specified PID intact even if another process was listed."""
        context = self._context("Close PID 1234 for notepad.exe")
        context = AgentContext(
            request=context.request,
            mode=context.mode,
            persona_capabilities=context.persona_capabilities,
            last_tool_result=cast(
                ToolResult,
                {
                    "tool_call_id": "list-processes",
                    "tool_id": "local_list_processes",
                    "status": AgentToolExecutionStatus.SUCCEEDED,
                    "output": {
                        "processes": [{"pid": 4321, "name": "notepad.exe"}],
                        "truncated": False,
                    },
                },
            ),
        )
        call: ToolCall = {
            "id": "terminate",
            "name": "local_terminate_process",
            "arguments": {"pid": 1234, "expected_name": "notepad.exe"},
        }

        recovered = recover_local_tool_call(call, context)

        assert recovered["arguments"]["pid"] == 1234

    def test_follow_up_path_recovery_uses_the_resolved_task_request(self) -> None:
        """Restore an earlier target resolved into a self-contained follow-up."""
        report = str((Path.cwd() / "reports" / "report.txt").resolve())
        context = self._context(
            f"Summarize {report}.",
            "Could you summarize that report again?",
        )
        read_call: ToolCall = {
            "id": "read",
            "name": "local_read_text",
            "arguments": {"path": "/wrong/report.txt"},
        }

        recovered_read = recover_local_tool_call(read_call, context)

        assert recovered_read["arguments"]["path"] == report

        source = str((Path.cwd() / "reports" / "report.txt").resolve())
        destination = str((Path.cwd() / "archive" / "report.txt").resolve())
        copy_context = self._context(
            f"Copy {source} to {destination}.",
            f"Copy that report to {destination}.",
        )
        copy_call: ToolCall = {
            "id": "copy",
            "name": "local_copy",
            "arguments": {
                "source": "/wrong/source.txt",
                "destination": "/wrong/dest.txt",
            },
        }

        recovered_copy = recover_local_tool_call(copy_call, copy_context)

        assert recovered_copy["arguments"] == {
            "source": source,
            "destination": destination,
        }

    def test_launch_process_recovers_requested_executable_and_empty_arguments(
        self,
    ) -> None:
        """Keep launch arguments aligned with the user's explicit request."""
        request = (
            "Use local_launch_process to launch notepad.exe "
            "with no arguments and no shell."
        )
        context = self._context("Launch the requested process.", request)
        call: ToolCall = {
            "id": "launch",
            "name": "local_launch_process",
            "arguments": {
                "executable": "notepad.exe with no arguments",
                "arguments": ["and", "no", "shell"],
                "cwd": "no shell",
            },
        }

        recovered = recover_local_tool_call(call, context)
        inferred = infer_local_tool_arguments("local_launch_process", context)

        assert recovered["arguments"] == {"executable": "notepad.exe", "arguments": []}
        assert inferred == {"executable": "notepad.exe", "arguments": []}

    def test_launch_process_recovers_only_an_explicit_working_directory(self) -> None:
        """Discard a generated cwd unless the request names its directory."""
        requested_directory = str(Path.cwd().resolve() / "explicit cwd")
        request = (
            "Use local_launch_process to launch notepad.exe with working directory "
            f'"{requested_directory}", no arguments, and no shell.'
        )
        context = self._context("Launch the requested process.", request)
        call: ToolCall = {
            "id": "launch",
            "name": "local_launch_process",
            "arguments": {
                "executable": "notepad.exe",
                "arguments": [],
                "cwd": "invented directory",
            },
        }

        recovered = recover_local_tool_call(call, context)

        assert recovered["arguments"]["cwd"] == requested_directory

    def test_launch_process_ignores_generated_working_directory(self) -> None:
        """Launch in the inherited process directory when cwd was not requested."""
        context = self._context(
            "Launch the requested process.",
            "Use local_launch_process to launch notepad.exe with no arguments and no shell.",
        )
        call: ToolCall = {
            "id": "launch",
            "name": "local_launch_process",
            "arguments": {
                "executable": "notepad.exe",
                "arguments": [],
                "cwd": "no shell",
            },
        }
        with (
            mock.patch(
                "celune.agent.tools._resolve_local_executable",
                return_value="C:/Windows/notepad.exe",
            ),
            mock.patch("celune.agent.tools.subprocess.Popen") as process,
        ):
            process.return_value.pid = 4321
            result = self._tool("local_launch_process").execute(call, context)

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert process.call_args.kwargs["cwd"] is None
        assert process.call_args.kwargs["shell"] is False
        assert process.call_args.args[0] == ["C:/Windows/notepad.exe"]

    @WINDOWS_ONLY
    def test_launch_process_searches_the_windows_system_directory(self) -> None:
        """Find a bare executable in Windows system folders outside PATH."""
        resolved = r"C:\Windows\System32\notepad.exe"
        call: ToolCall = {
            "id": "launch",
            "name": "local_launch_process",
            "arguments": {"executable": "notepad.exe", "arguments": []},
        }
        with (
            mock.patch.dict(os.environ, {"WINDIR": r"C:\Windows"}),
            mock.patch(
                "celune.agent.tools.shutil.which",
                side_effect=[None, resolved],
            ) as which,
            mock.patch("celune.agent.tools.subprocess.Popen") as process,
        ):
            process.return_value.pid = 4321
            result = self._tool("local_launch_process").execute(
                call,
                self._context(),
            )

        assert result["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert cast(dict[str, object], result["output"])["executable"] == resolved
        assert process.call_args.args[0] == [resolved]
        assert process.call_args.kwargs["shell"] is False
        assert (
            str(Path(r"C:\Windows") / "System32")
            in which.call_args_list[1].kwargs["path"]
        )

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

    def test_single_path_file_tools_recover_the_requested_path(self) -> None:
        """Recover paths and return explicit local ISO timestamps for metadata."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory).resolve()
            source = root / "source.txt"
            source.write_text("source contents", encoding="utf-8")
            modified_timestamp = datetime.datetime(
                2026,
                11,
                8,
                17,
                47,
                tzinfo=datetime.UTC,
            ).timestamp()
            os.utime(source, (modified_timestamp, modified_timestamp))
            metadata = self._tool("local_file_metadata").execute(
                {
                    "id": "metadata-recovered-path",
                    "name": "local_file_metadata",
                    "arguments": {"path": "relative/source.txt"},
                },
                self._context("Inspect the requested file.", f'Inspect "{source}".'),
            )
            assert metadata["status"] == AgentToolExecutionStatus.SUCCEEDED
            metadata_output = cast(dict[str, object], metadata["output"])
            source_stat = source.stat()
            expected_created_timestamp = getattr(source_stat, "st_birthtime", None)
            expected_created = (
                None
                if expected_created_timestamp is None
                else datetime.datetime.fromtimestamp(
                    expected_created_timestamp,
                    tz=datetime.UTC,
                )
                .astimezone()
                .isoformat(timespec="seconds")
            )
            expected_modified = (
                datetime.datetime.fromtimestamp(
                    modified_timestamp,
                    tz=datetime.UTC,
                )
                .astimezone()
                .isoformat(timespec="seconds")
            )
            assert metadata_output["path"] == str(source)
            assert metadata_output["created_at"] == expected_created
            assert metadata_output["modified_at"] == expected_modified
            assert "modified" not in metadata_output

            read = self._tool("local_read_text").execute(
                {
                    "id": "read-recovered-path",
                    "name": "local_read_text",
                    "arguments": {"path": "relative/source.txt"},
                },
                self._context("Read the requested file.", f'Read "{source}".'),
            )

            target = root / "written.txt"
            write = self._tool("local_write_text").execute(
                {
                    "id": "write-recovered-path",
                    "name": "local_write_text",
                    "arguments": {"path": "relative/written.txt", "text": "saved"},
                },
                self._context(
                    "Write the requested file.",
                    f'Write the requested file "{target}".',
                ),
            )
            deleted = root / "deleted.txt"
            deleted.write_text("remove me", encoding="utf-8")
            delete = self._tool("local_delete").execute(
                {
                    "id": "delete-recovered-path",
                    "name": "local_delete",
                    "arguments": {"path": "relative/deleted.txt"},
                },
                self._context(
                    "Delete the requested file.",
                    f'Delete the file "{deleted}".',
                ),
            )

            assert target.read_text(encoding="utf-8") == "saved"
            assert not deleted.exists()

        assert read["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert cast(dict[str, object], read["output"])["text"] == "source contents"
        assert write["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert delete["status"] == AgentToolExecutionStatus.SUCCEEDED

    def test_copy_and_move_recover_source_and_destination_paths(self) -> None:
        """Use two requested paths in order when Needle mangles a copy or move."""
        with TemporaryDirectory(prefix="celune agent ") as directory:
            root = Path(directory).resolve()
            source = root / "source.txt"
            copied = root / "copied.txt"
            moved = root / "moved.txt"
            source.write_text("file contents", encoding="utf-8")

            copy = self._tool("local_copy").execute(
                {
                    "id": "copy-recovered-paths",
                    "name": "local_copy",
                    "arguments": {
                        "source": "relative/source.txt",
                        "destination": "relative/copied.txt",
                    },
                },
                self._context(
                    "Copy the requested file.", f'Copy "{source}" to "{copied}".'
                ),
            )
            move = self._tool("local_move").execute(
                {
                    "id": "move-recovered-paths",
                    "name": "local_move",
                    "arguments": {
                        "source": "relative/copied.txt",
                        "destination": "relative/moved.txt",
                    },
                },
                self._context(
                    "Move the requested file.", f'Move "{copied}" to "{moved}".'
                ),
            )

            assert source.read_text(encoding="utf-8") == "file contents"
            assert not copied.exists()
            assert moved.read_text(encoding="utf-8") == "file contents"

        assert copy["status"] == AgentToolExecutionStatus.SUCCEEDED
        assert move["status"] == AgentToolExecutionStatus.SUCCEEDED

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

    def test_make_directory_recovers_unquoted_path_before_sentence_punctuation(
        self,
    ) -> None:
        """Ignore sentence punctuation after an unquoted requested path."""
        with TemporaryDirectory(prefix="celune-agent-") as directory:
            root = Path(directory).resolve()
            target = root / "created"
            result = self._tool("local_make_directory").execute(
                {
                    "id": "make-unquoted-path",
                    "name": "local_make_directory",
                    "arguments": {"path": "relative/created"},
                },
                self._context(
                    "Create the requested directory.",
                    f"Create the directory {target}.",
                ),
            )

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
