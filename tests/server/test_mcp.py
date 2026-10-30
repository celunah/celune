# SPDX-License-Identifier: Apache-2.0
"""Tests for Celune's Model Context Protocol tools."""

import queue
import base64
import asyncio
from typing import cast
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import pytest
import numpy as np
import mcp.types as mcp_types
import mcp.server.mcpserver.exceptions as mcp_exceptions
from fastapi.testclient import TestClient

from celune import api, mcp
from celune.typing.pipeline import SpeechStreamQueue
from ..support import CeluneTestCase

AudioContent = mcp_types.AudioContent
CallToolResult = mcp_types.CallToolResult
ToolError = mcp_exceptions.ToolError


class TestMcpServer(CeluneTestCase):
    """Verify the registered MCP tools share the Celune runtime."""

    def test_server_registers_the_seven_core_tools(self) -> None:
        """Expose lifecycle, speech, singing, Persona, voice, and backend tools."""
        tools = asyncio.run(mcp.mcp_server.list_tools())

        assert [tool.name for tool in tools] == [
            "status",
            "wake",
            "speak",
            "sing",
            "run_persona",
            "change_voice",
            "change_backend",
        ]

    def test_status_reports_sleep_and_readiness(self) -> None:
        """Report current runtime readiness without waking a sleeping engine."""
        engine = SimpleNamespace(
            cur_state="sleeping",
            loaded=False,
            sleeping=True,
            locked=False,
            model_ready=SimpleNamespace(is_set=mock.Mock(return_value=True)),
        )

        with mock.patch.object(mcp, "_require_celune", return_value=engine):
            result = cast(
                CallToolResult,
                asyncio.run(mcp.mcp_server.call_tool("status", {})),
            )

        assert result.structured_content == {
            "engine_state": "sleeping",
            "loaded": False,
            "sleeping": True,
            "locked": False,
            "ready": False,
        }

    def test_wake_waits_for_core_wake_and_reports_resulting_status(self) -> None:
        """Wake through Celune's asynchronous lifecycle before returning status."""
        engine = SimpleNamespace(
            cur_state="idle",
            loaded=True,
            sleeping=False,
            locked=False,
            model_ready=SimpleNamespace(is_set=mock.Mock(return_value=True)),
            wake_from_sleep_async=mock.AsyncMock(return_value=True),
        )

        with mock.patch.object(mcp, "_require_celune", return_value=engine):
            result = cast(
                CallToolResult,
                asyncio.run(mcp.mcp_server.call_tool("wake", {})),
            )

        assert result.structured_content == {
            "engine_state": "idle",
            "loaded": True,
            "sleeping": False,
            "locked": False,
            "ready": True,
            "awake": True,
        }
        engine.wake_from_sleep_async.assert_awaited_once_with()

    def test_http_mount_runs_inside_the_api_lifespan(self) -> None:
        """Start the MCP session manager with the parent REST API lifecycle."""
        previous_token = api.server_session.auth_token
        api.server_session.auth_token = "test-token"
        try:
            with TestClient(
                api.api,
                base_url="http://127.0.0.1:2060",
            ) as client:
                response = client.get(
                    "/mcp",
                    headers={
                        "Authorization": "Bearer test-token",
                        "Accept": "text/event-stream",
                    },
                )
        finally:
            api.server_session.auth_token = previous_token

        assert response.status_code == 400
        assert "Missing session ID" in response.text

    def test_speak_returns_flac_audio_content(self) -> None:
        """Return generated audio as a native MCP audio result."""
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=False),
            say_stream_async=mock.AsyncMock(return_value=chunks),
            playback_done=mock.Mock(wait=mock.Mock(return_value=True)),
        )

        with mock.patch.object(mcp, "_require_celune", return_value=engine):
            result = cast(
                CallToolResult,
                asyncio.run(
                    mcp.mcp_server.call_tool(
                        "speak",
                        {"content": "hello", "save": False},
                    )
                ),
            )

        assert isinstance(result.content[0], AudioContent)
        assert result.content[0].mime_type == "audio/flac"
        assert base64.b64decode(result.content[0].data)[:4] == b"fLaC"
        engine.say_stream_async.assert_awaited_once_with("hello", save=False)

    def test_sing_returns_flac_audio_content_after_validation(self) -> None:
        """Parse CESN before queueing and return the normal FLAC payload."""
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            say_stream_async=mock.AsyncMock(return_value=chunks),
            playback_done=mock.Mock(wait=mock.Mock(return_value=True)),
        )
        with (
            mock.patch.object(mcp, "_require_celune", return_value=engine),
            mock.patch.object(mcp, "validate_singing_notation", return_value=None),
        ):
            result = cast(
                CallToolResult,
                asyncio.run(
                    mcp.mcp_server.call_tool(
                        "sing",
                        {"content": "[c4:4][a] Ah", "save": False},
                    )
                ),
            )

        assert isinstance(result.content[0], AudioContent)
        assert result.content[0].mime_type == "audio/flac"
        assert base64.b64decode(result.content[0].data)[:4] == b"fLaC"
        engine.say_stream_async.assert_awaited_once_with("[c4:4][a] Ah", save=False)

    def test_empty_audio_is_returned_as_a_tool_error(self) -> None:
        """Report a missing generated payload as a readable tool failure."""
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(None)
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=False),
            say_stream_async=mock.AsyncMock(return_value=chunks),
        )

        with (
            mock.patch.object(mcp, "_require_celune", return_value=engine),
            pytest.raises(ToolError, match="request could not be accepted"),
        ):
            asyncio.run(
                mcp.mcp_server.call_tool(
                    "speak",
                    {"content": "hello", "save": False},
                )
            )

    def test_change_voice_accepts_both_current_pack_entries(self) -> None:
        """Route either active-pack voice through Celune's voice reload method."""
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=False),
            voices=("Voice A", "Voice B"),
            set_voice_async=mock.AsyncMock(return_value=True),
        )

        async def change_both() -> list[str]:
            with mock.patch.object(mcp, "_require_celune", return_value=engine):
                return [
                    await mcp.change_voice(voice_name="Voice A"),
                    await mcp.change_voice(voice_name="Voice B"),
                ]

        results = asyncio.run(change_both())

        assert len(results) == 2
        assert [call.args[0] for call in engine.set_voice_async.await_args_list] == [
            "Voice A",
            "Voice B",
        ]

    def test_change_voice_can_select_an_installed_pack(self, tmp_path: Path) -> None:
        """Resolve a pack name inside the voice directory and reload through Celune."""
        voice_directory = tmp_path / "voices"
        voice_directory.mkdir()
        (voice_directory / "alternate.cevoice").touch()
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=False),
            set_cevoice_async=mock.AsyncMock(return_value=True),
        )
        with (
            mock.patch.object(
                mcp,
                "bundled_voices_dir",
                return_value=voice_directory,
            ),
            mock.patch.object(mcp, "_require_celune", return_value=engine),
        ):
            result = asyncio.run(mcp.change_voice(pack_name="alternate"))

        assert result
        engine.set_cevoice_async.assert_awaited_once_with("alternate.cevoice")

    def test_run_persona_uses_the_async_core_action(self) -> None:
        """Queue a Persona request through Celune's normal async entrypoint."""
        engine = SimpleNamespace(think_async=mock.AsyncMock(return_value=True))
        with mock.patch.object(mcp, "_require_celune", return_value=engine):
            result = asyncio.run(mcp.run_persona("say hello"))

        assert result
        engine.think_async.assert_awaited_once_with("say hello")

    def test_run_persona_rejects_singing_backend(self) -> None:
        """Do not submit spoken Persona text to the singing pipeline."""
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            think_async=mock.AsyncMock(return_value=True),
        )
        with (
            mock.patch.object(mcp, "_require_celune", return_value=engine),
            pytest.raises(ToolError, match="Persona replies are unavailable"),
        ):
            asyncio.run(mcp.run_persona("say hello"))

        engine.think_async.assert_not_awaited()

    def test_change_backend_uses_a_registered_backend_name(self) -> None:
        """Switch backends only through Celune's installed backend registry."""
        engine = SimpleNamespace(set_backend_async=mock.AsyncMock(return_value=True))
        with mock.patch.object(mcp, "_require_celune", return_value=engine):
            result = asyncio.run(mcp.change_backend("mini"))

        assert result
        engine.set_backend_async.assert_awaited_once_with("mini")
