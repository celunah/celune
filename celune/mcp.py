# SPDX-License-Identifier: Apache-2.0
"""Expose Celune's live runtime through Model Context Protocol tools."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional, Union

import psutil
from mcp.server import MCPServer
from mcp.server.mcpserver import Audio
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from .exceptions import CEVoiceError
from .i18n import string
from .backends.vc import VC_BACKENDS
from .backends.tts import BACKENDS
from .singing import validate_singing_notation
from .threads import run_in_daemon_thread
from .typing.pipeline import SpeechStreamQueue
from .cevoice import CEVoice, bundled_voices_dir


if TYPE_CHECKING:
    from .celune import Celune

_transport_security = TransportSecuritySettings(
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
)


class _MCPApplication:
    """Dispatch requests to the server attached to the active API lifespan."""

    def __init__(self) -> None:
        self._app: Optional[ASGIApp] = None

    def set_app(self, app: Optional[ASGIApp]) -> None:
        """Set the Streamable HTTP application for the current API lifespan."""
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward one ASGI request or return unavailable while Celune is stopped."""
        app = self._app
        if app is None:
            await Response(status_code=503)(scope, receive, send)
            return
        await app(scope, receive, send)


mcp_app = _MCPApplication()


@asynccontextmanager
async def mcp_lifespan(_app: object) -> AsyncGenerator[None, None]:
    """Keep the MCP session manager running with Celune's API process."""
    server = _create_mcp_server()
    application = server.streamable_http_app(transport_security=_transport_security)
    async with server.session_manager.run():
        mcp_app.set_app(application)
        try:
            yield
        finally:
            mcp_app.set_app(None)


def configure_mcp_host_allowlist(bind_host: str) -> None:
    """Allow MCP requests addressed to loopback and this machine's interfaces."""
    hosts = {"127.0.0.1:*", "localhost:*", "[::1]:*"}
    if bind_host not in {"0.0.0.0", "::", ""}:
        address = _format_host(bind_host)
        if address is not None:
            hosts.add(f"{address}:*")

    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            try:
                parsed = ipaddress.ip_address(address.address.split("%", 1)[0])
            except ValueError:
                continue
            if parsed.is_unspecified:
                continue
            hosts.add(f"{_format_ip(parsed)}:*")

    _transport_security.allowed_hosts = sorted(hosts)


def _format_host(host: str) -> Optional[str]:
    """Return a normalized host literal suitable for an HTTP Host pattern."""
    try:
        return _format_ip(ipaddress.ip_address(host))
    except ValueError:
        if "/" in host or "\\" in host:
            return None
        return host.strip("[]") or None


def _format_ip(
    address: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
) -> str:
    """Format an address literal for the MCP HTTP Host allowlist."""
    if isinstance(address, ipaddress.IPv6Address):
        return f"[{address.compressed}]"
    return address.compressed


def _audio_bytes(chunks: SpeechStreamQueue) -> bytes:
    """Drain Celune's audio stream into one FLAC payload for MCP clients."""
    from .webui.runtime import audio_bytes

    return b"".join(audio_bytes(chunks))


def _require_celune() -> Celune:
    """Return the API-bound Celune instance without importing the API at startup."""
    from .webui.runtime import require_celune

    return require_celune()


async def _generate_audio(content: str, save: bool, *, singing: bool) -> Audio:
    """Generate one audio response through Celune's normal speech queue."""
    engine = _require_celune()
    backend_is_singing = bool(getattr(engine.backend, "singing", False))
    if singing:
        if not backend_is_singing:
            raise ToolError(string("api.singing_backend_required"))
        notation_error = validate_singing_notation(engine, content)
        if notation_error is not None:
            raise ToolError(notation_error)
    elif backend_is_singing:
        raise ToolError(string("mcp.speech_backend_required"))

    chunks = await engine.say_stream_async(content, save=save)
    if chunks is None:
        raise ToolError(string("mcp.request_rejected"))
    audio_data = await run_in_daemon_thread(_audio_bytes, chunks)
    if not audio_data:
        raise ToolError(string("mcp.request_rejected"))
    return Audio(data=audio_data, format="flac")


def _find_installed_pack(pack_name: str) -> Optional[Path]:
    """Find a voice pack by installed filename or stem, never by arbitrary path."""
    normalized_name = pack_name.casefold()
    if Path(pack_name).name != pack_name or "/" in pack_name or "\\" in pack_name:
        return None
    for pack_path in bundled_voices_dir().glob("*.cevoice"):
        if normalized_name in {pack_path.name.casefold(), pack_path.stem.casefold()}:
            return pack_path
    return None


async def speak(content: str, save: bool = True) -> Audio:
    """Generate speech audio from plain text."""
    return await _generate_audio(content, save, singing=False)


async def sing(content: str, save: bool = True) -> Audio:
    """Generate singing audio from CESN notation."""
    return await _generate_audio(content, save, singing=True)


async def run_persona(content: str) -> str:
    """Submit text to Persona and report whether Celune accepted it."""
    engine = _require_celune()
    if not await engine.think_async(content):
        raise ToolError(string("mcp.request_rejected"))
    return string("mcp.persona_accepted")


async def change_voice(
    voice_name: Optional[str] = None,
    pack_name: Optional[str] = None,
) -> str:
    """Select an active-pack voice or switch to an installed voice pack."""
    if (voice_name is None) == (pack_name is None):
        raise ToolError(string("mcp.change_voice_target_required"))

    engine = _require_celune()
    if voice_name is not None:
        if voice_name not in engine.voices:
            raise ToolError(string("api.invalid_voice"))
        if not await engine.set_voice_async(voice_name):
            raise ToolError(string("webui.cannot_change_voice_right_now"))
        return string("mcp.voice_changed", voice=voice_name)

    selected_pack = _find_installed_pack(pack_name or "")
    if selected_pack is None:
        raise ToolError(string("mcp.voice_pack_not_found", pack=pack_name or ""))

    if getattr(engine.backend, "singing", False):
        try:
            bundle = CEVoice.open(selected_pack)
        except (OSError, CEVoiceError) as error:
            raise ToolError(string("mcp.voice_pack_invalid")) from error
        if bundle.singing_voice is None:
            raise ToolError(string("mcp.voice_pack_singing_required"))

    if not await engine.set_cevoice_async(selected_pack.name):
        raise ToolError(string("webui.cannot_change_voice_right_now"))
    return string("mcp.voice_pack_changed", pack=selected_pack.stem)


async def change_backend(backend_name: str) -> str:
    """Switch Celune to an installed speech or voice-conversion backend."""
    normalized_backend = backend_name.strip().lower()
    available_backends = tuple(BACKENDS) + tuple(VC_BACKENDS)
    if normalized_backend not in available_backends:
        raise ToolError(
            string(
                "celune.unknown_backend",
                backend=backend_name,
                available=", ".join(available_backends),
            )
        )

    engine = _require_celune()
    if not await engine.set_backend_async(normalized_backend):
        raise ToolError(string("mcp.request_rejected"))
    return string("mcp.backend_changed", backend=normalized_backend)


def _create_mcp_server() -> MCPServer:
    """Create an MCP server with Celune's five public control tools."""
    server = MCPServer("Celune")
    server.add_tool(speak, description=string("mcp.speak_description"))
    server.add_tool(sing, description=string("mcp.sing_description"))
    server.add_tool(run_persona, description=string("mcp.run_persona_description"))
    server.add_tool(change_voice, description=string("mcp.change_voice_description"))
    server.add_tool(
        change_backend, description=string("mcp.change_backend_description")
    )
    return server


mcp_server = _create_mcp_server()
