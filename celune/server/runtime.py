# SPDX-License-Identifier: Apache-2.0
"""Synchronous bridges and process lifecycle operations for server clients."""

from __future__ import annotations
import asyncio
import inspect
from collections.abc import Callable, Awaitable

from .session import ServerSession, server_session

from ..typing.aliases import LogLevel
from ..typing.common import JSONSerializable


def run_async_runtime_call(
    awaitable: Awaitable[JSONSerializable],
) -> JSONSerializable:
    """Run one async runtime call from a synchronous API or WebUI callback."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:

        async def await_result() -> JSONSerializable:
            return await awaitable

        return asyncio.run(await_result())
    raise RuntimeError("synchronous runtime calls cannot run on an active event loop")


def invoke_message_callback(
    callback: Callable[..., None],
    msg: str,
    severity: str,
    loglevel: LogLevel,
) -> None:
    """Invoke a message callback while preserving legacy two-argument callbacks."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(msg, severity, loglevel=loglevel)
        return

    try:
        signature.bind(msg, severity, loglevel=loglevel)
    except TypeError:
        callback(msg, severity)
    else:
        callback(msg, severity, loglevel=loglevel)


def shutdown_api_for_fatal_error(session: ServerSession = server_session) -> None:
    """Stop the API and browser UI after a fatal Celune runtime failure."""
    session.bound_celune = None
    if session.current_api_server is not None:
        session.current_api_server.should_exit = True
        session.current_api_server.force_exit = True
