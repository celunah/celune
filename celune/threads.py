# SPDX-License-Identifier: Apache-2.0
"""Thread helpers for blocking Celune operations."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from collections.abc import Callable


async def run_in_daemon_thread[**Parameters, Result](
    function: Callable[Parameters, Result],
    *args: Parameters.args,
    **kwargs: Parameters.kwargs,
) -> Result:
    """Run one blocking operation without borrowing asyncio's default executor.

    The worker is daemonized so a cancelled or shutdown-bound operation cannot
    keep an asyncio runner waiting for its default executor. Backend-owned
    cancellation should still be requested before the owning runtime is closed.

    Args:
        function: Blocking callable to execute.
        args: Positional arguments for ``function``.
        kwargs: Keyword arguments for ``function``.

    Returns:
        _Result: The value returned by ``function``.
    """
    loop = asyncio.get_running_loop()
    result: asyncio.Future[Result] = loop.create_future()
    context = contextvars.copy_context()

    def complete(value: Result) -> None:
        if not result.done():
            result.set_result(value)

    def fail(error: BaseException) -> None:
        if not result.done():
            result.set_exception(error)

    def run() -> None:
        try:
            value = context.run(function, *args, **kwargs)
        except BaseException as error:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(fail, error)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(complete, value)

    threading.Thread(target=run, name="celune-async", daemon=True).start()
    return await result
