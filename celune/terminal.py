# SPDX-License-Identifier: Apache-2.0
"""Terminal handling helpers for Celune."""

import ctypes
import sys
from collections.abc import Callable
from typing import IO, Optional, cast


def supports_ansi(stream: Optional[IO[str]] = None) -> bool:
    """Return whether the current terminal supports ANSI escape codes.

    Args:
        stream: The terminal's stream.

    Returns:
        bool: Whether the underlying terminal supports ANSI.
    """
    output = sys.stdout if stream is None else stream
    is_tty = hasattr(output, "isatty") and output.isatty()
    if not is_tty:
        return False

    if sys.platform != "win32":
        return True

    windll = getattr(ctypes, "WinDLL", None)
    if not callable(windll):
        return False

    windll = cast(Callable[..., ctypes.CDLL], windll)

    kernel32 = windll("kernel32", use_last_error=True)
    handle_id = -12 if output is sys.stderr else -11
    stdout_handle = kernel32.GetStdHandle(handle_id)
    invalid_handle = ctypes.c_void_p(-1).value
    if stdout_handle in (0, invalid_handle):
        return False

    mode = ctypes.c_uint32(0)
    if not kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):
        return False

    enable_virtual_terminal_processing = 0x0004
    if mode.value & enable_virtual_terminal_processing:
        return True

    return bool(
        kernel32.SetConsoleMode(
            stdout_handle, mode.value | enable_virtual_terminal_processing
        )
    )
