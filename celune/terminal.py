# SPDX-License-Identifier: Apache-2.0
"""Terminal handling helpers for Celune."""

import ctypes
import re
import sys
from collections.abc import Callable
from typing import IO, Optional, cast

from .i18n import string

_TERMINAL_TITLE_MAX_LENGTH = 40
_TERMINAL_TITLE_STATUS_MAX_LENGTH = 20
_TERMINAL_TITLE_CONTROL_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
    r"|[\x00-\x1f\x7f\x80-\x9f\r\n]"
)

RUNTIME_LOG_FILTER_MESSAGES = frozenset(
    {
        "`torch_dtype` is deprecated! Use `dtype` instead!",
        "Skipped loading some keys due to shape mismatch:",
        "cfm loaded",
        "length_regulator loaded",
        "Removing weight norm...",
        "Loading weights from",
        "Loading Text2Semantic weights from",
        "Loading Text2Semantic Weights from",
        "min value is",
        "max value is",
        "generation flags are not valid and may be ignored:",
        "it/s]",
        "s/it]",
        "inputs will be cast",
        "Ignoring clean_up_tokenization_spaces=True for BPE tokenizer",
        "You are sending unauthenticated requests",
        "triton not found",
        "A custom logits processor of type",
    }
)


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


def terminal_title_escape(status: tuple[str, str, str]) -> str:
    """Build an OSC title escape sequence from an app state and action.

    Args:
        status: Application name, stable state label, and current action.

    Returns:
        str: The sanitized OSC 0 terminal-title escape sequence.
    """
    app_name, state, action = (
        _TERMINAL_TITLE_CONTROL_RE.sub("", part).strip() for part in status
    )
    title_key = "osc.title_state_only" if state == action else "osc.title"
    title = string(title_key, app_name=app_name, state=state, action=action)
    if title.startswith(app_name):
        title_prefix = app_name
        title_suffix = title[len(app_name) :]
    else:
        title_prefix = ""
        title_suffix = title
    if len(title_suffix) > _TERMINAL_TITLE_STATUS_MAX_LENGTH:
        title = (
            f"{title_prefix}{title_suffix[: _TERMINAL_TITLE_STATUS_MAX_LENGTH - 1]}…"
        )
    if len(title) > _TERMINAL_TITLE_MAX_LENGTH:
        title = f"{title[: _TERMINAL_TITLE_MAX_LENGTH - 1]}…"
    return f"\x1b]0;{title}\x07"


def set_terminal_title(
    status: tuple[str, str, str],
    output: Optional[IO[str]] = None,
) -> None:
    """Set the current terminal title to an app state and action.

    Args:
        status: Application name, stable state label, and current action.
        output: Terminal stream to receive the escape sequence.
    """
    target = sys.stdout if output is None else output
    target.write(terminal_title_escape(status))
    target.flush()
