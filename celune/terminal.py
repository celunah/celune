# SPDX-License-Identifier: Apache-2.0
"""Terminal handling helpers for Celune."""

import re
import sys
import base64
import ctypes

# noinspection PyProtectedMember
from typing import IO, Optional, cast
from collections.abc import Callable

from .i18n import string

_TERMINAL_TITLE_MAX_LENGTH = 40
_TERMINAL_TITLE_STATUS_MAX_LENGTH = 20
_TERMINAL_TITLE_CONTROL_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])"
    r"|[\x00-\x1f\x7f\x80-\x9f\r\n]"
)
_OSC_7501_STATES = frozenset({"idle", "working", "done", "blocked", "error"})
_OSC_7501_APP_RE = re.compile(r"[^A-Za-z0-9_.+/-]")
_OSC_7501_MESSAGE_MAX_BYTES = 2048

RUNTIME_LOG_FILTER_MESSAGES = frozenset(
    {
        "`torch_dtype` is deprecated",
        "Skipped loading some keys due to shape mismatch",
        "cfm loaded",
        "length_regulator loaded",
        "Removing weight norm",
        "Loading weights from",
        "Loading Text2Semantic weights from",
        "min value is",
        "max value is",
        "generation flags are not valid and may be ignored",
        "it/s]",
        "s/it]",
        "inputs will be cast",
        "Ignoring clean_up_tokenization_spaces=True for BPE tokenizer",
        "You are sending unauthenticated requests",
        "triton not found",
        "A custom logits processor of type",
        "FireRedTTS3 (text front-end) loaded",
        "with an incorrect regex pattern",
        "SyntaxWarning: invalid escape sequence",
        "regex for parameter names, must start with",
        "Redirects are currently not supported in Windows or Mac",
        "flash-attn is not installed",
        "Could not infer sample rate from base model",
        "Warming up predictor",
        "Capturing CUDA graph",
        "CUDA graph captured",
        "Warming up talker graph",
        "********",
        "The tokenizer class you load from this checkpoint is",
        "The class this function is called from",
        "HTTP Request:",
    }
)


def infer_log_severity(message: str, default_severity: str) -> str:
    """Infer severity from message content before using its stream default."""
    lowered = message.casefold()

    level_match = re.search(
        r"\[(?P<bracket>info|notice|warn(?:ing)?|error|critical|fatal)]"
        r"|(?:^|\s)(?P<plain>info|notice|warn(?:ing)?|error|critical|fatal)"
        r"(?:\s*:|\s+-)",
        lowered,
    )
    if level_match is not None:
        level = level_match.group("bracket") or level_match.group("plain")
        if level in {"error", "critical", "fatal"}:
            return "error"
        if level in {"warn", "warning"}:
            return "warning"
        return "info"

    if "traceback (most recent call last):" in lowered:
        return "error"
    if re.search(
        r"\b[a-z_][a-z0-9_.]*(?:error|exception):",
        lowered,
    ) or re.search(r"\b(?:error|exception|fatal(?: error)?)\b", lowered):
        return "error"
    if re.search(
        (
            r"\b(?:warn(?:ing)?|futurewarning|deprecationwarning|"
            r"pendingdeprecationwarning|runtimewarning|resourcewarning|"
            r"userwarning|syntaxwarning|importwarning|unicodewarning|byteswarning|"
            r"deprecated|deprecation)\b"
        ),
        lowered,
    ):
        return "warning"
    return default_severity


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


def _program_status_state(state: str) -> str:
    """Map Celune's runtime state name to the OSC 7501 state vocabulary."""
    if state in _OSC_7501_STATES:
        return state
    if state == "awaiting":
        return "blocked"
    if state == "exiting":
        return "done"
    if state in {"ready", "sleeping", "stopped", "paused", "warning"}:
        return "idle"
    return "working"


def _program_status_state_from_title(state: str) -> str:
    """Infer the OSC 7501 state from a localized title label."""
    state_by_label = {
        string("osc.state_awaiting").casefold(): "blocked",
        string("osc.state_error").casefold(): "error",
        string("osc.state_exiting").casefold(): "done",
        string("osc.state_ready").casefold(): "idle",
        string("osc.state_sleeping").casefold(): "idle",
        string("osc.state_stopped").casefold(): "idle",
        string("osc.state_paused").casefold(): "idle",
    }
    return state_by_label.get(state.casefold(), "working")


def terminal_status_escape(
    status: tuple[str, str, str],
    *,
    program_state: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """Build terminal title, progress, and OSC 7501 program-status escapes.

    Args:
        status: Application name, visible state label, and concise action label.
        program_state: Celune state name, mapped to the OSC 7501 vocabulary.
        message: Full current status text to include as base64 UTF-8 in ``msg``.

    Returns:
        str: The combined OSC 0, OSC 9;4, and OSC 7501 escape sequences.
    """
    app_name = _TERMINAL_TITLE_CONTROL_RE.sub("", status[0]).strip()
    visible_state = _TERMINAL_TITLE_CONTROL_RE.sub("", status[1]).strip()
    action = _TERMINAL_TITLE_CONTROL_RE.sub("", status[2]).strip()
    osc_state = (
        _program_status_state(program_state)
        if program_state is not None
        else _program_status_state_from_title(visible_state)
    )
    if osc_state not in _OSC_7501_STATES:
        osc_state = "working"

    app_value = _OSC_7501_APP_RE.sub("", app_name)[:32]
    status_message = action if message is None else message
    status_message = _TERMINAL_TITLE_CONTROL_RE.sub("", status_message).strip()
    message_bytes = status_message.encode("utf-8")[:_OSC_7501_MESSAGE_MAX_BYTES]
    status_message = message_bytes.decode("utf-8", errors="ignore")
    encoded_message = base64.b64encode(status_message.encode("utf-8")).decode("ascii")

    if osc_state == "working":
        progress = "3"
    elif osc_state == "error":
        progress = "2"
    else:
        progress = "0"

    return (
        f"{terminal_title_escape((app_name, visible_state, action))}"
        f"\x1b]9;4;{progress}\x07"
        f"\x1b]7501;state={osc_state}:app={app_value}:msg={encoded_message}\x1b\\"
    )


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


def set_terminal_status(
    status: tuple[str, str, str],
    output: Optional[IO[str]] = None,
    *,
    program_state: Optional[str] = None,
    message: Optional[str] = None,
) -> None:
    """Publish Celune's title, progress indicator, and detailed program status.

    Args:
        status: Application name, visible state label, and concise action label.
        output: Terminal stream to receive the escape sequences.
        program_state: Celune state name, mapped to the OSC 7501 vocabulary.
        message: Full current status text to include as base64 UTF-8 in ``msg``.
    """
    target = sys.stdout if output is None else output
    target.write(
        terminal_status_escape(
            status,
            program_state=program_state,
            message=message,
        )
    )
    target.flush()
