# SPDX-License-Identifier: MIT
"""Terminal color capability helpers for Celune."""

import os
import sys
import ctypes
from collections.abc import Mapping
from typing import IO, Final, Literal, Optional, Callable, Any, cast

from .config import config_value
from .constants import JSONSerializable

type Config = Mapping[str, JSONSerializable]
type ColorMode = Literal["auto", "truecolor", "terminal-default", "ansi", "none"]
type ResolvedColorMode = Literal["truecolor", "terminal-default", "ansi", "none"]

VALID_COLOR_MODES: Final[frozenset[str]] = frozenset(
    {"auto", "truecolor", "terminal-default", "ansi", "none"}
)
_TRUECOLOR_HINTS: Final[tuple[str, ...]] = ("truecolor", "24bit", "24-bit")
_ANSI_TERM_HINTS: Final[tuple[str, ...]] = (
    "color",
    "ansi",
    "xterm",
    "screen",
    "tmux",
    "rxvt",
    "vt100",
    "linux",
    "cygwin",
)


def normalize_color_mode(value: JSONSerializable) -> Optional[ColorMode]:
    """Normalize one configured color-mode value.

    Args:
        value: The color-mode value.

    Returns:
        Optional[ColorMode]: The normalized color-mode value.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in VALID_COLOR_MODES:
        return cast(ColorMode, normalized)
    return None


def configured_color_mode(config: Optional[Config] = None) -> ColorMode:
    """Return the requested Celune color mode before capability detection.

    Args:
        config: Celune's current configuration.

    Returns:
        ColorMode: The currently selected color mode. It isn't used at this time.
    """
    env_value = normalize_color_mode(os.getenv("CELUNE_COLOR_MODE"))
    if env_value is not None:
        return env_value

    config_value_mode = normalize_color_mode(config_value(config, "color_mode", "auto"))
    if config_value_mode is not None:
        return config_value_mode
    return "auto"


def no_color_requested() -> bool:
    """Return whether the ``NO_COLOR`` convention disables color output.

    Returns:
        bool: Whether Celune was requested to run without colors.
    """
    return "NO_COLOR" in os.environ


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

    windll = cast(Callable[..., Any], windll)

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


def _normalize_reported_color_system(value: Optional[str]) -> Optional[str]:
    """Normalize a Rich/Textual-reported color-system label."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered or None
    return None


def _truecolor_hint_from_environment() -> bool:
    """Return whether environment variables strongly suggest true-color support."""
    colorterm = os.getenv("COLORTERM", "").strip().lower()
    term = os.getenv("TERM", "").strip().lower()
    if any(hint in colorterm for hint in _TRUECOLOR_HINTS):
        return True
    return "direct" in term or "truecolor" in term


def _ansi_hint_from_environment() -> bool:
    """Return whether environment variables suggest ANSI color support."""
    colorterm = os.getenv("COLORTERM", "").strip().lower()
    term = os.getenv("TERM", "").strip().lower()
    if term in {"", "dumb"}:
        return False
    if colorterm:
        return True
    return any(hint in term for hint in _ANSI_TERM_HINTS)


def resolve_color_mode(
    config: Optional[Config] = None,
    *,
    reported_color_system: Optional[str] = None,
    stream: Optional[IO[str]] = None,
) -> ResolvedColorMode:
    """Resolve the color mode Celune should actually use on this terminal.

    Args:
        config: Celune's current configuration.
        reported_color_system: The reported color system.
        stream: The terminal's stream.

    Returns:
        ResolvedColorMode: The color mode Celune should actually use at this time.
    """
    configured = configured_color_mode(config)
    if no_color_requested() and configured != "none":
        return "none"
    if configured != "auto":
        return cast(ResolvedColorMode, configured)

    detected = _normalize_reported_color_system(reported_color_system)
    if detected == "truecolor":
        return "truecolor"
    if detected in {"standard", "256", "windows"}:
        return "terminal-default"

    if _truecolor_hint_from_environment():
        return "truecolor"
    if _ansi_hint_from_environment() or supports_ansi(stream):
        return "terminal-default"
    return "none"


def supports_truecolor(
    config: Optional[Config] = None,
    *,
    reported_color_system: Optional[str] = None,
    stream: Optional[IO[str]] = None,
) -> bool:
    """Return whether Celune should emit true-color styles.

    Args:
        config: Celune's current configuration.
        reported_color_system: The reported color system.
        stream: The terminal's stream.

    Returns:
        bool: Whether the underlying supports True Color.
    """
    return (
        resolve_color_mode(
            config,
            reported_color_system=reported_color_system,
            stream=stream,
        )
        == "truecolor"
    )
