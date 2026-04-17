"""Celune common utility functions."""

import subprocess


def get_revision() -> str:
    """Get the current Git repository revision.

    Returns:
        str: The short commit hash, suffixed with ``*`` when the worktree is dirty,
            or an empty string when Git metadata is unavailable.
    """
    try:
        rev = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        dirty = "*" if status else ""
        return f"{rev}{dirty}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def format_number(num: float, precision: int = 0) -> str:
    """Format a number without trailing zeroes.

    Args:
        num: The numeric value to format.
        precision: The number of decimal places to preserve before trimming.

    Returns:
        str: The formatted numeric string.
    """
    if precision < 0:
        raise ValueError("precision must be >= 0")

    digits = precision if precision > 0 else 12
    text = f"{num:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def to_rgb(color: str) -> tuple[int, ...]:
    """Convert a hexadecimal color code to an RGB tuple.

    Args:
        color: A 3-digit or 6-digit hexadecimal color string, optionally prefixed
            with ``#`` or ``0x``.

    Returns:
        tuple[int, ...]: The parsed ``(red, green, blue)`` color components.
    """
    color = color.strip()

    if color.startswith("#"):
        color = color[1:]
    elif color.lower().startswith("0x"):
        color = color[2:]

    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6 or any(c.lower() not in "0123456789abcdef" for c in color):
        raise ValueError(f"expected a 3 or 6-character hex code, found {color}")

    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
