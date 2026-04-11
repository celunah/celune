"""Celune common utility functions."""

import subprocess


def get_revision() -> str:
    """Get current Git repo revision."""
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
    """Format a number without trailing zeroes."""
    if precision < 0:
        raise ValueError("precision must be >= 0")

    digits = precision if precision > 0 else 12
    text = f"{num:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def to_rgb(color: str) -> tuple[int, ...]:
    """Convert hex code to RGB tuple."""
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
