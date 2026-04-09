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
    precision_digits = len(str(num).split(".", maxsplit=1)[1])

    while precision_digits > 0:
        str_rep = str(round(num, precision or precision_digits))
        if str_rep[-1] == "0":
            precision_digits -= 1
            continue
        return str_rep
    return str(int(num))


def to_rgb(color: str) -> tuple[int, int, int]:
    """Convert hex code to RGB tuple."""
    if color.startswith("#"):
        color = color[1:]
    elif color.lower().startswith("0x"):
        color = color[2:]

    color = color.strip()

    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6 or any(c.lower() not in "0123456789abcdef" for c in color):
        raise ValueError(f"expected a 3 or 6-character hex code, found {color}")

    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
