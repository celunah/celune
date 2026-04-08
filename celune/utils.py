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
