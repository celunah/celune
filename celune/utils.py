"""Celune common utility functions."""

import math
import datetime
import subprocess
import multiprocessing
from typing import Union, Callable

from celune.constants import REFERENCE_NEW_MOON


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


def lunar_info(dt: datetime.datetime) -> tuple[float, float, float]:
    """Get lunar state from the given date and time.

    Args:
        dt: The date and time to use.

    Returns:
        tuple[float, float, float]: The lunar phase, illumination level and days until a full moon.
    """
    frac_dt = dt.astimezone(datetime.timezone.utc)
    since_ref = (frac_dt - REFERENCE_NEW_MOON).total_seconds() / 86400
    cycle_days = 29.530588
    phase = (since_ref / cycle_days) % 1.0
    illumination = 0.5 * (1 - math.cos(2 * math.pi * phase))
    days_until_full = ((0.5 - phase) % 1.0) * cycle_days

    return phase, illumination, days_until_full


def lunar_phase(phase: float) -> str:
    """Convert a phase float to a phase name.

    Args:
        phase: The floating point phase.
    Returns:
        str: The corresponding phase name.
    """
    if phase < 0.03 or phase >= 0.97:
        return "new moon"
    if phase < 0.22:
        return "waxing crescent"
    if phase < 0.28:
        return "first quarter"
    if phase < 0.47:
        return "waxing gibbous"
    if phase < 0.53:
        return "full moon"
    if phase < 0.72:
        return "waning gibbous"
    if phase < 0.78:
        return "last quarter"

    return "waning crescent"


def celune_day_status(now: datetime.datetime) -> str:
    """Return a formatted Celune Day status message.

    Args:
        now: The current date and time.
    Returns:
        str: The formatted Celune Day status message.
    """
    celune_day_this_year = datetime.datetime(now.year, 6, 2)

    if now.date() == celune_day_this_year.date():
        return f"today is Celune Day {now.year}"

    if now > celune_day_this_year:
        next_celune_day = datetime.datetime(now.year + 1, 6, 2)
    else:
        next_celune_day = celune_day_this_year

    days_until = (next_celune_day.date() - now.date()).days
    suffix = "s" if days_until != 1 else ""
    return f"{days_until} day{suffix} until Celune Day {next_celune_day.year}"


def range_interpolated(
    value: float, lo: Union[int, float], hi: Union[int, float], power: float = 3.0
) -> Union[int, float]:
    """Get interpolated number within a specified range.

    Args:
        value: The number (0-1) to convert to interpolated value.
        lo: The lower bound of the interpolated range.
        hi: The upper bound of the interpolated range.
        power: How strongly to interpolate the number.

    Returns:
        Union[int, float]: The interpolated number.
    """
    clamped = max(0.0, min(1.0, value))
    value = clamped**power
    return lo + value * (hi - lo)


def cuda_architecture(capability: tuple[int, int]) -> str:
    """Convert a CUDA capability tuple to an architecture name.

    Args:
        capability: CUDA capability formatted as tuple.

    Returns:
        str: The architecture name.
    """

    major, minor = capability

    if major in [10, 11, 12] and minor == 0:
        return "Blackwell"
    if major == 9 and minor == 0:
        return "Hopper"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 8 and minor in [0, 6, 7]:  # CELINE INVADED THE CUDA ZONE!
        return "Ampere"
    if major < 8:
        raise NotImplementedError("capability not supported")

    raise ValueError("invalid capability")


def run_async(
    func: Callable, *args, daemon: bool = True, **kwargs
) -> multiprocessing.Process:
    """Run a function asynchronously.

    Args:
        func: The function to call. The function cannot reuse the current process's state.
        args: The arguments to pass to the function.
        daemon: Whether to use a daemon process. Defaults to True.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        multiprocessing.Process: The process object.
    """
    proc = multiprocessing.Process(
        target=func,
        args=args,
        kwargs=kwargs,
        daemon=daemon,
    )
    proc.start()
    return proc
