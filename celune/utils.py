"""Celune common utility functions."""

import sys
import math
import inspect
import datetime
import traceback
import subprocess
import multiprocessing
from pathlib import Path
from typing import Union, Callable, Optional, Literal, TypedDict

from celune import colors
from celune.constants import REFERENCE_NEW_MOON


class CallerInfo(TypedDict):
    """Caller information type annotation."""

    function: str
    filename: str
    line: int


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

    Raises:
        ValueError: ``precision`` is negative.
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

    Raises:
        ValueError: ``color`` is not a valid 3- or 6-character hex code.
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

    Raises:
        NotImplementedError: The CUDA capability is below Celune's supported
            minimum.
        ValueError: The CUDA capability is not recognized.
    """

    major, minor = capability

    if major in [10, 11, 12] and minor == 0:  # recommended family
        return "Blackwell"
    if major == 9 and minor == 0:
        return "Hopper"
    if major == 8 and minor == 9:
        return "Ada Lovelace"
    if major == 8 and minor in [0, 6, 7]:  # CELINE INVADED THE CUDA ZONE!
        return "Ampere"
    if major < 8:  # too old
        raise NotImplementedError("capability not supported")

    raise ValueError(
        "invalid capability"
    )  # non-CUDA GPU reported a capability not known to Celune


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


def supports_ansi() -> bool:
    """Does the terminal support ANSI color codes?

    Returns:
        bool: Whether the terminal supports ANSI color codes.
    """
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not is_tty:  # non-interactive terminals don't support ANSI
        return False

    if (
        sys.platform != "win32"
    ):  # interactive terminals on Linux systems should already be ANSI capable
        return True

    try:
        import ctypes
    except ModuleNotFoundError:
        return False

    # get stdout handle
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    stdout_handle = kernel32.GetStdHandle(-11)
    invalid_handle = ctypes.c_void_p(-1).value
    # no handle found
    if stdout_handle in (0, invalid_handle):
        return False

    # check stdout handle, bail out if none found
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode)):  # invalid handle
        return False

    enable_virtual_terminal_processing = 0x0004
    if (
        mode.value & enable_virtual_terminal_processing
    ):  # try to enable ANSI mode, bail out if not set
        return True  # ANSI mode was enabled

    return bool(  # try to enable ANSI mode using alternative call syntax, bail out if not set
        kernel32.SetConsoleMode(
            stdout_handle, mode.value | enable_virtual_terminal_processing
        )
    )


def format_error(e: Exception, dev: bool) -> str:
    """Format an error message.

    Args:
        e: The exception to format.
        dev: Whether developer mode is enabled.

    Returns:
        str: Either the full traceback or the exception text.
    """
    if dev:
        trace = traceback.format_exc()
        with open("celune_traceback.txt", "w", encoding="utf-8") as f:
            f.write(trace)

    details = str(e) or "no error description"
    return traceback.format_exc() if dev else details


def indent(text: str, spaces: int, direction: Literal["left", "right"] = "left") -> str:
    """Indent a string from left or the right.

    Args:
        text: The text to indent.
        spaces: How many spaces to indent with.
        direction: The direction to indent from, must be horizontal.

    Returns:
        str: The indented string.

    Raises:
        ValueError: Invalid indenting direction.
    """
    if direction == "left":
        return " " * spaces + text
    if direction == "right":
        return text + " " * spaces

    raise ValueError("can't indent from this direction")


def get_caller() -> Optional[CallerInfo]:
    """Get information on the caller importing a package.

    Returns:
        dict: The caller's information.
    """
    for frame in inspect.stack():
        filename = frame.filename
        current = Path(__file__).resolve().parts[-2]

        if "importlib" in filename or filename.startswith("<frozen"):
            continue

        if current in filename:
            continue

        return {
            "function": frame.function,
            "filename": frame.filename,
            "line": frame.lineno,
        }

    return None


def caller_is_repl() -> bool:
    """Is the caller importing Celune the Python REPL?

    Returns:
        bool: If the caller is the Python REPL.
    """
    caller = get_caller()

    if caller is not None:
        return caller["filename"].startswith("<python-input-")

    return False


def random_hex() -> str:
    """Return a random six-digit hex color.

    Returns:
        str: The random hex color.
    """
    return colors.random_hex()
