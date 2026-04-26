# pylint: disable=R0913, R0914, R0917, W0718
"""Runtime and environment validation helpers for Celune."""

import sys
import platform
import datetime
from typing import Callable

import torch

from . import __codename__, __comment__, __version__
from .utils import cuda_architecture, lunar_info, celune_day_status, lunar_phase


def log_runtime_banner(log: Callable[[str, str], None], backend_name: str) -> None:
    """Log high-level version and environment information.

    Args:
        log: Logging callback that receives the generated banner lines.
        backend_name: Optional backend name shown in the runtime banner.

    Returns:
        None: This function emits startup information through the log callback.
    """
    cuda_version = torch.version.cuda

    cuda_line = f", CUDA {cuda_version}" if cuda_version else ""

    log(
        f"Celune {__version__} "
        f"on backend {backend_name}, "
        f"Python {platform.python_version()}, "
        f"PyTorch {torch.__version__}"
        f"{cuda_line}",  # NOTE: may concatenate an empty string if CUDA support is not present
        "info",
    )
    log(
        f'{__codename__} - "{__comment__}"',
        "info",
    )

    # Celune reports the state of the moon and when the next Celune Day will occur below
    now = datetime.datetime.now()

    lunar = lunar_info(now)
    days_until_full_moon = int(lunar[2])
    prefix = "is" if days_until_full_moon == 1 else "are"
    suffix = "s" if days_until_full_moon != 1 else ""
    phase = lunar[0]
    celune_day_message = celune_day_status(now)

    log(
        f"Today is {now.strftime('%A, %B %d, %Y')}, it is a {lunar_phase(phase)}.",
        "info",
    )

    log(
        f"Celune reports there {prefix} {days_until_full_moon} day{suffix} until a full moon, "
        f"{celune_day_message}.",
        "info",
    )

    log("Environment test...", "info")


def check_supported_backends() -> tuple[str, bool]:
    """Check any supported backends and report if Celune can use them.

    Returns:
        tuple[str, bool]: A backend name and whether Celune can use the backend.
    """

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            return "ROCm", False
        return "CUDA", True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS", False

    return "CPU", False


def validate_runtime(
    log: Callable[[str, str], None],
    error: Callable[[str], None],
    set_state: Callable[[str], None],
    glow_connect_failed: bool,
    format_error: Callable[[Exception, bool], str],
    dev: bool,
) -> bool:
    """Validate Celune's Python, CUDA, and GPU environment.

    Args:
        log: Logging callback for informational and error messages.
        error: Error callback for surfaced user-facing failures.
        set_state: Callback used to update Celune's runtime state.
        glow_connect_failed: Whether the OpenRGB glow backend failed to connect.
        format_error: Error formatter used for exception messages.
        dev: Whether developer mode is enabled.

    Returns:
        bool: ``True`` when the runtime environment is supported and usable,
            otherwise ``False``.
    """
    cuda_version = torch.version.cuda

    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        log(
            f"Celune does not currently support Python {platform.python_version()}.",
            "error",
        )
        log(
            "Run `uv sync` in Celune's directory to set up the environment, then restart Celune.",
            "error",
        )
        set_state("error")
        error("Incompatible Python version")
        return False

    backend, usable = check_supported_backends()
    log(f"Current system supports {backend} execution.", "info")

    if not usable:
        log(f"Celune does not currently support {backend} execution.", "error")
        set_state("error")
        error("No supported backend found")
        return False

    cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
    if cuda_version_tuple != (12, 8):
        log(
            f"Celune only supports CUDA 12.8, found version {torch.version.cuda}.",
            "error",
        )
        set_state("error")
        error("Incompatible CUDA version")
        return False

    cuda_avail = torch.cuda.is_available()
    log(f"CUDA available: {cuda_avail}", "info")

    devices = torch.cuda.device_count()
    if devices == 0:
        log("No GPUs found.", "error")
        set_state("error")
        error("CUDA is not available")
        return False

    try:
        for i in range(devices):
            gpu = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            try:
                log(
                    f"GPU {i}: {gpu} ({cuda_architecture((major, minor))}) - CUDA capability: {major}.{minor}",
                    "info",
                )
            except (ValueError, NotImplementedError):
                log(
                    f"GPU {i}: {gpu} (not supported) - CUDA capability: {major}.{minor}",
                    "info",
                )
                log("Celune does not support this GPU.", "error")
                log("Celune requires Ampere or newer.", "error")
                log(
                    "If you have another supported GPU, set CUDA_VISIBLE_DEVICES appropriately.",
                    "error",
                )
                set_state("error")
                error("Unsupported GPU")
                return False

            if devices > 1:
                unused_devices = devices - 1
                log(f"{unused_devices} GPUs will not be used.", "warning")

        log("Compute test...", "info")
        x = torch.rand(256, 256, device="cuda")
        y = x @ x
        log(f"Compute test succeeded on {y.device}", "info")
    except Exception as e:
        log(f"Compute test failed: {format_error(e, dev)}", "error")
        set_state("error")
        error("CUDA device is not usable")
        return False

    if glow_connect_failed:
        log("OpenRGB is not available.", "warning")

    try:
        __import__("flash_attn")

        has_flash_attn = True
    except ModuleNotFoundError:
        has_flash_attn = False

    if has_flash_attn:
        log(
            "Celune has detected that Flash Attention is installed, however it is not currently supported.",
            "warning",
        )

    return True
