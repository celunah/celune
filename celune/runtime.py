# pylint: disable=R0913, R0914, R0917, W0718
"""Runtime and environment validation helpers for Celune."""

import os
import sys
import platform
from typing import Callable

import torch

from . import __codename__, __comment__, __version__


def log_runtime_banner(log: Callable[[str, str], None]) -> None:
    """Log high-level version and environment information."""
    cuda_version = torch.version.cuda
    quotation_marks = ("\u201c", "\u201d") if \
        (os.getenv("CELUNE_HEADLESS") not in {"1", "true", "on"} or sys.stdout.isatty()) else ('"', '"')

    log(
        f"Celune {__version__}, "
        f"Python {platform.python_version()}, "
        f"PyTorch {torch.__version__}, "
        f"CUDA {cuda_version}",
        "info",
    )
    log(f'{__codename__} - {quotation_marks[0]}{__comment__}{quotation_marks[1]}', "info")
    log("Environment test...", "info")


def validate_runtime(
    log: Callable[[str, str], None],
    error: Callable[[str], None],
    set_state: Callable[[str], None],
    glow_connect_failed: bool,
    format_error: Callable[[Exception, bool], str],
    dev: bool,
) -> bool:
    """Validate Celune's Python, CUDA, and GPU environment."""
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

    if cuda_version is None:
        log(
            "The currently installed PyTorch build does not include CUDA support.",
            "error",
        )
        set_state("error")
        error("PyTorch has no CUDA support")
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

    if not cuda_avail:
        log("No GPUs found.", "error")
        set_state("error")
        error("CUDA is not available")
        return False

    try:
        current_gpu = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        log(f"GPU: {current_gpu} (Capability: {major}.{minor})", "info")

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
