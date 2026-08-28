# SPDX-License-Identifier: Apache-2.0
"""Runtime and environment validation helpers."""

import sys
import platform
from typing import Union
from collections.abc import Callable

import torch

from . import __comment__, __version__, __codename__
from .i18n import string
from .utils import (
    format_number,
    cuda_architecture,
)
from ._version import DEVELOPMENT
from .constants import APP_NAME, NVIDIA_DEVICE_KEYWORDS
from .backends.vc import CeluneVCBackend
from .backends.tts import CeluneBackend
from .typing.modes import BackendMode
from .typing.aliases import LogLevel, LogCallback


def log_runtime_banner(
    log: Callable[[str, str], None],
    backend: Union[CeluneBackend, CeluneVCBackend],
    backend_mode: BackendMode = "normal",
) -> None:
    """Log high-level version and environment information.

    Args:
        log: Logging callback that receives the generated banner lines.
        backend: The backend with which Celune was started.
        backend_mode: The restricted Celune backend mode, when applicable.
    """
    cuda_version = torch.version.cuda

    cuda_line = f", CUDA {cuda_version}" if cuda_version else ""
    if backend_mode == "agent_test":
        backend_line = string("runtime.agent_test_backend")
    elif backend_mode == "ui_test" or backend.is_fake:
        backend_line = string("runtime.ui_test_backend")
    else:
        backend_line = f"on backend {backend.name}, "

    log(
        f"{APP_NAME} {__version__} "
        f"{backend_line}"
        f"Python {platform.python_version()}, "
        f"PyTorch {torch.__version__}"
        f"{cuda_line}",  # NOTE: may concatenate an empty string if CUDA support is not present
        "info",
    )
    log(
        f'{__codename__} - "{__comment__}"',
        "info",
    )

    if DEVELOPMENT:
        log(string("celune.development_version"), "warning")

    if not backend.is_fake:
        log(string("celune.testing_environment"), "info")


def check_supported_backends() -> tuple[str, bool]:
    """Check supported backends and report whether the app can use them.

    Returns:
        tuple[str, bool]: A backend name and whether the app can use the backend.
    """

    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None) is not None:
            return "ROCm", False

        try:
            device_name = torch.cuda.get_device_name(0).lower()
        except (RuntimeError, AssertionError):
            # sometimes CUDA can be available but not yet usable, bail out here
            return "CUDA", False

        if any(keyword in device_name for keyword in NVIDIA_DEVICE_KEYWORDS):
            return "CUDA", True
        return "ZLUDA", True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS", False

    return "CPU", False


def _run_compute_test(device: Union[torch.device, str]) -> torch.Tensor:
    """Run a small stress test on the selected device to determine whether CUDA is usable."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    x = torch.randn(2048, 2048, device=device, dtype=dtype)
    y = torch.randn(2048, 2048, device=device, dtype=dtype)

    torch.cuda.synchronize()
    z = x @ y

    torch.cuda.synchronize()
    for _ in range(1000):
        z = x @ y
        z = torch.relu(z)

    torch.cuda.synchronize()
    chk = z.mean().detach()

    return chk


def validate_runtime(
    log: LogCallback,
    error: Callable[[str], None],
    set_state: Callable[[str], None],
    glow_connect_failed: bool,
    format_error: Callable[[Exception, Union[LogLevel, bool]], str],
    log_level: LogLevel,
    backend_name: str = "qwen3",
) -> bool:
    """Validate the app's Python, CUDA, and GPU environment.

    Args:
        log: Logging callback for informational and error messages.
        error: Error callback for surfaced user-facing failures.
        set_state: Callback used to update the app runtime state.
        glow_connect_failed: Whether the OpenRGB glow backend failed to connect.
        format_error: Error formatter used for exception messages.
        log_level: The active Celune log level.
        backend_name: The active app backend name selected for this session.

    Returns:
        bool: ``True`` when the runtime environment is supported and usable, otherwise ``False``.
    """
    cuda_version = torch.version.cuda
    _, separator, torch_variant = torch.__version__.partition("+")

    def format_runtime_error(message: str, error: Exception) -> str:
        """Attach the configured runtime exception detail to one message."""
        details = format_error(error, log_level)
        if not details:
            return message
        separator = "\n" if log_level == "debug" else ": "
        return f"{message}{separator}{details}"

    if sys.version_info < (3, 12) or sys.version_info >= (3, 15):
        log(
            f"{APP_NAME} does not currently support Python "
            f"{platform.python_version()}.",
            "error",
        )
        log(
            f"Run `uv sync` in {APP_NAME}'s directory to set up the environment, "
            f"then restart {APP_NAME}.",
            "error",
        )
        set_state("error")
        error(string("runtime.incompatible_python"))
        return False

    backend, usable = check_supported_backends()
    if backend == "ZLUDA":
        log(string("runtime.zluda_supported"), "info")
    else:
        log(f"Current system supports {backend} execution.", "info")

    allow_cpu_mini = backend == "CPU" and backend_name.strip().lower() == "mini"
    if allow_cpu_mini:
        log(string("runtime.mini_startup", app_name=APP_NAME), "info")
        usable = True

    if not usable:
        log(f"{APP_NAME} does not currently support {backend} execution.", "error")
        set_state("error")
        error(string("runtime.no_supported_backend"))
        return False

    if allow_cpu_mini:
        if glow_connect_failed:
            log(
                string("runtime.openrgb_unavailable"),
                "warning",
            )
        return True

    if cuda_version is None:
        log(string("runtime.cuda_missing", app_name=APP_NAME), "error")

        if separator and torch_variant == "cpu":
            log(string("runtime.pytorch_cpu_build"), "error")
        else:
            log(string("runtime.pytorch_unsupported_build"), "error")

        set_state("error")
        error(string("runtime.no_cuda_runtime"))
        return False

    if backend == "ZLUDA":
        log(
            string("runtime.zluda_detected"),
            "warning",
        )
        log(string("runtime.zluda_performance", app_name=APP_NAME), "warning")

    cuda_version_tuple = tuple(map(int, cuda_version.split(".")))
    if cuda_version_tuple not in {(12, 8), (13, 0)}:
        log(
            f"{APP_NAME} only supports CUDA 12.8 or 13.0, "
            f"found version {torch.version.cuda}.",
            "error",
        )
        set_state("error")
        error(string("runtime.incompatible_cuda"))
        return False

    cuda_avail = torch.cuda.is_available()
    log(f"CUDA available: {cuda_avail}", "info")

    # devices to test
    devices = torch.cuda.device_count()
    if devices == 0:
        log(string("runtime.no_gpus"), "error")
        set_state("error")
        error(string("runtime.cuda_unavailable"))
        return False

    for i in range(devices):
        try:
            gpu = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
        except Exception as exc:
            log(
                format_runtime_error(f"Could not query GPU {i} capability", exc),
                "error",
            )
            set_state("error")
            error(string("runtime.unsupported_gpu"))
            return False
        try:
            log(
                f"GPU {i}: {gpu} ({cuda_architecture((major, minor))}) - "
                f"CUDA capability: {major}.{minor}",
                "info",
            )
        except (ValueError, NotImplementedError):
            log(
                f"GPU {i}: {gpu} (not supported) - CUDA capability: {major}.{minor}",
                "info",
            )
            log(f"{APP_NAME} does not support this GPU.", "error")
            log(f"{APP_NAME} requires Ampere or newer.", "error")
            log(
                "If you have another supported GPU, set CUDA_VISIBLE_DEVICES "
                "appropriately.",
                "error",
            )
            set_state("error")
            error(string("runtime.unsupported_gpu"))
            return False

        try:
            log(f"Testing GPU {i}...", "info")
            vtensor = _run_compute_test(f"cuda:{i}")
            if log_level != "info":
                log(
                    f"Compute test for GPU {i} succeeded, "
                    f"result: {format_number(float(vtensor.item()))}",
                    "info",
                    loglevel="verbose",
                )
            else:
                log(f"Compute test for GPU {i} succeeded", "info")
        except Exception as e:
            log(
                format_runtime_error(f"Compute test for GPU {i} failed", e),
                "warning",
            )

            # decrement device count if tests fail
            devices -= 1

        # usable devices
        if devices == 0:
            log(
                "GPU tests reported that all GPUs have failed their runtime checks.",
                "error",
            )
            return False

    # found several usable devices
    if devices > 1:
        unused_devices = devices - 1
        log(f"{unused_devices} working GPUs will not be used.", "warning")

    if glow_connect_failed:
        log(string("runtime.openrgb_unavailable"), "warning")

    try:
        __import__("flash_attn")

        has_flash_attn = True
    except ModuleNotFoundError:
        has_flash_attn = False

    if has_flash_attn:
        log(
            "Flash Attention is not compatible.",
            "warning",
        )

    return True
