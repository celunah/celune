# SPDX-License-Identifier: Apache-2.0
"""Runtime and environment validation helpers."""

import platform
import sys
from collections.abc import Callable
from typing import Union

import torch

from . import __codename__, __comment__, __version__
from ._version import DEVELOPMENT
from .backends.tts import CeluneBackend
from .backends.vc import CeluneVCBackend
from .constants import APP_NAME, NVIDIA_DEVICE_KEYWORDS
from .i18n import string
from .utils import cuda_architecture, format_number
from .typing.aliases import LogCallback, LogLevel
from .typing.modes import BackendMode


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

    if sys.version_info < (3, 12) or sys.version_info >= (3, 15):
        log(
            string(
                "runtime.python_unsupported",
                app_name=APP_NAME,
                version=platform.python_version(),
            ),
            "error",
        )
        log(string("runtime.python_setup", app_name=APP_NAME), "error")
        set_state("error")
        error(string("runtime.incompatible_python"))
        return False

    backend, usable = check_supported_backends()
    if backend == "ZLUDA":
        log(string("runtime.zluda_supported"), "info")
    else:
        log(string("runtime.backend_supported", backend=backend), "info")

    allow_cpu_mini = backend == "CPU" and backend_name.strip().lower() == "mini"
    if allow_cpu_mini:
        log(string("runtime.mini_startup", app_name=APP_NAME), "info")
        usable = True

    if not usable:
        log(
            string("runtime.backend_unsupported", app_name=APP_NAME, backend=backend),
            "error",
        )
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
            string(
                "runtime.cuda_version_unsupported",
                app_name=APP_NAME,
                version=torch.version.cuda,
            ),
            "error",
        )
        set_state("error")
        error(string("runtime.incompatible_cuda"))
        return False

    cuda_avail = torch.cuda.is_available()
    log(string("runtime.cuda_available", available=cuda_avail), "info")

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
                string(
                    "runtime.gpu_capability_failed",
                    index=i,
                    error=format_error(exc, log_level),
                ),
                "error",
            )
            set_state("error")
            error(string("runtime.unsupported_gpu"))
            return False
        try:
            log(
                string(
                    "runtime.gpu_supported",
                    index=i,
                    name=gpu,
                    architecture=cuda_architecture((major, minor)),
                    major=major,
                    minor=minor,
                ),
                "info",
            )
        except (ValueError, NotImplementedError):
            log(
                string(
                    "runtime.gpu_unsupported",
                    index=i,
                    name=gpu,
                    major=major,
                    minor=minor,
                ),
                "info",
            )
            log(string("runtime.gpu_not_supported", app_name=APP_NAME), "error")
            log(string("runtime.gpu_ampere_required", app_name=APP_NAME), "error")
            log(
                string("runtime.cuda_visible_devices"),
                "error",
            )
            set_state("error")
            error(string("runtime.unsupported_gpu"))
            return False

        try:
            log(string("runtime.testing_gpu", index=i), "info")
            vtensor = _run_compute_test(f"cuda:{i}")
            if log_level != "info":
                log(
                    string(
                        "runtime.compute_test_success_detail",
                        index=i,
                        result=format_number(float(vtensor.item())),
                    ),
                    "info",
                    loglevel="verbose",
                )
            else:
                log(string("runtime.compute_test_success", index=i), "info")
        except Exception as e:
            log(
                string(
                    "runtime.compute_test_failed",
                    index=i,
                    error=format_error(e, log_level),
                ),
                "warning",
            )

            # decrement device count if tests fail
            devices -= 1

        # usable devices
        if devices == 0:
            log(
                string("runtime.all_gpu_tests_failed"),
                "error",
            )
            return False

    # found several usable devices
    if devices > 1:
        unused_devices = devices - 1
        log(string("runtime.unused_gpus", count=unused_devices), "warning")

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
