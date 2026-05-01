"""Resource footer data for the Textual UI."""

from __future__ import annotations

import datetime
import shutil
import subprocess
from typing import TYPE_CHECKING, Optional

import psutil
import torch

from ..utils import celune_day_status, lunar_info, lunar_phase

if TYPE_CHECKING:
    from ..celune import Celune

_NVIDIA_SMI: Optional[str] = shutil.which("nvidia-smi")
_NVIDIA_SMI_PROC: Optional[subprocess.Popen[str]] = None
_NVIDIA_SMI_USAGE: Optional[int] = None


def format_vram() -> str:
    """Return available CUDA memory in a compact display format."""
    if not torch.cuda.is_available():
        return "VRAM: nothing to fetch"

    try:
        device = torch.cuda.current_device()
        avail, total = torch.cuda.mem_get_info(device)
    except (AssertionError, RuntimeError, ValueError):
        return "VRAM: cannot fetch"

    return f"VRAM: {avail / 1024**3:.2f}/{total / 1024**3:.2f} GB available"


def gpu_usage() -> Optional[int]:
    """Read GPU utilization from nvidia-smi when it is available."""
    global _NVIDIA_SMI_PROC, _NVIDIA_SMI_USAGE  # pylint: disable=global-statement

    if not _NVIDIA_SMI:
        return None

    proc = _NVIDIA_SMI_PROC
    if proc is not None:
        if proc.poll() is None:
            return _NVIDIA_SMI_USAGE

        stdout, _ = proc.communicate()
        _NVIDIA_SMI_PROC = None

        if proc.returncode != 0:
            _NVIDIA_SMI_USAGE = None
            return None

        first_line = stdout.strip().splitlines()[0:1]
        if not first_line:
            _NVIDIA_SMI_USAGE = None
            return None

        try:
            _NVIDIA_SMI_USAGE = int(first_line[0].strip())
        except ValueError:
            _NVIDIA_SMI_USAGE = None

        return _NVIDIA_SMI_USAGE

    try:
        _NVIDIA_SMI_PROC = subprocess.Popen(  # pylint: disable=consider-using-with
            [
                _NVIDIA_SMI,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        _NVIDIA_SMI_USAGE = None

    return _NVIDIA_SMI_USAGE


def format_usage() -> str:
    """Return CPU/GPU utilization in a compact display format."""
    cpu = psutil.cpu_percent(interval=None)
    gpu = gpu_usage()
    gpu_text = f"{gpu}%" if gpu is not None else "N/A"
    return f"CPU: {cpu:.0f}% • GPU: {gpu_text}"


def prime_usage() -> None:
    """Prime psutil CPU sampling for later footer updates."""
    psutil.cpu_percent(interval=None)


def format_seed(celune: Celune) -> str:
    """Return the current backend seed when Celune exposes one."""
    seed = getattr(celune.backend, "current_seed", None)
    return f"Seed: {seed}" if seed is not None else "Seed: N/A"


def resource_pages(celune: Optional[Celune]) -> tuple[str, ...]:
    """Return resource footer pages in their display order."""
    pages = [format_vram(), format_usage()]

    if celune is not None and getattr(celune.backend, "current_seed", None) is not None:
        pages.append(format_seed(celune))

    now = datetime.datetime.now()
    phase, _, days = lunar_info(now)
    suffix = "s" if int(days) != 1 else ""

    pages.append(now.strftime("%A, %B %d, %Y"))
    pages.append(celune_day_status(now))
    pages.append(lunar_phase(phase).title())
    if lunar_phase(phase) != "full moon":
        pages.append(f"{int(days)} day{suffix} until full moon")

    pages.append("/help commands")
    if celune is not None:
        pages.append(
            f"CTRL+C/CTRL+Q exit • CTRL+T {celune.config.get('theme', 'dark')} • CTRL+ENTER say"
        )

    return tuple(pages)
