# SPDX-License-Identifier: MIT
"""Resource footer data for the Textual UI."""

from __future__ import annotations

import shutil
import datetime
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from typing import TYPE_CHECKING, Optional

import torch
import psutil

from ..constants import APP_NAME, COST_EQUIVALENTS
from ..i18n import string
from ..persona.impl import persona_talkback_enabled
from ..utils import celune_day_status, lunar_info, lunar_phase

if TYPE_CHECKING:
    from ..celune import Celune

_NVIDIA_SMI: Optional[str] = shutil.which("nvidia-smi")
_NVIDIA_SMI_PROC: Optional[subprocess.Popen[str]] = None
_NVIDIA_SMI_USAGE: Optional[int] = None
FOOTER_ROTATE_SECONDS = 2.06


def _format_cost(amount: float) -> str:
    """Return one compact USD amount string for footer displays."""
    if amount <= 0.0:
        return "0.00"
    decimal_amount = Decimal(str(amount))
    if amount < 0.01:
        formatted = str(
            decimal_amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        )
    else:
        formatted = str(
            decimal_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )
    return formatted.rstrip("0").rstrip(".")


def format_cost_equivalent_pages(celune: Celune) -> tuple[str, ...]:
    """Return footer lines describing session and total saved cost by provider.

    Args:
        celune: The active Celune runtime state.

    Returns:
        tuple[str, ...]: Footer lines for session and total savings by provider.
    """
    session_speech_seconds = max(
        0.0,
        float(getattr(celune, "total_generated_speech_seconds", 0.0)),
    )
    historical_speech_seconds = max(
        0.0,
        float(getattr(celune, "historical_generated_speech_seconds", 0.0)),
    )
    session_speech_minutes = session_speech_seconds / 60.0
    total_speech_minutes = (historical_speech_seconds + session_speech_seconds) / 60.0

    pages = []
    for provider, cost_per_minute in COST_EQUIVALENTS.items():
        pages.append(
            string(
                "ui.footer_cost_equivalent",
                app_name=APP_NAME,
                cost=_format_cost(session_speech_minutes * cost_per_minute),
                provider=provider,
            )
        )
        pages.append(
            string(
                "ui.footer_total_cost_equivalent",
                app_name=APP_NAME,
                cost=_format_cost(total_speech_minutes * cost_per_minute),
                provider=provider,
            )
        )

    return tuple(pages)


def format_vram() -> str:
    """Return available CUDA memory in a compact display format.

    Returns:
        str: The formatted CUDA memory usage.
    """
    if not torch.cuda.is_available():
        return "VRAM: nothing to fetch"

    try:
        device = torch.cuda.current_device()
        avail, total = torch.cuda.mem_get_info(device)
    except (AssertionError, RuntimeError, ValueError):
        return "VRAM: cannot fetch"

    return f"VRAM: {avail / 1024**3:.2f}/{total / 1024**3:.2f} GB available"


def gpu_usage() -> Optional[int]:
    """Read GPU utilization from nvidia-smi when it is available.

    Returns:
        Optional[int]: The GPU utilization, or ``None`` if unavailable.
    """
    global _NVIDIA_SMI_PROC, _NVIDIA_SMI_USAGE

    if not _NVIDIA_SMI:
        return None

    proc = _NVIDIA_SMI_PROC
    if proc is not None:
        if proc.poll() is None:
            return _NVIDIA_SMI_USAGE

        try:
            stdout, _ = proc.communicate()
        except (OSError, ValueError, subprocess.SubprocessError):
            _NVIDIA_SMI_PROC = None
            _NVIDIA_SMI_USAGE = None
            return None

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
        _NVIDIA_SMI_PROC = subprocess.Popen(  # pylint: disable=R1732
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
    """Return CPU/GPU utilization in a compact display format.

    Returns:
        str: The formatted CPU/GPU utilization.
    """
    cpu = psutil.cpu_percent(interval=None)
    gpu = gpu_usage()
    gpu_text = f"{gpu}%" if gpu is not None else "N/A"
    return f"CPU: {cpu:.0f}% • GPU: {gpu_text}"


def prime_usage() -> None:
    """Prime psutil CPU sampling for later footer updates."""
    psutil.cpu_percent(interval=None)


def format_seed(celune: Celune) -> str:
    """Return the current backend seed when Celune exposes one.

    Args:
        celune: The instance of Celune to get the generation seed from.

    Returns:
        str: The formatted seed for UI displays.
    """
    seed = getattr(celune.backend, "current_seed", None)
    return f"Seed: {seed}" if seed is not None else "Seed: N/A"


def resource_pages(celune: Celune, theme_name: Optional[str] = None) -> tuple[str, ...]:
    """Return resource footer pages in their display order.

    Args:
        celune: The instance of Celune to get relevant data from.
        theme_name: The current theme name.

    Returns:
        tuple[str, ...]: A variable amount of resource pages formatted as text.
    """
    pages = [format_vram(), format_usage()]

    if getattr(celune.backend, "current_seed", None) is not None:
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
    pages.extend(format_cost_equivalent_pages(celune))
    if celune is not None:
        if getattr(celune, "input_mode", "text_to_speech") == "voice_conversion":
            pages.append(string("ui.footer_toggle_recording"))
        elif getattr(celune, "vision", None) is not None and persona_talkback_enabled(
            celune.config
        ):
            pages.append(string("ui.footer_voice_input"))

        active_theme = theme_name
        enter_action = "skip" if celune.is_in_tutorial else "say"

        if active_theme is None:
            configured_theme = celune.config.get("theme", "dark")
            active_theme = "celune_light" if configured_theme == "light" else "celune"

        if active_theme == "celune_april_fools":
            pages.append(f"CTRL+Q exit • CTRL+ENTER {enter_action}")
        else:
            other_theme = "light" if active_theme == "celune" else "dark"
            pages.append(
                f"CTRL+Q exit • CTRL+T {other_theme} • CTRL+ENTER {enter_action}"
            )

    return tuple(pages)
