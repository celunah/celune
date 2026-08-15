# SPDX-License-Identifier: MIT
"""Resource footer data for the Textual UI."""

from __future__ import annotations

import datetime
import shutil
import subprocess
import threading
import time
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Optional

import psutil
import torch

from ..constants import APP_NAME, COST_EQUIVALENTS
from ..i18n import string
from ..persona.impl import persona_talkback_enabled
from ..utils import celune_day_status, lunar_info, lunar_phase

if TYPE_CHECKING:
    from ..celune import Celune

_NVIDIA_SMI: Optional[str] = shutil.which("nvidia-smi")
_NVIDIA_SMI_THREAD: Optional[threading.Thread] = None
_NVIDIA_SMI_USAGE: Optional[int] = None
NVIDIA_SMI_TIMEOUT_SECONDS = 2.0
FOOTER_ROTATE_SECONDS = 2.06
RESOURCE_PAGE_CACHE_SECONDS = 0.25
_RESOURCE_PAGE_CACHE: Optional[
    tuple[
        tuple[
            int,
            Optional[str],
            Optional[int],
            str,
            bool,
            bool,
            str,
            float,
            float,
        ],
        float,
        tuple[str, ...],
    ]
] = None


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


def _query_gpu_usage() -> Optional[int]:
    """Read one GPU utilization sample with a bounded subprocess lifetime."""
    if not _NVIDIA_SMI:
        return None

    try:
        result = subprocess.run(
            [
                _NVIDIA_SMI,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    first_line = result.stdout.strip().splitlines()[0:1]
    if not first_line:
        return None

    try:
        return int(first_line[0].strip())
    except ValueError:
        return None


def _update_gpu_usage() -> None:
    """Update the cached GPU utilization from a worker thread."""
    global _NVIDIA_SMI_USAGE
    _NVIDIA_SMI_USAGE = _query_gpu_usage()


def gpu_usage() -> Optional[int]:
    """Return cached GPU utilization while sampling asynchronously.

    Returns:
        Optional[int]: The GPU utilization, or ``None`` if unavailable.
    """
    global _NVIDIA_SMI_THREAD

    if not _NVIDIA_SMI:
        return None

    thread = _NVIDIA_SMI_THREAD
    if thread is not None and thread.is_alive():
        return _NVIDIA_SMI_USAGE

    _NVIDIA_SMI_THREAD = threading.Thread(
        target=_update_gpu_usage,
        name="celune-nvidia-smi",
        daemon=True,
    )
    _NVIDIA_SMI_THREAD.start()

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
    global _RESOURCE_PAGE_CACHE

    backend = celune.backend
    raw_current_seed = getattr(backend, "current_seed", None)
    current_seed = raw_current_seed if isinstance(raw_current_seed, int) else None
    raw_input_mode = getattr(celune, "input_mode", "text_to_speech")
    input_mode = raw_input_mode if isinstance(raw_input_mode, str) else "text_to_speech"
    persona_ready = bool(
        getattr(celune, "persona_ready", getattr(celune, "vision", None) is not None)
    )
    tutorial = bool(getattr(celune, "is_in_tutorial", False))
    raw_configured_theme = celune.config.get("theme", "dark")
    configured_theme = (
        raw_configured_theme if isinstance(raw_configured_theme, str) else "dark"
    )
    session_speech_seconds = float(
        getattr(celune, "total_generated_speech_seconds", 0.0)
    )
    historical_speech_seconds = float(
        getattr(celune, "historical_generated_speech_seconds", 0.0)
    )
    now_monotonic = time.monotonic()
    cache_key = (
        id(celune),
        theme_name,
        current_seed,
        input_mode,
        persona_ready,
        tutorial,
        configured_theme,
        session_speech_seconds,
        historical_speech_seconds,
    )
    cached = _RESOURCE_PAGE_CACHE
    if (
        cached is not None
        and cached[0] == cache_key
        and now_monotonic - cached[1] < RESOURCE_PAGE_CACHE_SECONDS
    ):
        return cached[2]

    pages = [format_vram(), format_usage()]

    if current_seed is not None:
        pages.append(format_seed(celune))

    now = datetime.datetime.now(datetime.UTC)
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
        if input_mode == "voice_conversion":
            pages.append(string("ui.footer_toggle_recording"))
        elif (
            getattr(
                celune, "persona_ready", getattr(celune, "vision", None) is not None
            )
            and persona_ready
            and persona_talkback_enabled(celune.config)
        ):
            pages.append(string("ui.footer_voice_input"))

        active_theme = theme_name
        enter_action = "skip" if tutorial else "say"

        if active_theme is None:
            active_theme = "celune_light" if configured_theme == "light" else "celune"

        if active_theme == "celune_april_fools":
            pages.append(f"CTRL+Q exit • CTRL+ENTER {enter_action}")
        else:
            other_theme = "light" if active_theme == "celune" else "dark"
            pages.append(
                f"CTRL+Q exit • CTRL+T {other_theme} • CTRL+ENTER {enter_action}"
            )

    result = tuple(pages)
    _RESOURCE_PAGE_CACHE = (cache_key, now_monotonic, result)
    return result
