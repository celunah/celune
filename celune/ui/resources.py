# SPDX-License-Identifier: Apache-2.0
"""Resource footer data for the Textual UI."""

from __future__ import annotations

import asyncio
import time
import shutil
import datetime
import contextlib
import subprocess
from typing import TYPE_CHECKING, Optional
from decimal import ROUND_HALF_UP, Decimal

import torch
import psutil

from ..i18n import string
from ..constants import APP_NAME, COST_EQUIVALENTS
from ..persona.impl import persona_talkback_enabled
from ..utils import lunar_info, lunar_phase, celune_day_status

if TYPE_CHECKING:
    from ..celune import Celune

_NVIDIA_SMI: Optional[str] = shutil.which("nvidia-smi")
_NVIDIA_SMI_TASKS: dict[
    asyncio.AbstractEventLoop,
    asyncio.Task[None],
] = {}
_NVIDIA_SMI_USAGE: Optional[int] = None
NVIDIA_SMI_TIMEOUT_SECONDS = 2.0
NVIDIA_SMI_POLL_INTERVAL_SECONDS = 1.0
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


async def _terminate_gpu_process(process: asyncio.subprocess.Process) -> None:
    """Terminate one timed-out GPU query and drain its subprocess pipes."""
    if process.returncode is None:
        with contextlib.suppress(OSError, ProcessLookupError):
            process.kill()
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        await process.communicate()


async def _query_gpu_usage() -> Optional[int]:
    """Read one GPU utilization sample through an async subprocess."""
    if not _NVIDIA_SMI:
        return None

    process: Optional[asyncio.subprocess.Process] = None
    try:
        process = await asyncio.create_subprocess_exec(
            _NVIDIA_SMI,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        if process is not None:
            await _terminate_gpu_process(process)
        return None
    except (OSError, ValueError, subprocess.SubprocessError):
        if process is not None:
            await _terminate_gpu_process(process)
        return None
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_gpu_process(process)
        raise

    if process.returncode != 0:
        return None

    if not isinstance(stdout, bytes):
        return None

    first_line = stdout.decode(errors="replace").strip().splitlines()[0:1]
    if not first_line:
        return None

    try:
        return int(first_line[0].strip())
    except ValueError:
        return None


async def _update_gpu_usage() -> None:
    """Update the cached GPU utilization from the current event loop."""
    global _NVIDIA_SMI_USAGE
    _NVIDIA_SMI_USAGE = await _query_gpu_usage()


async def _gpu_usage_worker() -> None:
    """Continuously sample GPU utilization without blocking the event loop."""
    while True:
        await _update_gpu_usage()
        await asyncio.sleep(NVIDIA_SMI_POLL_INTERVAL_SECONDS)


def start_gpu_usage_worker() -> None:
    """Start one native async GPU sampler on the current event loop."""
    if not _NVIDIA_SMI:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    current = _NVIDIA_SMI_TASKS.get(loop)
    if current is not None and not current.done():
        return

    task = loop.create_task(_gpu_usage_worker(), name="celune-nvidia-smi")
    _NVIDIA_SMI_TASKS[loop] = task

    def clear_task(completed: asyncio.Task[None]) -> None:
        """Forget one completed sampler and consume unexpected task errors."""
        if _NVIDIA_SMI_TASKS.get(loop) is completed:
            _NVIDIA_SMI_TASKS.pop(loop, None)
        with contextlib.suppress(asyncio.CancelledError):
            completed.exception()

    task.add_done_callback(clear_task)


def stop_gpu_usage_worker() -> None:
    """Cancel the native async GPU sampler on the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    task = _NVIDIA_SMI_TASKS.pop(loop, None)
    if task is not None:
        task.cancel()


def gpu_usage() -> Optional[int]:
    """Return the latest cached GPU utilization sample.

    The owning UI or API event loop starts the native async sampler with
    :func:`start_gpu_usage_worker`.

    Returns:
        Optional[int]: The GPU utilization, or ``None`` if unavailable.
    """
    if not _NVIDIA_SMI:
        return None
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
