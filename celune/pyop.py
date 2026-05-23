# SPDX-License-Identifier: MIT
"""Detached PYOP companion process helpers."""

from __future__ import annotations

import os
import sys
import contextlib
import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx

from .utils import discard
from .constants import PYOP_MODEL_ID

PYOP_HOST = "127.0.0.1"
PYOP_PORT = 2061
PYOP_BASE_URL = f"http://{PYOP_HOST}:{PYOP_PORT}"  # noqa
PYOP_ENDPOINT = "/generate"
PYOP_TIMEOUT = httpx.Timeout(connect=2, timeout=60)
PYOP_STARTUP_TIMEOUT = 20.0
PYOP_VENV = "pyop/.venv"
PYOP_SCRIPT = "pyop/run_api.py"
PYOP_QUANTIZATION = "4bit"


def _project_root() -> Path:
    """Return the repository root containing Celune and PYOP."""
    return Path(__file__).resolve().parent.parent


def pyop_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized PYOP user-facing configuration block."""
    raw = config.get("pyop", {})
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raw = {}

    return dict(raw)


def pyop_enabled(config: dict[str, Any]) -> bool:
    """Return whether Celune should try to use PYOP."""
    return bool(pyop_config(config).get("enabled", True))


def pyop_talkback_enabled(config: dict[str, Any]) -> bool:
    """Return whether regular UI input should go through persona talkback."""
    return bool(pyop_config(config).get("talkback", True))


def pyop_base_url(config: dict[str, Any]) -> str:
    """Return the PYOP base URL."""
    discard(config)
    return PYOP_BASE_URL


def pyop_endpoint(config: dict[str, Any]) -> str:
    """Return the PYOP generation endpoint."""
    discard(config)
    return PYOP_ENDPOINT


def pyop_model_id(config: dict[str, Any]) -> str:
    """Return the PYOP model id the detached API should load."""
    discard(config)
    return PYOP_MODEL_ID


def pyop_quantization(config: dict[str, Any]) -> str:
    """Return the quantization mode the detached PYOP API should use."""
    discard(config)
    return PYOP_QUANTIZATION


def pyop_python(config: dict[str, Any]) -> Optional[Path]:
    """Return the separate PYOP virtual-environment interpreter."""
    discard(config)
    venv = Path(PYOP_VENV)
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"

    return venv / "bin" / "python"


def pyop_entrypoint(config: dict[str, Any]) -> Optional[str]:
    """Return the PYOP script to launch."""
    discard(config)
    return str(_project_root() / PYOP_SCRIPT)


def pyop_is_available(base_url: str, timeout: httpx.Timeout = PYOP_TIMEOUT) -> bool:
    """Return whether the local PYOP API responds."""
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.get("/")
            return response.status_code < 500
    except httpx.RequestError:
        return False


def start_pyop_detached(config: dict[str, Any]) -> Optional[subprocess.Popen[bytes]]:
    """Start the configured PYOP API in its own virtual environment."""
    python = pyop_python(config)
    entrypoint = pyop_entrypoint(config)
    if python is None or entrypoint is None or not python.exists():
        return None
    if python.resolve() == Path(sys.executable).resolve():
        return None

    args = [str(python)]
    if entrypoint.startswith("-m "):
        args.extend(["-m", entrypoint[3:]])
    else:
        args.append(entrypoint)

    process_cwd = str(_project_root())

    creationflags = 0
    start_new_session = True
    if sys.platform == "win32":
        creationflags = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        start_new_session = False

    env = os.environ.copy()
    env["PYOP_MODEL"] = pyop_model_id(config)
    env["PYOP_QUANTIZATION"] = pyop_quantization(config)
    env["PYOP_QUANTIZED"] = "1"

    return subprocess.Popen(  # pylint: disable=consider-using-with
        args,
        cwd=process_cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=start_new_session,
        creationflags=creationflags,
    )


def stop_pyop_process(process: subprocess.Popen[bytes], timeout: float = 5.0) -> None:
    """Terminate the PYOP process tree started by Celune."""
    if process.poll() is not None:
        return

    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            return

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, 9)
