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
    """Return the normalized configuration block for the persona system.

    Args:
        config: The configuration data for the persona system.

    Returns:
        dict[str, Any]: The normalized configuration data for the persona system.
    """
    raw = config.get("pyop", {})
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raw = {}

    return dict(raw)


def pyop_enabled(config: dict[str, Any]) -> bool:
    """Return whether Celune should try to use personas.

    Args:
        config: Celune's current persona system configuration.

    Returns:
        bool: Whether personas are currently enabled.
    """
    return bool(pyop_config(config).get("enabled", True))


def pyop_talkback_enabled(config: dict[str, Any]) -> bool:
    """Return whether regular UI input should go through persona talkback.

    Args:
        config: Celune's current persona system configuration.

    Returns:
        bool: Whether personas are in use, or fall back to speaking directly.
    """
    return bool(pyop_config(config).get("talkback", True))


def pyop_base_url() -> str:
    """Return the persona system base URL.

    Returns:
        str: The URL Celune should ask for persona outputs.
    """
    return PYOP_BASE_URL


def pyop_endpoint() -> str:
    """Return the persona system generation endpoint.

    Returns:
        str: The endpoint Celune should ask for persona outputs.
    """
    return PYOP_ENDPOINT


def pyop_model_id() -> str:
    """Return the PYOP model ID the detached API should load.

    Returns:
        str: The PYOP model ID currently in use.
    """
    return PYOP_MODEL_ID


def pyop_python() -> Path:
    """Return the separate persona system's separated interpreter.
    This interpreter does not depend on Celune directly.

    Returns:
        str: The path to the interpreter.
    """
    venv = Path(PYOP_VENV)
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"

    return venv / "bin" / "python"


def pyop_entrypoint() -> str:
    """Return the persona system's API entrypoint.

    Returns:
        str: The path to the API entrypoint.
    """
    return str(_project_root() / PYOP_SCRIPT)


def pyop_is_available(base_url: str, timeout: httpx.Timeout = PYOP_TIMEOUT) -> bool:
    """Check whether the persona system can be used.

    Args:
        base_url: The URL to be pinged.
        timeout: The timeout object.

    Returns:
        bool: Whether the persona system is usable.
    """
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.get("/")
            return response.status_code < 500
    except httpx.RequestError:
        return False


def start_pyop_detached() -> Optional[subprocess.Popen[bytes]]:
    """Start the configured persona system API in its own virtual environment.

    Returns:
        Optional[subprocess.Popen[bytes]]: A detached process object, or ``None`` if it was
            not found, or cannot be started.
    """
    python = pyop_python()
    entrypoint = pyop_entrypoint()
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
    env["PYOP_MODEL"] = pyop_model_id()
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
    """Terminate the persona system's process tree started by Celune.

    Args:
        process: The process object holding the API.
        timeout: How long to wait before the process exits.

    Returns:
        None: This method terminates the persona system API.
    """
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
