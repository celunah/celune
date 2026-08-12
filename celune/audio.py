# SPDX-License-Identifier: Apache-2.0
"""Helpers for restarting the host operating system's audio server."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence

_WINDOWS_AUDIO_SERVICE = "Audiosrv"
_WINDOWS_ELEVATED_RESTART_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "try { "
    "$child = Start-Process -FilePath 'powershell.exe' "
    "-Verb RunAs -WindowStyle Hidden -Wait -PassThru "
    "-ArgumentList @('-NoProfile', '-NonInteractive', '-Command', "
    "'Restart-Service -Name Audiosrv -Force'); "
    "exit $child.ExitCode "
    "} catch { "
    "[Console]::Error.WriteLine($_.Exception.Message); "
    "exit 1 "
    "}"
)
_LINUX_AUDIO_UNITS = (
    "pipewire.service",
    "pipewire-pulse.service",
    "wireplumber.service",
    "pulseaudio.service",
)
_COMMAND_TIMEOUT_SECONDS = 15.0


def _run_command(
    command: Sequence[str], timeout: float = _COMMAND_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run one audio-server command and capture its result."""
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(str(error)) from error


def _command_error(result: subprocess.CompletedProcess[str]) -> RuntimeError:
    """Build an error from a failed system command result."""
    detail = (result.stderr or result.stdout).strip()
    return RuntimeError(detail or f"command exited with status {result.returncode}")


def _restart_windows_audio() -> None:
    """Restart the Windows Audio service."""
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if powershell is None:
        raise RuntimeError("PowerShell is not available")

    result = _run_command(
        (
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _WINDOWS_ELEVATED_RESTART_SCRIPT.replace(
                "Audiosrv", _WINDOWS_AUDIO_SERVICE
            ),
        )
    )
    if result.returncode != 0:
        raise _command_error(result)


def _active_linux_audio_units(systemctl: str) -> list[str]:
    """Return active user audio-service units known to systemd."""
    active_units: list[str] = []
    for unit in _LINUX_AUDIO_UNITS:
        result = _run_command(
            (systemctl, "--user", "is-active", "--quiet", unit),
            timeout=5.0,
        )
        if result.returncode == 0:
            active_units.append(unit)
    return active_units


def _restart_linux_audio() -> None:
    """Restart the active per-user Linux audio server."""
    systemctl = shutil.which("systemctl")
    if systemctl is not None:
        active_units = _active_linux_audio_units(systemctl)
        if active_units:
            result = _run_command((systemctl, "--user", "restart", *active_units))
            if result.returncode != 0:
                raise _command_error(result)
            return

    pulseaudio = shutil.which("pulseaudio")
    if pulseaudio is not None:
        check = _run_command((pulseaudio, "--check"), timeout=5.0)
        if check.returncode == 0:
            result = _run_command((pulseaudio, "--kill"))
            if result.returncode != 0:
                raise _command_error(result)
            return

    raise RuntimeError("No active PipeWire or PulseAudio server was found")


def restart_audio_server() -> None:
    """Restart the supported host operating system audio server.

    Raises:
        RuntimeError: If the platform, required command, or audio server is unavailable, or if restarting it fails.
    """
    if os.name == "nt":
        _restart_windows_audio()
        return

    if sys.platform.startswith("linux"):
        _restart_linux_audio()
        return

    raise RuntimeError(f"unsupported platform: {sys.platform}")
