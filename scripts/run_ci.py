# SPDX-License-Identifier: Apache-2.0
"""Run the Poe CI task and terminate its complete process tree safely."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from ctypes import wintypes
from typing import cast, Optional


TIMEOUT = 600
GRACE_PERIOD = 2.0
POE_COMMAND = ["uv", "run", "poe", "ci"]
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_HANDLE_ATTRIBUTE = "_celune_ci_job_handle"


class _JobObjectBasicLimitInformation(ctypes.Structure):
    """Subset of Windows job limits needed to kill a job on handle close."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    """Windows job I/O counters included in the extended job limits structure."""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    """Windows structure used to enable kill-on-close job behavior."""

    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _windows_kernel32() -> Optional[ctypes.CDLL]:
    """Return a configured Windows kernel API handle."""
    if not hasattr(ctypes, "WinDLL"):
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _attach_windows_job(process: subprocess.Popen[str]) -> None:
    """Attach a Windows process tree to a kill-on-close job object."""
    kernel32 = _windows_kernel32()
    if kernel32 is None:
        return

    if not hasattr(ctypes, "WinError") or not hasattr(ctypes, "get_last_error"):
        return

    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    process_handle = getattr(process, "_handle", None)
    if not isinstance(process_handle, int):
        return

    job_handle = kernel32.CreateJobObjectW(None, None)
    if not job_handle:
        raise ctypes.WinError(ctypes.get_last_error())

    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job_handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job_handle)
        raise error

    if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(job_handle)
        raise error

    setattr(process, _JOB_HANDLE_ATTRIBUTE, job_handle)


def _close_windows_job(process: subprocess.Popen[str]) -> bool:
    """Close the process job, killing its descendants when configured."""
    if os.name != "nt":
        return False

    job_handle = getattr(process, _JOB_HANDLE_ATTRIBUTE, None)
    if job_handle is None:
        return False

    kernel32 = _windows_kernel32()
    if kernel32 is None:
        return False

    close_handle = getattr(kernel32, "CloseHandle", None)
    setattr(process, _JOB_HANDLE_ATTRIBUTE, None)

    if close_handle is not None:
        return bool(close_handle(job_handle))
    return False


def _start_process() -> subprocess.Popen[str]:
    """Start the Poe CI task while keeping console interrupts visible to the runner."""
    if os.name == "nt":
        process = subprocess.Popen(POE_COMMAND, text=True)  # pylint: disable=R1732
        with suppress(OSError):
            _attach_windows_job(process)
        return process

    return subprocess.Popen(
        POE_COMMAND,
        text=True,
        start_new_session=True,
    )


def stop_process_tree(process: subprocess.Popen[str]) -> None:
    """Immediately terminate Poe and every child process it spawned."""
    if os.name == "nt":
        if not _close_windows_job(process):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)
        return

    getpgid = cast(Callable[[int], int], getattr(os, "getpgid", None))
    killpg = cast(Callable[[int, int], None], getattr(os, "killpg", None))
    sigkill = cast(int, getattr(signal, "SIGKILL", signal.SIGTERM))
    if getpgid is None or killpg is None:
        with suppress(OSError):
            process.kill()
        return

    process_group_id = process.pid
    try:
        process_group_id = getpgid(process.pid)
    except ProcessLookupError:
        with suppress(OSError):
            process.kill()
        return

    with suppress(ProcessLookupError):
        killpg(process_group_id, sigkill)
    with suppress(OSError):
        process.kill()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=GRACE_PERIOD)


def _wait_for_process(process: subprocess.Popen[str], timeout: float) -> int:
    """Wait in short intervals so console interrupts reach the runner promptly."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(remaining, 0.1))
        except subprocess.TimeoutExpired:
            continue


def main() -> int:
    """Run Poe CI and return a shell-compatible status code."""
    process = _start_process()
    try:
        exit_code = _wait_for_process(process, TIMEOUT)
    except KeyboardInterrupt:
        previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_process_tree(process)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        print("\nSLOP - Interrupted", flush=True)
        return 130
    except subprocess.TimeoutExpired:
        stop_process_tree(process)
        print(f"\nSLOP - Timed out after {TIMEOUT} seconds", flush=True)
        return 1
    finally:
        _close_windows_job(process)

    if exit_code == 0:
        print("LGTM - Everything is OK", flush=True)
    else:
        print(f"SLOP - Exit code {exit_code}", flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
