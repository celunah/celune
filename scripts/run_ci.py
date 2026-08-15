# SPDX-License-Identifier: MIT
"""Run the Poe CI task and terminate its complete process tree safely."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from typing import cast
from collections.abc import Callable


TIMEOUT = 600
GRACE_PERIOD = 2.0
POE_COMMAND = ["uv", "run", "poe", "ci"]


def _start_process() -> subprocess.Popen[str]:
    """Start the Poe CI task in a process group that can be terminated as a unit."""
    if os.name == "nt":
        return subprocess.Popen(
            POE_COMMAND,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    return subprocess.Popen(
        POE_COMMAND,
        text=True,
        start_new_session=True,
    )


def stop_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate Poe and every child process it spawned."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)
        return

    getpgid = cast(Callable[[int], int], getattr(os, "getpgid", None))
    killpg = cast(Callable[[int, int], None], getattr(os, "killpg", None))
    sigkill = cast(int, getattr(signal, "SIGKILL", signal.SIGTERM))
    if getpgid is None or killpg is None:
        return

    process_group_id = process.pid
    try:
        process_group_id = getpgid(process.pid)
    except ProcessLookupError:
        return

    with suppress(ProcessLookupError):
        killpg(process_group_id, signal.SIGTERM)

    try:
        process.wait(timeout=GRACE_PERIOD)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            killpg(process_group_id, sigkill)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)


def main() -> int:
    """Run Poe CI and return a shell-compatible status code."""
    process = _start_process()
    try:
        exit_code = process.wait(timeout=TIMEOUT)
    except KeyboardInterrupt:
        stop_process_tree(process)
        print("\nSLOP - Interrupted", flush=True)
        return 130
    except subprocess.TimeoutExpired:
        stop_process_tree(process)
        print(f"\nSLOP - Timed out after {TIMEOUT} seconds", flush=True)
        return 1

    if exit_code == 0:
        print("LGTM - Everything is OK", flush=True)
    else:
        print(f"SLOP - Exit code {exit_code}", flush=True)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
