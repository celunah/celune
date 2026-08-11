# SPDX-License-Identifier: MIT
"""Run CI automatically and cleanly terminate it when interrupted."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from typing import Optional, cast

from tqdm.contrib import tzip


TIMEOUT = 300
GRACE_PERIOD = 2.0

CI_COMMANDS = (
    ("ruff", "format", "--check"),
    ("ruff", "check"),
    ("pylint",),
    ("pyrefly", "check"),
    ("pytest", "-v"),
)

CI_PATHS = (
    (".",),
    (".",),
    ("celune", "tests"),
    (),
    ("tests",),
)

_AGENT_ERROR_MARKERS = (
    "Access is denied",
    "Permission denied",
    "Operation not permitted",
)


def _agent_permission_marker(output: str) -> Optional[str]:
    """Return the first permission marker found in command output."""
    normalized_output = output.lower()
    for marker in _AGENT_ERROR_MARKERS:
        if marker.lower() in normalized_output:
            return marker
    return None


def _start_process(command: list[str]) -> subprocess.Popen[str]:
    """Start one CI command in a process group that can be terminated as a unit."""
    if os.name == "nt":
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def stop_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a CI command and every child it spawned."""
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

    get_process_group = cast(
        Optional[Callable[[int], int]],
        getattr(os, "getpgid", None),
    )
    kill_process_group = cast(
        Optional[Callable[[int, int], None]],
        getattr(os, "killpg", None),
    )
    if get_process_group is None:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)
        return
    if kill_process_group is None:
        process.terminate()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)
        return

    get_process_group_fn = cast(Callable[[int], int], get_process_group)
    kill_process_group_fn = cast(Callable[[int, int], None], kill_process_group)
    process_group = -1
    try:
        process_group = get_process_group_fn(process.pid)
    except ProcessLookupError:
        return

    sigterm = cast(int, getattr(signal, "SIGTERM", signal.SIGINT))
    sigkill = cast(int, getattr(signal, "SIGKILL", sigterm))
    with suppress(ProcessLookupError):
        kill_process_group_fn(process_group, sigterm)

    try:
        process.wait(timeout=GRACE_PERIOD)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            kill_process_group_fn(process_group, sigkill)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=GRACE_PERIOD)


def _run_process(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one command while retaining output and interrupt-safe cleanup."""
    process = _start_process(command)
    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT)
    except KeyboardInterrupt:
        stop_process_tree(process)
        with suppress(OSError, subprocess.TimeoutExpired):
            process.communicate(timeout=GRACE_PERIOD)
        raise
    except subprocess.TimeoutExpired as timeout:
        stop_process_tree(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command,
            TIMEOUT,
            output=stdout,
            stderr=stderr,
        ) from timeout

    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _run_uv_command(*command: str) -> None:
    """Run one uv-backed CI command, retrying without cache on permission errors."""
    requested_command = list(command)
    base_command = ["uv", "run", *requested_command]
    result = _run_process(base_command)
    if result.returncode == 0:
        return

    combined_output = f"{result.stdout}\n{result.stderr}"
    marker = _agent_permission_marker(combined_output)
    if marker is None:
        raise subprocess.CalledProcessError(
            result.returncode,
            base_command,
            output=result.stdout,
            stderr=result.stderr,
        )

    no_cache_command = ["uv", "--no-cache", "run", *requested_command]
    result = _run_process(no_cache_command)
    if result.returncode == 0:
        return

    combined_output = f"{result.stdout}\n{result.stderr}"
    marker = _agent_permission_marker(combined_output)
    if marker is not None:
        raise RuntimeError(f"agent has no permissions to run CI: {marker}")
    raise subprocess.CalledProcessError(
        result.returncode,
        no_cache_command,
        output=result.stdout,
        stderr=result.stderr,
    )


def main() -> int:
    """Run every configured CI command and return a shell-compatible status."""
    if len(CI_COMMANDS) != len(CI_PATHS):
        raise RuntimeError(
            f"CI configuration mismatch: {len(CI_COMMANDS)} commands for "
            f"{len(CI_PATHS)} path entries"
        )

    commands_failed = 0
    total_errors: list[str] = []

    try:
        command_iterator = tzip(
            CI_COMMANDS,
            CI_PATHS,
            desc="Running CI commands",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}",
        )
        for command, paths in command_iterator:
            try:
                _run_uv_command(*command, *paths)
            except subprocess.CalledProcessError as failed:
                commands_failed += 1
                if failed.stdout:
                    total_errors.append(failed.stdout)
                if failed.stderr:
                    total_errors.append(failed.stderr)
            except subprocess.TimeoutExpired:
                commands_failed += 1
                total_errors.append(f"{' '.join(command)} has timed out")
    except KeyboardInterrupt:
        print("\nSLOP - Interrupted", flush=True)
        return 130

    if commands_failed:
        print()
        print("######## SLOP DETECTED! ########")
        print("Are you vibe coding?")
        print()
        print(f"{commands_failed} command(s) failed:")
        print()
        print("\n\n".join(total_errors))
        return 1

    print("LGTM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
