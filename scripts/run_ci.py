# SPDX-License-Identifier: MIT
"""Run CI automatically."""

import subprocess
import sys
import signal
from typing import Optional

from tqdm.contrib import tzip

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

cmds_failed = 0
total_errors = []

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


def _run_uv_command(*command: str) -> None:
    """Run one uv-backed CI command, retrying without cache on permission errors."""
    base_cmd = ["uv", "run", *command]
    try:
        subprocess.run(
            base_cmd,
            check=True,
            text=True,
            timeout=300,
            capture_output=True,
        )
        return
    except subprocess.CalledProcessError as failed_process:
        combined_output = f"{failed_process.stdout}\n{failed_process.stderr}"
        marker = _agent_permission_marker(combined_output)
        if marker is None:
            raise

    try:
        subprocess.run(
            ["uv", "--no-cache", "run", *cmd],
            check=True,
            text=True,
            timeout=300,
            capture_output=True,
        )
    except subprocess.CalledProcessError as failed_process:
        combined_output = f"{failed_process.stdout}\n{failed_process.stderr}"
        marker = _agent_permission_marker(combined_output)
        if marker is not None:
            raise RuntimeError(
                f"agent has no permissions to run CI: {marker}"
            ) from failed_process
        raise


if len(CI_COMMANDS) != len(CI_PATHS):
    raise RuntimeError(
        f"CI configuration mismatch: {len(CI_COMMANDS)} commands for {len(CI_PATHS)} path entries"
    )

for cmd, paths in tzip(
    CI_COMMANDS,
    CI_PATHS,
    desc="Running CI commands",
    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}",
):
    try:
        _run_uv_command(*cmd, *paths)
    except subprocess.CalledProcessError as failed:
        cmds_failed += 1

        exit_code = failed.returncode
        if exit_code >= 127:
            signal_name = signal.Signals(exit_code % 128).name

            if signal_name in (
                "SIGILL",
                "SIGSEGV",
                "SIGABRT",
                "SIGBUS",
            ):
                print(f"Caught a fatal signal {signal_name} ({exit_code % 128})")
                print("CI cannot continue.")
                print()
                raise RuntimeError(f"fatal signal {signal_name}") from failed

        if failed.stdout:
            total_errors.append(failed.stdout)
        if failed.stderr:
            total_errors.append(failed.stderr)
    except subprocess.TimeoutExpired:
        cmds_failed += 1
        total_errors.append(f"{' '.join(cmd)} has timed out")


if cmds_failed:
    print()
    print("######## SLOP DETECTED! ########")
    print("Are you vibe coding?")
    print()
    print(f"{cmds_failed} command(s) failed:")
    print()
    print("\n\n".join(total_errors))
    sys.exit(1)

print("LGTM")
