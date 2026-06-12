# SPDX-License-Identifier: MIT
"""Run CI automatically."""

import sys
import subprocess

from tqdm.contrib import tzip

CI_COMMANDS = (
    ("ruff", "format", "--check"),
    ("pylint",),
    ("pyrefly", "check"),
    ("pytest", "-v"),
)

CI_PATHS = ((".",), ("celune", "tests"), ("celune", "tests"), ("tests",))

cmds_failed = 0
total_errors = []

_CACHE_PERMISSION_MARKERS = (
    "Access is denied.",
    "Access is denied",
    "Permission denied",
)


def _run_uv_command(*cmd: str) -> None:
    """Run one uv-backed CI command, retrying without cache on permission errors."""
    base_cmd = ["uv", "run", *cmd]
    try:
        subprocess.run(
            base_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=300,
        )
        return
    except subprocess.CalledProcessError as failed:
        combined_output = f"{failed.stdout}\n{failed.stderr}"
        if not any(marker in combined_output for marker in _CACHE_PERMISSION_MARKERS):
            raise
    except subprocess.TimeoutExpired:
        raise

    subprocess.run(
        ["uv", "--no-cache", "run", *cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
        timeout=300,
    )


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
        if failed.stdout:
            total_errors.append(failed.stdout)
        if failed.stderr:
            total_errors.append(failed.stderr)
    except subprocess.TimeoutExpired as failed:
        cmds_failed += 1
        total_errors.append(f"{' '.join(cmd)} has timed out")


if cmds_failed:
    print()
    print("######## SLOP DETECTED! ########")
    print(f"{cmds_failed} command(s) failed:")
    print()
    print("\n\n".join(total_errors))
    sys.exit(1)

print("LGTM")
