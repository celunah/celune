"""Run CI automatically."""

import sys
import subprocess

from tqdm.contrib import tzip

CI_COMMANDS = (
    ("ruff", "format", "--check"),
    ("pylint",),
    ("pyright",),
    ("pytest", "-v"),
)

CI_PATHS = ((".",), ("celune", "tests"), ("celune", "tests"), ("tests",))

cmds_failed = 0
total_errors = []

if len(CI_COMMANDS) != len(CI_PATHS):
    raise RuntimeError(
        "CI configuration mismatch: {len(CI_COMMANDS)} commands for {len(CI_PATHS)} path entries"
    )

for cmd, paths in tzip(
    CI_COMMANDS,
    CI_PATHS,
    desc="Running CI commands",
    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt}",
):
    try:
        subprocess.run(
            ["uv", "run", *cmd, *paths],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            timeout=300,
        )
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
