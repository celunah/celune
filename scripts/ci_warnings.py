# SPDX-License-Identifier: MIT
"""Run a command while turning warning-like output into GitHub annotations."""

from __future__ import annotations

import re
import sys
import subprocess
from typing import Optional

_WARNING_LINE = re.compile(
    r"(?:\b\w*warn(?:ing)?\b|\bdeprecated\b|\bW\d{4}\b)",
    re.IGNORECASE,
)


def _is_warning_line(line: str) -> bool:
    """Return whether one command-output line describes a warning."""
    return bool(_WARNING_LINE.search(line))


def _annotation_message(line: str) -> str:
    """Escape one output line for a GitHub Actions warning command."""
    return line.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(arguments: Optional[list[str]] = None) -> int:
    """Run the requested command and preserve its exit status.

    Args:
        arguments: Command arguments excluding the Python executable and script path.

    Returns:
        int: The wrapped command's exit status, or ``2`` when no command is supplied.
    """
    command = list(sys.argv[1:] if arguments is None else arguments)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("Usage: ci_warnings.py -- command [args ...]", file=sys.stderr)
        return 2

    process = subprocess.Popen(  # pylint: disable=R1732
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            print(line, flush=True)
            if _is_warning_line(line) and not line.startswith("::warning::"):
                print(f"::warning::{_annotation_message(line)}", flush=True)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130
    finally:
        process.stdout.close()

    return process.wait()


if __name__ == "__main__":
    sys.exit(main())
