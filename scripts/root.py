# SPDX-License-Identifier: Apache-2.0
"""Maintain the repository marker used by the native Celune launchers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

_VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"$', re.MULTILINE)


def _repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parent.parent


def _git_value(root: Path, *arguments: str) -> str:
    """Return one trimmed value from Git in ``root``."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def update_marker(root: Optional[Path] = None) -> str:
    """Write and return the current repository's Celune marker contents."""
    repository = root or _repository_root()
    project = (repository / "pyproject.toml").read_text(encoding="utf-8")
    version_match = _VERSION_PATTERN.search(project)
    if version_match is None:
        raise ValueError("pyproject.toml does not define a project version")

    version = version_match.group(1)
    commit = _git_value(repository, "rev-parse", "--short", "HEAD")
    commit_date = _git_value(repository, "log", "-1", "--format=%cs")
    year, month, day = commit_date.split("-")
    marker = f"v{version} ({commit}), {day}/{month}/{year}\n"
    marker_path = repository / ".celune-root"
    current_marker = (
        marker_path.read_text(encoding="utf-8") if marker_path.exists() else None
    )
    if current_marker != marker:
        marker_path.write_text(marker, encoding="utf-8")
    return marker.rstrip("\n")


if __name__ == "__main__":
    print(update_marker())
