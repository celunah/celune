# SPDX-License-Identifier: Apache-2.0
"""Maintain the repository marker used by the native Celune launchers."""

from __future__ import annotations

import re
import sys
import json
import subprocess
from pathlib import Path
from collections.abc import Sequence
from typing import Optional, cast

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


def _git_revision(root: Path) -> str:
    """Return HEAD only when the script root is the checkout root."""
    checkout_root = Path(_git_value(root, "rev-parse", "--show-toplevel")).resolve()
    if checkout_root != root.resolve():
        raise subprocess.CalledProcessError(
            1,
            "git rev-parse --show-toplevel",
        )
    revision = _git_value(root, "rev-parse", "--verify", "HEAD^{commit}")
    if not revision:
        raise subprocess.CalledProcessError(1, "git rev-parse HEAD^{commit}")
    return revision


def _git_checkout_messages() -> tuple[str, str]:
    """Read localized Git checkout instructions without importing the app."""
    language_path = _repository_root() / "celune" / "lang" / "en.json"
    translations = cast(
        dict[str, str],
        json.loads(language_path.read_text(encoding="utf-8")),
    )
    return (
        translations["git.checkout_required"],
        translations["git.clone_upstream"],
    )


def update_marker(root: Optional[Path] = None) -> str:
    """Write and return the current repository's Celune marker contents."""
    repository = root or _repository_root()
    project = (repository / "pyproject.toml").read_text(encoding="utf-8")
    version_match = _VERSION_PATTERN.search(project)
    if version_match is None:
        raise ValueError("pyproject.toml does not define a project version")

    version = version_match.group(1)
    commit = _git_revision(repository)[:7]
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


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Update the marker or print the current revision for build scripts.

    Args:
        arguments: Optional command-line arguments, primarily ``--revision``.

    Returns:
        int: Zero on success, one without a valid checkout, or two for invalid arguments.
    """
    args = list(sys.argv[1:] if arguments is None else arguments)
    repository = _repository_root()
    try:
        if args == ["--revision"]:
            print(_git_revision(repository))
        elif not args:
            print(update_marker(repository))
        else:
            return 2
    except (FileNotFoundError, subprocess.CalledProcessError):
        checkout_required, clone_upstream = _git_checkout_messages()
        print(checkout_required, file=sys.stderr)
        print(clone_upstream, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
