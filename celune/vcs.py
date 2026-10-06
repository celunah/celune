# SPDX-License-Identifier: Apache-2.0
"""Small GitPython helpers shared by Celune's repository-aware modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from git import Repo


def _open_repository(path: Optional[Path] = None) -> Repo:
    """Open a repository at ``path`` or discover one from the current directory."""
    from git import Repo

    return Repo(
        str(path) if path is not None else None,
        search_parent_directories=path is None,
    )


def get_revision(path: Optional[Path] = None) -> str:
    """Return the short Git revision and dirty marker, or empty when unavailable.

    Args:
        path: Optional path used to locate the repository.

    Returns:
        str: Seven-character revision plus ``*`` when dirty, or ``""`` if unavailable.
    """
    try:
        from git.exc import GitError
    except ModuleNotFoundError as package:
        if package.name != "git":
            raise
        return ""

    try:
        with _open_repository(path) as repository:
            revision = repository.head.commit.hexsha[:7]
            dirty = "*" if repository.is_dirty(untracked_files=True) else ""
            return f"{revision}{dirty}"
    except (GitError, OSError, TypeError, ValueError):
        return ""


def get_branch(path: Optional[Path] = None) -> str:
    """Return the current Git repository's branch name.

    Args:
        path: Optional path used to locate the repository.

    Returns:
        str: The repository's branch name.
    """

    from git.exc import GitError

    try:
        with _open_repository(path) as repository:
            return repository.active_branch.name
    except (GitError, OSError, TypeError, ValueError):
        return ""
