# SPDX-License-Identifier: Apache-2.0
"""Small GitPython helpers shared by Celune's repository-aware modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .i18n import string

if TYPE_CHECKING:
    from git import Repo

UPSTREAM_REPOSITORY_URL = "https://github.com/celunah/celune"


def _open_repository(path: Optional[Path] = None) -> Repo:
    """Open a repository at ``path`` or discover one from the current directory."""
    from git import Repo

    return Repo(
        str(path) if path is not None else None,
        search_parent_directories=path is None,
    )


def _has_git_metadata(path: Path) -> bool:
    """Return whether ``path`` has a Git directory or worktree pointer."""
    marker = path / ".git"
    try:
        if marker.is_dir():
            git_dir = marker
        elif marker.is_file():
            pointer = marker.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return False
            git_dir = Path(pointer.partition(":")[2].strip())
            if not git_dir.is_absolute():
                git_dir = path / git_dir
        else:
            return False
        return (git_dir / "HEAD").is_file()
    except OSError:
        return False


def is_git_checkout(path: Optional[Path] = None) -> bool:
    """Return whether ``path`` is a committed Git working tree.

    Args:
        path: Optional project path that must be the checkout root.

    Returns:
        bool: Whether the path is the root of a working tree with a valid commit.
    """
    checkout_path = (path or Path.cwd()).resolve()
    try:
        from git.exc import GitError
    except ModuleNotFoundError as package:
        if package.name != "git":
            raise
        return _has_git_metadata(checkout_path)

    try:
        with _open_repository(checkout_path) as repository:
            return (
                not repository.bare
                and repository.working_tree_dir is not None
                and Path(repository.working_tree_dir).resolve() == checkout_path
                and bool(repository.head.commit.hexsha)
            )
    except (GitError, OSError, TypeError, ValueError):
        return False


def git_checkout_hint() -> str:
    """Return the localized command for cloning Celune's upstream repository.

    Returns:
        str: The localized upstream clone command.
    """
    return string("git.clone_upstream")


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
