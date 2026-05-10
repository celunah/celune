"""Celune self-update helpers."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .exceptions import UpdateError

REMOTE_URL = "https://github.com/celunah/celune.git"
SHORT_HASH_LENGTH = 7


@dataclass(frozen=True)
class UpdateInfo:
    """Information about an available Celune update."""

    local_version: str
    local_revision: str
    local_tag: str
    latest_version: str
    latest_revision: str
    latest_tag: str


@dataclass(frozen=True)
class VersionKey:
    """Structured representation of a Git tag version."""

    numbers: tuple[int, ...]
    suffix: str = ""


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent


def _run_git(args: list[str], timeout: int = 15) -> str:
    """Run a Git command in the repository root.

    Args:
        args: Git arguments excluding the ``git`` executable.
        timeout: Maximum seconds to wait for the command.

    Returns:
        str: Trimmed stdout from the Git command.

    Raises:
        subprocess.CalledProcessError: Git exits with a non-zero status.
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=_repo_root(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _format_git_error(exc: subprocess.CalledProcessError) -> str:
    """Format a Git process failure for display.

    Args:
        exc: Git process exception to describe.

    Returns:
        str: Human-readable Git error message.
    """
    details = "\n".join(
        part.strip()
        for part in (exc.stderr, exc.stdout)
        if isinstance(part, str) and part.strip()
    )
    command = " ".join(str(part) for part in exc.cmd)
    if details:
        return f"{command} failed:\n{details}"

    return f"{command} failed with exit code {exc.returncode}."


def _git_succeeds(args: list[str], timeout: int = 15) -> bool:
    """Return whether a Git command succeeds.

    Args:
        args: Git arguments excluding the ``git`` executable.
        timeout: Maximum seconds to wait for the command.

    Returns:
        bool: ``True`` when Git exits with status code zero.

    Raises:
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=_repo_root(),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


def _short_revision(revision: str) -> str:
    """Shorten a Git revision for display.

    Args:
        revision: Full Git revision string.

    Returns:
        str: Short revision or ``"unknown"``.
    """
    return revision[:SHORT_HASH_LENGTH] if revision else "unknown"


def _base_version(version: str) -> str:
    """Return the public version without local build metadata.

    Args:
        version: Version string to normalize.

    Returns:
        str: Version before the first ``+`` suffix.
    """
    return version.split("+", 1)[0]


def _normalize_tag(tag: str) -> str:
    """Normalize a Git tag or ref into a version string.

    Args:
        tag: Tag or tag ref to normalize.

    Returns:
        str: Tag without ``refs/tags/`` or leading ``v``.
    """
    return tag.removeprefix("refs/tags/").removeprefix("v")


def _version_key(tag: str) -> VersionKey:
    """Convert a tag into a comparable version key.

    Args:
        tag: Version tag to parse.

    Returns:
        VersionKey: Parsed numeric version and suffix.
    """
    normalized = _normalize_tag(tag)
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", normalized)
    if not match:
        return VersionKey((), normalized)

    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    return VersionKey(numbers, suffix)


def _is_newer_version_tag(candidate: str, current: str) -> bool:
    """Return whether one version tag is newer than another.

    Args:
        candidate: Candidate version tag.
        current: Current version tag.

    Returns:
        bool: ``True`` when the candidate should be considered newer.
    """
    candidate_key = _version_key(candidate)
    current_key = _version_key(current)
    if candidate_key.numbers != current_key.numbers:
        return candidate_key.numbers > current_key.numbers

    return candidate_key.suffix > current_key.suffix


def _latest_remote_tag() -> tuple[str, str]:
    """Find the latest version tag available on the remote repository.

    Returns:
        tuple[str, str]: Latest tag and its revision, or empty strings.

    Raises:
        subprocess.CalledProcessError: Git cannot read remote tags.
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    output = _run_git(["ls-remote", "--tags", "--refs", REMOTE_URL], timeout=20)
    tags: list[tuple[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        revision, ref = line.split(maxsplit=1)
        tags.append((_normalize_tag(ref), revision))

    if not tags:
        return "", ""

    latest_tag, latest_revision = tags[0]
    for tag, revision in tags[1:]:
        if _is_newer_version_tag(tag, latest_tag):
            latest_tag = tag
            latest_revision = revision

    return latest_tag, latest_revision


def _remote_head_revision() -> str:
    """Read the remote HEAD revision.

    Returns:
        str: Remote HEAD revision, or an empty string.

    Raises:
        subprocess.CalledProcessError: Git cannot read remote HEAD.
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    output = _run_git(["ls-remote", REMOTE_URL, "HEAD"], timeout=20)
    if not output:
        return ""

    return output.split(maxsplit=1)[0]


def _local_tag() -> str:
    """Return the exact local tag for HEAD when one exists.

    Returns:
        str: Normalized local tag, or an empty string.
    """
    try:
        return _normalize_tag(_run_git(["describe", "--tags", "--exact-match", "HEAD"]))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _local_revision() -> str:
    """Return the current local Git revision.

    Returns:
        str: Current HEAD revision.

    Raises:
        subprocess.CalledProcessError: Git cannot read the local revision.
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    return _run_git(["rev-parse", "HEAD"])


def _has_local_changes() -> bool:
    """Return whether the Git checkout has uncommitted changes.

    Returns:
        bool: ``True`` when ``git status --porcelain`` has output.

    Raises:
        subprocess.CalledProcessError: Git cannot read local status.
        subprocess.TimeoutExpired: Git does not finish before the timeout.
    """
    return bool(_run_git(["status", "--porcelain"]))


def _is_git_checkout() -> bool:
    """Return whether the project is running from a Git checkout.

    Returns:
        bool: ``True`` when the project root is inside a Git work tree.
    """
    try:
        return _run_git(["rev-parse", "--is-inside-work-tree"]) == "true"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def check_for_update() -> Optional[UpdateInfo]:
    """Check GitHub for a newer Celune revision or tag.

    Returns:
        Optional[UpdateInfo]: Information about the update, or ``None`` when Celune
            appears current or update metadata cannot be read.
    """
    if os.getenv("CELUNE_SKIP_UPDATE") in {"1", "true", "on", "yes", "enabled"}:
        return None

    if not _is_git_checkout():
        return None

    try:
        if _has_local_changes():
            return None

        local_revision = _local_revision()
        local_tag = _local_tag()
        remote_revision = _remote_head_revision()
        latest_tag, latest_tag_revision = _latest_remote_tag()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ):
        return None

    local_version = _base_version(__version__)
    has_new_revision = bool(remote_revision and local_revision != remote_revision)
    has_new_tag = bool(
        latest_tag and _is_newer_version_tag(latest_tag, local_tag or local_version)
    )
    if not has_new_revision and not has_new_tag:
        return None

    latest_revision = remote_revision if has_new_revision else latest_tag_revision
    if not latest_revision:
        return None

    latest_version = latest_tag or _base_version(__version__)
    return UpdateInfo(
        local_version=local_version,
        local_revision=_short_revision(local_revision),
        local_tag=local_tag,
        latest_version=latest_version,
        latest_revision=_short_revision(latest_revision),
        latest_tag=latest_tag,
    )


def update_to_latest() -> None:
    """Fast-forward the local checkout to GitHub's current Celune revision.

    Raises:
        UpdateError: Celune cannot be updated safely.
    """
    if not _is_git_checkout():
        raise UpdateError("did not find a repository")

    if _has_local_changes():
        raise UpdateError("repository not committed")

    try:
        _run_git(["fetch", "--prune", REMOTE_URL, "HEAD"], timeout=120)
    except subprocess.CalledProcessError as exc:
        raise UpdateError(_format_git_error(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"timed out fetching the repository after {exc.timeout} seconds"
        ) from exc
    except FileNotFoundError as exc:
        raise UpdateError("git is not available") from exc

    try:
        can_fast_forward = _git_succeeds(
            ["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"]
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"timed out validating the update after {exc.timeout} seconds"
        ) from exc
    except FileNotFoundError as exc:
        raise UpdateError("git is not available") from exc

    if not can_fast_forward:
        raise UpdateError("repository is not able to be fast-forwarded")

    try:
        _run_git(["merge", "--ff-only", "FETCH_HEAD"], timeout=120)
    except subprocess.CalledProcessError as exc:
        raise UpdateError(_format_git_error(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(f"timed out merging after {exc.timeout} seconds") from exc
    except FileNotFoundError as exc:
        raise UpdateError("git is not available") from exc
