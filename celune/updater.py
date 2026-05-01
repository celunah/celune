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
    return Path(__file__).resolve().parent.parent


def _run_git(args: list[str], timeout: int = 15) -> str:
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
    return revision[:SHORT_HASH_LENGTH] if revision else "unknown"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _normalize_tag(tag: str) -> str:
    return tag.removeprefix("refs/tags/").removeprefix("v")


def _version_key(tag: str) -> VersionKey:
    normalized = _normalize_tag(tag)
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", normalized)
    if not match:
        return VersionKey((), normalized)

    numbers = tuple(int(part) for part in match.group(1).split("."))
    suffix = match.group(2)
    return VersionKey(numbers, suffix)


def _is_newer_version_tag(candidate: str, current: str) -> bool:
    candidate_key = _version_key(candidate)
    current_key = _version_key(current)
    if candidate_key.numbers != current_key.numbers:
        return candidate_key.numbers > current_key.numbers

    return candidate_key.suffix > current_key.suffix


def _latest_remote_tag() -> tuple[str, str]:
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
    output = _run_git(["ls-remote", REMOTE_URL, "HEAD"], timeout=20)
    if not output:
        return ""

    return output.split(maxsplit=1)[0]


def _local_tag() -> str:
    try:
        return _normalize_tag(_run_git(["describe", "--tags", "--exact-match", "HEAD"]))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _local_revision() -> str:
    return _run_git(["rev-parse", "HEAD"])


def _has_local_changes() -> bool:
    return bool(_run_git(["status", "--porcelain"]))


def _is_git_checkout() -> bool:
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
        raise UpdateError("Celune did not find a Git repository.")

    if _has_local_changes():
        raise UpdateError(
            "Celune has determined the local Git repository has not been committed yet."
        )

    try:
        _run_git(["fetch", "--prune", REMOTE_URL, "HEAD"], timeout=120)
    except subprocess.CalledProcessError as exc:
        raise UpdateError(_format_git_error(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(f"Git fetch timed out after {exc.timeout} seconds.") from exc
    except FileNotFoundError as exc:
        raise UpdateError("Celune could not find Git on this system.") from exc

    try:
        can_fast_forward = _git_succeeds(
            ["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"]
        )
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"Git update validation timed out after {exc.timeout} seconds."
        ) from exc
    except FileNotFoundError as exc:
        raise UpdateError("Celune could not find Git on this system.") from exc

    if not can_fast_forward:
        raise UpdateError(
            "Celune cannot update automatically because the local branch cannot "
            "be fast-forwarded."
        )

    try:
        _run_git(["merge", "--ff-only", "FETCH_HEAD"], timeout=120)
    except subprocess.CalledProcessError as exc:
        raise UpdateError(_format_git_error(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(f"Git merge timed out after {exc.timeout} seconds.") from exc
    except FileNotFoundError as exc:
        raise UpdateError("Celune could not find Git on this system.") from exc
