# SPDX-License-Identifier: MIT
"""Celune automatic update helpers."""

from __future__ import annotations

import os
import re
import sys
import json
import time
import ctypes
import shutil
import hashlib
import zipfile
import tempfile
import subprocess
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

from . import __version__
from .exceptions import UpdateError
from .typing.common import JSONSerializable
from .paths import project_root, running_compiled

REMOTE_URL = "https://github.com/celunah/celune.git"
ARTIFACT_BASE_URL = "https://nightly.link/celunah/celune/workflows/ci"
UPDATE_MANIFEST_NAME = "celune-update.json"
SHORT_HASH_LENGTH = 7
UPDATE_BRANCHES = {"main", "master"}
DOWNLOAD_TIMEOUT = 30


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


@dataclass(frozen=True)
class BundleManifest:
    """Compiled bundle metadata distributed with launcher artifacts."""

    version: str
    revision: str
    artifact: str
    files: dict[str, str]


def _repo_root() -> Path:
    """Return where the Git repository root is located."""
    return project_root()


def _bundle_dir() -> Path:
    """Return the compiled bundle directory used by the launcher."""
    executable = Path(sys.argv[0]).resolve()
    if executable.is_dir():
        return executable
    return executable.parent


def _manifest_path(bundle_dir: Optional[Path] = None) -> Path:
    """Return the bundled update manifest path."""
    return (bundle_dir or _bundle_dir()) / UPDATE_MANIFEST_NAME


def _platform_artifact_name() -> str:
    """Return the CI artifact name for the current platform."""
    if os.name == "nt":
        return "Celune-win-x64"
    return "Celune-linux-x64"


def _artifact_download_url(branch: str, artifact: str) -> str:
    """Return the direct nightly.link ZIP URL for one workflow artifact."""
    return f"{ARTIFACT_BASE_URL}/{branch}/{artifact}.zip"


def _run_git(args: list[str], timeout: int = 15) -> str:
    """Run a Git command on the repository."""
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
    """Format an error from Git."""
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
    """Check if Git succeeded this command."""
    try:
        _run_git(args, timeout=timeout)
        return True
    except subprocess.CalledProcessError:
        return False


def _short_revision(revision: str) -> str:
    """Return current commit revision."""
    return revision[:SHORT_HASH_LENGTH] if revision else "unknown"


def _base_version(version: str) -> str:
    """Return current Celune version info from local repository."""
    return version.split("+", 1)[0]


def _normalize_tag(tag: str) -> str:
    """Convert a tag identifier to a usable format."""
    return tag.removeprefix("refs/tags/").removeprefix("v")


def _version_key(tag: str) -> VersionKey:
    """Return a Celune version for the given tag."""
    normalized = _normalize_tag(tag)
    rmatch = re.match(r"^(\d+(?:\.\d+)*)(.*)$", normalized)
    if not rmatch:
        return VersionKey((), normalized)

    numbers = tuple(int(part) for part in rmatch.group(1).split("."))
    suffix = rmatch.group(2)
    return VersionKey(numbers, suffix)


def _is_newer_version_tag(candidate: str, current: str) -> bool:
    """Is this revision a newer Celune version?"""
    candidate_key = _version_key(candidate)
    current_key = _version_key(current)
    if candidate_key.numbers != current_key.numbers:
        return candidate_key.numbers > current_key.numbers

    return candidate_key.suffix > current_key.suffix


def _latest_remote_tag() -> tuple[str, str]:
    """Return the latest revision from the remote repository."""
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


def _remote_revision(ref: str) -> str:
    """Get current remote revision."""
    output = _run_git(["ls-remote", REMOTE_URL, ref], timeout=20)
    if not output:
        return ""
    return output.split(maxsplit=1)[0]


def _remote_head_revision() -> str:
    """Get current remote revision from HEAD."""
    return _remote_revision("HEAD")


def _remote_branch_revision(branch: str) -> str:
    """Return current remote branch revision."""
    return _remote_revision(f"refs/heads/{branch}")


def _current_branch() -> str:
    """Get current branch."""
    return _run_git(["branch", "--show-current"])


def _local_tag() -> str:
    """Get current local tag."""
    try:
        return _normalize_tag(_run_git(["describe", "--tags", "--exact-match", "HEAD"]))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _local_revision() -> str:
    """Get current local revision."""
    return _run_git(["rev-parse", "HEAD"])


def _has_local_changes() -> bool:
    """Does the local repository have any changes pending for commit?"""
    return bool(_run_git(["status", "--porcelain"]))


def _has_new_remote_revision(local_revision: str, remote_revision: str) -> bool:
    """Return whether the remote revision is a fast-forward update for HEAD."""
    if not remote_revision or local_revision == remote_revision:
        return False

    if _git_succeeds(["merge-base", "--is-ancestor", remote_revision, "HEAD"]):
        return False

    return _git_succeeds(["merge-base", "--is-ancestor", "HEAD", remote_revision])


def _is_git_checkout() -> bool:
    """Can the repository be checked out?"""
    try:
        return _run_git(["rev-parse", "--is-inside-work-tree"]) == "true"
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return False


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_checksums(
    bundle_dir: Path,
    filenames: Union[tuple[str, ...], list[str]],
) -> dict[str, str]:
    """Return checksums for bundle files present in one install directory."""
    checksums: dict[str, str] = {}
    for filename in filenames:
        path = bundle_dir / filename
        if path.is_file():
            checksums[filename] = _sha256_file(path)
    return checksums


def _parse_bundle_manifest(raw: JSONSerializable) -> Optional[BundleManifest]:
    """Convert raw JSON-like data into bundle metadata."""
    if not isinstance(raw, dict):
        return None

    version = raw.get("version")
    revision = raw.get("revision")
    artifact = raw.get("artifact")
    files = raw.get("files")
    if not (
        isinstance(version, str)
        and isinstance(revision, str)
        and isinstance(artifact, str)
        and isinstance(files, dict)
    ):
        return None

    normalized_files: dict[str, str] = {}
    for name, digest in files.items():
        if isinstance(name, str) and isinstance(digest, str):
            normalized_files[name] = digest

    if not normalized_files:
        return None

    return BundleManifest(
        version=version,
        revision=revision,
        artifact=artifact,
        files=normalized_files,
    )


def _load_local_bundle_manifest(
    bundle_dir: Optional[Path] = None,
) -> Optional[BundleManifest]:
    """Load the local compiled bundle manifest when available."""
    manifest_file = _manifest_path(bundle_dir)
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return _parse_bundle_manifest(payload)


def _download_to_file(url: str, destination: Path) -> None:
    """Download one URL into the given destination path."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Celune-Updater/1.0"},
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _download_artifact_zip(branch: str, artifact: str, destination: Path) -> None:
    """Download the latest launcher artifact ZIP for one branch."""
    _download_to_file(_artifact_download_url(branch, artifact), destination)


def _manifest_from_zip(zip_path: Path) -> Optional[BundleManifest]:
    """Load bundled update metadata from an artifact ZIP file."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if Path(name).name != UPDATE_MANIFEST_NAME:
                    continue
                with archive.open(name) as handle:
                    payload = json.loads(handle.read().decode("utf-8"))
                return _parse_bundle_manifest(payload)
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
        return None
    return None


def _read_remote_bundle_manifest(branch: str) -> Optional[BundleManifest]:
    """Download the latest artifact manifest for this platform."""
    artifact = _platform_artifact_name()
    with tempfile.TemporaryDirectory(prefix="celune-update-check-") as temp_dir:
        zip_path = Path(temp_dir) / f"{artifact}.zip"
        try:
            _download_artifact_zip(branch, artifact, zip_path)
        except OSError:
            return None
        return _manifest_from_zip(zip_path)


def _compiled_bundle_matches_remote(
    bundle_dir: Path,
    remote_manifest: BundleManifest,
) -> bool:
    """Return whether the installed bundle already matches the remote artifact."""
    local_files = _bundle_checksums(bundle_dir, list(remote_manifest.files))
    return bool(local_files) and local_files == remote_manifest.files


def _check_for_compiled_update() -> Optional[UpdateInfo]:
    """Check whether the packaged launcher bundle differs from the latest artifact."""
    local_manifest = _load_local_bundle_manifest()
    if local_manifest is None:
        return None

    branch = "main"
    remote_revision = ""
    latest_tag = ""
    latest_tag_revision = ""
    try:
        if _is_git_checkout():
            branch = _current_branch() or branch
            if branch and branch not in UPDATE_BRANCHES:
                return None
            remote_revision = (
                _remote_branch_revision(branch) if branch else _remote_head_revision()
            )
            latest_tag, latest_tag_revision = _latest_remote_tag()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ):
        branch = "main"

    remote_manifest = _read_remote_bundle_manifest(branch or "main")
    if remote_manifest is None:
        return None

    if _compiled_bundle_matches_remote(_bundle_dir(), remote_manifest):
        return None

    latest_revision = remote_manifest.revision or remote_revision or latest_tag_revision
    latest_version = remote_manifest.version or latest_tag or _base_version(__version__)
    return UpdateInfo(
        local_version=local_manifest.version,
        local_revision=_short_revision(local_manifest.revision),
        local_tag="",
        latest_version=latest_version,
        latest_revision=_short_revision(latest_revision),
        latest_tag=latest_tag,
    )


def check_for_update() -> Optional[UpdateInfo]:
    """Check for a newer Celune revision or packaged launcher bundle.

    Returns:
        Optional[UpdateInfo]: Metadata describing the available update, or ``None`` when no safe update path is
        currently available.
    """
    if os.getenv("CELUNE_SKIP_UPDATE") in {"1", "true", "on", "yes", "enabled"}:
        return None

    if running_compiled():
        return _check_for_compiled_update()

    if not _is_git_checkout():
        return None

    try:
        branch = _current_branch()
        if branch and branch not in UPDATE_BRANCHES:
            return None

        if _has_local_changes():
            return None

        local_revision = _local_revision()
        local_tag = _local_tag()
        remote_revision = (
            _remote_branch_revision(branch) if branch else _remote_head_revision()
        )
        latest_tag, latest_tag_revision = _latest_remote_tag()
        has_new_revision = _has_new_remote_revision(local_revision, remote_revision)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ):
        return None

    local_version = _base_version(__version__)
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


def _extract_artifact_root(zip_path: Path, destination: Path) -> Path:
    """Extract one artifact ZIP and return the directory containing the manifest."""
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)

    for manifest in destination.rglob(UPDATE_MANIFEST_NAME):
        return manifest.parent

    raise UpdateError("downloaded artifact is missing update metadata")


def _replace_path(source: Path, destination: Path) -> None:
    """Replace one file or directory in the install directory."""
    if destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _apply_compiled_update(install_dir: Optional[Path] = None) -> None:
    """Download and replace the packaged launcher bundle in place."""
    bundle_dir = (install_dir or _bundle_dir()).resolve()
    branch = "main"
    if _is_git_checkout():
        try:
            branch = _current_branch() or branch
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            branch = "main"

    artifact = _platform_artifact_name()
    with tempfile.TemporaryDirectory(prefix="celune-update-") as temp_dir:
        temp_root = Path(temp_dir)
        zip_path = temp_root / f"{artifact}.zip"
        try:
            _download_artifact_zip(branch, artifact, zip_path)
        except OSError as exc:
            raise UpdateError(f"could not download the latest artifact: {exc}") from exc

        try:
            extracted_root = _extract_artifact_root(zip_path, temp_root / "artifact")
        except (OSError, zipfile.BadZipFile, UpdateError) as exc:
            raise UpdateError(f"could not unpack the latest artifact: {exc}") from exc

        for source in extracted_root.iterdir():
            _replace_path(source, bundle_dir / source.name)


def update_to_latest(install_dir: Optional[Path] = None) -> None:
    """Update Celune either from Git or from the latest packaged artifact.

    Args:
        install_dir: Optional compiled-install directory to replace in place.

    Raises:
        UpdateError: Raised when the repository or packaged install cannot be updated safely.
    """
    if running_compiled() or install_dir is not None:
        _apply_compiled_update(install_dir=install_dir)
        return

    if not _is_git_checkout():
        raise UpdateError("did not find a repository")

    if _has_local_changes():
        raise UpdateError("repository not committed")

    try:
        branch = _current_branch()
    except subprocess.CalledProcessError as exc:
        raise UpdateError(_format_git_error(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError(
            f"timed out checking the current branch after {exc.timeout} seconds"
        ) from exc
    except FileNotFoundError as exc:
        raise UpdateError("git is not available") from exc

    if branch and branch not in UPDATE_BRANCHES:
        raise UpdateError(f"automatic updates are disabled on branch '{branch}'")

    fetch_ref = f"refs/heads/{branch}" if branch else "HEAD"

    try:
        _run_git(["fetch", "--prune", REMOTE_URL, fetch_ref], timeout=120)
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


def _wait_for_pid_exit(pid: int, timeout: float = 120.0) -> None:
    """Wait until the launcher process fully exits before replacing it."""
    if pid <= 0:
        return

    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return
        try:
            wait_ms = max(0, int(timeout * 1000))
            result = kernel32.WaitForSingleObject(handle, wait_ms)
            if result == 0x00000102:
                raise UpdateError("timed out waiting for the launcher to exit")
        finally:
            kernel32.CloseHandle(handle)
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.1)

    raise UpdateError("timed out waiting for the launcher to exit")


short_revision = _short_revision
normalize_tag = _normalize_tag
is_newer_version_tag = _is_newer_version_tag
has_new_remote_revision = _has_new_remote_revision
sha256_file = _sha256_file


def apply_update_and_restart(
    parent_pid: int,
    launcher_path: Path,
    launcher_args: list[str],
) -> int:
    """Wait for the launcher, apply the update, then restart it.

    Args:
        parent_pid: Process ID of the launcher that must fully exit first.
        launcher_path: Path to the outer ``celune`` launcher binary to restart.
        launcher_args: Original launcher arguments to pass back into the restart.

    Returns:
        int: Process exit code to return from the updater helper.
    """
    _wait_for_pid_exit(parent_pid)
    update_to_latest(install_dir=launcher_path.resolve().parent)
    subprocess.Popen([str(launcher_path), *launcher_args], cwd=_repo_root())  # pylint: disable=R1732
    return 0
