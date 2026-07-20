# SPDX-License-Identifier: MIT
"""Celune automatic update helpers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from . import __version__
from .constants import CELUNE_UA
from .exceptions import UpdateError
from .i18n import string
from .paths import project_root, running_compiled
from .typing.common import JSONSerializable

REMOTE_URL = "https://github.com/celunah/celune.git"
RELEASES_API_URL = "https://api.github.com/repos/celunah/celune/releases?per_page=100"
UPDATE_MANIFEST_NAME = "celune-update.json"
SHORT_HASH_LENGTH = 7
UPDATE_BRANCHES = {"main", "master"}
DOWNLOAD_TIMEOUT = 30
FORCE_DISABLE_UPDATES = False
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


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


@dataclass(frozen=True)
class ReleaseInfo:
    """Published release metadata used by the updater."""

    tag: str
    version: str
    revision: str
    asset_url: str


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


def _parse_release(raw: JSONSerializable) -> Optional[ReleaseInfo]:
    """Parse one GitHub release when it is a published SemVer release."""
    if not isinstance(raw, dict):
        return None

    tag = raw.get("tag_name")
    if not isinstance(tag, str) or not _is_semver(tag):
        return None
    if raw.get("draft") is True:
        return None

    asset_url = ""
    assets = raw.get("assets")
    if isinstance(assets, list):
        expected_name = f"{_platform_artifact_name()}.zip"
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            if asset.get("name") != expected_name:
                continue
            browser_download_url = asset.get("browser_download_url")
            if isinstance(browser_download_url, str):
                asset_url = browser_download_url
                break

    target_commitish = raw.get("target_commitish")
    revision = (
        target_commitish
        if isinstance(target_commitish, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", target_commitish)
        else ""
    )
    return ReleaseInfo(
        tag=tag,
        version=_normalize_tag(tag),
        revision=revision,
        asset_url=asset_url,
    )


def _latest_release() -> Optional[ReleaseInfo]:
    """Return the newest published GitHub release with a SemVer tag."""
    request = urllib.request.Request(
        RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": CELUNE_UA,
        },
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        releases = json.loads(response.read().decode("utf-8"))

    if not isinstance(releases, list):
        return None

    latest: Optional[ReleaseInfo] = None
    for raw in releases:
        release = _parse_release(raw)
        if release is None:
            continue
        if latest is None or _is_newer_version_tag(release.tag, latest.tag):
            latest = release
    return latest


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
    """Return a structured SemVer key for the given tag."""
    normalized = _normalize_tag(tag)
    rmatch = SEMVER_PATTERN.fullmatch(normalized)
    if not rmatch:
        return VersionKey((), normalized)

    numbers = tuple(int(part) for part in rmatch.group(1, 2, 3))
    suffix = rmatch.group(4) or ""
    return VersionKey(numbers, suffix)


def _is_semver(tag: str) -> bool:
    """Return whether a tag contains a valid SemVer version."""
    return bool(SEMVER_PATTERN.fullmatch(_normalize_tag(tag)))


def _compare_prerelease(candidate: str, current: str) -> int:
    """Compare two SemVer prerelease identifiers."""
    if not candidate and not current:
        return 0
    if not candidate:
        return 1
    if not current:
        return -1

    candidate_parts = candidate.split(".")
    current_parts = current.split(".")
    for candidate_part, current_part in zip(candidate_parts, current_parts):
        if candidate_part == current_part:
            continue
        candidate_numeric = candidate_part.isdigit()
        current_numeric = current_part.isdigit()
        if candidate_numeric and current_numeric:
            return (int(candidate_part) > int(current_part)) - (
                int(candidate_part) < int(current_part)
            )
        if candidate_numeric != current_numeric:
            return -1 if candidate_numeric else 1
        return (candidate_part > current_part) - (candidate_part < current_part)

    return (len(candidate_parts) > len(current_parts)) - (
        len(candidate_parts) < len(current_parts)
    )


def _is_newer_version_tag(candidate: str, current: str) -> bool:
    """Return whether one valid SemVer tag is newer than another."""
    if not _is_semver(candidate) or not _is_semver(current):
        return False

    candidate_key = _version_key(candidate)
    current_key = _version_key(current)
    if candidate_key.numbers != current_key.numbers:
        return candidate_key.numbers > current_key.numbers

    return _compare_prerelease(candidate_key.suffix, current_key.suffix) > 0


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
        headers={"User-Agent": CELUNE_UA},
    )
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _download_release_zip(release: ReleaseInfo, destination: Path) -> None:
    """Download the current-platform ZIP attached to one GitHub release."""
    if not release.asset_url:
        raise UpdateError(string("cli.update_no_platform_zip"))
    _download_to_file(release.asset_url, destination)


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


def _read_remote_bundle_manifest(release: ReleaseInfo) -> Optional[BundleManifest]:
    """Download a release ZIP and read its bundled update metadata."""
    artifact = _platform_artifact_name()
    with tempfile.TemporaryDirectory(prefix="celune-update-check-") as temp_dir:
        zip_path = Path(temp_dir) / f"{artifact}.zip"
        try:
            _download_release_zip(release, zip_path)
        except (OSError, UpdateError):
            return None
        return _manifest_from_zip(zip_path)


def _compiled_bundle_matches_manifest(
    bundle_dir: Path,
    manifest: BundleManifest,
) -> bool:
    """Return whether the installed bundle matches the supplied manifest."""
    local_files = _bundle_checksums(bundle_dir, list(manifest.files))
    return bool(local_files) and local_files == manifest.files


def _release_manifest_matches(
    release: ReleaseInfo,
    manifest: BundleManifest,
) -> bool:
    """Return whether a bundle manifest belongs to the selected release."""
    return _is_semver(manifest.version) and _base_version(
        manifest.version
    ) == _base_version(release.version)


def _check_for_compiled_update() -> Optional[UpdateInfo]:
    """Check whether a newer SemVer release bundle is available."""
    local_manifest = _load_local_bundle_manifest()
    if local_manifest is None:
        return None

    try:
        release = _latest_release()
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    if (
        release is None
        or not release.asset_url
        or not _is_newer_version_tag(release.version, local_manifest.version)
    ):
        return None

    remote_manifest = _read_remote_bundle_manifest(release)
    if remote_manifest is None or not _release_manifest_matches(
        release, remote_manifest
    ):
        return None
    if not _is_newer_version_tag(remote_manifest.version, local_manifest.version):
        return None

    bundle_dir = _bundle_dir()
    if _compiled_bundle_matches_manifest(bundle_dir, remote_manifest):
        return None

    latest_revision = remote_manifest.revision or release.revision
    return UpdateInfo(
        local_version=local_manifest.version,
        local_revision=_short_revision(local_manifest.revision),
        local_tag="",
        latest_version=release.version,
        latest_revision=_short_revision(latest_revision),
        latest_tag=release.tag,
    )


def _get_latest_release() -> Optional[ReleaseInfo]:
    """Return the newest SemVer release, suppressing network and payload errors."""
    try:
        return _latest_release()
    except (
        OSError,
        ValueError,
    ):
        return None


def check_for_update() -> Optional[UpdateInfo]:
    """Check for a newer published SemVer release with a platform ZIP.

    Returns:
        Optional[UpdateInfo]: Metadata describing the available update, or ``None`` when no safe update path is
        currently available.
    """
    if FORCE_DISABLE_UPDATES:
        return None

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
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ):
        return None

    release = _get_latest_release()
    local_version = _base_version(__version__)
    if (
        release is None
        or not release.asset_url
        or not _is_newer_version_tag(release.version, local_version)
    ):
        return None

    return UpdateInfo(
        local_version=local_version,
        local_revision=_short_revision(local_revision),
        local_tag=local_tag,
        latest_version=release.version,
        latest_revision=_short_revision(release.revision),
        latest_tag=release.tag,
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
    """Download and replace the current-platform ZIP from a SemVer release."""
    bundle_dir = (install_dir or _bundle_dir()).resolve()
    artifact = _platform_artifact_name()
    release = _get_latest_release()
    if release is None or not release.asset_url:
        raise UpdateError(string("cli.update_no_platform_zip"))

    with tempfile.TemporaryDirectory(prefix="celune-update-") as temp_dir:
        temp_root = Path(temp_dir)
        zip_path = temp_root / f"{artifact}.zip"
        try:
            _download_release_zip(release, zip_path)
        except OSError as exc:
            raise UpdateError(
                string("cli.update_download_failed", error=str(exc))
            ) from exc

        release_manifest = _manifest_from_zip(zip_path)
        if release_manifest is None or not _release_manifest_matches(
            release, release_manifest
        ):
            raise UpdateError(string("cli.update_invalid_release"))

        try:
            extracted_root = _extract_artifact_root(zip_path, temp_root / "artifact")
        except (OSError, zipfile.BadZipFile, UpdateError) as exc:
            raise UpdateError(
                string("cli.update_unpack_failed", error=str(exc))
            ) from exc

        for source in extracted_root.iterdir():
            _replace_path(source, bundle_dir / source.name)


def update_to_latest(install_dir: Optional[Path] = None) -> None:
    """Update Celune from a published SemVer release.

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

    release = _get_latest_release()
    if (
        release is None
        or not release.asset_url
        or not _is_newer_version_tag(release.version, _base_version(__version__))
    ):
        raise UpdateError(string("cli.update_no_newer_release"))

    fetch_ref = f"refs/tags/{release.tag}"

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
