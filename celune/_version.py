# SPDX-License-Identifier: Apache-2.0
"""Celune build version metadata."""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional

from .vcs import (
    get_revision as _get_revision,
    open_repository as _open_repository,
)

REVISION = _get_revision(Path(__file__).resolve().parent.parent)
VERSION = "5.1.0"
EXTRA_IDENTIFIER: str = "realtime"


def _is_annotated_release(path: Path, version: str) -> bool:
    """Return whether a clean release commit has its matching annotated tag.

    The build process refreshes ``.celune-root`` before compilation, so that
    generated marker is the only tracked file allowed to differ from ``HEAD``.

    Args:
        path: Repository root to inspect.
        version: Project version expected in the annotated release tag.

    Returns:
        bool: Whether the current commit matches a clean annotated release tag.
    """
    try:
        from git.exc import GitError
        from git.objects.tag import TagObject
    except ModuleNotFoundError as package:
        if package.name != "git":
            raise
        return False

    try:
        with _open_repository(path) as repository:
            release_tag_names = {version, f"v{version}"}
            head_revision = repository.head.commit.hexsha
            if not any(
                tag.name in release_tag_names
                and isinstance(tag.object, TagObject)
                and tag.commit.hexsha == head_revision
                for tag in repository.tags
            ):
                return False

            changed_paths: set[str] = set()
            for change in repository.index.diff("HEAD"):
                changed_paths.update(
                    changed_path
                    for changed_path in (change.a_path, change.b_path)
                    if changed_path is not None
                )
            for change in repository.head.commit.diff(None):
                changed_paths.update(
                    changed_path
                    for changed_path in (change.a_path, change.b_path)
                    if changed_path is not None
                )
            return not repository.untracked_files and changed_paths <= {".celune-root"}
    except (GitError, OSError, TypeError, ValueError):
        return False


def _bundle_development_flag() -> Optional[bool]:
    """Read release status from the compiled bundle's update manifest.

    Returns:
        Optional[bool]: Manifest status when it matches this build, otherwise ``None``.
    """
    main_module = sys.modules.get("__main__")
    if not getattr(main_module, "__compiled__", False) or not sys.executable:
        return None

    manifest_path = Path(sys.executable).resolve().parent / "celune-update.json"
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(metadata, dict) or metadata.get("version") != VERSION:
        return None
    development = metadata.get("development")
    revision = metadata.get("revision")
    if not isinstance(development, bool) or not isinstance(revision, str):
        return None
    if REVISION and not revision.startswith(REVISION.rstrip("*")):
        return None
    return development


_bundle_development = _bundle_development_flag()
DEVELOPMENT = (
    _bundle_development
    if _bundle_development is not None
    else not _is_annotated_release(Path(__file__).resolve().parent.parent, VERSION)
)

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"{VERSION}+{EXTRA_IDENTIFIER}{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"
