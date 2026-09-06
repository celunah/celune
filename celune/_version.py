# SPDX-License-Identifier: Apache-2.0
"""Celune build version metadata."""

import subprocess


def _get_revision() -> str:
    """Return the current Git revision, including a dirty-worktree marker."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{revision}{'*' if status else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


REVISION = _get_revision()
VERSION = "5.0.1"
DEVELOPMENT = False

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"{VERSION}+{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"
