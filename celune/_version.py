# SPDX-License-Identifier: Apache-2.0
"""Celune build version metadata."""

from pathlib import Path

from .vcs import get_revision as _get_revision


REVISION = _get_revision(Path(__file__).resolve().parent.parent)
VERSION = "5.1.0"
DEVELOPMENT = True

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"{VERSION}+{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"
