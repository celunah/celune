# SPDX-License-Identifier: Apache-2.0
"""Celune build version metadata."""

from pathlib import Path

from .vcs import get_revision as _get_revision

REVISION = _get_revision(Path(__file__).resolve().parent.parent)
VERSION = "5.1.0"
EXTRA_IDENTIFIER = "realtime"

# TODO: force ``DEVELOPMENT = False`` if this Celune build was already released on GitHub
# read the local repository's annotated tags to determine whether the build matches a known published release
DEVELOPMENT = True

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    _identifier = f".{EXTRA_IDENTIFIER}" if EXTRA_IDENTIFIER else ""
    __version__ = f"{VERSION}+{_identifier}{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"
