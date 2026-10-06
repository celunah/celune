# SPDX-License-Identifier: Apache-2.0
"""Celune build version metadata."""

from .vcs import get_revision as _get_revision


REVISION = _get_revision()
VERSION = "5.1.0"
DEVELOPMENT = True

if REVISION:
    _local = REVISION.rstrip("*")
    _dirty = ".dirty" if REVISION.endswith("*") else ""
    __version__ = f"{VERSION}+{_local}{_dirty}"
else:
    __version__ = f"{VERSION}+unknown"
