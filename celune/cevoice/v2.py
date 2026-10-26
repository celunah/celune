# SPDX-License-Identifier: Apache-2.0
"""Compatibility rules for legacy CECHAR v2 bundles."""

from __future__ import annotations

from typing import Final

from ..typing.cevoice import ManifestValue

MAGIC: Final[bytes] = b"CECHAR\0\0"
VERSION: Final[int] = 2
FORMAT_NAME: Final[str] = "CECHAR"


def supports_header(magic: bytes, version: int) -> bool:
    """Return whether a legacy header identifies CECHAR v2."""
    return magic == MAGIC and version == VERSION


def supports_manifest(format_name: ManifestValue, version: ManifestValue) -> bool:
    """Return whether a manifest declares CECHAR v2."""
    return format_name == FORMAT_NAME and version == VERSION
