# SPDX-License-Identifier: Apache-2.0
"""Compatibility rules for the original CEVOICE v1 format."""

from __future__ import annotations

from typing import Final

from ..typing.cevoice import ManifestValue

MAGIC: Final[bytes] = b"CEVOICE\0"
VERSION: Final[int] = 1
FORMAT_NAME: Final[str] = "CEVOICE"


def supports_header(magic: bytes, version: int) -> bool:
    """Return whether a legacy header identifies CEVOICE v1."""
    return magic == MAGIC and version == VERSION


def supports_manifest(format_name: ManifestValue, version: ManifestValue) -> bool:
    """Return whether a manifest declares CEVOICE v1."""
    return format_name == FORMAT_NAME and version == VERSION
