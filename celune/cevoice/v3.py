# SPDX-License-Identifier: Apache-2.0
"""CECHAR v3 writer and compatibility rules."""

from __future__ import annotations

from typing import Final, Union, Optional
from pathlib import Path
from collections.abc import Mapping

from .base import _write_legacy_cevoice
from ..typing.cevoice import ManifestValue

MAGIC: Final[bytes] = b"CECHAR\0\0"
VERSION: Final[int] = 3
FORMAT_NAME: Final[str] = "CECHAR"


def supports_header(magic: bytes, version: int) -> bool:
    """Return whether a legacy header identifies CECHAR v3."""
    return magic == MAGIC and version == VERSION


def supports_manifest(format_name: ManifestValue, version: ManifestValue) -> bool:
    """Return whether a manifest declares CECHAR v3."""
    return format_name == FORMAT_NAME and version == VERSION


def write_cevoice(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]] = None,
) -> Path:
    """Write a CECHAR v3 package from per-voice binary assets.

    Args:
        path: The CECHAR v3 package to save as.
        voices: The voice files to bundle into the package.
        metadata: The metadata to bundle into the package.
        voice_metadata: Extra metadata stored beside each voice's assets.
        bundle_assets: Extra top-level Persona Markdown assets.

    Returns:
        Path: The path to the created CECHAR v3 package.

    Raises:
        CEVoiceError: The package data contains unsupported names or assets.
    """
    return _write_legacy_cevoice(
        path,
        voices,
        metadata,
        voice_metadata,
        bundle_assets,
        MAGIC,
        VERSION,
        None,
    )
