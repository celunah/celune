# SPDX-License-Identifier: MIT
"""Filesystem helpers for Persona character overrides and debug data."""

from __future__ import annotations

import re
from pathlib import Path

from ..cevoice import SUPPORTED_PERSONA_FILENAMES
from ..paths import persona_data_dir


def persona_character_slug(character_name: str) -> str:
    """Return a filesystem-safe identifier for one Persona character name.

    Args:
        character_name: The character name to normalize for a directory name.

    Returns:
        str: A lowercase hyphen-separated character identifier.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", character_name.casefold()).strip("-")
    return cleaned or "unknown"


def persona_character_dir(character_name: str, create: bool = False) -> Path:
    """Return the app-data directory for one Persona character.

    Args:
        character_name: The active character name.
        create: Whether the character directory should be created.

    Returns:
        Path: The character-specific Persona data directory.
    """
    path = persona_data_dir(create=create) / persona_character_slug(character_name)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def persona_memory_dir(character_name: str, create: bool = False) -> Path:
    """Return the debug memory directory for one Persona character.

    Args:
        character_name: The active character name.
        create: Whether the memory directory should be created.

    Returns:
        Path: The character-specific Persona memory directory.
    """
    path = persona_character_dir(character_name, create=create) / "memory"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def persona_override_files(character_name: str) -> dict[str, str]:
    """Read supported Markdown overrides for one Persona character.

    Args:
        character_name: The active character name whose overrides should be read.

    Returns:
        dict[str, str]: Non-empty supported Markdown files keyed by filename.
    """
    directory = persona_character_dir(character_name)
    files: dict[str, str] = {}
    for filename in SUPPORTED_PERSONA_FILENAMES:
        path = directory / filename
        try:
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if content:
            files[filename] = content
    return files
