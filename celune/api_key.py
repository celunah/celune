# SPDX-License-Identifier: Apache-2.0
"""Cryptographically secure Celune API-key generation and storage."""

import os
import time
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from .paths import api_key_path

API_KEY_PREFIX = "C-"
API_KEY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
API_KEY_LENGTH = 25


def generate_api_key() -> str:
    """Generate a Celune API key with 25 secure base-36 characters.

    Returns:
        str: A key with the ``C-`` prefix and 25 uppercase base-36 characters.
    """
    return API_KEY_PREFIX + "".join(
        secrets.choice(API_KEY_ALPHABET) for _ in range(API_KEY_LENGTH)
    )


def _valid_generated_key(value: str) -> bool:
    """Return whether a value matches Celune's generated-key format."""
    if not value.startswith(API_KEY_PREFIX):
        return False
    suffix = value[len(API_KEY_PREFIX) :]
    return len(suffix) == API_KEY_LENGTH and all(
        character in API_KEY_ALPHABET for character in suffix
    )


def _replace_key_file(path: Path, key: str) -> None:
    """Atomically replace a malformed generated-key file."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(f"{key}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_or_create_api_key(
    configured_key: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Return a configured key or load and persist a generated key.

    Args:
        configured_key: Optional key from Celune's configuration.
        path: Optional storage path, primarily useful for isolated callers.

    Returns:
        str: The configured or persisted Celune API key.
    """
    for candidate in (configured_key, os.getenv("CELUNE_API_TOKEN")):
        if candidate is not None and candidate.strip():
            return candidate.strip()

    key_file = path or api_key_path(create_parent=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            key_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        for _attempt in range(20):
            existing = key_file.read_text(encoding="utf-8").strip()
            if _valid_generated_key(existing):
                return existing
            time.sleep(0.01)
        generated = generate_api_key()
        _replace_key_file(key_file, generated)
        return generated

    generated = generate_api_key()
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(f"{generated}\n")
        output.flush()
        os.fsync(output.fileno())
    return generated
