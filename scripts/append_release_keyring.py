#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Append Celune's root-signed release keyring to a launcher executable."""

from __future__ import annotations

import os
import argparse
from pathlib import Path

from celune.signature import append_trusted_keyring


def _default_root_key_path() -> Path:
    """Return the protected path used for Celune's pinned keyring root."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "Celune" / "keys" / "release-keyring-root.pem"
    return Path.home() / ".celune" / "release-keyring-root.pem"


def main() -> int:
    """Append a signed keyring to the specified native launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("--root-key", type=Path, default=_default_root_key_path())
    arguments = parser.parse_args()
    try:
        append_trusted_keyring(arguments.executable, arguments.root_key)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
