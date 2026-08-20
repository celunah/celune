#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write update metadata for a compiled Celune bundle."""

from __future__ import annotations

import json
import hashlib
import argparse
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: File whose contents should be hashed.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output_dir: Path,
    version: str,
    revision: str,
    artifact: str,
    filenames: list[str],
) -> None:
    """Write formatted update metadata for compiled bundle files.

    Args:
        output_dir: Directory containing the compiled bundle files.
        version: Version represented by the compiled bundle.
        revision: Git revision used to build the bundle.
        artifact: Platform-specific artifact identifier.
        filenames: Bundle filenames to include when present.
    """
    files = {
        filename: sha256_file(output_dir / filename)
        for filename in filenames
        if (output_dir / filename).is_file()
    }
    manifest = {
        "version": version,
        "revision": revision,
        "artifact": artifact,
        "files": files,
    }
    (output_dir / "celune-update.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse compiled bundle metadata arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--file", dest="filenames", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    """Write the compiled bundle update manifest."""
    arguments = parse_args()
    write_manifest(
        output_dir=arguments.output_dir,
        version=arguments.version,
        revision=arguments.revision,
        artifact=arguments.artifact,
        filenames=arguments.filenames,
    )


if __name__ == "__main__":
    main()
