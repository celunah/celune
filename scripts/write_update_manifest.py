#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write update metadata for a compiled Celune bundle."""

from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional

from celune.signature import sign_manifest


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
    signing_key: Optional[Path] = None,
    require_signature: bool = False,
) -> None:
    """Write formatted update metadata for compiled bundle files.

    Args:
        output_dir: Directory containing the compiled bundle files.
        version: Version represented by the compiled bundle.
        revision: Git revision used to build the bundle.
        artifact: Platform-specific artifact identifier.
        filenames: Bundle filenames to include when present.
        signing_key: Optional trusted Ed25519 private-key file.
        require_signature: Whether to fail when no key is supplied.

    Raises:
        ValueError: if a signature is required but no signing key is supplied
    """
    files = {
        filename: sha256_file(output_dir / filename)
        for filename in filenames
        if (output_dir / filename).is_file()
    }
    if require_signature and signing_key is None:
        raise ValueError("a release signing key is required for this build")

    manifest = {
        "version": version,
        "revision": revision,
        "artifact": artifact,
        "files": files,
        "signature_required": signing_key is not None,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest_path = output_dir / "celune-update.json"
    manifest_path.write_bytes(manifest_bytes)

    signature_path = output_dir / "celune-update.sig"
    if signing_key is None:
        signature_path.unlink(missing_ok=True)
    else:
        signature_path.write_bytes(sign_manifest(manifest_bytes, signing_key))


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
    parser.add_argument("--signing-key", type=Path)
    parser.add_argument("--require-signature", action="store_true")
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
        signing_key=arguments.signing_key,
        require_signature=arguments.require_signature,
    )


if __name__ == "__main__":
    main()
