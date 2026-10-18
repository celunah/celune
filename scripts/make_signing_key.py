#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a private release key and add its public key to the signed catalog."""

from __future__ import annotations

import os
import sys
import json
import stat
import hashlib
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, cast

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _repository_root() -> Path:
    """Return the checkout containing the signing-key script."""
    return Path(__file__).resolve().parent.parent


def default_output_path() -> Path:
    """Return the platform's default protected release-key destination."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "Celune" / "keys" / "release-ed25519-private.pem"
    return Path.home() / ".celune" / "release-signing-key.pem"


def _localized_string(key: str, **values: str) -> str:
    """Resolve a signing-tool message from Celune's English localization file."""
    translations = cast(
        dict[str, str],
        json.loads(
            (_repository_root() / "celune" / "lang" / "en.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    return translations[key].format(**values)


def _key_id(public_key: bytes) -> str:
    """Return the stable identifier used by the manifest signature envelope."""
    return f"ed25519-{hashlib.sha256(public_key).hexdigest()[:16]}"


def _load_trusted_keys(path: Path) -> dict[str, str]:
    """Read and validate the JSON release-key catalog."""
    keys = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(keys, dict):
        raise TypeError("the trusted release key catalog must be a dictionary")
    if len(keys) > 128:
        raise ValueError("the trusted release key catalog contains too many keys")
    validated: dict[str, str] = {}
    for key_id, public_key_hex in keys.items():
        if not isinstance(key_id, str) or not isinstance(public_key_hex, str):
            raise TypeError("the trusted release key registry has invalid entries")
        public_key = bytes.fromhex(public_key_hex)
        if len(public_key) != 32 or _key_id(public_key) != key_id:
            raise ValueError(f"the trusted release key entry is invalid: {key_id}")
        validated[key_id] = public_key_hex.lower()
    return validated


def _render_key_catalog(keys: dict[str, str]) -> str:
    """Render the canonical release-key catalog."""
    return json.dumps(dict(sorted(keys.items())), indent=2) + "\n"


def _write_bytes_atomically(path: Path, contents: bytes) -> None:
    """Replace a file atomically while preserving its mode."""
    mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(contents)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_source_atomically(path: Path, contents: str) -> None:
    """Replace a tracked source file atomically while preserving its mode."""
    _write_bytes_atomically(path, contents.encode("utf-8"))


def _secure_windows_key_file(path: Path) -> None:
    """Restrict a generated Windows key file to the current account."""
    account = subprocess.run(
        ["whoami"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"{account}:(F)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_private_key(path: Path, private_key_pem: bytes) -> None:
    """Create a private-key file without overwriting an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise ValueError("the signing key directory cannot be a symbolic link")
    if os.name != "nt":
        directory_info = path.parent.stat()
        if directory_info.st_uid != os.getuid() or directory_info.st_mode & 0o022:
            raise ValueError(
                "the signing key directory must be owned by this user and not group-writable"
            )

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.name == "nt":
            os.close(descriptor)
            _secure_windows_key_file(path)
            with path.open("wb") as file:
                file.write(private_key_pem)
                file.flush()
                os.fsync(file.fileno())
        else:
            with os.fdopen(descriptor, "wb") as file:
                file.write(private_key_pem)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(path, 0o600)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def make_signing_key(
    output_path: Path,
    project_root: Optional[Path] = None,
) -> str:
    """Create a protected Ed25519 key and add its public key to the catalog.

    Args:
        output_path: Destination for the private PEM file; existing files are never replaced.
        project_root: Checkout whose Python and native trust registries are updated.

    Returns:
        str: Stable identifier for the new public key.

    Raises:
        FileExistsError: if the private-key destination already exists.
        OSError: if a key or trust registry cannot be written.
        ValueError: if the catalog is invalid or the key directory is unsafe.
    """
    repository = project_root or _repository_root()
    key_catalog = repository / "celune" / "release_keyring.json"
    requested_destination = output_path.expanduser().absolute()
    if requested_destination.is_symlink():
        raise ValueError("the signing key destination cannot be a symbolic link")
    destination = requested_destination.resolve(strict=False)
    try:
        destination.relative_to(repository.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("the signing key must be stored outside the repository")
    if destination.exists():
        raise FileExistsError(destination)

    trusted_keys = _load_trusted_keys(key_catalog)
    original_key_catalog = key_catalog.read_bytes()

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    identifier = _key_id(public_key)
    if identifier in trusted_keys:
        raise ValueError("the generated key identifier is already trusted")

    trusted_keys[identifier] = public_key.hex()
    private_key_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    _write_private_key(destination, private_key_pem)
    try:
        _write_source_atomically(key_catalog, _render_key_catalog(trusted_keys))
    except Exception:
        _write_bytes_atomically(key_catalog, original_key_catalog)
        destination.unlink(missing_ok=True)
        raise
    return identifier


def parse_args() -> argparse.Namespace:
    """Parse signing-key creation options."""
    parser = argparse.ArgumentParser(
        description=_localized_string("release.signing_key_description"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help=_localized_string("release.signing_key_output_help"),
    )
    return parser.parse_args()


def main() -> int:
    """Create a signing key, register its public key, and report its location."""
    arguments = parse_args()
    try:
        identifier = make_signing_key(arguments.output)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(
            _localized_string("release.signing_key_failed"),
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 1

    print(
        _localized_string(
            "release.signing_key_created",
            key_id=identifier,
            path=str(arguments.output.expanduser()),
        )
    )
    print(_localized_string("release.signing_key_registered"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
