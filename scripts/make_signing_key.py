#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare release signing keys and configure the signed release build."""

from __future__ import annotations

import os
import re
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


def default_root_key_path() -> Path:
    """Return the platform's default protected keyring-root destination."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        return Path(local_app_data) / "Celune" / "keys" / "release-keyring-root.pem"
    return Path.home() / ".celune" / "release-keyring-root.pem"


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


def _private_key_for_path(path: Path) -> Ed25519PrivateKey:
    """Load an unencrypted Ed25519 private key from a PEM file."""
    private_key = serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError(f"the private key is not Ed25519: {path}")
    return private_key


def _public_key_bytes(private_key: Ed25519PrivateKey) -> bytes:
    """Return the raw Ed25519 public key matching a private key."""
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _load_root_pin_sources(
    repository: Path,
) -> tuple[Path, Path, str, str, bytes]:
    """Read and compare the Python and launcher root-key trust anchors."""
    python_path = repository / "celune" / "constants.py"
    header_path = repository / "launcher" / "release_keys.h"
    python_source = python_path.read_text(encoding="utf-8")
    header_source = header_path.read_text(encoding="utf-8")
    python_match = re.search(
        r"(?m)^RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX = \(\r?\n"
        r'    "([0-9a-fA-F]{64})"\r?\n\)',
        python_source,
    )
    header_match = re.search(
        r"(?ms)^static const unsigned char launcher_release_keyring_root\[32\] = \{\n"
        r"(.*?)\n\};",
        header_source,
    )
    if python_match is None or header_match is None:
        raise ValueError("the release keyring root pin could not be read")

    python_public_key = bytes.fromhex(python_match.group(1))
    header_values = re.findall(r"0x([0-9a-fA-F]{2})", header_match.group(1))
    if len(header_values) != 32:
        raise ValueError("the launcher release keyring root pin is invalid")
    header_public_key = bytes.fromhex("".join(header_values))
    if python_public_key != header_public_key:
        raise ValueError("the Python and launcher release root pins differ")
    return (
        python_path,
        header_path,
        python_source,
        header_source,
        python_public_key,
    )


def _render_root_pin_sources(
    python_source: str,
    header_source: str,
    public_key: bytes,
) -> tuple[str, str]:
    """Update both verifier source files to trust the same root public key."""
    public_key_hex = public_key.hex()
    python_source, python_count = re.subn(
        r"(?m)^RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX = \(\r?\n"
        r'    "[0-9a-fA-F]{64}"\r?\n\)',
        f'RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX = (\n    "{public_key_hex}"\n)',
        python_source,
        count=1,
    )
    if python_count != 1:
        raise ValueError("the Python release root pin could not be updated")

    formatted_bytes = (
        ",\n".join(
            "    "
            + ", ".join(f"0x{value:02x}" for value in public_key[offset : offset + 8])
            for offset in range(0, len(public_key), 8)
        )
        + ","
    )
    header_source, header_count = re.subn(
        r"(?ms)^static const unsigned char launcher_release_keyring_root\[32\] = \{\n"
        r".*?\n\};",
        "static const unsigned char launcher_release_keyring_root[32] = {\n"
        + formatted_bytes
        + "\n};",
        header_source,
        count=1,
    )
    if header_count != 1:
        raise ValueError("the launcher release root pin could not be updated")
    return python_source, header_source


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


def _validated_destination(path: Path, repository: Path, role: str) -> Path:
    """Resolve a private-key destination and keep it outside the checkout."""
    requested_destination = path.expanduser().absolute()
    if requested_destination.is_symlink():
        raise ValueError(f"the {role} destination cannot be a symbolic link")
    destination = requested_destination.resolve(strict=False)
    try:
        destination.relative_to(repository.resolve())
    except ValueError:
        return destination
    raise ValueError(f"the {role} must be stored outside the repository")


def make_signing_key(
    output_path: Path,
    project_root: Optional[Path] = None,
    root_key_path: Optional[Path] = None,
    trust_existing_root: bool = False,
) -> str:
    """Prepare protected release and root keys and synchronize their trust data.

    Args:
        output_path: Release private-key path; an existing Ed25519 key is reused.
        project_root: Checkout whose release catalog and verifier pins are updated.
        root_key_path: Root private-key path; a missing root is generated.
        trust_existing_root: Update verifier pins to trust an existing root key.

    Returns:
        str: Stable identifier for the release public key.

    Raises:
        OSError: if a key or trust registry cannot be read or written.
        TypeError: if an existing private key is not Ed25519.
        ValueError: if key paths, trust pins, or the key catalog are invalid.
    """
    repository = project_root or _repository_root()
    key_catalog = repository / "celune" / "release_keyring.json"
    release_destination = _validated_destination(
        output_path,
        repository,
        "release signing key",
    )
    root_destination = _validated_destination(
        root_key_path or default_root_key_path(),
        repository,
        "keyring root key",
    )
    if release_destination == root_destination:
        raise ValueError(
            "the release signing key and keyring root must be separate files"
        )
    if trust_existing_root and not root_destination.exists():
        raise FileNotFoundError(root_destination)

    (
        python_pin_path,
        header_pin_path,
        python_pin_source,
        header_pin_source,
        pinned_root_public_key,
    ) = _load_root_pin_sources(repository)
    root_is_new = not root_destination.exists()
    if root_is_new:
        root_private_key = Ed25519PrivateKey.generate()
    else:
        root_private_key = _private_key_for_path(root_destination)
    root_public_key = _public_key_bytes(root_private_key)
    if root_public_key != pinned_root_public_key and not (
        root_is_new or trust_existing_root
    ):
        raise ValueError(
            "the existing root key differs from the pinned verifier key; "
            "use --trust-existing-root to update the trust anchor"
        )
    release_is_new = not release_destination.exists()
    if release_is_new:
        release_private_key = Ed25519PrivateKey.generate()
    else:
        release_private_key = _private_key_for_path(release_destination)
    release_public_key = _public_key_bytes(release_private_key)
    if root_public_key == release_public_key:
        raise ValueError(
            "the release signing key and keyring root must be different keys"
        )

    trusted_keys = _load_trusted_keys(key_catalog)
    root_key_id = _key_id(root_public_key)
    release_key_id = _key_id(release_public_key)
    trusted_keys[root_key_id] = root_public_key.hex()
    trusted_keys[release_key_id] = release_public_key.hex()
    if root_key_id == release_key_id and root_public_key != release_public_key:
        raise ValueError("the release catalog key identifier is ambiguous")
    if len(trusted_keys) > 128:
        raise ValueError("the trusted release key catalog contains too many keys")

    python_pin_source, header_pin_source = _render_root_pin_sources(
        python_pin_source,
        header_pin_source,
        root_public_key,
    )
    original_sources = {
        python_pin_path: python_pin_path.read_bytes(),
        header_pin_path: header_pin_path.read_bytes(),
        key_catalog: key_catalog.read_bytes(),
    }
    created_keys: list[Path] = []
    try:
        if root_is_new:
            root_private_pem = root_private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            _write_private_key(root_destination, root_private_pem)
            created_keys.append(root_destination)
        if release_is_new:
            release_private_pem = release_private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            _write_private_key(release_destination, release_private_pem)
            created_keys.append(release_destination)

        _write_source_atomically(python_pin_path, python_pin_source)
        _write_source_atomically(header_pin_path, header_pin_source)
        _write_source_atomically(key_catalog, _render_key_catalog(trusted_keys))
    except Exception:
        for path, contents in original_sources.items():
            _write_bytes_atomically(path, contents)
        for path in created_keys:
            path.unlink(missing_ok=True)
        raise
    return release_key_id


def configure_signed_build(
    project_root: Path,
    signing_key_path: Path,
    root_key_path: Path,
) -> None:
    """Configure the signed CMake preset with the prepared private keys."""
    subprocess.run(
        [
            "cmake",
            "--preset",
            "signed",
            f"-DCELUNE_RELEASE_SIGNING_KEY_FILE={signing_key_path.resolve()}",
            f"-DCELUNE_RELEASE_ROOT_KEY_FILE={root_key_path.resolve()}",
        ],
        cwd=project_root,
        check=True,
    )


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
    parser.add_argument(
        "--root-key",
        type=Path,
        default=default_root_key_path(),
        help=_localized_string("release.signing_key_root_help"),
    )
    parser.add_argument(
        "--trust-existing-root",
        action="store_true",
        help=_localized_string("release.signing_key_trust_root_help"),
    )
    parser.add_argument(
        "--no-configure",
        action="store_true",
        help=_localized_string("release.signing_key_no_configure_help"),
    )
    return parser.parse_args()


def main() -> int:
    """Prepare release keys, update trust data, and configure the signed preset."""
    arguments = parse_args()
    repository = _repository_root()
    signing_path = arguments.output.expanduser().absolute()
    root_path = arguments.root_key.expanduser().absolute()
    try:
        identifier = make_signing_key(
            arguments.output,
            repository,
            arguments.root_key,
            arguments.trust_existing_root,
        )
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(
            _localized_string("release.signing_key_failed"),
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 1

    signing_path = signing_path.resolve(strict=False)
    root_path = root_path.resolve(strict=False)
    print(
        _localized_string(
            "release.signing_key_ready",
            key_id=identifier,
            path=str(signing_path),
        )
    )
    print(
        _localized_string(
            "release.root_key_ready",
            path=str(root_path),
        )
    )
    if arguments.no_configure:
        return 0

    try:
        configure_signed_build(repository, signing_path, root_path)
    except (OSError, subprocess.SubprocessError) as error:
        print(
            _localized_string("release.signed_preset_failed"),
            file=sys.stderr,
        )
        print(str(error), file=sys.stderr)
        return 1

    print(_localized_string("release.signed_preset_configured"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
