# SPDX-License-Identifier: Apache-2.0
"""Tests for release signing-key creation."""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from celune import signature
from scripts.make_signing_key import (
    configure_signed_build,
    default_output_path,
    default_root_key_path,
    make_signing_key,
)
from scripts import make_signing_key as signing_tool


def _empty_project(root: Path) -> None:
    """Create an isolated checkout with matching root trust anchors."""
    (root / "celune").mkdir(parents=True)
    (root / "launcher").mkdir()
    root_public_key = b"\x01" * 32
    key_id = f"ed25519-{hashlib.sha256(root_public_key).hexdigest()[:16]}"
    (root / "celune" / "release_keyring.json").write_text(
        json.dumps({key_id: root_public_key.hex()}, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "celune" / "constants.py").write_text(
        f'RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX = (\n    "{root_public_key.hex()}"\n)\n',
        encoding="utf-8",
    )
    formatted_root = ",\n".join(
        "    "
        + ", ".join(f"0x{value:02x}" for value in root_public_key[offset : offset + 8])
        for offset in range(0, len(root_public_key), 8)
    )
    (root / "launcher" / "release_keys.h").write_text(
        "static const unsigned char launcher_release_keyring_root[32] = {\n"
        f"{formatted_root}\n"
        "};\n",
        encoding="utf-8",
    )


def test_generated_key_is_private_and_registered_in_the_key_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create signing keys and verify the resulting signed catalog."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    key_path = tmp_path / "home" / ".celune" / "release-signing-key.pem"
    root_path = tmp_path / "home" / ".celune" / "release-keyring-root.pem"

    key_id = make_signing_key(key_path, project_root, root_path)

    private_key = serialization.load_pem_private_key(
        key_path.read_bytes(),
        password=None,
    )
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    expected_key_id = f"ed25519-{hashlib.sha256(public_key).hexdigest()[:16]}"
    catalog = json.loads(
        (project_root / "celune" / "release_keyring.json").read_text(encoding="utf-8")
    )
    root_private_key = serialization.load_pem_private_key(
        root_path.read_bytes(),
        password=None,
    )
    root_public_key = root_private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )

    assert key_id == expected_key_id
    assert catalog[key_id] == public_key.hex()
    assert (
        catalog[f"ed25519-{hashlib.sha256(root_public_key).hexdigest()[:16]}"]
        == root_public_key.hex()
    )
    assert root_public_key.hex() in (
        project_root / "celune" / "constants.py"
    ).read_text(encoding="utf-8")
    assert (
        bytes.fromhex(
            "".join(
                re.findall(
                    r"0x([0-9a-fA-F]{2})",
                    (project_root / "launcher" / "release_keys.h").read_text(
                        encoding="utf-8"
                    ),
                )
            )
        )
        == root_public_key
    )
    monkeypatch.setattr(signature, "_ROOT_PUBLIC_KEY", root_public_key)
    monkeypatch.setattr(
        signature,
        "_CATALOG_PATH",
        project_root / "celune" / "release_keyring.json",
    )
    verified_keyring = signature.decode_signed_keyring(
        signature.create_signed_keyring(root_path)
    )
    assert verified_keyring[key_id] == public_key
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600
        assert root_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "nt", reason="The Windows default uses LOCALAPPDATA.")
def test_default_key_path_uses_the_existing_celune_key_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Select the Celune LocalAppData key path on Windows."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert default_output_path() == (
        tmp_path / "Celune" / "keys" / "release-ed25519-private.pem"
    )
    assert default_root_key_path() == (
        tmp_path / "Celune" / "keys" / "release-keyring-root.pem"
    )


def test_release_signing_and_keyring_root_defaults_are_separate() -> None:
    """Keep the generated manifest key separate from the pinned keyring root."""
    assert default_root_key_path() != default_output_path()


def test_existing_release_key_is_reused_and_registered(
    tmp_path: Path,
) -> None:
    """Reuse a valid existing release key without changing its private bytes."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    key_path = tmp_path / "home" / ".celune" / "release-signing-key.pem"
    root_path = tmp_path / "home" / ".celune" / "release-keyring-root.pem"
    private_key = Ed25519PrivateKey.generate()
    private_key_bytes = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    key_path.parent.mkdir(parents=True)
    key_path.write_bytes(private_key_bytes)

    key_id = make_signing_key(key_path, project_root, root_path)

    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    catalog = json.loads(
        (project_root / "celune" / "release_keyring.json").read_text(encoding="utf-8")
    )
    assert key_path.read_bytes() == private_key_bytes
    assert key_id == f"ed25519-{hashlib.sha256(public_key).hexdigest()[:16]}"
    assert catalog[key_id] == public_key.hex()


def test_existing_root_key_requires_explicit_trust_update(
    tmp_path: Path,
) -> None:
    """Do not silently replace a checkout's pinned root with an unrelated key."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    release_path = tmp_path / "release.pem"
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    original_catalog = (project_root / "celune" / "release_keyring.json").read_bytes()
    original_python_pin = (project_root / "celune" / "constants.py").read_bytes()
    original_launcher_pin = (project_root / "launcher" / "release_keys.h").read_bytes()

    with pytest.raises(ValueError, match="trust anchor"):
        make_signing_key(release_path, project_root, root_path)

    assert not release_path.exists()
    assert (
        project_root / "celune" / "release_keyring.json"
    ).read_bytes() == original_catalog
    assert (
        project_root / "celune" / "constants.py"
    ).read_bytes() == original_python_pin
    assert (
        project_root / "launcher" / "release_keys.h"
    ).read_bytes() == original_launcher_pin


def test_existing_root_key_can_be_trusted(
    tmp_path: Path,
) -> None:
    """Adopt an existing root key and preserve earlier catalog entries."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    original_catalog = json.loads(
        (project_root / "celune" / "release_keyring.json").read_text(encoding="utf-8")
    )
    release_path = tmp_path / "release.pem"
    root_path = tmp_path / "root.pem"
    root_private_key = Ed25519PrivateKey.generate()
    root_path.write_bytes(
        root_private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    make_signing_key(
        release_path,
        project_root,
        root_path,
        trust_existing_root=True,
    )

    root_public_key = root_private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    assert root_public_key.hex() in (
        project_root / "celune" / "constants.py"
    ).read_text(encoding="utf-8")
    catalog = json.loads(
        (project_root / "celune" / "release_keyring.json").read_text(encoding="utf-8")
    )
    root_key_id = f"ed25519-{hashlib.sha256(root_public_key).hexdigest()[:16]}"
    assert catalog[root_key_id] == root_public_key.hex()
    assert all(
        catalog[key_id] == public_key for key_id, public_key in original_catalog.items()
    )


def test_signed_build_configuration_receives_both_key_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pass both prepared key paths to the signed CMake configure command."""
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(
        signing_tool.subprocess,
        "run",
        lambda command, cwd, check: calls.append((command, cwd, check)),
    )
    release_path = tmp_path / "release.pem"
    root_path = tmp_path / "root.pem"

    configure_signed_build(tmp_path, release_path, root_path)

    assert calls == [
        (
            [
                "cmake",
                "--preset",
                "signed",
                f"-DCELUNE_RELEASE_SIGNING_KEY_FILE={release_path.resolve()}",
                f"-DCELUNE_RELEASE_ROOT_KEY_FILE={root_path.resolve()}",
            ],
            tmp_path,
            True,
        )
    ]


def test_key_creation_rejects_destinations_inside_the_checkout(
    tmp_path: Path,
) -> None:
    """Keep generated private keys out of the project checkout."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    key_catalog = project_root / "celune" / "release_keyring.json"
    original_catalog = key_catalog.read_bytes()

    with pytest.raises(ValueError, match="outside the repository"):
        make_signing_key(project_root / "release-signing-key.pem", project_root)

    assert not (project_root / "release-signing-key.pem").exists()
    assert key_catalog.read_bytes() == original_catalog


@pytest.mark.skipif(
    os.name == "nt", reason="Creating Windows symlinks may require elevation."
)
def test_key_creation_rejects_a_symlink_destination(tmp_path: Path) -> None:
    """Do not follow a private-key destination symlink."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    destination = tmp_path / "release-signing-key.pem"
    external_target = tmp_path / "external-key.pem"
    destination.symlink_to(external_target)

    with pytest.raises(ValueError, match="symbolic link"):
        make_signing_key(destination, project_root)

    assert not external_target.exists()
