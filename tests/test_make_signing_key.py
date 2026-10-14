# SPDX-License-Identifier: Apache-2.0
"""Tests for release signing-key creation."""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization

from scripts.make_signing_key import _default_output_path, make_signing_key


def _empty_project(root: Path) -> None:
    """Create an empty release-key catalog for an isolated checkout fixture."""
    (root / "celune").mkdir(parents=True)
    (root / "celune" / "release_keyring.json").write_text("{}\n", encoding="utf-8")


def test_generated_key_is_private_and_registered_in_the_key_catalog(
    tmp_path: Path,
) -> None:
    """Create one protected private key and register its matching public key."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    key_path = tmp_path / "home" / ".celune" / "release-signing-key.pem"

    key_id = make_signing_key(key_path, project_root)

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

    assert key_id == expected_key_id
    assert catalog[key_id] == public_key.hex()
    if os.name != "nt":
        assert key_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name != "nt", reason="The Windows default uses LOCALAPPDATA.")
def test_default_key_path_uses_the_existing_celune_key_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Select the Celune LocalAppData key path on Windows."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert _default_output_path() == (
        tmp_path / "Celune" / "keys" / "release-ed25519-private.pem"
    )


def test_key_creation_never_overwrites_an_existing_private_key(
    tmp_path: Path,
) -> None:
    """Preserve an existing destination when key creation is repeated."""
    project_root = tmp_path / "project"
    _empty_project(project_root)
    key_path = tmp_path / "home" / ".celune" / "release-signing-key.pem"
    key_path.parent.mkdir(parents=True)
    key_path.write_bytes(b"keep this key")
    key_catalog = project_root / "celune" / "release_keyring.json"
    original_catalog = key_catalog.read_bytes()

    with pytest.raises(FileExistsError):
        make_signing_key(key_path, project_root)

    assert key_path.read_bytes() == b"keep this key"
    assert key_catalog.read_bytes() == original_catalog


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
