# SPDX-License-Identifier: Apache-2.0
"""Tests for signed compiled update manifests."""

from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from celune import signature, updater
from celune.exceptions import UpdateError


def _trusted_test_key(monkeypatch: pytest.MonkeyPatch) -> Ed25519PrivateKey:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    key_id = signature.signing_key_id(public_key)
    monkeypatch.setattr(
        signature,
        "_runtime_release_keys",
        lambda: {key_id: public_key},
    )
    return private_key


def test_appended_keyring_requires_root_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a trusted keyring trailer and reject a modified trailer signature."""
    root_key = Ed25519PrivateKey.generate()
    root_public_key = root_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    root_id = signature.signing_key_id(root_public_key)
    release_key = Ed25519PrivateKey.generate()
    release_public_key = release_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    release_id = signature.signing_key_id(release_public_key)
    monkeypatch.setattr(signature, "_ROOT_PUBLIC_KEY", root_public_key)
    monkeypatch.setattr(
        signature,
        "_load_source_release_keys",
        lambda: {root_id: root_public_key, release_id: release_public_key},
    )
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(
        root_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    executable = tmp_path / "celune"
    executable.write_bytes(b"binary payload")
    signature.append_trusted_keyring(executable, root_path)

    assert signature.read_appended_keyring(executable) == {
        root_id: root_public_key,
        release_id: release_public_key,
    }
    first_size = executable.stat().st_size
    signature.append_trusted_keyring(executable, root_path)
    assert executable.stat().st_size == first_size

    corrupted_trailer = bytearray(executable.read_bytes())
    corrupted_trailer[-signature.KEYRING_FOOTER_SIZE - 1] ^= 1
    executable.write_bytes(corrupted_trailer)
    with pytest.raises(ValueError, match="root signature"):
        signature.read_appended_keyring(executable)


def _manifest_data(files: dict[str, str]) -> bytes:
    manifest = {
        "artifact": "Celune-test-x64",
        "files": files,
        "revision": "0123456789abcdef",
        "signature_required": True,
        "version": "5.1.0",
    }
    return (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _signature_data(
    manifest_data: bytes,
    private_key: Ed25519PrivateKey,
) -> bytes:
    return signature.signature_payload(manifest_data, private_key)


def test_signature_verification_rejects_changed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a manifest whose bytes differ from the signed payload."""
    private_key = _trusted_test_key(monkeypatch)
    manifest_data = _manifest_data({"celune-bin": "0" * 64})
    signature_data = _signature_data(manifest_data, private_key)

    assert signature.verify_manifest_signature(manifest_data, signature_data)
    assert not signature.verify_manifest_signature(manifest_data + b" ", signature_data)


def test_manifest_parser_requires_valid_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept only manifests with a valid signature when required."""
    private_key = _trusted_test_key(monkeypatch)
    manifest_data = _manifest_data({"celune-bin": "0" * 64})
    signature_data = _signature_data(manifest_data, private_key)

    parsed_manifest = updater._parse_bundle_manifest_data(
        manifest_data,
        signature_data,
        require_signature=True,
    )
    assert parsed_manifest is not None
    assert parsed_manifest.signature_verified
    assert (
        updater._parse_bundle_manifest_data(
            manifest_data,
            None,
            require_signature=True,
        )
        is None
    )
    assert (
        updater._parse_bundle_manifest_data(
            manifest_data + b" ",
            signature_data,
            require_signature=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("payload_files", "manifest_files"),
    [
        (
            {"celune-bin": b"modified runtime"},
            {"celune-bin": hashlib.sha256(b"expected runtime").hexdigest()},
        ),
        (
            {"celune-bin": b"expected runtime"},
            {
                "celune-bin": hashlib.sha256(b"expected runtime").hexdigest(),
                "celune": hashlib.sha256(b"missing launcher").hexdigest(),
            },
        ),
        (
            {
                "celune-bin": b"expected runtime",
                "unlisted.bin": b"unlisted payload",
            },
            {"celune-bin": hashlib.sha256(b"expected runtime").hexdigest()},
        ),
    ],
)
def test_compiled_update_rejects_unverified_archive_without_replacing_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_files: dict[str, bytes],
    manifest_files: dict[str, str],
) -> None:
    """Reject changed, missing, or unlisted ZIP payloads before installation."""
    private_key = _trusted_test_key(monkeypatch)
    manifest_data = _manifest_data(manifest_files)
    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(updater.UPDATE_MANIFEST_NAME, manifest_data)
        archive.writestr(
            signature.SIGNATURE_NAME,
            _signature_data(manifest_data, private_key),
        )
        for filename, contents in payload_files.items():
            archive.writestr(filename, contents)

    install_dir = tmp_path / "install"
    install_dir.mkdir()
    for filename in ("celune", "celune-bin", "existing.data"):
        (install_dir / filename).write_bytes(f"installed {filename}".encode())
    before = {path.name: path.read_bytes() for path in install_dir.iterdir()}

    release = updater.ReleaseInfo(
        tag="v5.1.0",
        version="5.1.0",
        revision="0123456789abcdef",
        asset_url="https://example.invalid/release.zip",
    )
    monkeypatch.setattr(updater, "_get_latest_release", lambda: release)
    monkeypatch.setattr(
        updater,
        "_download_release_zip",
        lambda _release, destination: destination.write_bytes(
            archive_path.read_bytes()
        ),
    )

    with pytest.raises(UpdateError):
        updater._apply_compiled_update(install_dir)

    assert {path.name: path.read_bytes() for path in install_dir.iterdir()} == before


def test_compiled_update_installs_matching_signed_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a signed archive only when every payload digest matches."""
    private_key = _trusted_test_key(monkeypatch)
    payload_files = {
        "celune": b"launcher payload",
        "celune-bin": b"runtime payload",
    }
    manifest_files = {
        filename: hashlib.sha256(contents).hexdigest()
        for filename, contents in payload_files.items()
    }
    manifest_data = _manifest_data(manifest_files)
    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(updater.UPDATE_MANIFEST_NAME, manifest_data)
        archive.writestr(
            signature.SIGNATURE_NAME,
            _signature_data(manifest_data, private_key),
        )
        for filename, contents in payload_files.items():
            archive.writestr(filename, contents)

    install_dir = tmp_path / "install"
    install_dir.mkdir()
    release = updater.ReleaseInfo(
        tag="v5.1.0",
        version="5.1.0",
        revision="0123456789abcdef",
        asset_url="https://example.invalid/release.zip",
    )
    monkeypatch.setattr(updater, "_get_latest_release", lambda: release)
    monkeypatch.setattr(
        updater,
        "_download_release_zip",
        lambda _release, destination: destination.write_bytes(
            archive_path.read_bytes()
        ),
    )

    updater._apply_compiled_update(install_dir)

    assert {
        filename: (install_dir / filename).read_bytes() for filename in payload_files
    } == payload_files


def test_local_bundle_verification_detects_changed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a signed local bundle when a shipped file has changed."""
    private_key = _trusted_test_key(monkeypatch)
    payload = b"compiled Celune runtime"
    runtime_path = tmp_path / "celune-bin"
    runtime_path.write_bytes(payload)
    manifest_data = _manifest_data(
        {
            runtime_path.name: hashlib.sha256(payload).hexdigest(),
        }
    )
    (tmp_path / updater.UPDATE_MANIFEST_NAME).write_bytes(manifest_data)
    (tmp_path / signature.SIGNATURE_NAME).write_bytes(
        _signature_data(manifest_data, private_key)
    )

    updater.verify_local_bundle(tmp_path)

    runtime_path.write_bytes(b"modified runtime")
    with pytest.raises(UpdateError):
        updater.verify_local_bundle(tmp_path)


def test_invalid_local_signature_rejects_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject startup when a present local signature is invalid."""
    private_key = _trusted_test_key(monkeypatch)
    manifest_data = _manifest_data({"celune-bin": "0" * 64})
    (tmp_path / updater.UPDATE_MANIFEST_NAME).write_bytes(manifest_data)
    invalid_envelope = json.loads(_signature_data(manifest_data, private_key))
    invalid_envelope["signature"] = "AA=="
    (tmp_path / signature.SIGNATURE_NAME).write_bytes(
        (json.dumps(invalid_envelope) + "\n").encode("utf-8")
    )

    with pytest.raises(UpdateError):
        updater.verify_local_bundle(tmp_path)


def test_legacy_manifest_without_signature_warns_and_continues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warn about an unsigned legacy manifest without blocking startup."""
    manifest = {
        "artifact": "Celune-test-x64",
        "files": {"celune-bin": "0" * 64},
        "revision": "0123456789abcdef",
        "version": "4.1.0",
    }
    (tmp_path / updater.UPDATE_MANIFEST_NAME).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    updater.verify_local_bundle(tmp_path)

    assert "unsigned" in capsys.readouterr().err
