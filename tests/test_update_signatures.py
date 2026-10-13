# SPDX-License-Identifier: Apache-2.0
"""Tests for signed compiled update manifests."""

from __future__ import annotations

import json
import hashlib
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
    monkeypatch.setitem(signature._TRUSTED_RELEASE_KEYS, key_id, public_key)
    return private_key


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
