# SPDX-License-Identifier: Apache-2.0
"""Sign and verify compiled-release update manifests."""

from __future__ import annotations

import json
import base64
import hashlib
import binascii
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .release_keys import TRUSTED_RELEASE_KEYS_HEX

SIGNATURE_NAME = "celune-update.sig"
SIGNATURE_FORMAT = 1
SIGNATURE_ALGORITHM = "Ed25519"
_TRUSTED_RELEASE_KEYS = {
    key_id: bytes.fromhex(public_key_hex)
    for key_id, public_key_hex in TRUSTED_RELEASE_KEYS_HEX.items()
}


def signing_key_id(public_key: bytes) -> str:
    """Return the stable key identifier for raw Ed25519 public-key bytes."""
    return f"ed25519-{hashlib.sha256(public_key).hexdigest()[:16]}"


def signature_payload(manifest: bytes, private_key: Ed25519PrivateKey) -> bytes:
    """Return the detached signature envelope for manifest bytes."""
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    signature = private_key.sign(manifest)
    envelope = {
        "algorithm": SIGNATURE_ALGORITHM,
        "format": SIGNATURE_FORMAT,
        "key_id": signing_key_id(public_key),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    return (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def sign_manifest(manifest: bytes, private_key_path: Path) -> bytes:
    """Sign manifest bytes using the trusted release private key.

    Args:
        manifest: Exact bytes written to ``celune-update.json``.
        private_key_path: PEM-encoded Ed25519 private-key file.

    Returns:
        bytes: Detached JSON signature envelope.

    Raises:
        TypeError: if the key is not an Ed25519 private key
        ValueError: if the key does not match a trusted release key
    """
    key = serialization.load_pem_private_key(
        private_key_path.read_bytes(),
        password=None,
    )
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("the release signing key must be Ed25519")

    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if signing_key_id(public_key) not in _TRUSTED_RELEASE_KEYS:
        raise ValueError("the release signing key does not match a trusted key")

    return signature_payload(manifest, key)


def verify_manifest_signature(manifest: bytes, signature_data: bytes) -> bool:
    """Return whether the detached signature matches a trusted release key."""
    try:
        envelope = json.loads(signature_data)
        if not isinstance(envelope, dict):
            return False
        if (
            envelope.get("format") != SIGNATURE_FORMAT
            or envelope.get("algorithm") != SIGNATURE_ALGORITHM
        ):
            return False

        key_id = envelope.get("key_id")
        encoded_signature = envelope.get("signature")
        if not isinstance(key_id, str) or not isinstance(encoded_signature, str):
            return False
        public_key_bytes = _TRUSTED_RELEASE_KEYS.get(key_id)
        if public_key_bytes is None:
            return False

        signature = base64.b64decode(encoded_signature, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature,
            manifest,
        )
        return True
    except (
        InvalidSignature,
        ValueError,
        TypeError,
        binascii.Error,
        json.JSONDecodeError,
    ):
        return False


def load_signature(path: Path) -> Optional[bytes]:
    """Read a detached signature file, returning ``None`` when it is absent."""
    try:
        return path.read_bytes()
    except OSError:
        return None
