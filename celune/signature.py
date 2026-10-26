# SPDX-License-Identifier: Apache-2.0
"""Sign release manifests and verify executable-appended keyrings."""

from __future__ import annotations

import os
import sys
import json
import stat
import base64
import struct
import binascii
import hashlib
import tempfile
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .constants import RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX

SIGNATURE_NAME = "celune-update.sig"
SIGNATURE_FORMAT = 1
SIGNATURE_ALGORITHM = "Ed25519"
KEYRING_CATALOG_NAME = "release_keyring.json"
KEYRING_PAYLOAD_MAGIC = b"CELUNE-KEYRING-1"
KEYRING_TRAILER_MAGIC = b"CELUNE-KEYS-END1"
KEYRING_KEY_ID_SIZE = 24
KEYRING_ENTRY_SIZE = KEYRING_KEY_ID_SIZE + 32
KEYRING_MAX_ENTRIES = 128
KEYRING_SIGNATURE_SIZE = 64
KEYRING_FOOTER_SIZE = 8 + len(KEYRING_TRAILER_MAGIC)
KEYRING_MAX_BODY_SIZE = (
    len(KEYRING_PAYLOAD_MAGIC)
    + 2
    + KEYRING_MAX_ENTRIES * KEYRING_ENTRY_SIZE
    + KEYRING_SIGNATURE_SIZE
)
_ROOT_PUBLIC_KEY = bytes.fromhex(RELEASE_KEYRING_ROOT_PUBLIC_KEY_HEX)
_CATALOG_PATH = Path(__file__).with_name(KEYRING_CATALOG_NAME)


def signing_key_id(public_key: bytes) -> str:
    """Return the stable key identifier for raw Ed25519 public-key bytes."""
    return f"ed25519-{hashlib.sha256(public_key).hexdigest()[:16]}"


def _load_source_release_keys() -> dict[str, bytes]:
    """Load and validate release public keys used by source tools."""
    raw_keys = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_keys, dict) or not 1 <= len(raw_keys) <= KEYRING_MAX_ENTRIES:
        raise ValueError("the release key catalog has an invalid shape")

    trusted_keys: dict[str, bytes] = {}
    for key_id, public_key_hex in raw_keys.items():
        if not isinstance(key_id, str) or not isinstance(public_key_hex, str):
            raise TypeError("the release key catalog has invalid entries")
        public_key = bytes.fromhex(public_key_hex)
        if (
            len(public_key) != 32
            or signing_key_id(public_key) != key_id
            or key_id in trusted_keys
        ):
            raise ValueError("the release key catalog has an invalid public key")
        trusted_keys[key_id] = public_key

    root_id = signing_key_id(_ROOT_PUBLIC_KEY)
    if trusted_keys.get(root_id) != _ROOT_PUBLIC_KEY:
        raise ValueError("the release key catalog does not contain its trusted root")
    return trusted_keys


def _encode_keyring_payload(trusted_keys: dict[str, bytes]) -> bytes:
    """Encode the canonical, root-signed release-keyring payload."""
    if not 1 <= len(trusted_keys) <= KEYRING_MAX_ENTRIES:
        raise ValueError("the release keyring contains an invalid number of keys")

    payload = bytearray(KEYRING_PAYLOAD_MAGIC)
    payload.extend(struct.pack("<H", len(trusted_keys)))
    for key_id, public_key in sorted(trusted_keys.items()):
        key_id_bytes = key_id.encode("ascii")
        if (
            len(key_id_bytes) != KEYRING_KEY_ID_SIZE
            or len(public_key) != 32
            or signing_key_id(public_key) != key_id
        ):
            raise ValueError("the release keyring contains an invalid key")
        payload.extend(key_id_bytes)
        payload.extend(public_key)
    return bytes(payload)


def create_signed_keyring(root_private_key_path: Path) -> bytes:
    """Create a signed keyring body from the source key catalog.

    Args:
        root_private_key_path: PEM file for the pinned Ed25519 root key.

    Returns:
        bytes: Canonical keyring payload followed by its Ed25519 signature.

    Raises:
        TypeError: if the root key is not Ed25519
        ValueError: if the root key or source catalog is not trusted
    """
    root_key = serialization.load_pem_private_key(
        root_private_key_path.read_bytes(),
        password=None,
    )
    if not isinstance(root_key, Ed25519PrivateKey):
        raise TypeError("the release keyring root must be Ed25519")
    public_key = root_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if public_key != _ROOT_PUBLIC_KEY:
        raise ValueError("the release keyring root does not match Celune's pinned root")

    payload = _encode_keyring_payload(_load_source_release_keys())
    return payload + root_key.sign(payload)


def decode_signed_keyring(
    keyring_data: bytes,
) -> dict[str, bytes]:
    """Verify and decode a keyring body signed by Celune's pinned root.

    Args:
        keyring_data: Payload and detached root signature.
    Returns:
        dict[str, bytes]: Trusted release keys indexed by their stable IDs.

    Raises:
        ValueError: if the keyring is malformed or its root signature is invalid
    """
    payload_size = len(keyring_data) - KEYRING_SIGNATURE_SIZE
    minimum_payload_size = len(KEYRING_PAYLOAD_MAGIC) + 2 + KEYRING_ENTRY_SIZE
    if not minimum_payload_size <= payload_size:
        raise ValueError("the release keyring is truncated")

    payload = keyring_data[:payload_size]
    signature = keyring_data[payload_size:]
    if payload[: len(KEYRING_PAYLOAD_MAGIC)] != KEYRING_PAYLOAD_MAGIC:
        raise ValueError("the release keyring header is invalid")

    key_count = struct.unpack_from("<H", payload, len(KEYRING_PAYLOAD_MAGIC))[0]
    expected_payload_size = (
        len(KEYRING_PAYLOAD_MAGIC) + 2 + key_count * KEYRING_ENTRY_SIZE
    )
    if (
        not 1 <= key_count <= KEYRING_MAX_ENTRIES
        or payload_size != expected_payload_size
    ):
        raise ValueError("the release keyring has an invalid size")

    try:
        Ed25519PublicKey.from_public_bytes(_ROOT_PUBLIC_KEY).verify(
            signature,
            payload,
        )
    except (InvalidSignature, ValueError) as error:
        raise ValueError("the release keyring root signature is invalid") from error

    trusted_keys: dict[str, bytes] = {}
    offset = len(KEYRING_PAYLOAD_MAGIC) + 2
    for _ in range(key_count):
        key_id_bytes = payload[offset : offset + KEYRING_KEY_ID_SIZE]
        public_key = payload[offset + KEYRING_KEY_ID_SIZE : offset + KEYRING_ENTRY_SIZE]
        offset += KEYRING_ENTRY_SIZE
        try:
            key_id = key_id_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(
                "the release keyring contains a non-ASCII key ID"
            ) from error
        if signing_key_id(public_key) != key_id or key_id in trusted_keys:
            raise ValueError("the release keyring contains an invalid or duplicate key")
        trusted_keys[key_id] = public_key
    return trusted_keys


def read_appended_keyring(executable_path: Path) -> dict[str, bytes]:
    """Read and verify the keyring trailer appended to an executable.

    Args:
        executable_path: Launcher executable containing the keyring trailer.

    Returns:
        dict[str, bytes]: Trusted release keys indexed by their stable IDs.

    Raises:
        OSError: if the executable cannot be read
        ValueError: if the trailer is missing, malformed, or untrusted
    """
    with executable_path.open("rb") as executable:
        executable.seek(0, os.SEEK_END)
        file_size = executable.tell()
        if file_size < KEYRING_FOOTER_SIZE:
            raise ValueError("the executable has no trusted keyring trailer")
        executable.seek(-KEYRING_FOOTER_SIZE, os.SEEK_END)
        footer = executable.read(KEYRING_FOOTER_SIZE)
        body_size = struct.unpack_from("<Q", footer)[0]
        if (
            footer[8:] != KEYRING_TRAILER_MAGIC
            or body_size > KEYRING_MAX_BODY_SIZE
            or body_size > file_size - KEYRING_FOOTER_SIZE
        ):
            raise ValueError("the executable keyring trailer is invalid")
        executable.seek(-(KEYRING_FOOTER_SIZE + body_size), os.SEEK_END)
        keyring_data = executable.read(body_size)
        if len(keyring_data) != body_size:
            raise ValueError("the executable keyring trailer is truncated")
    return decode_signed_keyring(keyring_data)


def _strip_existing_keyring(contents: bytes) -> bytes:
    """Return executable bytes before an existing valid keyring trailer."""
    if len(contents) < KEYRING_FOOTER_SIZE:
        return contents
    footer = contents[-KEYRING_FOOTER_SIZE:]
    if footer[8:] != KEYRING_TRAILER_MAGIC:
        return contents
    body_size = struct.unpack_from("<Q", footer)[0]
    trailer_size = body_size + KEYRING_FOOTER_SIZE
    if body_size > KEYRING_MAX_BODY_SIZE or trailer_size > len(contents):
        raise ValueError("the existing executable keyring trailer is invalid")
    decode_signed_keyring(contents[-trailer_size:-KEYRING_FOOTER_SIZE])
    return contents[:-trailer_size]


def append_trusted_keyring(executable_path: Path, root_private_key_path: Path) -> None:
    """Append a root-signed public-key ring to a launcher executable.

    Args:
        executable_path: Launcher binary that will carry the trusted keyring.
        root_private_key_path: PEM file for Celune's pinned Ed25519 root key.

    Raises:
        OSError: if the executable or root key cannot be read or written
        ValueError: if the root key, catalog, or existing trailer is invalid
    """
    path = executable_path.resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    original = _strip_existing_keyring(path.read_bytes())
    body = create_signed_keyring(root_private_key_path)
    trailer = body + struct.pack("<Q", len(body)) + KEYRING_TRAILER_MAGIC
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(original)
            temporary.write(trailer)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _is_compiled_runtime() -> bool:
    """Return whether the active entrypoint is a compiled Celune binary."""
    main_module = sys.modules.get("__main__")
    return bool(getattr(main_module, "__compiled__", False))


def _keyring_executable_path() -> Optional[Path]:
    """Resolve the adjacent launcher whose trailer contains release keys."""
    configured_path = os.environ.get("CELUNE_LAUNCHER_EXE")
    if configured_path:
        return Path(configured_path)
    if not _is_compiled_runtime():
        return None
    executable = Path(sys.argv[0]).resolve()
    launcher_name = "celune.exe" if os.name == "nt" else "celune"
    return executable.parent / launcher_name


def _runtime_release_keys() -> dict[str, bytes]:
    """Load trusted release keys from the compiled launcher's trailer."""
    keyring_executable = _keyring_executable_path()
    if keyring_executable is not None:
        return read_appended_keyring(keyring_executable)
    if _is_compiled_runtime():
        raise ValueError("the compiled launcher has no trusted keyring")
    return _load_source_release_keys()


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
    """Sign manifest bytes using a key in the release key catalog.

    Args:
        manifest: Exact bytes written to ``celune-update.json``.
        private_key_path: PEM-encoded Ed25519 private-key file.

    Returns:
        bytes: Detached JSON signature envelope.

    Raises:
        TypeError: if the key is not an Ed25519 private key
        ValueError: if the key does not match a catalogued release key
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
    if _load_source_release_keys().get(signing_key_id(public_key)) != public_key:
        raise ValueError("the release signing key is not in the key catalog")

    return signature_payload(manifest, key)


def verify_manifest_signature(manifest: bytes, signature_data: bytes) -> bool:
    """Return whether a trusted keyring key signed the manifest bytes."""
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
        public_key_bytes = _runtime_release_keys().get(key_id)
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
        OSError,
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
