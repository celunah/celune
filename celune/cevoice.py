# SPDX-License-Identifier: MIT
"""CEVOICE/CECHAR bundle writer, parser, and lazy file loader."""

from __future__ import annotations

import atexit
import contextlib
import gzip
import hashlib
import io
import json
import lzma
import re
import shutil
import struct
import tempfile
import threading
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, Optional, Union, cast

from .exceptions import CEVoiceError
from .paths import project_root, temp_data_dir, voices_data_dir
from .typing.cevoice import Manifest, ManifestValue, VoiceManifest

# CECHAR v2/v3 legacy container. Keep these names stable for callers that
# construct or inspect legacy packs.
MAGIC: Final[bytes] = b"CECHAR\0\0"
VERSION: Final[int] = 3
FORMAT_NAME: Final[str] = "CECHAR"
COMPATIBLE_CECHAR_VERSIONS: Final[frozenset[int]] = frozenset({2, 3})

# CEVOICE v1 spec (Celune v3.5 format)
LEGACY_MAGIC: Final[bytes] = b"CEVOICE\0"
LEGACY_VERSION: Final[int] = 1
LEGACY_FORMAT_NAME: Final[str] = "CEVOICE"

HEADER = struct.Struct("<8sHI")
V4_MAGIC: Final[bytes] = b"CECHAR\0"
V4_VERSION: Final[int] = 4
V4_HEADER = struct.Struct("<7sBB")
V4_FILE_ENTRY = struct.Struct("<II")
V4_COMPRESSION_NONE: Final[int] = 0
V4_COMPRESSION_GZIP: Final[int] = 1
V4_COMPRESSION_ZSTANDARD: Final[int] = 2
V4_COMPRESSION_XZ: Final[int] = 3
V4_COMPRESSION_OPENZL: Final[int] = 4
V4_COMPRESSION_MODES: Final[frozenset[int]] = frozenset(range(5))
ALLOWED_ASSET_KINDS = {"wav", "pt"}
V4_ALLOWED_ASSET_SUFFIXES: Final[frozenset[str]] = frozenset({".wav", ".pt", ".md"})
SUPPORTED_PERSONA_FILENAMES: Final[tuple[str, ...]] = (
    "identity.md",
    "soul.md",
    "personality.md",
    "speech_style.md",
    "boundaries.md",
    "examples.md",
)
DEFAULT_CEVOICE_PACK_SHA256: Final[str] = (
    "52e15b5c1749e4cac0f81298632395b94f6b32e8373a8a901188c209a122ba5d"
)


@dataclass(frozen=True)
class CEVoiceAsset:
    """One binary asset stored inside a CEVOICE/CECHAR package."""

    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class CEVoice:
    """Parsed CEVOICE/CECHAR package metadata and payload access."""

    path: Path
    metadata: Manifest
    payload_offset: int
    _payload: Optional[bytes] = field(default=None, repr=False, compare=False)
    _asset_entries: Mapping[tuple[str, str], CEVoiceAsset] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _bundle_asset_entries: Mapping[str, CEVoiceAsset] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def open(cls, path: Union[str, Path]) -> CEVoice:
        """Parse and validate a CEVOICE/CECHAR package.

        Args:
            path: The CEVOICE/CECHAR package to load.

        Returns:
            CEVoice: The CEVoice object.

        Raises:
            CEVoiceError: The CEVOICE/CECHAR package is malformed and could not be loaded.
        """
        bundle_path = Path(path)
        with bundle_path.open("rb") as stream:
            prefix = stream.read(V4_HEADER.size)
            if len(prefix) >= len(V4_MAGIC) and prefix[: len(V4_MAGIC)] == V4_MAGIC:
                if len(prefix) >= V4_HEADER.size and prefix[7] != 0:
                    return cls._open_v4(bundle_path, prefix, stream.read())
            stream.seek(0)
            magic, version, metadata_length = _read_header(stream)
            if magic not in {MAGIC, LEGACY_MAGIC}:
                raise CEVoiceError("invalid CEVOICE magic")
            if not (
                (magic == MAGIC and version in COMPATIBLE_CECHAR_VERSIONS)
                or (magic == LEGACY_MAGIC and version == LEGACY_VERSION)
            ):
                raise CEVoiceError(f"unsupported CEVOICE version {version}")

            metadata_bytes = stream.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise CEVoiceError("truncated CEVOICE metadata")

            try:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CEVoiceError("invalid CEVOICE metadata") from error

            if metadata.get("version") != version:
                raise CEVoiceError("metadata format/version mismatch")

            payload_offset = HEADER.size + metadata_length
            _validate_metadata(bundle_path, metadata, payload_offset)

        return cls(bundle_path, metadata, payload_offset)

    @classmethod
    def _open_v4(
        cls,
        path: Path,
        header: bytes,
        stored_payload: bytes,
    ) -> CEVoice:
        """Open a CECHAR v4 archive after its fixed header was identified."""
        magic, version, compression = V4_HEADER.unpack(header)
        if magic != V4_MAGIC:
            raise CEVoiceError("invalid CECHAR v4 magic")
        if version != V4_VERSION:
            raise CEVoiceError(f"unsupported CECHAR version {version}")
        if compression not in V4_COMPRESSION_MODES:
            raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")

        payload = _decompress_v4_payload(stored_payload, compression)
        metadata, asset_entries, bundle_asset_entries = _parse_v4_payload(payload)
        return cls(
            path,
            metadata,
            V4_HEADER.size,
            payload,
            asset_entries,
            bundle_asset_entries,
        )

    @property
    def voices(self) -> VoiceManifest:  # noqa
        """Return the voice manifest.

        Returns:
            VoiceManifest: The voice manifest of this CEVOICE/CECHAR package.

        Raises:
            CEVoiceError: The CEVOICE/CECHAR package does not contain a valid voice manifest.
        """
        voices = self.metadata.get("voices")
        if not isinstance(voices, dict):
            raise CEVoiceError("metadata voices must be an object")
        return cast(VoiceManifest, voices)

    @property
    def voice_order(self) -> tuple[str, ...]:  # noqa
        """Return the preferred user-facing voice order.

        Returns:
            tuple[str, ...]: The preferred user-facing voice order.
        """
        order = self.metadata.get("voice_order")
        if isinstance(order, list) and all(isinstance(voice, str) for voice in order):
            return tuple(cast(list[str], order))
        return tuple(self.voices)

    def asset(self, voice: str, kind: str) -> CEVoiceAsset:
        """Return metadata for one named voice asset.

        Args:
            voice: The voice name to find an asset for.
            kind: The type of asset to find.

        Returns:
            CEVoiceAsset: The corresponding CEVoiceAsset object.

        Raises:
            KeyError: The specified voice name does not have this kind of asset.
        """
        if self._asset_entries:
            try:
                return self._asset_entries[(voice, kind)]
            except KeyError as error:
                raise KeyError(
                    f"asset '{kind}' for voice '{voice}' not found"
                ) from error

        try:
            assets = cast(dict[str, Manifest], self.voices[voice]["assets"])
            raw_asset = assets[kind]
        except KeyError as error:
            raise KeyError(f"asset '{kind}' for voice '{voice}' not found") from error

        return CEVoiceAsset(
            offset=cast(int, raw_asset["offset"]),
            length=cast(int, raw_asset["length"]),
            sha256=cast(str, raw_asset["sha256"]),
        )

    @property
    def assets(self) -> Manifest:  # noqa
        """Return the top-level bundle asset manifest.

        Returns:
            Manifest containing the bundle's top-level assets.
        """
        assets = self.metadata.get("assets")
        if not isinstance(assets, dict):
            return {}
        return cast(Manifest, assets)

    def bundle_asset(self, name: str) -> CEVoiceAsset:
        """Return metadata for one top-level bundle asset.

        Args:
            name: The bundle asset filename to resolve.

        Returns:
            CEVoiceAsset: The parsed asset metadata for the requested bundle asset.

        Raises:
            KeyError: Raised when the named bundle asset does not exist.
        """
        if self._bundle_asset_entries:
            try:
                return self._bundle_asset_entries[name]
            except KeyError as error:
                raise KeyError(f"bundle asset '{name}' not found") from error

        try:
            raw_asset = self.assets[name]
        except KeyError as error:
            raise KeyError(f"bundle asset '{name}' not found") from error
        if not isinstance(raw_asset, dict):
            raise KeyError(f"bundle asset '{name}' not found")
        return CEVoiceAsset(
            offset=cast(int, raw_asset["offset"]),
            length=cast(int, raw_asset["length"]),
            sha256=cast(str, raw_asset["sha256"]),
        )

    def read_asset(self, voice: str, kind: str) -> bytes:
        """Read and checksum one asset payload.

        Args:
            voice: The voice name to load an asset for.
            kind: The type of asset to load.

        Returns:
            bytes: The asset payload streamed into memory.

        Raises:
            CEVoiceError: The asset was truncated, or its checksum validation failed.
        """
        asset = self.asset(voice, kind)
        data = self._read_asset_bytes(asset)

        if len(data) != asset.length:
            raise CEVoiceError(f"truncated asset '{kind}' for voice '{voice}'")
        if hashlib.sha256(data).hexdigest().casefold() != asset.sha256.casefold():
            raise CEVoiceError(
                f"checksum mismatch for asset '{kind}' of voice '{voice}'"
            )
        return data

    def read_bundle_asset(self, name: str) -> bytes:
        """Read and checksum one top-level bundle asset payload.

        Args:
            name: The bundle asset filename to read.

        Returns:
            bytes: The decoded bundle asset payload bytes.

        Raises:
            CEVoiceError: Raised when the asset is truncated or fails checksum validation.
        """
        asset = self.bundle_asset(name)
        data = self._read_asset_bytes(asset)

        if len(data) != asset.length:
            raise CEVoiceError(f"truncated bundle asset '{name}'")
        if hashlib.sha256(data).hexdigest().casefold() != asset.sha256.casefold():
            raise CEVoiceError(f"checksum mismatch for bundle asset '{name}'")
        return data

    def _read_asset_bytes(self, asset: CEVoiceAsset) -> bytes:
        """Read an asset from either a legacy file payload or a v4 buffer."""
        if self._payload is not None:
            return self._payload[asset.offset : asset.offset + asset.length]
        with self.path.open("rb") as stream:
            stream.seek(self.payload_offset + asset.offset)
            return stream.read(asset.length)


@dataclass(frozen=True, slots=True)
class PersonaIdentity:
    """Identity details supplied by a CEVOICE/CECHAR pack."""

    name: str = ""
    age: str = ""
    gender: str = ""
    profile: str = ""


@dataclass(frozen=True, slots=True)
class PersonaStyleValues:
    """Baseline speaking-style values supplied by a CEVOICE/CECHAR pack."""

    warmth: str = ""
    directness: str = ""
    humor: str = ""
    detail: str = ""
    formality: str = ""
    enthusiasm: str = ""


@dataclass(frozen=True, slots=True)
class CEVoicePersona:
    """Persona metadata supplied by a CEVOICE/CECHAR pack."""

    identity: PersonaIdentity = field(default_factory=PersonaIdentity)
    speaking_style: str = ""
    boundaries: tuple[str, ...] = ()
    prompt_rules: tuple[str, ...] = ()
    example_dialogue: tuple[str, ...] = ()
    style: PersonaStyleValues = field(default_factory=PersonaStyleValues)


class CEVoiceLoader:
    """Lazily materialize CEVOICE assets as real files for path-only consumers."""

    def __init__(self, bundle: CEVoice) -> None:
        self.bundle = bundle
        self._directory = Path(
            tempfile.mkdtemp(
                prefix="celune-cevoice-",
                dir=str(temp_data_dir(create=True)),
            )
        )
        register_protected_temp_path(self._directory)
        self._paths: dict[tuple[str, str], Path] = {}
        atexit.register(self.close)

    def materialize(self, voice: str, kind: str, suffix: Optional[str] = None) -> Path:
        """Extract one asset once and return its temporary path.

        Args:
            voice: The named voice assets to extract.
            kind: The type of voice assets to extract.
            suffix: The extension of the voice asset to extract.

        Returns:
            Path: The path to the extracted voice asset.

        Raises:
            CEVoiceError: The CEVOICE/CECHAR package contains path delimiters.
        """
        key = (voice, kind)
        path = self._paths.get(key)
        if path is None or not path.is_file():
            if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
                raise CEVoiceError(f"invalid voice name '{voice}'")
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind '{kind}'")
            extension = suffix or f".{kind}"
            safe_voice = Path(voice).name
            register_protected_temp_path(self._directory)
            self._directory.mkdir(parents=True, exist_ok=True)
            path = self._directory / f"{safe_voice}{extension}"
            path.write_bytes(self.bundle.read_asset(voice, kind))
            self._paths[key] = path
        return path

    def close(self) -> None:
        """Remove extracted temporary files."""
        unregister_protected_temp_path(self._directory)
        shutil.rmtree(self._directory, ignore_errors=True)


def write_cevoice(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]] = None,
) -> Path:
    """Write a CEVOICE/CECHAR package from per-voice binary assets.

    Args:
        path: The CEVOICE/CECHAR package to save as.
        voices: The voice files to bundle into this CEVOICE/CECHAR package.
        metadata: The metadata to bundle into this CEVOICE/CECHAR package.
        voice_metadata: Extra metadata stored beside each voice's assets.
        bundle_assets: Extra top-level payload assets such as legacy Persona Markdown.

    Returns:
        Path: The path to the created CEVOICE/CECHAR package.

    Raises:
        CEVoiceError: The CEVOICE/CECHAR package contains path delimiters.
    """
    payload = bytearray()
    manifest_voices: VoiceManifest = {}
    unknown_voice_metadata = set(voice_metadata or {}) - set(voices)
    if unknown_voice_metadata:
        unknown = sorted(unknown_voice_metadata)[0]  # noqa: FURB192
        raise CEVoiceError(f"voice metadata provided for unknown voice '{unknown}'")

    for voice, assets in voices.items():
        if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
            raise CEVoiceError(f"invalid voice name '{voice}'")
        manifest_assets: dict[str, Manifest] = {}
        for kind, source in assets.items():
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind for voice '{voice}'")
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            data = _read_source(source)
            manifest_assets[kind] = {
                "offset": len(payload),
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            payload.extend(data)
        voice_entry = dict((voice_metadata or {}).get(voice, {}))
        voice_entry["assets"] = cast(ManifestValue, manifest_assets)
        manifest_voices[voice] = voice_entry

    manifest_bundle_assets: dict[str, Manifest] = {}
    for name, source in (bundle_assets or {}).items():
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise CEVoiceError(f"invalid bundle asset name '{name}'")
        if Path(name).suffix.lower() != ".md":
            raise CEVoiceError(f"unsupported bundle asset '{name}'")
        if Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
            raise CEVoiceError(f"unsupported bundle asset '{name}'")
        data = _read_source(source)
        manifest_bundle_assets[name] = {
            "offset": len(payload),
            "length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        payload.extend(data)

    manifest = dict(metadata or {})
    manifest["format"] = FORMAT_NAME
    manifest["version"] = VERSION
    manifest["voices"] = cast(ManifestValue, manifest_voices)
    if manifest_bundle_assets:
        manifest["assets"] = cast(ManifestValue, manifest_bundle_assets)
    metadata_bytes = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    output_path = Path(path)
    with output_path.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, len(metadata_bytes)))
        stream.write(metadata_bytes)
        stream.write(payload)
    return output_path


def write_cechar_v4(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]] = None,
    compression: int = V4_COMPRESSION_NONE,
) -> Path:
    """Write a CECHAR v4 archive with a name-only JSON manifest and jump table.

    Args:
        path: The CECHAR v4 package to save as.
        voices: The voice files to bundle into the package.
        metadata: The semantic package metadata to bundle into the package.
        voice_metadata: Extra metadata stored beside each voice's assets.
        bundle_assets: Persona Markdown files stored as top-level assets.
        compression: One of the CECHAR v4 compression mode constants.

    Returns:
        Path: The path to the created CECHAR v4 package.

    Raises:
        CEVoiceError: The package data cannot be represented or validated as CECHAR v4.
    """
    if compression not in V4_COMPRESSION_MODES:
        raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")

    unknown_voice_metadata = set(voice_metadata or {}) - set(voices)
    if unknown_voice_metadata:
        unknown = sorted(unknown_voice_metadata)[0]  # noqa: FURB192
        raise CEVoiceError(f"voice metadata provided for unknown voice '{unknown}'")

    payload = bytearray()
    file_entries: list[dict[str, str]] = []
    file_lengths: list[int] = []
    manifest_voices: VoiceManifest = {}
    for voice, assets in voices.items():
        _validate_v4_name(voice, "voice")
        manifest_assets: dict[str, ManifestValue] = {}
        seen_names: set[str] = set()
        for kind, source in assets.items():
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            name = f"{voice}.{kind}"
            data = _read_source(source)
            _validate_v4_file_data(name, data)
            payload.extend(data)

            folded = name.casefold()
            if folded in seen_names:
                raise CEVoiceError(f"{name} already exists")
            seen_names.add(folded)

            file_entries.append(
                {"name": name, "sha256": hashlib.sha256(data).hexdigest()}
            )
            file_lengths.append(len(data))
            manifest_assets[kind] = name

        voice_entry = dict((voice_metadata or {}).get(voice, {}))
        voice_entry["assets"] = cast(ManifestValue, manifest_assets)
        manifest_voices[voice] = voice_entry

    manifest_bundle_assets: dict[str, ManifestValue] = {}
    for name, source in (bundle_assets or {}).items():
        _validate_v4_name(name, "bundle asset")
        if Path(name).suffix.lower() != ".md":
            raise CEVoiceError(f"unsupported bundle asset '{name}'")
        if Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
            raise CEVoiceError(f"unsupported bundle asset '{name}'")
        data = _read_source(source)
        _validate_v4_file_data(name, data)
        payload.extend(data)
        file_entries.append({"name": name, "sha256": hashlib.sha256(data).hexdigest()})
        file_lengths.append(len(data))
        manifest_bundle_assets[name] = name

    manifest = dict(metadata or {})
    manifest["format"] = FORMAT_NAME
    manifest["version"] = V4_VERSION
    manifest["files"] = cast(
        ManifestValue,
        cast(list[ManifestValue], file_entries),
    )
    manifest["voices"] = cast(ManifestValue, manifest_voices)
    if manifest_bundle_assets:
        manifest["assets"] = cast(ManifestValue, manifest_bundle_assets)
    else:
        manifest.pop("assets", None)

    metadata_bytes = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(metadata_bytes) > 0xFFFFFFFF or len(file_entries) > 0xFFFFFFFF:
        raise CEVoiceError("CECHAR v4 metadata exceeds uint32 limits")

    table_size = 4 + len(metadata_bytes) + 4 + V4_FILE_ENTRY.size * len(file_entries)
    if table_size + len(payload) > 0xFFFFFFFF:
        raise CEVoiceError("CECHAR v4 payload exceeds uint32 limits")

    logical_payload = bytearray()
    logical_payload.extend(struct.pack("<I", len(metadata_bytes)))
    logical_payload.extend(metadata_bytes)
    logical_payload.extend(struct.pack("<I", len(file_entries)))
    offset = table_size
    for length in file_lengths:
        logical_payload.extend(V4_FILE_ENTRY.pack(offset, length))
        offset += length
    logical_payload.extend(payload)

    stored_payload = _compress_v4_payload(bytes(logical_payload), compression)
    output_path = Path(path)
    with output_path.open("wb") as stream:
        stream.write(V4_HEADER.pack(V4_MAGIC, V4_VERSION, compression))
        stream.write(stored_payload)
    return output_path


def _validate_v4_file_data(name: str, data: bytes) -> None:
    """Validate one file before placing it into a CECHAR v4 archive."""
    suffix = Path(name).suffix.lower()
    if suffix not in V4_ALLOWED_ASSET_SUFFIXES:
        raise CEVoiceError(f"unsupported CECHAR v4 asset extension '{suffix}'")
    if suffix == ".md" and Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
        raise CEVoiceError(f"unsupported CECHAR v4 persona asset '{name}'")
    if suffix == ".wav":
        _validate_v4_wav(name, data)
    elif suffix == ".pt":
        _validate_v4_embedding(name, data)
    else:
        _validate_v4_persona_markdown(name, data)


def _compress_v4_payload(payload: bytes, compression: int) -> bytes:
    """Compress one logical v4 payload using its selected storage codec."""
    if compression == V4_COMPRESSION_NONE:
        return payload
    if compression == V4_COMPRESSION_GZIP:
        return gzip.compress(payload)
    if compression == V4_COMPRESSION_ZSTANDARD:
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as error:
            raise CEVoiceError("Zstandard compression is unavailable") from error
        return zstandard.ZstdCompressor().compress(payload)
    if compression == V4_COMPRESSION_XZ:
        return lzma.compress(payload)
    if compression == V4_COMPRESSION_OPENZL:
        try:
            import openzl.ext as zl  # type: ignore[import-not-found]
        except ImportError as error:
            raise CEVoiceError("OpenZL compression is unavailable") from error
        compressor = zl.Compressor()
        compressor.select_starting_graph(zl.graphs.Compress()(compressor))
        context = zl.CCtx()
        context.ref_compressor(compressor)
        context.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
        return bytes(context.compress([zl.Input(zl.Type.Serial, payload)]))
    raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")


def _decompress_v4_payload(stored_payload: bytes, compression: int) -> bytes:
    """Decompress one complete CECHAR v4 stored payload."""
    try:
        if compression == V4_COMPRESSION_NONE:
            return stored_payload
        if compression == V4_COMPRESSION_GZIP:
            return gzip.decompress(stored_payload)
        if compression == V4_COMPRESSION_ZSTANDARD:
            try:
                import zstandard  # type: ignore[import-not-found]
            except ImportError as error:
                raise CEVoiceError("Zstandard compression is unavailable") from error
            return zstandard.ZstdDecompressor().decompress(stored_payload)
        if compression == V4_COMPRESSION_XZ:
            return lzma.decompress(stored_payload)
        if compression == V4_COMPRESSION_OPENZL:
            try:
                import openzl.ext as zl  # type: ignore[import-not-found]
            except ImportError as error:
                raise CEVoiceError("OpenZL compression is unavailable") from error
            outputs = zl.DCtx().decompress(stored_payload)
            if len(outputs) != 1 or outputs[0].type != zl.Type.Serial:
                raise CEVoiceError("invalid OpenZL CECHAR v4 payload")
            return bytes(outputs[0].content.as_bytes())
    except CEVoiceError:
        raise
    except Exception as error:
        raise CEVoiceError("invalid compressed CECHAR v4 payload") from error
    raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")


def _parse_v4_payload(
    payload: bytes,
) -> tuple[
    Manifest,
    dict[tuple[str, str], CEVoiceAsset],
    dict[str, CEVoiceAsset],
]:
    """Parse, validate, and index one decompressed CECHAR v4 payload."""
    if len(payload) < 4:
        raise CEVoiceError("truncated CECHAR v4 metadata length")
    metadata_length = struct.unpack_from("<I", payload)[0]
    metadata_start = 4
    metadata_end = metadata_start + metadata_length
    if metadata_end > len(payload):
        raise CEVoiceError("truncated CECHAR v4 metadata")
    metadata_bytes = payload[metadata_start:metadata_end]
    if metadata_bytes.startswith(b"\xef\xbb\xbf"):
        raise CEVoiceError("CECHAR v4 metadata must not contain a byte-order mark")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CEVoiceError("invalid CECHAR v4 metadata") from error
    if not isinstance(metadata, dict):
        raise CEVoiceError("CECHAR v4 metadata root must be an object")
    if metadata.get("format") != FORMAT_NAME or metadata.get("version") != V4_VERSION:
        raise CEVoiceError("CECHAR v4 metadata format/version mismatch")

    file_list = metadata.get("files")
    if not isinstance(file_list, list):
        raise CEVoiceError("CECHAR v4 metadata files must be a list")
    file_count_offset = metadata_end
    if len(payload) < file_count_offset + 4:
        raise CEVoiceError("truncated CECHAR v4 file count")
    file_count = struct.unpack_from("<I", payload, file_count_offset)[0]
    if len(file_list) != file_count:
        raise CEVoiceError("CECHAR v4 metadata file count mismatch")

    jump_table_start = file_count_offset + 4
    jump_table_end = jump_table_start + V4_FILE_ENTRY.size * file_count
    if jump_table_end > len(payload):
        raise CEVoiceError("truncated CECHAR v4 jump table")

    file_names: list[str] = []
    file_digests: list[Optional[str]] = []
    for item in file_list:
        if not isinstance(item, dict):
            raise CEVoiceError("CECHAR v4 file entry must be an object")
        if "offset" in item or "length" in item:
            raise CEVoiceError("CECHAR v4 file entries must not store physical offsets")
        name = item.get("name")
        if not isinstance(name, str):
            raise CEVoiceError("CECHAR v4 file entry must contain a name")
        _validate_v4_name(name, "file")
        suffix = Path(name).suffix.lower()
        if suffix not in V4_ALLOWED_ASSET_SUFFIXES:
            raise CEVoiceError(f"unsupported CECHAR v4 asset extension '{suffix}'")
        if suffix == ".md" and Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
            raise CEVoiceError(f"unsupported CECHAR v4 persona asset '{name}'")
        if name.casefold() in {existing.casefold() for existing in file_names}:
            raise CEVoiceError(f"duplicate CECHAR v4 file name '{name}'")
        digest = item.get("sha256")
        if digest is not None and not _is_sha256(digest):
            raise CEVoiceError(f"invalid CECHAR v4 checksum for '{name}'")
        file_names.append(name)
        file_digests.append(digest.casefold() if isinstance(digest, str) else None)

    entries: dict[str, CEVoiceAsset] = {}
    ranges: list[tuple[int, int, str]] = []
    for index, name in enumerate(file_names):
        offset, length = V4_FILE_ENTRY.unpack_from(
            payload, jump_table_start + index * 8
        )
        if (
            offset < jump_table_end
            or offset > len(payload)
            or length > len(payload) - offset
        ):
            raise CEVoiceError(
                f"CECHAR v4 file range for '{name}' is outside the payload"
            )
        end = offset + length
        ranges.append((offset, end, name))
        data = payload[offset:end]
        digest = hashlib.sha256(data).hexdigest()
        declared_digest = file_digests[index]
        if declared_digest is not None and digest.casefold() != declared_digest:
            raise CEVoiceError(f"checksum mismatch for CECHAR v4 file '{name}'")
        entries[name] = CEVoiceAsset(offset, length, digest)

    for first, second in zip(sorted(ranges), sorted(ranges)[1:]):
        if second[0] < first[1]:
            raise CEVoiceError(
                f"overlapping CECHAR v4 file ranges '{first[2]}' and '{second[2]}'"
            )

    _validate_v4_assets(payload, entries)
    normalized_metadata = _normalize_v4_metadata(metadata, entries)
    asset_entries, bundle_asset_entries = _index_v4_manifest(
        normalized_metadata,
        entries,
    )
    return normalized_metadata, asset_entries, bundle_asset_entries


def _normalize_v4_metadata(
    metadata: dict,
    entries: Mapping[str, CEVoiceAsset],
) -> Manifest:
    """Normalize v4 file references into the runtime's existing manifest shape."""
    normalized = cast(Manifest, metadata)
    raw_voices = normalized.get("voices")
    if raw_voices is None:
        derived_voices: VoiceManifest = {}
        for name in entries:
            suffix = Path(name).suffix.lower()
            if suffix not in {".wav", ".pt"}:
                continue
            voice = Path(name).stem
            voice_entry = derived_voices.setdefault(voice, {})
            assets = cast(
                dict[str, ManifestValue], voice_entry.setdefault("assets", {})
            )
            assets[suffix[1:]] = name
        normalized["voices"] = cast(ManifestValue, derived_voices)
    elif not isinstance(raw_voices, dict):
        raise CEVoiceError("CECHAR v4 metadata voices must be an object")

    raw_assets = normalized.get("assets")
    if raw_assets is None:
        normalized["assets"] = cast(
            ManifestValue,
            {name: name for name in entries if Path(name).suffix.lower() == ".md"},
        )
    elif not isinstance(raw_assets, dict):
        raise CEVoiceError("CECHAR v4 metadata assets must be an object")
    return normalized


def _index_v4_manifest(
    metadata: Manifest,
    entries: Mapping[str, CEVoiceAsset],
) -> tuple[dict[tuple[str, str], CEVoiceAsset], dict[str, CEVoiceAsset]]:
    """Resolve name-only v4 manifest references to validated jump-table entries."""
    indexed_voices: dict[tuple[str, str], CEVoiceAsset] = {}
    voices = metadata.get("voices")
    if not isinstance(voices, dict):
        raise CEVoiceError("CECHAR v4 metadata voices must be an object")
    voices = cast(dict[str, Manifest], voices)
    _validate_v4_semantic_metadata(metadata, voices)
    for voice, voice_data in voices.items():
        if not isinstance(voice, str) or not isinstance(voice_data, dict):
            raise CEVoiceError("invalid CECHAR v4 voice entry")
        assets = voice_data.get("assets")
        if not isinstance(assets, dict):
            raise CEVoiceError(f"voice '{voice}' assets must be an object")
        for kind, reference in assets.items():
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            name = _v4_reference_name(reference, f"voice '{voice}' asset '{kind}'")
            if name not in entries or Path(name).suffix.lower() != f".{kind}":
                raise CEVoiceError(f"missing CECHAR v4 asset '{name}'")
            indexed_voices[(voice, kind)] = entries[name]

    indexed_bundle_assets: dict[str, CEVoiceAsset] = {}
    raw_assets = metadata.get("assets")
    if not isinstance(raw_assets, dict):
        raise CEVoiceError("CECHAR v4 metadata assets must be an object")
    for name, reference in raw_assets.items():
        if not isinstance(name, str):
            raise CEVoiceError("invalid CECHAR v4 bundle asset name")
        referenced_name = _v4_reference_name(reference, f"bundle asset '{name}'")
        if referenced_name != name or referenced_name not in entries:
            raise CEVoiceError(f"missing CECHAR v4 bundle asset '{name}'")
        indexed_bundle_assets[name] = entries[referenced_name]
    return indexed_voices, indexed_bundle_assets


def _validate_v4_semantic_metadata(
    metadata: Manifest,
    voices: Mapping[str, Manifest],
) -> None:
    """Validate v4 metadata fields that describe runtime voice selection."""
    default_voice = metadata.get("default_voice")
    if default_voice is not None and default_voice not in voices:
        raise CEVoiceError("metadata default_voice must name a defined voice")

    voice_order = metadata.get("voice_order")
    if voice_order is not None:
        if not isinstance(voice_order, list) or not all(
            isinstance(voice, str) for voice in voice_order
        ):
            raise CEVoiceError("metadata voice_order must be a list of voice names")
        order = cast(list[str], voice_order)
        if len(set(order)) != len(order):
            raise CEVoiceError("metadata voice_order must not contain duplicates")
        if any(voice not in voices for voice in order):
            raise CEVoiceError("metadata voice_order must only name defined voices")
        order.extend(voice for voice in voices if voice not in order)

    theme = metadata.get("theme")
    if theme is not None:
        if not isinstance(theme, dict):
            raise CEVoiceError("metadata theme must be an object")
        if (
            theme.get("faded_accent") is None
            and theme.get("sleeping_color") is not None
        ):
            theme["faded_accent"] = str(theme.get("sleeping_color", "#9c88ce"))
        for key in ("background", "accent", "glow_color", "faded_accent"):
            value = theme.get(key)
            if key in {"glow_color", "faded_accent"} and value is None:
                continue
            if not _is_hex_color(value):
                raise CEVoiceError(f"metadata theme '{key}' must be a hex color")

    persona = metadata.get("persona")
    if persona is not None:
        _validate_persona_metadata(persona)

    for voice, voice_data in voices.items():
        if not isinstance(voice, str) or not isinstance(voice_data, dict):
            raise CEVoiceError("invalid voice entry")
        _validate_v4_name(voice, "voice")
        cfg_scale = voice_data.get("cfg_scale")
        if cfg_scale is not None and (
            not isinstance(cfg_scale, (int, float))
            or isinstance(cfg_scale, bool)
            or cfg_scale <= 0
        ):
            raise CEVoiceError(f"voice '{voice}' cfg_scale must be a positive number")
        reference_text = voice_data.get("reference_text")
        if reference_text is not None and (
            not isinstance(reference_text, str) or not reference_text.strip()
        ):
            raise CEVoiceError(
                f"voice '{voice}' reference_text must be a non-empty string"
            )
        voice_persona = voice_data.get("persona")
        if voice_persona is not None:
            _validate_persona_metadata(voice_persona)


def _v4_reference_name(value: ManifestValue, field_name: str) -> str:
    """Read a v4 logical file name while rejecting physical table fields."""
    name: object
    if isinstance(value, str):
        name = value
    elif isinstance(value, dict):
        if "offset" in value or "length" in value:
            raise CEVoiceError(f"{field_name} must not store physical offsets")
        name = value.get("file")
        if not isinstance(name, str):
            name = value.get("name")
    else:
        name = None
    if not isinstance(name, str):
        raise CEVoiceError(f"{field_name} must reference a file name")
    _validate_v4_name(name, field_name)
    return name


def _validate_v4_name(value: object, field_name: str) -> None:
    """Validate one CECHAR v4 logical name against traversal and null-byte rules."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise CEVoiceError(f"invalid CECHAR v4 {field_name} name")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise CEVoiceError(f"invalid CECHAR v4 {field_name} name '{value}'")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise CEVoiceError(f"invalid CECHAR v4 {field_name} name '{value}'")


def _validate_v4_assets(
    payload: bytes,
    entries: Mapping[str, CEVoiceAsset],
) -> None:
    """Validate each v4 asset according to its extension."""
    for name, entry in entries.items():
        data = payload[entry.offset : entry.offset + entry.length]
        suffix = Path(name).suffix.lower()
        if suffix == ".wav":
            _validate_v4_wav(name, data)
        elif suffix == ".pt":
            _validate_v4_embedding(name, data)
        elif suffix == ".md":
            _validate_v4_persona_markdown(name, data)


def _validate_v4_wav(name: str, data: bytes) -> None:
    """Validate one canonical CECHAR v4 PCM WAV asset."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CEVoiceError(f"invalid CECHAR v4 WAV asset '{name}'")
    fmt_data: Optional[bytes] = None
    has_data_chunk = False
    cursor = 12
    while cursor + 8 <= len(data):
        chunk_id = data[cursor : cursor + 4]
        chunk_length = struct.unpack_from("<I", data, cursor + 4)[0]
        chunk_start = cursor + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise CEVoiceError(f"invalid CECHAR v4 WAV asset '{name}'")
        if chunk_id == b"fmt " and fmt_data is None:
            fmt_data = data[chunk_start:chunk_end]
        if chunk_id == b"data":
            has_data_chunk = True
        cursor = chunk_end + (chunk_length & 1)
    if fmt_data is None or len(fmt_data) < 16 or not has_data_chunk:
        raise CEVoiceError(f"invalid CECHAR v4 WAV asset '{name}'")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", fmt_data[:16]
    )
    if (audio_format, channels, sample_rate, bits) != (1, 1, 24000, 16):
        raise CEVoiceError(f"invalid CECHAR v4 WAV properties for '{name}'")
    if (
        byte_rate != sample_rate * channels * bits // 8
        or block_align != channels * bits // 8
    ):
        raise CEVoiceError(f"invalid CECHAR v4 WAV properties for '{name}'")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getframerate() != 24000
                or wav_file.getsampwidth() != 2
                or wav_file.getcomptype() != "NONE"
            ):
                raise CEVoiceError(f"invalid CECHAR v4 WAV properties for '{name}'")
    except (EOFError, wave.Error) as error:
        raise CEVoiceError(f"invalid CECHAR v4 WAV asset '{name}'") from error


def _validate_v4_embedding(name: str, data: bytes) -> None:
    """Validate one restricted tensor-only CECHAR v4 voice embedding."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise CEVoiceError("PyTorch not found, can't validate .pt assets") from error
    try:
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise CEVoiceError(f"invalid CECHAR v4 PT asset '{name}'") from error
    if isinstance(value, dict):
        if set(value) != {"speaker_embedding"}:
            raise CEVoiceError(f"invalid CECHAR v4 PT keys for '{name}'")
        value = value["speaker_embedding"]
    if not isinstance(value, torch.Tensor):
        raise CEVoiceError(f"invalid CECHAR v4 PT asset '{name}'")
    if (
        value.dtype != torch.float32
        or value.numel() != 2048
        or value.ndim not in {1, 2}
    ):
        raise CEVoiceError(f"invalid CECHAR v4 PT shape or dtype for '{name}'")
    if value.ndim == 2 and value.shape != (1, 2048):
        raise CEVoiceError(f"invalid CECHAR v4 PT shape for '{name}'")
    if value.ndim == 1 and value.shape != (2048,):
        raise CEVoiceError(f"invalid CECHAR v4 PT shape for '{name}'")
    if not bool(torch.isfinite(value).all()):
        raise CEVoiceError(f"invalid CECHAR v4 PT values for '{name}'")


_V4_FORBIDDEN_PERSONA_FIELD = re.compile(
    r"(?im)^\s*(?:agent|agents|capabilities|filesystem|network|process|shell|"
    r"tools?|permissions?|exec(?:ute)?|code)\s*[:=]"
)


def _validate_v4_persona_markdown(name: str, data: bytes) -> None:
    """Validate a CECHAR v4 Persona Markdown asset as inert character text."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CEVoiceError(f"invalid CECHAR v4 Persona Markdown '{name}'") from error
    if not text.strip() or "\0" in text or "```" in text:
        raise CEVoiceError(f"invalid CECHAR v4 Persona Markdown '{name}'")
    if _V4_FORBIDDEN_PERSONA_FIELD.search(text):
        raise CEVoiceError(f"unsupported CECHAR v4 Persona declaration in '{name}'")
    if re.search(r"(?im)^\s*(?:[$>]\s*)?(?:powershell|cmd(?:\.exe)?|bash|sh)\b", text):
        raise CEVoiceError(f"executable CECHAR v4 Persona content in '{name}'")


def _is_sha256(value: object) -> bool:
    """Return whether a value is a hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _read_header(stream: BinaryIO) -> tuple[bytes, int, int]:
    raw_header = stream.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise CEVoiceError("truncated CEVOICE header")
    magic, version, metadata_length = HEADER.unpack(raw_header)
    return magic, version, metadata_length


def _read_source(source: Union[bytes, str, Path]) -> bytes:
    if isinstance(source, bytes):
        return source
    return Path(source).read_bytes()


def _validate_metadata(
    path: Path, metadata: ManifestValue, payload_offset: int
) -> None:
    if not isinstance(metadata, dict):
        raise CEVoiceError("metadata root must be an object")
    format_name = metadata.get("format")
    manifest_version = metadata.get("version")
    if not (
        (
            format_name == FORMAT_NAME
            and isinstance(manifest_version, int)
            and manifest_version in COMPATIBLE_CECHAR_VERSIONS
        )
        or (format_name, manifest_version) == (LEGACY_FORMAT_NAME, LEGACY_VERSION)
    ):
        raise CEVoiceError("metadata format/version mismatch")

    voices = metadata.get("voices")
    if not isinstance(voices, dict):
        raise CEVoiceError("metadata voices must be an object")

    default_voice = metadata.get("default_voice")
    if default_voice is not None and default_voice not in voices:
        raise CEVoiceError("metadata default_voice must name a defined voice")

    voice_order = metadata.get("voice_order")
    if voice_order is not None:
        if not isinstance(voice_order, list) or not all(
            isinstance(voice, str) for voice in voice_order
        ):
            raise CEVoiceError("metadata voice_order must be a list of voice names")
        if len(set(voice_order)) != len(voice_order):
            raise CEVoiceError("metadata voice_order must not contain duplicates")
        if any(voice not in voices for voice in voice_order):
            raise CEVoiceError("metadata voice_order must only name defined voices")
        voice_order.extend(voice for voice in voices if voice not in voice_order)

    theme = metadata.get("theme")
    if theme is not None:
        if not isinstance(theme, dict):
            raise CEVoiceError("metadata theme must be an object")
        if (
            theme.get("faded_accent") is None
            and theme.get("sleeping_color") is not None
        ):
            theme["faded_accent"] = str(theme.get("sleeping_color", "#9c88ce"))
        for key in ("background", "accent", "glow_color", "faded_accent"):
            value = theme.get(key)
            if key in {"glow_color", "faded_accent"} and value is None:
                continue
            if not _is_hex_color(value):
                raise CEVoiceError(f"metadata theme '{key}' must be a hex color")

    persona = metadata.get("persona")
    if persona is not None:
        _validate_persona_metadata(persona)
    bundle_assets = metadata.get("assets")
    if bundle_assets is not None:
        _validate_bundle_assets_metadata(bundle_assets)

    payload_length = path.stat().st_size - payload_offset
    for voice, voice_data in voices.items():
        if not isinstance(voice, str) or not isinstance(voice_data, dict):
            raise CEVoiceError("invalid voice entry")
        cfg_scale = voice_data.get("cfg_scale")
        if cfg_scale is not None and (
            not isinstance(cfg_scale, (int, float))
            or isinstance(cfg_scale, bool)
            or cfg_scale <= 0
        ):
            raise CEVoiceError(f"voice '{voice}' cfg_scale must be a positive number")
        reference_text = voice_data.get("reference_text")
        if reference_text is not None and (
            not isinstance(reference_text, str) or not reference_text.strip()
        ):
            raise CEVoiceError(
                f"voice '{voice}' reference_text must be a non-empty string"
            )
        voice_persona = voice_data.get("persona")
        if voice_persona is not None:
            _validate_persona_metadata(voice_persona)
        assets = voice_data.get("assets")
        if not isinstance(assets, dict):
            raise CEVoiceError(f"voice '{voice}' assets must be an object")
        if "wav" in assets and reference_text is None:
            raise CEVoiceError(
                f"voice '{voice}' reference_text is required when a wav asset is present"
            )
        for kind, asset in assets.items():
            if not isinstance(kind, str) or not isinstance(asset, dict):
                raise CEVoiceError(f"invalid asset entry for voice '{voice}'")
            if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
                raise CEVoiceError("invalid voice name")
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind for voice '{voice}'")
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            offset = asset.get("offset")
            length = asset.get("length")
            digest = asset.get("sha256")
            if (
                not isinstance(offset, int)
                or offset < 0
                or not isinstance(length, int)
                or length < 0
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdefABCDEF" for character in digest
                )
            ):
                raise CEVoiceError(f"invalid asset metadata for voice '{voice}'")
            if offset + length > payload_length:
                raise CEVoiceError(
                    f"asset '{kind}' for voice '{voice}' exceeds payload"
                )
    for name, asset in cast(Manifest, metadata.get("assets", {})).items():
        _validate_bundle_asset_entry(name, asset, payload_length)


def _is_hex_color(value: ManifestValue) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 7
        and value.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in value[1:])
    )


def _validate_optional_string(
    value: ManifestValue,
    field_name: str,
) -> None:
    """Validate one optional metadata field that must be a string."""
    if value is not None and not isinstance(value, str):
        raise CEVoiceError(f"{field_name} must be a string")


def _validate_string_list_or_text(
    value: ManifestValue,
    field_name: str,
) -> None:
    """Validate one optional metadata field that may be text or text lines."""
    if value is None or isinstance(value, str):
        return
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return
    raise CEVoiceError(f"{field_name} must be a string or list of strings")


def _validate_bundle_asset_entry(
    name: str,
    asset: ManifestValue,
    payload_length: int,
) -> None:
    """Validate one top-level bundle asset manifest entry."""
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise CEVoiceError(f"invalid bundle asset name '{name}'")
    if (
        Path(name).suffix.lower() != ".md"
        or Path(name).name not in SUPPORTED_PERSONA_FILENAMES
    ):
        raise CEVoiceError(f"unsupported bundle asset '{name}'")
    if not isinstance(asset, dict):
        raise CEVoiceError(f"invalid bundle asset '{name}'")
    offset = asset.get("offset")
    length = asset.get("length")
    digest = asset.get("sha256")
    if (
        not isinstance(offset, int)
        or offset < 0
        or not isinstance(length, int)
        or length < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ):
        raise CEVoiceError(f"invalid bundle asset '{name}'")
    if offset + length > payload_length:
        raise CEVoiceError(f"bundle asset '{name}' exceeds payload")


def _validate_bundle_assets_metadata(value: ManifestValue) -> None:
    """Validate the top-level bundle asset table."""
    if not isinstance(value, dict):
        raise CEVoiceError("metadata assets must be an object")


def _validate_persona_metadata(persona: ManifestValue) -> None:
    """Validate the optional CEVOICE persona metadata block."""
    if not isinstance(persona, dict):
        raise CEVoiceError("metadata persona must be an object")

    identity = persona.get("identity")
    if identity is not None:
        if not isinstance(identity, dict):
            raise CEVoiceError("metadata persona identity must be an object")
        for key in ("name", "age", "gender", "profile"):
            _validate_optional_string(
                cast(Manifest, identity).get(key),
                f"metadata persona identity '{key}'",
            )

    _validate_optional_string(persona.get("profile"), "metadata persona 'profile'")
    _validate_optional_string(
        persona.get("speaking_style"),
        "metadata persona 'speaking_style'",
    )
    _validate_string_list_or_text(
        persona.get("boundaries"),
        "metadata persona 'boundaries'",
    )
    _validate_string_list_or_text(
        persona.get("prompt_rules"),
        "metadata persona 'prompt_rules'",
    )
    _validate_string_list_or_text(
        persona.get("example_dialogue"),
        "metadata persona 'example_dialogue'",
    )
    style = persona.get("style")
    if style is not None:
        if not isinstance(style, dict):
            raise CEVoiceError("metadata persona style must be an object")
        for key in (
            "warmth",
            "directness",
            "humor",
            "detail",
            "formality",
            "enthusiasm",
        ):
            _validate_optional_string(
                cast(Manifest, style).get(key),
                f"metadata persona style '{key}'",
            )


def _text_tuple(value: ManifestValue) -> tuple[str, ...]:
    """Normalize one text or text-list manifest value into a tuple."""
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        lines = [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
        return tuple(lines)
    return ()


def persona_metadata_from_manifest(
    metadata: Mapping[str, ManifestValue],
) -> Optional[CEVoicePersona]:
    """Return typed persona metadata from a CEVOICE manifest when present.

    Args:
        metadata: The CEVOICE/CECHAR package manifest.

    Returns:
        Optional[CEVoicePersona]: The Persona metadata from the current CEVOICE/CECHAR package.
    """
    raw_persona = metadata.get("persona")
    if not isinstance(raw_persona, dict):
        return None

    return _persona_from_manifest(cast(Manifest, raw_persona))


def persona_metadata_from_voice(
    bundle: CEVoice,
    voice: Optional[str],
) -> Optional[CEVoicePersona]:
    """Return optional Persona style metadata attached to one voice entry.

    Args:
        bundle: The active CEVOICE/CECHAR package.
        voice: The active voice name.

    Returns:
        Optional[CEVoicePersona]: The voice-specific Persona metadata when present.
    """
    if not isinstance(voice, str) or not voice.strip():
        return None
    voice_data = bundle.voices.get(voice)
    if not isinstance(voice_data, dict):
        return None
    raw_persona = voice_data.get("persona")
    if not isinstance(raw_persona, dict):
        return None
    return _persona_from_manifest(cast(Manifest, raw_persona))


def merge_persona_metadata(
    base: Optional[CEVoicePersona],
    voice_override: Optional[CEVoicePersona],
) -> Optional[CEVoicePersona]:
    """Layer one voice's Persona style metadata on top of the base persona.

    Args:
        base: The shared character Persona metadata.
        voice_override: Optional voice-specific Persona metadata.

    Returns:
        Optional[CEVoicePersona]: The effective Persona metadata for the voice.
    """
    if base is None:
        return voice_override
    if voice_override is None:
        return base

    def merge_lines(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*first, *second)))

    base_style = base.style
    voice_style = voice_override.style
    return CEVoicePersona(
        identity=base.identity,
        speaking_style="\n\n".join(
            value
            for value in (
                base.speaking_style.strip(),
                voice_override.speaking_style.strip(),
            )
            if value
        ),
        boundaries=merge_lines(base.boundaries, voice_override.boundaries),
        prompt_rules=merge_lines(base.prompt_rules, voice_override.prompt_rules),
        example_dialogue=merge_lines(
            base.example_dialogue,
            voice_override.example_dialogue,
        ),
        style=PersonaStyleValues(
            warmth=voice_style.warmth or base_style.warmth,
            directness=voice_style.directness or base_style.directness,
            humor=voice_style.humor or base_style.humor,
            detail=voice_style.detail or base_style.detail,
            formality=voice_style.formality or base_style.formality,
            enthusiasm=voice_style.enthusiasm or base_style.enthusiasm,
        ),
    )


def _persona_from_manifest(raw_persona: Manifest) -> CEVoicePersona:
    """Build typed Persona metadata from one validated manifest object."""

    raw_identity = raw_persona.get("identity")
    identity = cast(Manifest, raw_identity) if isinstance(raw_identity, dict) else {}
    raw_style = raw_persona.get("style")
    style = cast(Manifest, raw_style) if isinstance(raw_style, dict) else {}

    profile = raw_persona.get("profile")
    identity_profile = identity.get("profile")
    return CEVoicePersona(
        identity=PersonaIdentity(
            name=_manifest_text(identity.get("name")),
            age=_manifest_text(identity.get("age")),
            gender=_manifest_text(identity.get("gender")),
            profile=_manifest_text(identity_profile or profile),
        ),
        speaking_style=_manifest_text(raw_persona.get("speaking_style")),
        boundaries=_text_tuple(raw_persona.get("boundaries")),
        prompt_rules=_text_tuple(raw_persona.get("prompt_rules")),
        example_dialogue=_text_tuple(raw_persona.get("example_dialogue")),
        style=PersonaStyleValues(
            warmth=_manifest_text(style.get("warmth")),
            directness=_manifest_text(style.get("directness")),
            humor=_manifest_text(style.get("humor")),
            detail=_manifest_text(style.get("detail")),
            formality=_manifest_text(style.get("formality")),
            enthusiasm=_manifest_text(style.get("enthusiasm")),
        ),
    )


def persona_files_from_manifest(
    metadata: Mapping[str, ManifestValue],
) -> dict[str, str]:
    """Return legacy whitelisted persona Markdown content embedded in metadata.

    Args:
        metadata: The CEVOICE/CECHAR package manifest.

    Returns:
        dict[str, str]: Supported persona filenames mapped to trimmed content.
    """
    persona_files: ManifestValue = metadata.get("persona_files")
    if not isinstance(persona_files, dict):
        raw_persona = metadata.get("persona")
        if isinstance(raw_persona, dict):
            persona_files = cast(Manifest, raw_persona).get("files")

    if not isinstance(persona_files, dict):
        return {}

    files = cast(Manifest, persona_files)
    return {
        filename: cast(str, files[filename]).strip()
        for filename in SUPPORTED_PERSONA_FILENAMES
        if isinstance(files.get(filename), str) and cast(str, files[filename]).strip()
    }


def persona_files_from_bundle(bundle: CEVoice) -> dict[str, str]:
    """Return whitelisted CECHAR Persona Markdown stored as bundle assets.

    Args:
        bundle: The CEVOICE/CECHAR package to inspect.

    Returns:
        dict[str, str]: Supported persona filenames mapped to decoded UTF-8 text.
    """
    files: dict[str, str] = {}
    for filename in SUPPORTED_PERSONA_FILENAMES:
        if filename not in bundle.assets:
            continue
        try:
            text = bundle.read_bundle_asset(filename).decode("utf-8").strip()
        except (UnicodeDecodeError, CEVoiceError, KeyError):
            continue
        if text:
            files[filename] = text

    if files:
        return files
    return persona_files_from_manifest(bundle.metadata)


def bundle_character_name(bundle: CEVoice) -> Optional[str]:
    """Return the active character name implied by one CEVOICE/CECHAR package.

    Args:
        bundle: The CEVOICE/CECHAR package to use.

    Returns:
        Optional[str]: The character name from the current CEVOICE/CECHAR package.
    """
    persona = persona_metadata_from_manifest(bundle.metadata)
    if persona is not None and persona.identity.name.strip():
        return persona.identity.name.strip()

    name = bundle.metadata.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _manifest_text(value: ManifestValue) -> str:
    """Return a manifest value only when it is meaningful text."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


_DEFAULT_LOADER: Optional[CEVoiceLoader] = None
_DEFAULT_LOADER_INITIALIZED = False
_DEFAULT_LOADER_ANNOUNCED = False
_DEFAULT_LOADER_FAILED = False
_SELECTED_BUNDLE: Optional[Path] = None
_SELECTED_BUNDLE_IS_NAMED = False
_DEFAULT_LOADER_FELL_BACK_FROM: Optional[Path] = None
_PROTECTED_TEMP_PATHS: set[Path] = set()
_PROTECTED_TEMP_PATHS_LOCK = threading.RLock()


def register_protected_temp_path(path: Union[str, Path]) -> Path:
    """Register one live temp path that Celune cleanup must not delete.

    Args:
        path: The live temp path to protect.

    Returns:
        Path: The normalized protected path.
    """
    resolved = Path(path).resolve()
    with _PROTECTED_TEMP_PATHS_LOCK:
        _PROTECTED_TEMP_PATHS.add(resolved)
    return resolved


def unregister_protected_temp_path(path: Union[str, Path]) -> None:
    """Remove one previously protected temp path from cleanup protection.

    Args:
        path: The temp path to unprotect.
    """
    resolved = Path(path).resolve()
    with _PROTECTED_TEMP_PATHS_LOCK:
        _PROTECTED_TEMP_PATHS.discard(resolved)


def is_protected_temp_path(path: Union[str, Path]) -> bool:
    """Return whether one temp path is protected from Celune cleanup.

    Args:
        path: The temp path to check.

    Returns:
        bool: ``True`` when the path is registered directly or nested under one that is.
    """
    resolved = Path(path).resolve()
    with _PROTECTED_TEMP_PATHS_LOCK:
        protected_paths = tuple(_PROTECTED_TEMP_PATHS)

    for protected in protected_paths:
        if resolved == protected:
            return True
        with contextlib.suppress(ValueError):
            resolved.relative_to(protected)
            return True
    return False


def default_bundle_path() -> Path:
    """Find where Celune's default voice bundle is located.

    Returns:
        Path: The absolute path to Celune's default voice bundle.
    """
    return bundled_voices_dir() / "default.cevoice"


def bundle_sha256(path: Union[str, Path]) -> str:
    """Return the SHA-256 checksum of one CEVOICE/CECHAR bundle file.

    Args:
        path: The bundle file to hash.

    Returns:
        str: The lowercase hexadecimal SHA-256 checksum for the bundle.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_matches_default_pack_checksum(path: Union[str, Path]) -> bool:
    """Return whether one bundle matches Celune's canonical default pack bytes.

    Args:
        path: The bundle file to compare against Celune's bundled default pack checksum.

    Returns:
        bool: ``True`` when the file checksum matches the canonical default CEVOICE pack.
    """
    try:
        return bundle_sha256(path) == DEFAULT_CEVOICE_PACK_SHA256
    except OSError:
        return False


def bundled_voices_dir() -> Path:
    """Return the user-local directory that stores bundled voice packs.

    Returns:
        Path: The absolute path to the user-local CEVOICE directory.
    """
    user_directory = voices_data_dir()
    if user_directory.is_dir():
        return user_directory

    repository_directory = project_root() / "voices"
    if not repository_directory.is_dir():
        return user_directory

    try:
        shutil.copytree(repository_directory, user_directory)
    except OSError:
        return repository_directory
    return user_directory


def resolve_bundle_path(bundle: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a configured CEVOICE/CECHAR package name or path.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or ``None`` to select Celune's default bundle.

    Returns:
        Path: The resolved CEVOICE/CECHAR package path.
    """
    if bundle is None:
        return default_bundle_path()

    candidate = Path(bundle).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate

    if candidate.suffix.lower() != ".cevoice":
        candidate = candidate.with_suffix(".cevoice")
    return bundled_voices_dir() / candidate


def select_voice_bundle(bundle: Optional[Union[str, Path]] = None) -> Path:
    """Select the CEVOICE/CECHAR package used by Celune's shared loader.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or ``None`` to restore Celune's default bundle.

    Returns:
        Path: The selected CEVOICE/CECHAR package path.
    """
    global _DEFAULT_LOADER, _DEFAULT_LOADER_INITIALIZED
    global _DEFAULT_LOADER_ANNOUNCED, _DEFAULT_LOADER_FAILED, _SELECTED_BUNDLE
    global _SELECTED_BUNDLE_IS_NAMED, _DEFAULT_LOADER_FELL_BACK_FROM

    selected = resolve_bundle_path(bundle)
    if selected == active_bundle_path():
        return selected

    if _DEFAULT_LOADER is not None:
        _DEFAULT_LOADER.close()

    _DEFAULT_LOADER = None
    _DEFAULT_LOADER_INITIALIZED = False
    _DEFAULT_LOADER_ANNOUNCED = False
    _DEFAULT_LOADER_FAILED = False
    _SELECTED_BUNDLE = selected
    _DEFAULT_LOADER_FELL_BACK_FROM = None
    if bundle is None:
        _SELECTED_BUNDLE_IS_NAMED = False
    else:
        candidate = Path(bundle).expanduser()
        _SELECTED_BUNDLE_IS_NAMED = (
            not candidate.is_absolute() and candidate.parent == Path(".")
        )
    return selected


def active_bundle_path() -> Path:
    """Return the currently selected CEVOICE/CECHAR package path.

    Returns:
        Path: The selected bundle path, or Celune's default bundle path.
    """
    return _SELECTED_BUNDLE or default_bundle_path()


def default_loader() -> Optional[CEVoiceLoader]:
    """Check if a default CEVOICE/CECHAR package can be loaded and return the loader.

    Returns:
        Optional[CEVoiceLoader]: The default CEVOICE/CECHAR package loader.
    """
    global _DEFAULT_LOADER, _DEFAULT_LOADER_INITIALIZED, _DEFAULT_LOADER_FAILED
    global _DEFAULT_LOADER_FELL_BACK_FROM
    if not _DEFAULT_LOADER_INITIALIZED:
        _DEFAULT_LOADER_INITIALIZED = True
        path = active_bundle_path()
        if not path.exists():
            fallback_path = default_bundle_path()
            if not _SELECTED_BUNDLE_IS_NAMED and path != fallback_path:
                return None
            if path == fallback_path or not fallback_path.exists():
                _DEFAULT_LOADER_FAILED = True
                return None
            _DEFAULT_LOADER_FELL_BACK_FROM = path
            path = fallback_path

        try:
            bundle = CEVoice.open(path)
        except (OSError, CEVoiceError):
            _DEFAULT_LOADER_FAILED = True
            return None

        _DEFAULT_LOADER = CEVoiceLoader(bundle)

    return _DEFAULT_LOADER


def close_default_loader() -> None:
    """Close and reset the shared CEVOICE/CECHAR loader when one is active."""
    global _DEFAULT_LOADER, _DEFAULT_LOADER_INITIALIZED
    global _DEFAULT_LOADER_ANNOUNCED, _DEFAULT_LOADER_FAILED
    global _DEFAULT_LOADER_FELL_BACK_FROM

    loader = _DEFAULT_LOADER
    _DEFAULT_LOADER = None
    _DEFAULT_LOADER_INITIALIZED = False
    _DEFAULT_LOADER_ANNOUNCED = False
    _DEFAULT_LOADER_FAILED = False
    _DEFAULT_LOADER_FELL_BACK_FROM = None
    if loader is not None:
        loader.close()


def announce_default_bundle(log: Callable[[str, str], None]) -> Optional[str]:
    """Log the default bundle result once at the caller's chosen lifecycle point.

    Args:
        log: The logging callback to the bound user interface.

    Returns:
        Optional[str]: The selected bundle's character name, or ``None`` if loading failed.
    """
    global _DEFAULT_LOADER_ANNOUNCED
    loader = default_loader()
    if _DEFAULT_LOADER_ANNOUNCED:
        return None

    if loader is not None:
        if _DEFAULT_LOADER_FELL_BACK_FROM is not None:
            log(
                "Voice pack "
                f"{_DEFAULT_LOADER_FELL_BACK_FROM.stem} not found, "
                "using default pack instead.",
                "warning",
            )
        name = loader.bundle.metadata.get("name", active_bundle_path().stem)
        if not isinstance(name, str):
            name = active_bundle_path().stem
        log(f"Loading voice pack: {name}", "info")
        _DEFAULT_LOADER_ANNOUNCED = True
        return name

    if _DEFAULT_LOADER_FAILED:
        log(
            "No compatible voice pack could be loaded.",
            "warning",
        )
        _DEFAULT_LOADER_ANNOUNCED = True
        return None

    return None
