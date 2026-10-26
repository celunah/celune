# SPDX-License-Identifier: Apache-2.0
"""CECHAR v4 indexed format implementation shared with v5."""

from __future__ import annotations

import io
import re
import gzip
import json
import lzma
import wave
import struct
import hashlib
from typing import Final, Union, Optional, cast
from pathlib import Path
from collections.abc import Mapping, Sequence

import numpy as np

from .base import (
    ALLOWED_ASSET_KINDS,
    SUPPORTED_ENTRY_SUFFIXES,
    SUPPORTED_PERSONA_FILENAMES,
    CEVoice,
    CEVoiceAsset,
    _read_source,
    _is_hex_color,
    _validate_persona_metadata,
)
from ..exceptions import CEVoiceError
from ..typing.cevoice import Manifest, ManifestValue, VoiceManifest

FORMAT_NAME: Final[str] = "CECHAR"
V4_MAGIC: Final[bytes] = b"CECHAR\0"
V4_VERSION: Final[int] = 4
INDEXED_CECHAR_VERSIONS: Final[frozenset[int]] = frozenset({V4_VERSION, 5})
V4_HEADER = struct.Struct("<7sBB")
V4_FILE_ENTRY = struct.Struct("<II")
V4_COMPRESSION_NONE: Final[int] = 0
V4_COMPRESSION_GZIP: Final[int] = 1
V4_COMPRESSION_ZSTANDARD: Final[int] = 2
V4_COMPRESSION_XZ: Final[int] = 3
V4_COMPRESSION_OPENZL: Final[int] = 4
V4_COMPRESSION_MODES: Final[frozenset[int]] = frozenset(range(5))
V4_MAX_DECOMPRESSED_BYTES: Final[int] = 0xFFFFFFFF
V4_ALLOWED_ASSET_SUFFIXES: Final[frozenset[str]] = SUPPORTED_ENTRY_SUFFIXES

_V4_FORBIDDEN_PERSONA_FIELD = re.compile(
    r"(?im)^\s*(?:agent|agents|capabilities|filesystem|network|process|shell|"
    r"tools?|permissions?|exec(?:ute)?|code)\s*[:=]"
)


def write_cechar_v4(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]] = None,
    compression: int = V4_COMPRESSION_NONE,
) -> Path:
    """Write a CECHAR v4 indexed archive.

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
        CEVoiceError: The package data cannot be represented as CECHAR v4.
    """
    return _write_indexed_cechar(
        path,
        voices,
        metadata,
        voice_metadata,
        bundle_assets,
        V4_VERSION,
        compression,
        None,
        None,
        None,
    )


def _write_indexed_cechar(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]],
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]],
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]],
    version: int,
    compression: int,
    entry_order: Optional[Sequence[str]],
    extra_assets: Optional[Mapping[str, bytes]],
    voice_asset_names: Optional[Mapping[tuple[str, str], str]],
) -> Path:
    """Write one version of the indexed CECHAR binary layout."""
    from .v5 import V5_VERSION, prepare_voice_metadata

    if compression not in V4_COMPRESSION_MODES:
        raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")

    unknown_voice_metadata = set(voice_metadata or {}) - set(voices)
    if unknown_voice_metadata:
        unknown = min(unknown_voice_metadata)
        raise CEVoiceError(f"voice metadata provided for unknown voice '{unknown}'")

    file_sources: dict[str, bytes] = {}
    natural_order: list[str] = []
    seen_names: set[str] = set()
    manifest_voices: VoiceManifest = {}
    for voice, assets in voices.items():
        _validate_v4_name(voice, "voice")
        manifest_assets: dict[str, ManifestValue] = {}
        for kind, source in assets.items():
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            name = (voice_asset_names or {}).get((voice, kind), f"{voice}.{kind}")
            data = _read_source(source)
            _validate_v4_file_data(name, data)

            folded = name.casefold()
            if folded in seen_names:
                raise CEVoiceError(f"{name} already exists")
            seen_names.add(folded)
            file_sources[name] = data
            natural_order.append(name)
            manifest_assets[kind] = name

        voice_entry = dict((voice_metadata or {}).get(voice, {}))
        if version == V5_VERSION:
            prepare_voice_metadata(voice, cast(Manifest, voice_entry))
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
        folded = name.casefold()
        if folded in seen_names:
            raise CEVoiceError(f"{name} already exists")
        seen_names.add(folded)
        file_sources[name] = data
        natural_order.append(name)
        manifest_bundle_assets[name] = name

    for name, data in (extra_assets or {}).items():
        _validate_v4_name(name, "file")
        _validate_v4_file_data(name, data)
        if Path(name).suffix.lower() not in V4_ALLOWED_ASSET_SUFFIXES:
            raise CEVoiceError(f"unsupported indexed CECHAR asset extension '{name}'")
        if (
            Path(name).suffix.lower() == ".md"
            and Path(name).name not in SUPPORTED_PERSONA_FILENAMES
        ):
            raise CEVoiceError(f"unsupported indexed CECHAR persona asset '{name}'")
        folded = name.casefold()
        if folded in seen_names:
            raise CEVoiceError(f"{name} already exists")
        seen_names.add(folded)
        file_sources[name] = data
        natural_order.append(name)

    names = natural_order if entry_order is None else list(entry_order)
    if len(names) != len(set(names)) or set(names) != set(file_sources):
        raise CEVoiceError("indexed CECHAR entry order does not match its files")
    payload = bytearray()
    file_entries: list[dict[str, str]] = []
    file_lengths: list[int] = []
    for name in names:
        data = file_sources[name]
        payload.extend(data)
        file_entries.append({"name": name, "sha256": hashlib.sha256(data).hexdigest()})
        file_lengths.append(len(data))

    manifest = dict(metadata or {})
    manifest["format"] = FORMAT_NAME
    manifest["version"] = version
    manifest["files"] = cast(
        ManifestValue,
        cast(list[ManifestValue], file_entries),
    )
    manifest["voices"] = cast(ManifestValue, manifest_voices)
    if manifest_bundle_assets:
        manifest["assets"] = cast(ManifestValue, manifest_bundle_assets)
    else:
        manifest.pop("assets", None)

    if version == V5_VERSION:
        _validate_v4_semantic_metadata(
            cast(Manifest, manifest),
            manifest_voices,
            version,
        )

    metadata_bytes = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(metadata_bytes) > 0xFFFFFFFF or len(file_entries) > 0xFFFFFFFF:
        raise CEVoiceError(f"CECHAR v{version} metadata exceeds uint32 limits")

    table_size = 4 + len(metadata_bytes) + 4 + V4_FILE_ENTRY.size * len(file_entries)
    if table_size + len(payload) > 0xFFFFFFFF:
        raise CEVoiceError(f"CECHAR v{version} payload exceeds uint32 limits")

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
        stream.write(V4_HEADER.pack(V4_MAGIC, version, compression))
        stream.write(stored_payload)
    return output_path


def open_indexed(
    cls: type[CEVoice],
    path: Path,
    header: bytes,
    stored_payload: bytes,
) -> CEVoice:
    """Open and validate an indexed CECHAR v4 or v5 archive."""
    magic, version, compression = V4_HEADER.unpack(header)
    if magic != V4_MAGIC:
        raise CEVoiceError("invalid indexed CECHAR magic")
    if version not in INDEXED_CECHAR_VERSIONS:
        raise CEVoiceError(f"unsupported CECHAR version {version}")
    if compression not in V4_COMPRESSION_MODES:
        raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")

    payload = _decompress_v4_payload(stored_payload, compression)
    metadata, asset_entries, bundle_asset_entries, file_entries = _parse_v4_payload(
        payload,
        version,
    )
    return cls(
        path,
        metadata,
        V4_HEADER.size,
        payload,
        asset_entries,
        bundle_asset_entries,
        file_entries,
        compression,
    )


def _validate_v4_file_data(name: str, data: bytes) -> None:
    """Validate one file before placing it into an indexed CECHAR archive."""
    suffix = Path(name).suffix.lower()
    if suffix not in V4_ALLOWED_ASSET_SUFFIXES:
        raise CEVoiceError(f"unsupported indexed CECHAR asset extension '{suffix}'")
    if suffix == ".md" and Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
        raise CEVoiceError(f"unsupported indexed CECHAR persona asset '{name}'")
    if suffix == ".wav":
        _validate_v4_wav(name, data)
    elif suffix == ".pt":
        _validate_v4_embedding(name, data)
    else:
        _validate_v4_persona_markdown(name, data)


def _compress_v4_payload(payload: bytes, compression: int) -> bytes:
    """Compress one logical indexed payload using its selected storage codec."""
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
        return context.compress(
            [zl.Input(zl.Type.Serial, np.frombuffer(payload, dtype=np.uint8))]
        )
    raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")


def _decompress_v4_payload(stored_payload: bytes, compression: int) -> bytes:
    """Decompress one complete indexed CECHAR stored payload."""

    def _check_size(payload: bytes) -> bytes:
        """Reject a logical payload larger than the indexed offset range."""
        if len(payload) > V4_MAX_DECOMPRESSED_BYTES:
            raise CEVoiceError("indexed CECHAR decompressed payload is too large")
        return payload

    try:
        if compression == V4_COMPRESSION_NONE:
            return _check_size(stored_payload)
        if compression == V4_COMPRESSION_GZIP:
            with gzip.GzipFile(fileobj=io.BytesIO(stored_payload)) as handle:
                return _check_size(handle.read(V4_MAX_DECOMPRESSED_BYTES + 1))
        if compression == V4_COMPRESSION_ZSTANDARD:
            try:
                import zstandard  # type: ignore[import-not-found]
            except ImportError as error:
                raise CEVoiceError("Zstandard compression is unavailable") from error
            return _check_size(
                zstandard.ZstdDecompressor().decompress(
                    stored_payload,
                    max_output_size=V4_MAX_DECOMPRESSED_BYTES + 1,
                )
            )
        if compression == V4_COMPRESSION_XZ:
            return _check_size(
                lzma.LZMADecompressor().decompress(
                    stored_payload,
                    max_length=V4_MAX_DECOMPRESSED_BYTES + 1,
                )
            )
        if compression == V4_COMPRESSION_OPENZL:
            try:
                import openzl.ext as zl  # type: ignore[import-not-found]
            except ImportError as error:
                raise CEVoiceError("OpenZL compression is unavailable") from error
            frame_info = zl.FrameInfo(stored_payload)
            if frame_info.num_outputs != 1:
                raise CEVoiceError("invalid OpenZL indexed CECHAR payload")
            if frame_info.output_content_size(0) > V4_MAX_DECOMPRESSED_BYTES:
                raise CEVoiceError("indexed CECHAR decompressed payload is too large")
            outputs = zl.DCtx().decompress(stored_payload)
            if len(outputs) != 1 or outputs[0].type != zl.Type.Serial:
                raise CEVoiceError("invalid OpenZL indexed CECHAR payload")
            return _check_size(outputs[0].content.as_bytes())
    except CEVoiceError:
        raise
    except Exception as error:
        raise CEVoiceError("invalid compressed indexed CECHAR payload") from error
    raise CEVoiceError(f"unsupported CECHAR compression mode {compression}")


def _parse_v4_payload(
    payload: bytes,
    expected_version: int,
) -> tuple[
    Manifest,
    dict[tuple[str, str], CEVoiceAsset],
    dict[str, CEVoiceAsset],
    dict[str, CEVoiceAsset],
]:
    """Parse, validate, and index one decompressed indexed CECHAR payload."""
    if len(payload) < 4:
        raise CEVoiceError("truncated indexed CECHAR metadata length")
    metadata_length = struct.unpack_from("<I", payload)[0]
    metadata_start = 4
    metadata_end = metadata_start + metadata_length
    if metadata_end > len(payload):
        raise CEVoiceError("truncated indexed CECHAR metadata")
    metadata_bytes = payload[metadata_start:metadata_end]
    if metadata_bytes.startswith(b"\xef\xbb\xbf"):
        raise CEVoiceError("indexed CECHAR metadata must not contain a byte-order mark")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CEVoiceError("invalid indexed CECHAR metadata") from error
    if not isinstance(metadata, dict):
        raise CEVoiceError("indexed CECHAR metadata root must be an object")
    if (
        metadata.get("format") != FORMAT_NAME
        or metadata.get("version") != expected_version
        or expected_version not in INDEXED_CECHAR_VERSIONS
    ):
        raise CEVoiceError("indexed CECHAR metadata format/version mismatch")

    file_list = metadata.get("files")
    if not isinstance(file_list, list):
        raise CEVoiceError("indexed CECHAR metadata files must be a list")
    file_count_offset = metadata_end
    if len(payload) < file_count_offset + 4:
        raise CEVoiceError("truncated indexed CECHAR file count")
    file_count = struct.unpack_from("<I", payload, file_count_offset)[0]
    if len(file_list) != file_count:
        raise CEVoiceError("indexed CECHAR metadata file count mismatch")

    jump_table_start = file_count_offset + 4
    jump_table_end = jump_table_start + V4_FILE_ENTRY.size * file_count
    if jump_table_end > len(payload):
        raise CEVoiceError("truncated indexed CECHAR jump table")

    file_names: list[str] = []
    file_digests: list[Optional[str]] = []
    for item in file_list:
        if not isinstance(item, dict):
            raise CEVoiceError("indexed CECHAR file entry must be an object")
        if "offset" in item or "length" in item:
            raise CEVoiceError(
                "indexed CECHAR file entries must not store physical offsets"
            )
        name = item.get("name")
        if not isinstance(name, str):
            raise CEVoiceError("indexed CECHAR file entry must contain a name")
        _validate_v4_name(name, "file")
        suffix = Path(name).suffix.lower()
        if suffix not in V4_ALLOWED_ASSET_SUFFIXES:
            raise CEVoiceError(f"unsupported indexed CECHAR asset extension '{suffix}'")
        if suffix == ".md" and Path(name).name not in SUPPORTED_PERSONA_FILENAMES:
            raise CEVoiceError(f"unsupported indexed CECHAR persona asset '{name}'")
        if name.casefold() in {existing.casefold() for existing in file_names}:
            raise CEVoiceError(f"duplicate indexed CECHAR file name '{name}'")
        digest = item.get("sha256")
        if digest is not None and not _is_sha256(digest):
            raise CEVoiceError(f"invalid indexed CECHAR checksum for '{name}'")
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
                f"indexed CECHAR file range for '{name}' is outside the payload"
            )
        end = offset + length
        ranges.append((offset, end, name))
        data = payload[offset:end]
        digest = hashlib.sha256(data).hexdigest()
        declared_digest = file_digests[index]
        if declared_digest is not None and digest.casefold() != declared_digest:
            raise CEVoiceError(f"checksum mismatch for indexed CECHAR file '{name}'")
        entries[name] = CEVoiceAsset(offset, length, digest)

    for first, second in zip(sorted(ranges), sorted(ranges)[1:]):
        if second[0] < first[1]:
            raise CEVoiceError(
                f"overlapping indexed CECHAR file ranges '{first[2]}' and '{second[2]}'"
            )

    _validate_v4_assets(payload, entries)
    normalized_metadata = _normalize_v4_metadata(metadata, entries)
    asset_entries, bundle_asset_entries = _index_v4_manifest(
        normalized_metadata,
        entries,
        expected_version,
    )
    return normalized_metadata, asset_entries, bundle_asset_entries, entries


def _normalize_v4_metadata(
    metadata: dict,
    entries: Mapping[str, CEVoiceAsset],
) -> Manifest:
    """Normalize indexed file references into the runtime manifest shape."""
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
        raise CEVoiceError("indexed CECHAR metadata voices must be an object")

    raw_assets = normalized.get("assets")
    if raw_assets is None:
        normalized["assets"] = cast(
            ManifestValue,
            {name: name for name in entries if Path(name).suffix.lower() == ".md"},
        )
    elif not isinstance(raw_assets, dict):
        raise CEVoiceError("indexed CECHAR metadata assets must be an object")
    return normalized


def _index_v4_manifest(
    metadata: Manifest,
    entries: Mapping[str, CEVoiceAsset],
    version: int,
) -> tuple[dict[tuple[str, str], CEVoiceAsset], dict[str, CEVoiceAsset]]:
    """Resolve indexed manifest references to validated jump-table entries."""
    indexed_voices: dict[tuple[str, str], CEVoiceAsset] = {}
    voices = metadata.get("voices")
    if not isinstance(voices, dict):
        raise CEVoiceError("indexed CECHAR metadata voices must be an object")
    voices = cast(dict[str, Manifest], voices)
    _validate_v4_semantic_metadata(metadata, voices, version)
    for voice, voice_data in voices.items():
        if not isinstance(voice, str) or not isinstance(voice_data, dict):
            raise CEVoiceError("invalid indexed CECHAR voice entry")
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
                raise CEVoiceError(f"missing indexed CECHAR asset '{name}'")
            indexed_voices[(voice, kind)] = entries[name]

    indexed_bundle_assets: dict[str, CEVoiceAsset] = {}
    raw_assets = metadata.get("assets")
    if not isinstance(raw_assets, dict):
        raise CEVoiceError("indexed CECHAR metadata assets must be an object")
    for name, reference in raw_assets.items():
        if not isinstance(name, str):
            raise CEVoiceError("invalid indexed CECHAR bundle asset name")
        referenced_name = _v4_reference_name(reference, f"bundle asset '{name}'")
        if referenced_name != name or referenced_name not in entries:
            raise CEVoiceError(f"missing indexed CECHAR bundle asset '{name}'")
        indexed_bundle_assets[name] = entries[referenced_name]
    return indexed_voices, indexed_bundle_assets


def _validate_v4_semantic_metadata(
    metadata: Manifest,
    voices: Mapping[str, Manifest],
    version: int,
) -> None:
    """Validate indexed metadata fields that describe runtime voice selection."""
    from .v5 import V5_VERSION, complete_voice_order, validate_global_semantics

    default_voice = metadata.get("default_voice")
    if default_voice is not None and (
        not isinstance(default_voice, str) or default_voice not in voices
    ):
        raise CEVoiceError("metadata default_voice must name a defined voice")
    if version == V5_VERSION:
        validate_global_semantics(metadata, voices)

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
        if version == V5_VERSION:
            complete_voice_order(order, voices)
        else:
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
        display_name = voice_data.get("display_name")
        if display_name is not None and (
            not isinstance(display_name, str) or not display_name.strip()
        ):
            raise CEVoiceError(
                f"voice '{voice}' display_name must be a non-empty string"
            )
        voice_persona = voice_data.get("persona")
        if voice_persona is not None:
            _validate_persona_metadata(voice_persona)


def _v4_reference_name(value: ManifestValue, field_name: str) -> str:
    """Read an indexed logical file name without physical table fields."""
    name: Optional[str] = None
    if isinstance(value, str):
        name = value
    elif isinstance(value, dict):
        if "offset" in value or "length" in value:
            raise CEVoiceError(f"{field_name} must not store physical offsets")
        candidate = value.get("file")
        if not isinstance(candidate, str):
            candidate = value.get("name")
        if isinstance(candidate, str):
            name = candidate
    if not isinstance(name, str):
        raise CEVoiceError(f"{field_name} must reference a file name")
    _validate_v4_name(name, field_name)
    return name


def _validate_v4_name(value: ManifestValue, field_name: str) -> None:
    """Validate one indexed CECHAR logical name against traversal and null-byte rules."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise CEVoiceError(f"invalid indexed CECHAR {field_name} name")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise CEVoiceError(f"invalid indexed CECHAR {field_name} name '{value}'")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise CEVoiceError(f"invalid indexed CECHAR {field_name} name '{value}'")


def _validate_v4_assets(
    payload: bytes,
    entries: Mapping[str, CEVoiceAsset],
) -> None:
    """Validate each indexed asset according to its extension."""
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
    """Validate one canonical indexed CECHAR PCM WAV asset."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise CEVoiceError(f"invalid indexed CECHAR WAV asset '{name}'")
    fmt_data: Optional[bytes] = None
    has_data_chunk = False
    cursor = 12
    while cursor + 8 <= len(data):
        chunk_id = data[cursor : cursor + 4]
        chunk_length = struct.unpack_from("<I", data, cursor + 4)[0]
        chunk_start = cursor + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(data):
            raise CEVoiceError(f"invalid indexed CECHAR WAV asset '{name}'")
        if chunk_id == b"fmt " and fmt_data is None:
            fmt_data = data[chunk_start:chunk_end]
        if chunk_id == b"data":
            has_data_chunk = True
        cursor = chunk_end + (chunk_length & 1)
    if fmt_data is None or len(fmt_data) < 16 or not has_data_chunk:
        raise CEVoiceError(f"invalid indexed CECHAR WAV asset '{name}'")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
        "<HHIIHH", fmt_data[:16]
    )
    if (audio_format, channels, sample_rate, bits) != (1, 1, 24000, 16):
        raise CEVoiceError(f"invalid indexed CECHAR WAV properties for '{name}'")
    if (
        byte_rate != sample_rate * channels * bits // 8
        or block_align != channels * bits // 8
    ):
        raise CEVoiceError(f"invalid indexed CECHAR WAV properties for '{name}'")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getframerate() != 24000
                or wav_file.getsampwidth() != 2
                or wav_file.getcomptype() != "NONE"
            ):
                raise CEVoiceError(
                    f"invalid indexed CECHAR WAV properties for '{name}'"
                )
    except (EOFError, wave.Error) as error:
        raise CEVoiceError(f"invalid indexed CECHAR WAV asset '{name}'") from error


def _validate_v4_embedding(name: str, data: bytes) -> None:
    """Validate one restricted tensor-only indexed CECHAR voice embedding."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise CEVoiceError("PyTorch not found, can't validate .pt assets") from error
    try:
        value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise CEVoiceError(f"invalid indexed CECHAR PT asset '{name}'") from error
    if isinstance(value, dict):
        if set(value) != {"speaker_embedding"}:
            raise CEVoiceError(f"invalid indexed CECHAR PT keys for '{name}'")
        value = value["speaker_embedding"]
    if not isinstance(value, torch.Tensor):
        raise CEVoiceError(f"invalid indexed CECHAR PT asset '{name}'")
    if (
        value.dtype != torch.float32
        or value.numel() != 2048
        or value.ndim not in {1, 2}
    ):
        raise CEVoiceError(f"invalid indexed CECHAR PT shape or dtype for '{name}'")
    if value.ndim == 2 and value.shape != (1, 2048):
        raise CEVoiceError(f"invalid indexed CECHAR PT shape for '{name}'")
    if value.ndim == 1 and value.shape != (2048,):
        raise CEVoiceError(f"invalid indexed CECHAR PT shape for '{name}'")
    if not bool(torch.isfinite(value).all()):
        raise CEVoiceError(f"invalid indexed CECHAR PT values for '{name}'")


def _validate_v4_persona_markdown(name: str, data: bytes) -> None:
    """Validate an indexed CECHAR Persona Markdown asset as inert character text."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CEVoiceError(
            f"invalid indexed CECHAR Persona Markdown '{name}'"
        ) from error
    if not text.strip() or "\0" in text or "```" in text:
        raise CEVoiceError(f"invalid indexed CECHAR Persona Markdown '{name}'")
    if _V4_FORBIDDEN_PERSONA_FIELD.search(text):
        raise CEVoiceError(
            f"unsupported indexed CECHAR Persona declaration in '{name}'"
        )
    if re.search(r"(?im)^\s*(?:[$>]\s*)?(?:powershell|cmd(?:\.exe)?|bash|sh)\b", text):
        raise CEVoiceError(f"executable indexed CECHAR Persona content in '{name}'")


def _is_sha256(value: ManifestValue) -> bool:
    """Return whether a value is a hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )
