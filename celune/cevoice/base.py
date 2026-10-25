# SPDX-License-Identifier: Apache-2.0
"""CEVOICE/CECHAR bundle writer, parser, and lazy file loader."""

from __future__ import annotations

import os
import copy
import json
import atexit
import shutil
import struct
import hashlib
import tempfile
import threading
import contextlib
from typing import Final, Union, BinaryIO, Optional, cast
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Callable, Sequence

from ..i18n import string
from ..paths import project_root, temp_data_dir, voices_data_dir
from ..exceptions import CEVoiceError
from ..typing.cevoice import Manifest, ManifestValue, VoiceManifest
from .registry import KNOWN_DEFAULT_CEVOICE_PACKS, OFFICIAL_CEVOICE_PACK_SHA256S

HEADER = struct.Struct("<8sHI")
ALLOWED_ASSET_KINDS = {"wav", "pt"}
SUPPORTED_ENTRY_SUFFIXES: Final[frozenset[str]] = frozenset({".wav", ".pt", ".md"})
SUPPORTED_PERSONA_FILENAMES: Final[tuple[str, ...]] = (
    "identity.md",
    "soul.md",
    "personality.md",
    "speech_style.md",
    "boundaries.md",
    "examples.md",
)


@dataclass(frozen=True)
class CEVoiceAsset:
    """One binary asset stored inside a CEVOICE/CECHAR package."""

    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class CEVoiceEntry:
    """One named file stored in a CEVOICE/CECHAR package."""

    name: str
    size: int


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
    _file_entries: Mapping[str, CEVoiceAsset] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _compression: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def is_indexed(self) -> bool:
        """Return whether the pack uses the indexed CECHAR payload layout."""
        return self._payload is not None

    @property
    def compression(self) -> Optional[int]:
        """Return the indexed payload compression mode, when the pack has one."""
        return self._compression

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
        from . import v4

        bundle_path = Path(path)
        with bundle_path.open("rb") as stream:
            prefix = stream.read(v4.V4_HEADER.size)
            if (
                len(prefix) >= len(v4.V4_MAGIC)
                and prefix[: len(v4.V4_MAGIC)] == v4.V4_MAGIC
                and len(prefix) >= v4.V4_HEADER.size
                and prefix[7] != 0
            ):
                return cls._open_indexed(bundle_path, prefix, stream.read())
            stream.seek(0)
            magic, version, metadata_length = _read_header(stream)
            if not _is_known_legacy_magic(magic):
                raise CEVoiceError("invalid CEVOICE magic")
            if not _is_supported_legacy_header(magic, version):
                raise CEVoiceError(f"unsupported CEVOICE version {version}")

            metadata_bytes = stream.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise CEVoiceError("truncated CEVOICE metadata")

            try:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CEVoiceError("invalid CEVOICE metadata") from error

            if not isinstance(metadata, dict):
                raise CEVoiceError("metadata root must be an object")
            if metadata.get("version") != version:
                raise CEVoiceError("metadata format/version mismatch")

            payload_offset = HEADER.size + metadata_length
            _validate_metadata(bundle_path, metadata, payload_offset)

        return cls(bundle_path, metadata, payload_offset)

    @classmethod
    def _open_indexed(
        cls,
        path: Path,
        header: bytes,
        stored_payload: bytes,
    ) -> CEVoice:
        """Open an indexed CECHAR archive after its fixed header was identified."""
        from .v4 import open_indexed

        return open_indexed(cls, path, header, stored_payload)

    @property
    def voices(self) -> VoiceManifest:
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
    def voice_order(self) -> tuple[str, ...]:
        """Return the preferred user-facing voice order.

        Returns:
            tuple[str, ...]: The preferred user-facing voice order.
        """
        from .v5 import V5_VERSION, voice_order

        if self.metadata.get("version") == V5_VERSION:
            return voice_order(self.metadata)
        order = self.metadata.get("voice_order")
        if isinstance(order, list) and all(isinstance(voice, str) for voice in order):
            voices = cast(list[str], order)
        else:
            voices = list(self.voices)
        return tuple(voices)

    @property
    def singing_voice(self) -> Optional[str]:
        """Return the reserved singing voice name when present.

        Returns:
            Optional[str]: The reserved singing voice name for CECHAR v5 packs.
        """
        from .v5 import singing_voice

        return singing_voice(self)

    @property
    def singing_pitch_range(self) -> Optional[tuple[int, int]]:
        """Return the singing voice's MIDI range, or the legacy full range."""
        from .v5 import singing_pitch_range

        return singing_pitch_range(self)

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
    def assets(self) -> Manifest:
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

    @property
    def entries(self) -> tuple[CEVoiceEntry, ...]:
        """Return named entries in the order stored by the container.

        Returns:
            tuple[CEVoiceEntry, ...]: The ordered file names and byte sizes.
        """
        asset_entries = self._entry_asset_map()
        if self._file_entries:
            names = list(self._file_entries)
        else:
            names = sorted(
                asset_entries,
                key=lambda name: asset_entries[name].offset,
            )
        folded_names: set[str] = set()
        result: list[CEVoiceEntry] = []
        for name in names:
            folded = name.casefold()
            if folded in folded_names:
                raise CEVoiceError(f"duplicate CEVOICE entry name '{name}'")
            folded_names.add(folded)
            result.append(CEVoiceEntry(name, asset_entries[name].length))
        return tuple(result)

    def read_entry(self, name: str) -> bytes:
        """Read and checksum one named container entry.

        Args:
            name: The exact name of the entry to read.

        Returns:
            bytes: The validated entry contents.

        Raises:
            KeyError: The container has no entry with this name.
            CEVoiceError: The entry payload is truncated or has an invalid checksum.
        """
        try:
            asset = self._entry_asset_map()[name]
        except KeyError as error:
            raise KeyError(f"bundle entry '{name}' not found") from error
        data = self._read_asset_bytes(asset)
        if len(data) != asset.length:
            raise CEVoiceError(f"truncated bundle entry '{name}'")
        if hashlib.sha256(data).hexdigest().casefold() != asset.sha256.casefold():
            raise CEVoiceError(f"checksum mismatch for bundle entry '{name}'")
        return data

    def _entry_asset_map(self) -> dict[str, CEVoiceAsset]:
        """Return file names mapped to validated payload locations."""
        if self._file_entries:
            return dict(self._file_entries)
        result: dict[str, CEVoiceAsset] = {}
        for voice, voice_data in self.voices.items():
            assets = cast(dict[str, Manifest], voice_data["assets"])
            for kind in assets:
                result[f"{voice}.{kind}"] = self.asset(voice, kind)
        for name in self.assets:
            result[name] = self.bundle_asset(name)
        return result

    def _read_asset_bytes(self, asset: CEVoiceAsset) -> bytes:
        """Read an asset from either a legacy payload or an indexed buffer."""
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


def _write_legacy_cevoice(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]],
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]],
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]],
    magic: bytes,
    version: int,
    entry_order: Optional[Sequence[str]],
) -> Path:
    """Write one legacy CEVOICE layout while retaining its original generation."""
    format_name = _legacy_format_for_header(magic, version)
    if format_name is None:
        raise CEVoiceError("unsupported legacy CEVOICE generation")

    manifest_voices: VoiceManifest = {}
    asset_names: dict[tuple[str, str], str] = {}
    source_data: dict[str, bytes] = {}
    natural_order: list[str] = []
    unknown_voice_metadata = set(voice_metadata or {}) - set(voices)
    if unknown_voice_metadata:
        unknown = min(unknown_voice_metadata)
        raise CEVoiceError(f"voice metadata provided for unknown voice '{unknown}'")

    for voice, assets in voices.items():
        if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
            raise CEVoiceError(f"invalid voice name '{voice}'")
        manifest_assets: dict[str, ManifestValue] = {}
        for kind, source in assets.items():
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind for voice '{voice}'")
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(
                    f"unsupported asset kind '{kind}' for voice '{voice}'"
                )
            name = f"{voice}.{kind}"
            data = _read_source(source)
            source_data[name] = data
            natural_order.append(name)
            asset_names[(voice, kind)] = name
            manifest_assets[kind] = {}
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
        source_data[name] = _read_source(source)
        natural_order.append(name)
        manifest_bundle_assets[name] = {}

    names = natural_order if entry_order is None else list(entry_order)
    if len(names) != len(set(names)) or set(names) != set(source_data):
        raise CEVoiceError("CEVOICE entry order does not match its files")

    payload = bytearray()
    entry_metadata: dict[str, Manifest] = {}
    for name in names:
        data = source_data[name]
        entry_metadata[name] = {
            "offset": len(payload),
            "length": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        payload.extend(data)
    for (voice, kind), name in asset_names.items():
        manifest_assets = cast(
            dict[str, ManifestValue],
            manifest_voices[voice]["assets"],
        )
        manifest_assets[kind] = cast(ManifestValue, entry_metadata[name])
    for name in manifest_bundle_assets:
        manifest_bundle_assets[name] = entry_metadata[name]

    manifest = dict(metadata or {})
    manifest["format"] = format_name
    manifest["version"] = version
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

    output_path = Path(path)
    with output_path.open("wb") as stream:
        stream.write(HEADER.pack(magic, version, len(metadata_bytes)))
        stream.write(metadata_bytes)
        stream.write(payload)
    return output_path


def add_cevoice_entries(
    path: Union[str, Path],
    sources: Sequence[Union[str, Path]],
    position: Optional[int] = None,
) -> CEVoice:
    """Append or insert supported files into a validated CEVOICE pack.

    Args:
        path: The pack to update in place.
        sources: Files to add in their requested order.
        position: Optional 1-based insertion position; omit it to append.

    Returns:
        CEVoice: The reopened and validated updated pack.

    Raises:
        CEVoiceError: An input cannot be represented while keeping the pack valid.
        FileNotFoundError: The pack or one of the input files does not exist.
        OSError: A pack or input file cannot be read or written.
    """
    if not sources:
        raise CEVoiceError("at least one input file is required")
    bundle = CEVoice.open(path)
    entries = list(bundle.entries)
    insert_at = len(entries) if position is None else position - 1
    if position is not None and not 1 <= position <= len(entries) + 1:
        raise CEVoiceError("entry insertion position is out of range")

    metadata, voices, voice_metadata, bundle_assets, extra_assets = (
        _bundle_mutation_data(bundle)
    )
    voice_asset_names = _voice_entry_names(bundle)
    existing_names = {entry.name.casefold() for entry in entries}
    additions: list[tuple[str, str, str, str, bytes]] = []
    for source in sources:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        suffix = source_path.suffix.lower()
        if suffix not in SUPPORTED_ENTRY_SUFFIXES:
            raise CEVoiceError(f"unsupported CEVOICE file '{source_path.name}'")
        if suffix == ".md":
            name = source_path.name
            if name not in SUPPORTED_PERSONA_FILENAMES:
                raise CEVoiceError(f"unsupported CEVOICE Persona file '{name}'")
            category, voice, kind = "persona", "", ""
        else:
            voice = source_path.stem
            kind = suffix[1:]
            name = f"{voice}.{kind}"
            category = "voice"
        folded_name = name.casefold()
        if folded_name in existing_names:
            raise CEVoiceError(f"CEVOICE entry '{name}' already exists")
        existing_names.add(folded_name)
        additions.append((name, category, voice, kind, source_path.read_bytes()))

    for name, category, voice, kind, data in additions:
        if category == "persona":
            bundle_assets[name] = data
        else:
            voice_assets = voices.setdefault(voice, {})
            if kind in voice_assets:
                raise CEVoiceError(
                    f"voice '{voice}' already contains an asset of kind '{kind}'"
                )
            if kind == "wav":
                details = voice_metadata.get(voice)
                reference_text = details.get("reference_text") if details else None
                if not isinstance(reference_text, str) or not reference_text.strip():
                    raise CEVoiceError(
                        f"voice '{voice}' needs valid reference_text metadata for a WAV file"
                    )
            voice_assets[kind] = data
            voice_asset_names[(voice, kind)] = name

    names = [entry.name for entry in entries]
    names[insert_at:insert_at] = [addition[0] for addition in additions]
    _atomic_rewrite_cevoice(
        path,
        bundle,
        metadata,
        voices,
        voice_metadata,
        bundle_assets,
        extra_assets,
        names,
        voice_asset_names,
    )
    return CEVoice.open(path)


def remove_cevoice_entries(
    path: Union[str, Path],
    names: Sequence[str],
) -> CEVoice:
    """Remove named files from a validated CEVOICE pack.

    Args:
        path: The pack to update in place.
        names: Exact entry names to remove.

    Returns:
        CEVoice: The reopened and validated updated pack.

    Raises:
        CEVoiceError: A name is missing or removal would make the pack invalid.
        FileNotFoundError: The pack does not exist.
        OSError: The pack cannot be read or written.
    """
    if not names or len(names) != len(set(names)):
        raise CEVoiceError("entry names must be present and unique")
    bundle = CEVoice.open(path)
    entry_names = [entry.name for entry in bundle.entries]
    if any(name not in entry_names for name in names):
        missing = next(name for name in names if name not in entry_names)
        raise CEVoiceError(f"CEVOICE entry '{missing}' was not found")

    metadata, voices, voice_metadata, bundle_assets, extra_assets = (
        _bundle_mutation_data(bundle)
    )
    removed = set(names)
    voice_entry_names = _voice_entry_names(bundle)
    removed_voices: set[str] = set()
    for (voice, kind), name in voice_entry_names.items():
        if name not in removed:
            continue
        voices[voice].pop(kind, None)
        if not voices[voice]:
            removed_voices.add(voice)
    for voice in removed_voices:
        voices.pop(voice, None)
        voice_metadata.pop(voice, None)
    for name in removed:
        bundle_assets.pop(name, None)
        extra_assets.pop(name, None)

    if metadata.get("default_voice") in removed_voices:
        metadata.pop("default_voice", None)
    voice_order = metadata.get("voice_order")
    if isinstance(voice_order, list):
        metadata["voice_order"] = [
            voice for voice in voice_order if voice not in removed_voices
        ]

    retained_order = [name for name in entry_names if name not in removed]
    voice_asset_names = {
        key: name for key, name in voice_entry_names.items() if name not in removed
    }
    _atomic_rewrite_cevoice(
        path,
        bundle,
        metadata,
        voices,
        voice_metadata,
        bundle_assets,
        extra_assets,
        retained_order,
        voice_asset_names,
    )
    return CEVoice.open(path)


def _bundle_mutation_data(
    bundle: CEVoice,
) -> tuple[
    Manifest,
    dict[str, dict[str, bytes]],
    dict[str, Manifest],
    dict[str, bytes],
    dict[str, bytes],
]:
    """Build writer inputs while retaining semantic metadata and all entries."""
    from .v4 import _v4_reference_name

    data = {entry.name: bundle.read_entry(entry.name) for entry in bundle.entries}
    metadata = copy.deepcopy(bundle.metadata)
    raw_voices = cast(dict[str, Manifest], metadata.pop("voices", {}))
    metadata.pop("assets", None)
    metadata.pop("files", None)
    metadata.pop("format", None)
    metadata.pop("version", None)

    voices: dict[str, dict[str, bytes]] = {}
    voice_metadata: dict[str, Manifest] = {}
    referenced: set[str] = set()
    for voice, voice_data in raw_voices.items():
        raw_assets = cast(dict[str, ManifestValue], voice_data.get("assets", {}))
        voices[voice] = {}
        for kind, reference in raw_assets.items():
            name = (
                _v4_reference_name(reference, f"voice '{voice}' asset '{kind}'")
                if bundle.is_indexed
                else f"{voice}.{kind}"
            )
            voices[voice][kind] = data[name]
            referenced.add(name)
        voice_metadata[voice] = {
            key: value for key, value in voice_data.items() if key != "assets"
        }

    from .v5 import V5_VERSION, normalize_singing_metadata

    if bundle.metadata.get("version") == V5_VERSION:
        normalize_singing_metadata(voice_metadata)

    bundle_assets: dict[str, bytes] = {}
    raw_bundle_assets = cast(
        dict[str, ManifestValue], bundle.metadata.get("assets", {})
    )
    for name, reference in raw_bundle_assets.items():
        entry_name = (
            _v4_reference_name(reference, f"bundle asset '{name}'")
            if bundle.is_indexed
            else name
        )
        bundle_assets[name] = data[entry_name]
        referenced.add(entry_name)

    extra_assets = (
        {name: content for name, content in data.items() if name not in referenced}
        if bundle.is_indexed
        else {}
    )
    return metadata, voices, voice_metadata, bundle_assets, extra_assets


def _voice_entry_names(bundle: CEVoice) -> dict[tuple[str, str], str]:
    """Map voice asset references to their stored filenames."""
    from .v4 import _v4_reference_name

    result: dict[tuple[str, str], str] = {}
    for voice, voice_data in bundle.voices.items():
        assets = cast(dict[str, ManifestValue], voice_data["assets"])
        for kind, reference in assets.items():
            name = (
                _v4_reference_name(reference, f"voice '{voice}' asset '{kind}'")
                if bundle.is_indexed
                else f"{voice}.{kind}"
            )
            result[(voice, kind)] = name
    return result


def _atomic_rewrite_cevoice(
    path: Union[str, Path],
    original: CEVoice,
    metadata: Manifest,
    voices: Mapping[str, Mapping[str, bytes]],
    voice_metadata: Mapping[str, Mapping[str, ManifestValue]],
    bundle_assets: Mapping[str, bytes],
    extra_assets: Mapping[str, bytes],
    entry_order: Sequence[str],
    voice_asset_names: Mapping[tuple[str, str], str],
) -> None:
    """Write, validate, and atomically replace a CEVOICE file."""
    pack_path = Path(path).resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{pack_path.name}.",
        suffix=".tmp",
        dir=pack_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        if original.is_indexed:
            from .v4 import _write_indexed_cechar

            _write_indexed_cechar(
                temporary_path,
                voices,
                metadata,
                voice_metadata,
                bundle_assets,
                cast(int, original.metadata["version"]),
                cast(int, original.compression),
                entry_order,
                extra_assets,
                voice_asset_names,
            )
        else:
            format_name = original.metadata["format"]
            version = cast(int, original.metadata["version"])
            magic = _legacy_magic_for_manifest(format_name, version)
            if magic is None:
                raise CEVoiceError("unsupported legacy CEVOICE generation")
            _write_legacy_cevoice(
                temporary_path,
                voices,
                metadata,
                voice_metadata,
                bundle_assets,
                magic,
                version,
                entry_order,
            )
        rewritten = CEVoice.open(temporary_path)
        for entry in rewritten.entries:
            rewritten.read_entry(entry.name)
        os.chmod(temporary_path, pack_path.stat().st_mode & 0o777)
        os.replace(temporary_path, pack_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_header(stream: BinaryIO) -> tuple[bytes, int, int]:
    raw_header = stream.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise CEVoiceError("truncated CEVOICE header")
    magic, version, metadata_length = HEADER.unpack(raw_header)
    return magic, version, metadata_length


def _is_known_legacy_magic(magic: bytes) -> bool:
    from . import v1, v2, v3

    return magic in {v1.MAGIC, v2.MAGIC, v3.MAGIC}


def _is_supported_legacy_header(magic: bytes, version: int) -> bool:
    from . import v1, v2, v3

    return any(codec.supports_header(magic, version) for codec in (v1, v2, v3))


def _legacy_format_for_header(magic: bytes, version: int) -> Optional[str]:
    from . import v1, v2, v3

    for codec in (v1, v2, v3):
        if codec.supports_header(magic, version):
            return codec.FORMAT_NAME
    return None


def _legacy_magic_for_manifest(
    format_name: ManifestValue,
    version: ManifestValue,
) -> Optional[bytes]:
    from . import v1, v2, v3

    for codec in (v1, v2, v3):
        if codec.supports_manifest(format_name, version):
            return codec.MAGIC
    return None


def _is_supported_legacy_manifest(
    format_name: ManifestValue,
    version: ManifestValue,
) -> bool:
    from . import v1, v2, v3

    return any(codec.supports_manifest(format_name, version) for codec in (v1, v2, v3))


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
    if not _is_supported_legacy_manifest(format_name, manifest_version):
        raise CEVoiceError("metadata format/version mismatch")

    voices = metadata.get("voices")
    if not isinstance(voices, dict):
        raise CEVoiceError("metadata voices must be an object")

    default_voice = metadata.get("default_voice")
    if default_voice is not None and (
        not isinstance(default_voice, str) or default_voice not in voices
    ):
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


def bundle_display_name(bundle: CEVoice) -> str:
    """Return a user-facing name for one CEVOICE/CECHAR package.

    Args:
        bundle: The CEVOICE/CECHAR package to label.

    Returns:
        str: The character name with the official bundle variant when applicable.
    """
    name = bundle_character_name(bundle) or bundle.path.stem
    if bundle_matches_default_pack_checksum(bundle.path):
        checksum = bundle_sha256(bundle.path)
        variant = next(
            (
                Path(filename).stem
                for filename, checksums in KNOWN_DEFAULT_CEVOICE_PACKS.items()
                if checksum in checksums
            ),
            bundle.path.stem,
        )
        return string(
            "cevoice.official_bundle_label",
            character=name,
            variant=variant,
        )
    return name


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
    """Return whether one bundle matches an official Celune pack's bytes.

    Args:
        path: The bundle file to compare against known pack checksums.

    Returns:
        bool: ``True`` when the file checksum matches a registered default pack.
    """
    try:
        return bundle_sha256(path) in OFFICIAL_CEVOICE_PACK_SHA256S
    except OSError:
        return False


def bundled_voices_dir() -> Path:
    """Return the user-local directory that stores bundled voice packs.

    Returns:
        Path: The absolute path to the user-local CEVOICE directory.
    """
    user_directory = voices_data_dir()
    repository_directory = project_root() / "voices"
    if not repository_directory.is_dir():
        return user_directory

    if user_directory.is_dir():
        try:
            for repository_pack in repository_directory.glob("*.cevoice"):
                user_pack = user_directory / repository_pack.name
                if repository_pack.is_file() and (
                    not user_pack.is_file()
                    or bundle_sha256(user_pack) != bundle_sha256(repository_pack)
                ):
                    shutil.copy2(repository_pack, user_pack)
        except OSError:
            pass
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
        name = bundle_display_name(loader.bundle)
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
