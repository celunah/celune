# SPDX-License-Identifier: MIT
"""CEVOICE bundle writer, parser, and lazy file loader."""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Optional, Union

from .exceptions import CEVoiceError

MAGIC = b"CEVOICE\0"
VERSION = 1
HEADER = struct.Struct("<8sHI")
ALLOWED_ASSET_KINDS = {"wav", "pt"}


@dataclass(frozen=True)
class CEVoiceAsset:
    """One binary asset stored inside a CEVOICE bundle."""

    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class CEVoice:
    """Parsed CEVOICE bundle metadata and payload access."""

    path: Path
    metadata: dict[str, Any]
    payload_offset: int

    @classmethod
    def open(cls, path: Union[str, Path]) -> "CEVoice":
        """Parse and validate a CEVOICE bundle.

        Args:
            path: The CEVOICE bundle to load.

        Returns:
            CEVoice: The CEVoice object.

        Raises:
            CEVoiceError: The CEVOICE bundle is malformed and could not be loaded.
        """
        bundle_path = Path(path)
        with bundle_path.open("rb") as stream:
            magic, version, metadata_length = _read_header(stream)
            if magic != MAGIC:
                raise CEVoiceError("invalid CEVOICE magic")
            if version != VERSION:
                raise CEVoiceError(f"unsupported CEVOICE version {version}")

            metadata_bytes = stream.read(metadata_length)
            if len(metadata_bytes) != metadata_length:
                raise CEVoiceError("truncated CEVOICE metadata")

            try:
                metadata = json.loads(metadata_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CEVoiceError("invalid CEVOICE metadata") from error

            payload_offset = HEADER.size + metadata_length
            _validate_metadata(bundle_path, metadata, payload_offset)

        return cls(bundle_path, metadata, payload_offset)

    @property
    def voices(self) -> dict[str, dict[str, Any]]:
        """Return the voice manifest.

        Returns:
            dict[str, dict[str, Any]]: The voice manifest of this CEVOICE bundle.

        Raises:
            CEVoiceError: The CEVOICE bundle does not contain a valid voice manifest.
        """
        voices = self.metadata.get("voices")
        if not isinstance(voices, dict):
            raise CEVoiceError("metadata voices must be an object")
        return voices

    @property
    def voice_order(self) -> tuple[str, ...]:
        """Return the preferred user-facing voice order."""
        order = self.metadata.get("voice_order")
        if isinstance(order, list) and all(isinstance(voice, str) for voice in order):
            return tuple(order)
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
        try:
            raw_asset = self.voices[voice]["assets"][kind]
        except KeyError as error:
            raise KeyError(f"asset '{kind}' for voice '{voice}' not found") from error

        return CEVoiceAsset(
            offset=raw_asset["offset"],
            length=raw_asset["length"],
            sha256=raw_asset["sha256"],
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
        with self.path.open("rb") as stream:
            stream.seek(self.payload_offset + asset.offset)
            data = stream.read(asset.length)

        if len(data) != asset.length:
            raise CEVoiceError(f"truncated asset '{kind}' for voice '{voice}'")
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise CEVoiceError(
                f"checksum mismatch for asset '{kind}' of voice '{voice}'"
            )
        return data


class CEVoiceLoader:
    """Lazily materialize CEVOICE assets as real files for path-only consumers."""

    def __init__(self, bundle: CEVoice) -> None:
        self.bundle = bundle
        self._directory = Path(tempfile.mkdtemp(prefix="celune-cevoice-"))
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
        """
        key = (voice, kind)
        if key not in self._paths:
            if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
                raise CEVoiceError(f"invalid voice name '{voice}'")
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind '{kind}'")
            extension = suffix or f".{kind}"
            safe_voice = Path(voice).name
            path = self._directory / f"{safe_voice}{extension}"
            path.write_bytes(self.bundle.read_asset(voice, kind))
            self._paths[key] = path
        return self._paths[key]

    def close(self) -> None:
        """Remove extracted temporary files.

        Returns:
            None: This function cleans up temporary voice assets."""
        shutil.rmtree(self._directory, ignore_errors=True)


def write_cevoice(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a CEVOICE bundle from per-voice binary assets.

    Args:
        path: The CEVOICE bundle to save as.
        voices: The voice files to bundle into this CEVOICE bundle.
        metadata: The metadata to bundle into this CEVOICE bundle.

    Returns:
        Path: The path to the created CEVOICE bundle.
    """
    payload = bytearray()
    manifest_voices: dict[str, dict[str, Any]] = {}

    for voice, assets in voices.items():
        manifest_assets: dict[str, dict[str, Any]] = {}
        for kind, source in assets.items():
            data = _read_source(source)
            manifest_assets[kind] = {
                "offset": len(payload),
                "length": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            payload.extend(data)
        manifest_voices[voice] = {"assets": manifest_assets}

    manifest = dict(metadata or {})
    manifest["format"] = "CEVOICE"
    manifest["version"] = VERSION
    manifest["voices"] = manifest_voices
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


def _read_header(stream: BinaryIO) -> tuple[bytes, int, int]:
    raw_header = stream.read(HEADER.size)
    if len(raw_header) != HEADER.size:
        raise CEVoiceError("truncated CEVOICE header")
    magic, version, metadata_length = HEADER.unpack(raw_header)
    return magic, version, metadata_length


def _read_source(source: bytes | str | Path) -> bytes:
    if isinstance(source, bytes):
        return source
    return Path(source).read_bytes()


def _validate_metadata(path: Path, metadata: Any, payload_offset: int) -> None:
    if not isinstance(metadata, dict):
        raise CEVoiceError("metadata root must be an object")
    if metadata.get("format") != "CEVOICE" or metadata.get("version") != VERSION:
        raise CEVoiceError("metadata format/version mismatch")

    voices = metadata.get("voices")
    if not isinstance(voices, dict):
        raise CEVoiceError("metadata voices must be an object")

    default_voice = metadata.get("default_voice")
    if default_voice is not None and default_voice not in voices:
        raise CEVoiceError("metadata default_voice must name a defined voice")

    voice_order = metadata.get("voice_order")
    if voice_order is not None:
        if (
            not isinstance(voice_order, list)
            or not all(isinstance(voice, str) for voice in voice_order)
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
        for key in ("background", "accent", "glow_color"):
            value = theme.get(key)
            if key == "glow_color" and value is None:
                continue
            if not _is_hex_color(value):
                raise CEVoiceError(f"metadata theme '{key}' must be a hex color")

    payload_length = path.stat().st_size - payload_offset
    for voice, voice_data in voices.items():
        if not isinstance(voice, str) or not isinstance(voice_data, dict):
            raise CEVoiceError("invalid voice entry")
        assets = voice_data.get("assets")
        if not isinstance(assets, dict):
            raise CEVoiceError(f"voice '{voice}' assets must be an object")
        for kind, asset in assets.items():
            if not isinstance(kind, str) or not isinstance(asset, dict):
                raise CEVoiceError(f"invalid asset entry for voice '{voice}'")
            if "/" in voice or "\\" in voice or voice in {"", ".", ".."}:
                raise CEVoiceError("invalid voice name")
            if "/" in kind or "\\" in kind or kind in {"", ".", ".."}:
                raise CEVoiceError(f"invalid asset kind for voice '{voice}'")
            if kind not in ALLOWED_ASSET_KINDS:
                raise CEVoiceError(f"unsupported asset kind '{kind}' for voice '{voice}'")
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
                raise CEVoiceError(f"invalid asset metadata for voice '{voice}'")
            if offset + length > payload_length:
                raise CEVoiceError(
                    f"asset '{kind}' for voice '{voice}' exceeds payload"
                )


def _is_hex_color(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 7
        and value.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in value[1:])
    )


_DEFAULT_LOADER: Optional[CEVoiceLoader] = None
_DEFAULT_LOADER_INITIALIZED = False
_DEFAULT_LOADER_ANNOUNCED = False
_DEFAULT_LOADER_FAILED = False
_SELECTED_BUNDLE: Optional[Path] = None


def default_bundle_path() -> Path:
    """Find where Celune's default voice bundle is located.

    Returns:
        Path: The absolute path to Celune's default voice bundle.
    """
    return Path(__file__).resolve().parent / "voices" / "default.cevoice"


def resolve_bundle_path(bundle: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a configured CEVOICE bundle name or path.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or
            ``None`` to select Celune's default bundle.

    Returns:
        Path: The resolved CEVOICE bundle path.
    """
    if bundle is None:
        return default_bundle_path()

    candidate = Path(bundle).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate

    if candidate.suffix.lower() != ".cevoice":
        candidate = candidate.with_suffix(".cevoice")
    return Path(__file__).resolve().parent / "voices" / candidate


def select_voice_bundle(bundle: Optional[Union[str, Path]] = None) -> Path:
    """Select the CEVOICE bundle used by Celune's shared loader.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or
            ``None`` to restore Celune's default bundle.

    Returns:
        Path: The selected CEVOICE bundle path.
    """
    global _DEFAULT_LOADER, _DEFAULT_LOADER_INITIALIZED
    global _DEFAULT_LOADER_ANNOUNCED, _DEFAULT_LOADER_FAILED, _SELECTED_BUNDLE

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
    return selected


def active_bundle_path() -> Path:
    """Return the currently selected CEVOICE bundle path.

    Returns:
        Path: The selected bundle path, or Celune's default bundle path.
    """
    return _SELECTED_BUNDLE or default_bundle_path()


def default_loader() -> Optional[CEVoiceLoader]:
    """Check if a default CEVOICE bundle can be loaded and return the loader.

    Returns:
        Optional[CEVoiceLoader]: The default CEVOICE bundle loader.
    """
    global _DEFAULT_LOADER, _DEFAULT_LOADER_INITIALIZED, _DEFAULT_LOADER_FAILED
    if not _DEFAULT_LOADER_INITIALIZED:
        _DEFAULT_LOADER_INITIALIZED = True
        path = active_bundle_path()
        if not path.exists():
            return None

        try:
            bundle = CEVoice.open(path)
        except (OSError, CEVoiceError):
            _DEFAULT_LOADER_FAILED = True
            return None

        _DEFAULT_LOADER = CEVoiceLoader(bundle)

    return _DEFAULT_LOADER


def announce_default_bundle(log: Callable[[str, str], None]) -> Optional[str]:
    """Log the default bundle result once at the caller's chosen lifecycle point.

    Args:
        log: The logging callback to the bound user interface.

    Returns:
        Optional[str]: The selected bundle's character name, ``None`` if loading failed, ``"Celune"`` if
            a fallback reference was loaded.
    """
    global _DEFAULT_LOADER_ANNOUNCED
    loader = default_loader()
    if _DEFAULT_LOADER_ANNOUNCED:
        return None

    if loader is not None:
        name = loader.bundle.metadata.get("name", active_bundle_path().stem)
        log(f"Loading voice bundle: {name}", "info")
        _DEFAULT_LOADER_ANNOUNCED = True
        return name

    if _DEFAULT_LOADER_FAILED:
        log(
            "No voice bundles found or voice loading failed. "
            "Loading a default character from loose references instead.",
            "warning",
        )
        _DEFAULT_LOADER_ANNOUNCED = True

        # this is a fallback, and only Celune is available as old format voice data
        # it only triggers if no CEVOICE is loadable, but the old format refs still exist
        return "Celune"

    return None
