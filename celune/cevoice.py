# SPDX-License-Identifier: MIT
"""CEVOICE/CECHAR bundle writer, parser, and lazy file loader."""

from __future__ import annotations

import json
import atexit
import shutil
import struct
import hashlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Final, Mapping, Optional, Union, cast

from .exceptions import CEVoiceError
from .constants import JSONSerializable
from .paths import temp_data_dir

MAGIC: Final[bytes] = b"CECHAR\0\0"
VERSION: Final[int] = 2
LEGACY_MAGIC: Final[bytes] = b"CEVOICE\0"
LEGACY_VERSION: Final[int] = 1
FORMAT_NAME: Final[str] = "CECHAR"
LEGACY_FORMAT_NAME: Final[str] = "CEVOICE"
HEADER = struct.Struct("<8sHI")
ALLOWED_ASSET_KINDS = {"wav", "pt"}
type ManifestValue = Union[JSONSerializable, "Manifest"]
type Manifest = dict[str, ManifestValue]
type VoiceManifest = dict[str, Manifest]


@dataclass(frozen=True)
class CEVoiceAsset:
    """One binary asset stored inside a CEVOICE package."""

    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class CEVoice:
    """Parsed CEVOICE package metadata and payload access."""

    path: Path
    metadata: Manifest
    payload_offset: int

    @classmethod
    def open(cls, path: Union[str, Path]) -> CEVoice:
        """Parse and validate a CEVOICE package.

        Args:
            path: The CEVOICE package to load.

        Returns:
            CEVoice: The CEVoice object.

        Raises:
            CEVoiceError: The CEVOICE package is malformed and could not be loaded.
        """
        bundle_path = Path(path)
        with bundle_path.open("rb") as stream:
            magic, version, metadata_length = _read_header(stream)
            if magic not in {MAGIC, LEGACY_MAGIC}:
                raise CEVoiceError("invalid CEVOICE magic")
            if (magic, version) not in {
                (MAGIC, VERSION),
                (LEGACY_MAGIC, LEGACY_VERSION),
            }:
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
    def voices(self) -> VoiceManifest:
        """Return the voice manifest.

        Returns:
            VoiceManifest: The voice manifest of this CEVOICE package.

        Raises:
            CEVoiceError: The CEVOICE package does not contain a valid voice manifest.
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


@dataclass(frozen=True, slots=True)
class PersonaIdentity:
    """Identity details supplied by a CEVOICE pack."""

    name: str = ""
    age: str = ""
    gender: str = ""
    profile: str = ""


@dataclass(frozen=True, slots=True)
class PersonaStyleValues:
    """Baseline speaking-style values supplied by a CEVOICE pack."""

    warmth: str = ""
    directness: str = ""
    humor: str = ""
    detail: str = ""
    formality: str = ""
    enthusiasm: str = ""


@dataclass(frozen=True, slots=True)
class CEVoicePersona:
    """Persona metadata supplied by a CEVOICE pack."""

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
            CEVoiceError: The CEVOICE package contains path delimiters.
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
        """Remove extracted temporary files."""
        shutil.rmtree(self._directory, ignore_errors=True)


def write_cevoice(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
) -> Path:
    """Write a CEVOICE package from per-voice binary assets.

    Args:
        path: The CEVOICE package to save as.
        voices: The voice files to bundle into this CEVOICE package.
        metadata: The metadata to bundle into this CEVOICE package.
        voice_metadata: Extra metadata stored beside each voice's assets.

    Returns:
        Path: The path to the created CEVOICE package.

    Raises:
        CEVoiceError: The CEVOICE package contains path delimiters.
    """
    payload = bytearray()
    manifest_voices: VoiceManifest = {}
    unknown_voice_metadata = set(voice_metadata or {}) - set(voices)
    if unknown_voice_metadata:
        unknown = sorted(unknown_voice_metadata)[0]
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

    manifest = dict(metadata or {})
    manifest["format"] = FORMAT_NAME
    manifest["version"] = VERSION
    manifest["voices"] = cast(ManifestValue, manifest_voices)
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
    if (format_name, manifest_version) not in {
        (FORMAT_NAME, VERSION),
        (LEGACY_FORMAT_NAME, LEGACY_VERSION),
    }:
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
        metadata: The CEVOICE package manifest.

    Returns:
        Optional[CEVoicePersona]: The Persona metadata from the current CEVOICE package.
    """
    raw_persona = metadata.get("persona")
    if not isinstance(raw_persona, dict):
        return None

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


def bundle_character_name(bundle: CEVoice) -> Optional[str]:
    """Return the active character name implied by one CEVOICE package.

    Args:
        bundle: The CEVOICE package to use.

    Returns:
        Optional[str]: The character name from the current CEVOICE package.
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


def default_bundle_path() -> Path:
    """Find where Celune's default voice bundle is located.

    Returns:
        Path: The absolute path to Celune's default voice bundle.
    """
    return Path(__file__).resolve().parent.parent / "voices" / "default.cevoice"


def bundled_voices_dir() -> Path:
    """Return the repository-level directory that stores bundled voice packs.

    Returns:
        Path: The absolute path to the bundled CEVOICE directory.
    """
    return Path(__file__).resolve().parent.parent / "voices"


def resolve_bundle_path(bundle: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a configured CEVOICE package name or path.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or ``None`` to select Celune's default bundle.

    Returns:
        Path: The resolved CEVOICE package path.
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
    """Select the CEVOICE package used by Celune's shared loader.

    Args:
        bundle: Either a built-in bundle name, an explicit bundle path, or ``None`` to restore Celune's default bundle.

    Returns:
        Path: The selected CEVOICE package path.
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
    """Return the currently selected CEVOICE package path.

    Returns:
        Path: The selected bundle path, or Celune's default bundle path.
    """
    return _SELECTED_BUNDLE or default_bundle_path()


def default_loader() -> Optional[CEVoiceLoader]:
    """Check if a default CEVOICE package can be loaded and return the loader.

    Returns:
        Optional[CEVoiceLoader]: The default CEVOICE package loader.
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
