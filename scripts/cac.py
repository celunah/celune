# SPDX-License-Identifier: Apache-2.0
"""Create a character pack for use in Celune."""

import io
import sys
import argparse
from math import gcd
from typing import TYPE_CHECKING, Union, Optional, TypedDict, cast
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

if TYPE_CHECKING:
    from celune.typing.cevoice import ManifestValue

try:
    from celune.i18n import string
    from celune.cevoice import (
        SINGING_VOICE_NAME,
        write_cevoice,
        write_cechar_v4,
        write_cechar_v5,
    )
    from celune.exceptions import CEVoiceError
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from celune.i18n import string
    from celune.cevoice import (
        SINGING_VOICE_NAME,
        write_cevoice,
        write_cechar_v4,
        write_cechar_v5,
    )
    from celune.exceptions import CEVoiceError


class ThemeMetadata(TypedDict, total=False):
    """Optional CEVOICE theme metadata."""

    background: str
    accent: str
    glow_color: str
    faded_accent: str


class PersonaIdentityMetadata(TypedDict, total=False):
    """Optional CEVOICE persona identity metadata."""

    name: str
    age: str
    gender: str
    profile: str


class PersonaStyleMetadata(TypedDict, total=False):
    """Optional CEVOICE persona style metadata."""

    warmth: str
    directness: str
    humor: str
    detail: str
    formality: str
    enthusiasm: str


class PersonaMetadata(TypedDict, total=False):
    """Optional CEVOICE persona metadata."""

    identity: PersonaIdentityMetadata
    profile: str
    speaking_style: str
    boundaries: Union[str, list[str]]
    prompt_rules: Union[str, list[str]]
    example_dialogue: Union[str, list[str]]
    style: PersonaStyleMetadata


class VoiceEntryMetadata(TypedDict, total=False):
    """Optional per-voice CEVOICE metadata."""

    cfg_scale: float
    reference_text: str
    display_name: str
    persona: PersonaMetadata
    cesn: str
    range: dict[str, int]


class BundleMetadata(TypedDict, total=False):
    """Optional CEVOICE bundle metadata."""

    name: str
    description: str
    default_voice: str
    voice_order: list[str]
    theme: ThemeMetadata
    persona: PersonaMetadata


class VoiceAssets(TypedDict, total=False):
    """Voice asset sources accepted by ``write_cevoice``."""

    wav: Union[Path, bytes]
    pt: Path


class CharacterData(TypedDict):
    """Collected wizard data ready for bundle creation."""

    output_path: Path
    voices: dict[str, VoiceAssets]
    metadata: BundleMetadata
    voice_metadata: dict[str, VoiceEntryMetadata]


DEFAULT_SIMPLE_VOICE_NAME = "Standard"
REFERENCE_SAMPLE_RATE = 24000


def ask(prompt: str) -> str:
    """Ask a question and return the value provided by the user."""
    print(prompt)
    return input("> ").strip()


def ask_required_text(prompt: str, error: str) -> str:
    """Repeatedly ask until the user provides non-empty text."""
    while True:
        value = ask(prompt)
        if value:
            return value
        print(error)


def ask_optional_text(prompt: str) -> Optional[str]:
    """Ask for optional text and normalize empty input to ``None``."""
    value = ask(prompt)
    return value or None


def ask_optional_text_list(prompt: str) -> Optional[list[str]]:
    """Ask for an optional ``|``-separated text list."""
    value = ask(prompt)
    if not value:
        return None
    parts = [item.strip() for item in value.split("|")]
    items = [item for item in parts if item]
    return items or None


def ask_positive_int(prompt: str) -> int:
    """Ask for a positive integer."""
    while True:
        raw_value = ask(prompt)
        try:
            value = int(raw_value)
        except ValueError:
            print(string("cac.integer_required"))
            continue
        if value <= 0:
            print(string("cac.positive_required"))
            continue
        return value


def ask_optional_float(prompt: str) -> Optional[float]:
    """Ask for an optional positive float."""
    while True:
        raw_value = ask(prompt)
        if not raw_value:
            return None
        try:
            value = float(raw_value)
        except ValueError:
            print(string("cac.number_optional"))
            continue
        if value <= 0:
            print(string("cac.positive_optional"))
            continue
        return value


def ask_existing_file(prompt: str, missing_error: str) -> Path:
    """Ask for a required file path and ensure it exists."""
    while True:
        raw_value = ask(prompt)
        if not raw_value:
            print(missing_error)
            continue
        path = Path(raw_value).expanduser()
        if not path.is_file():
            print(string("cac.file_missing", path=path))
            continue
        return path


def ask_optional_existing_file(prompt: str) -> Optional[Path]:
    """Ask for an optional file path and ensure it exists when provided."""
    while True:
        raw_value = ask(prompt)
        if not raw_value:
            return None
        path = Path(raw_value).expanduser()
        if not path.is_file():
            print(string("cac.file_missing", path=path))
            continue
        return path


def build_output_path(bundle_name: str) -> Path:
    """Return the output bundle path for the current character."""
    return Path(f"{bundle_name}.cevoice")


def normalize_reference_wav_asset(
    source_path: Path, canonical: bool = False
) -> Union[Path, bytes]:
    """Return 24 kHz mono PCM16 audio, rewriting headers for indexed packs.

    Args:
        source_path: The reference WAV to read.
        canonical: Always re-encode as canonical PCM WAV for CECHAR v4/v5.

    Returns:
        Union[Path, bytes]: The original compatible legacy reference or encoded WAV bytes.
    """
    audio, sample_rate = sf.read(source_path, dtype="float32")
    source_info = sf.info(source_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    needs_resample = sample_rate != REFERENCE_SAMPLE_RATE
    needs_remix = source_info.channels != 1

    if not needs_resample and not needs_remix and not canonical:
        return source_path

    if needs_resample:
        factor = gcd(sample_rate, REFERENCE_SAMPLE_RATE)
        audio = np.asarray(
            resample_poly(
                audio,
                up=REFERENCE_SAMPLE_RATE // factor,
                down=sample_rate // factor,
            ),
            dtype=np.float32,
        )

    output_buffer = io.BytesIO()
    sf.write(
        output_buffer,
        audio,
        REFERENCE_SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    normalized_bytes = output_buffer.getvalue()

    print(string("cac.normalized", path=source_path))
    return normalized_bytes


def build_simple_character_data(
    bundle_name: str,
    wav_path: Union[str, Path],
    reference_text: str,
) -> CharacterData:
    """Build a minimal single-voice CEVOICE payload from one name, WAV path, and transcript."""
    normalized_name = bundle_name.strip()
    if not normalized_name:
        raise ValueError(string("cac.name_required"))

    resolved_wav = Path(wav_path).expanduser()
    if not resolved_wav.is_file():
        raise ValueError(string("cac.file_missing", path=resolved_wav))
    normalized_reference_text = reference_text.strip()
    if not normalized_reference_text:
        raise ValueError(string("cac.transcript_required"))

    return {
        "output_path": build_output_path(normalized_name),
        "voices": {DEFAULT_SIMPLE_VOICE_NAME: {"wav": resolved_wav}},
        "metadata": {
            "name": normalized_name,
            "default_voice": DEFAULT_SIMPLE_VOICE_NAME,
            "voice_order": [DEFAULT_SIMPLE_VOICE_NAME],
        },
        "voice_metadata": {
            DEFAULT_SIMPLE_VOICE_NAME: {
                "reference_text": normalized_reference_text,
            }
        },
    }


def collect_theme_metadata() -> Optional[ThemeMetadata]:
    """Collect optional CEVOICE theme metadata."""
    theme: ThemeMetadata = {}
    for key, prompt in (
        ("background", string("cac.theme_background")),
        ("accent", string("cac.theme_accent")),
        ("glow_color", string("cac.theme_glow")),
        ("faded_accent", string("cac.theme_faded")),
    ):
        value = ask_optional_text(prompt)
        if value is not None:
            theme[key] = value
    return theme or None


def collect_persona_metadata() -> Optional[PersonaMetadata]:
    """Collect optional CEVOICE persona metadata."""
    persona: PersonaMetadata = {}

    identity: PersonaIdentityMetadata = {}
    for key, prompt in (
        ("name", string("cac.identity_name")),
        ("age", string("cac.identity_age")),
        ("gender", string("cac.identity_gender")),
        ("profile", string("cac.identity_profile")),
    ):
        value = ask_optional_text(prompt)
        if value is not None:
            identity[key] = value
    if identity:
        persona["identity"] = identity

    for key, prompt in (
        ("profile", string("cac.persona_profile")),
        ("speaking_style", string("cac.speaking_style")),
    ):
        value = ask_optional_text(prompt)
        if value is not None:
            persona[key] = value

    for key, prompt in (
        (
            "boundaries",
            string("cac.boundaries"),
        ),
        (
            "prompt_rules",
            string("cac.prompt_rules"),
        ),
        (
            "example_dialogue",
            string("cac.examples"),
        ),
    ):
        value = ask_optional_text_list(prompt)
        if value is not None:
            persona[key] = value

    style: PersonaStyleMetadata = {}
    for key in (
        "warmth",
        "directness",
        "humor",
        "detail",
        "formality",
        "enthusiasm",
    ):
        value = ask_optional_text(string("cac.persona_style", style=key))
        if value is not None:
            style[key] = value
    if style:
        persona["style"] = style

    return persona or None


def collect_voice_persona_metadata(voice_name: str) -> Optional[PersonaMetadata]:
    """Collect style overrides layered onto the shared Persona for one voice."""
    persona: PersonaMetadata = {}
    speaking_style = ask_optional_text(
        string("cac.voice_persona_speaking_style", voice_name=voice_name)
    )
    if speaking_style is not None:
        persona["speaking_style"] = speaking_style

    style: PersonaStyleMetadata = {}
    for key in (
        "warmth",
        "directness",
        "humor",
        "detail",
        "formality",
        "enthusiasm",
    ):
        value = ask_optional_text(
            string("cac.voice_persona_style", voice_name=voice_name, style=key)
        )
        if value is not None:
            style[key] = value
    if style:
        persona["style"] = style

    return persona or None


def collect_voice_assets(
    index: int, version: int = 3
) -> tuple[str, VoiceAssets, VoiceEntryMetadata]:
    """Collect one voice entry and its optional per-voice metadata."""
    voice_name = ask_required_text(
        string("cac.voice_name", index=index),
        string("cac.voice_name_required"),
    )
    wav_path = ask_existing_file(
        string("cac.wav_path", index=index),
        string("cac.wav_required"),
    )
    if version == 5 and voice_name == SINGING_VOICE_NAME:
        reference_text = ask_required_text(
            string("cac.sing_transcript"), string("cac.transcript_required")
        )
        cesn = ask_required_text(string("cac.sing_cesn"), string("cac.cesn_required"))
        pitches = {}
        for bound in ("min", "max"):
            while True:
                raw = ask(string("cac.sing_pitch", bound=bound))
                try:
                    pitch = int(raw)
                except ValueError:
                    print(string("cac.invalid_pitch"))
                    continue
                if 0 <= pitch <= 127 and (bound == "min" or pitch >= pitches["min"]):
                    pitches[bound] = pitch
                    break
                print(string("cac.invalid_pitch"))
        return (
            voice_name,
            {"wav": wav_path},
            {"reference_text": reference_text, "cesn": cesn, "range": pitches},
        )
    pt_path = ask_optional_existing_file(string("cac.embedding", index=index))
    cfg_scale = ask_optional_float(string("cac.cfg_scale", index=index))
    reference_text = ask_optional_text(string("cac.reference_text", index=index))
    display_name = ask_optional_text(string("cac.display_name", voice_name=voice_name))
    persona = collect_voice_persona_metadata(voice_name)

    assets: VoiceAssets = {"wav": wav_path}
    if pt_path is not None:
        assets["pt"] = pt_path

    metadata: VoiceEntryMetadata = {}
    if cfg_scale is not None:
        metadata["cfg_scale"] = cfg_scale
    if reference_text is not None:
        metadata["reference_text"] = reference_text
    if display_name is not None:
        metadata["display_name"] = display_name
    if persona is not None:
        metadata["persona"] = persona

    return voice_name, assets, metadata


def collect_voice_order(voice_names: list[str]) -> Optional[list[str]]:
    """Collect an optional explicit voice order."""
    order_text = ask_optional_text(string("cac.voice_order"))
    if order_text is None:
        return None

    requested = [item.strip() for item in order_text.split(",") if item.strip()]
    if not requested:
        print(string("cac.empty_order"))
        return None
    duplicates = {name for name in requested if requested.count(name) > 1}
    if duplicates:
        raise ValueError(
            string("cac.duplicate_order", names=", ".join(sorted(duplicates)))
        )
    unknown = [name for name in requested if name not in voice_names]
    if unknown:
        raise ValueError(string("cac.unknown_order", names=", ".join(unknown)))
    return requested


def collect_default_voice(voice_names: list[str]) -> Optional[str]:
    """Collect an optional default voice and validate that it exists."""
    default_voice = ask_optional_text(string("cac.default_voice"))
    if default_voice is None:
        return None
    if default_voice not in voice_names:
        raise ValueError(
            string(
                "cac.unknown_default", voice=default_voice, names=", ".join(voice_names)
            )
        )
    return default_voice


def collect_character_data(version: int = 3) -> CharacterData:
    """Run the CEVOICE wizard and collect validated bundle data."""
    bundle_name = ask_required_text(
        string("cac.name_prompt"),
        string("cac.name_required"),
    )
    description = ask_optional_text(string("cac.description_prompt"))
    voice_count = ask_positive_int(string("cac.voice_count"))

    voices: dict[str, VoiceAssets] = {}
    voice_metadata: dict[str, VoiceEntryMetadata] = {}
    for index in range(1, voice_count + 1):
        voice_name, assets, entry_metadata = collect_voice_assets(index, version)
        if voice_name in voices:
            raise ValueError(string("cac.duplicate_voice", voice_name=voice_name))
        voices[voice_name] = assets
        if entry_metadata:
            voice_metadata[voice_name] = entry_metadata

    voice_names = [
        name for name in voices if version != 5 or name != SINGING_VOICE_NAME
    ]
    default_voice = collect_default_voice(voice_names)
    voice_order = collect_voice_order(voice_names)
    theme = collect_theme_metadata()
    persona = collect_persona_metadata()

    metadata: BundleMetadata = {"name": bundle_name}
    if description is not None:
        metadata["description"] = description
    if default_voice is not None:
        metadata["default_voice"] = default_voice
    if voice_order is not None:
        metadata["voice_order"] = voice_order
    if theme is not None:
        metadata["theme"] = theme
    if persona is not None:
        metadata["persona"] = persona

    return {
        "output_path": build_output_path(bundle_name),
        "voices": voices,
        "metadata": metadata,
        "voice_metadata": voice_metadata,
    }


def create_cevoice(data: CharacterData, version: int = 3) -> Path:
    """Normalize reference audio and write the requested CECHAR generation.

    Args:
        data: Wizard output containing the destination, assets, and Persona metadata.
        version: CECHAR generation, either 3, 4, or 5; defaults to legacy v3.

    Returns:
        Path: The created pack path.

    Raises:
        ValueError: The requested generation is unsupported.
        CEVoiceError: The selected format rejects the supplied bundle data.
    """
    writers = {3: write_cevoice, 4: write_cechar_v4, 5: write_cechar_v5}
    if version not in writers:
        raise ValueError(f"unsupported CECHAR version {version}")
    normalized_voices: dict[str, dict[str, Union[bytes, str, Path]]] = {}
    for voice_name, assets in data["voices"].items():
        normalized_assets = cast(
            dict[str, Union[bytes, str, Path]],
            dict(assets),
        )
        wav_asset = normalized_assets.get("wav")
        if isinstance(wav_asset, Path):
            normalized_assets["wav"] = (
                normalize_reference_wav_asset(wav_asset, canonical=True)
                if version >= 4
                else normalize_reference_wav_asset(wav_asset)
            )
        normalized_voices[voice_name] = normalized_assets

    voices = cast(
        Mapping[str, Mapping[str, Union[bytes, str, Path]]],
        normalized_voices,
    )
    metadata = cast("Mapping[str, ManifestValue]", data["metadata"])
    voice_metadata = cast(
        "Mapping[str, Mapping[str, ManifestValue]]",
        data["voice_metadata"],
    )
    return writers[version](
        data["output_path"],
        voices,
        metadata,
        voice_metadata,
    )


def wizard(version: int = 3) -> None:
    """Run the CEVOICE creation wizard."""
    try:
        char_data = collect_character_data(version)
        print(string("cac.saving"))
        output_path = create_cevoice(char_data, version)
    except (ValueError, CEVoiceError) as error:
        print(f"{string('cac.error')} {error}")
        raise SystemExit(1) from error

    print(string("cac.saved", path=output_path))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the CEVOICE creator script."""
    parser = argparse.ArgumentParser(description=string("cac.description"))
    parser.add_argument("bundle_name", nargs="?", help=string("cac.name_help"))
    parser.add_argument("wav_path", nargs="?", help=string("cac.wav_help"))
    parser.add_argument(
        "reference_text",
        nargs="?",
        help=string("cac.transcript_help"),
    )
    parser.add_argument(
        "--format-version",
        type=int,
        choices=(3, 4, 5),
        default=3,
        help=string("cac.format_version_help"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the CEVOICE creator script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.bundle_name is None and args.wav_path is None:
        if args.format_version == 3:
            wizard()
        else:
            wizard(args.format_version)
        return 0

    if args.bundle_name is None or args.wav_path is None:
        parser.error(string("cac.simple_arguments"))

    try:
        reference_text = args.reference_text
        if reference_text is None:
            reference_text = ask_required_text(
                string("cac.transcript_prompt"),
                string("cac.transcript_required"),
            )
        char_data = build_simple_character_data(
            args.bundle_name,
            args.wav_path,
            reference_text,
        )
        print(string("cac.saving"))
        output_path = create_cevoice(char_data, args.format_version)
    except (ValueError, CEVoiceError) as error:
        print(f"{string('cac.error')} {error}")
        return 1

    print(string("cac.saved", path=output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
