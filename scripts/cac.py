"""Create a CEVOICE pack for use in Celune."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict, Mapping, Union, Optional, cast

try:
    from celune.cevoice import write_cevoice, ManifestValue
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from celune.cevoice import write_cevoice


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

    wav: Path
    pt: Path


class CharacterData(TypedDict):
    """Collected wizard data ready for bundle creation."""

    output_path: Path
    voices: dict[str, VoiceAssets]
    metadata: BundleMetadata
    voice_metadata: dict[str, VoiceEntryMetadata]


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
            print("Please enter a whole number.")
            continue
        if value <= 0:
            print("Please enter a number greater than zero.")
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
            print("Please enter a number or leave it empty.")
            continue
        if value <= 0:
            print("Please enter a positive number or leave it empty.")
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
            print(f"File not found: {path}")
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
            print(f"File not found: {path}")
            continue
        return path


def build_output_path(bundle_name: str) -> Path:
    """Return the output bundle path for the current character."""
    return Path(f"{bundle_name}.cevoice")


def collect_theme_metadata() -> Optional[ThemeMetadata]:
    """Collect optional CEVOICE theme metadata."""
    theme: ThemeMetadata = {}
    for key, prompt in (
        ("background", "Theme background hex color (#RRGGBB), leave empty to skip"),
        ("accent", "Theme accent hex color (#RRGGBB), leave empty to skip"),
        ("glow_color", "Theme glow color (#RRGGBB), leave empty to skip"),
        ("faded_accent", "Theme faded accent color (#RRGGBB), leave empty to skip"),
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
        ("name", "Persona identity name, leave empty to skip"),
        ("age", "Persona identity age, leave empty to skip"),
        ("gender", "Persona identity gender, leave empty to skip"),
        ("profile", "Persona identity profile, leave empty to skip"),
    ):
        value = ask_optional_text(prompt)
        if value is not None:
            identity[key] = value
    if identity:
        persona["identity"] = identity

    for key, prompt in (
        ("profile", "Persona profile text, leave empty to skip"),
        ("speaking_style", "Persona speaking style, leave empty to skip"),
    ):
        value = ask_optional_text(prompt)
        if value is not None:
            persona[key] = value

    for key, prompt in (
        (
            "boundaries",
            "Persona boundaries, separate multiple entries with | and leave empty to skip",
        ),
        (
            "prompt_rules",
            "Persona prompt rules, separate multiple entries with | and leave empty to skip",
        ),
        (
            "example_dialogue",
            "Persona example dialogue, separate multiple entries with | and leave empty to skip",
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
        value = ask_optional_text(f"Persona style {key}, leave empty to skip")
        if value is not None:
            style[key] = value
    if style:
        persona["style"] = style

    return persona or None


def collect_voice_assets(index: int) -> tuple[str, VoiceAssets, VoiceEntryMetadata]:
    """Collect one voice entry and its optional per-voice metadata."""
    voice_name = ask_required_text(
        f"Enter voice name for entry #{index}",
        "Voice name is required.",
    )
    wav_path = ask_existing_file(
        f"Enter WAV path for entry #{index}",
        "A WAV path is required.",
    )
    pt_path = ask_optional_existing_file(
        f"Enter embedding file (.pt) for entry #{index}, leave empty to skip"
    )
    cfg_scale = ask_optional_float(
        f"Enter cfg_scale for entry #{index}, leave empty to skip"
    )
    reference_text = ask_optional_text(
        f"Enter reference_text for entry #{index}, leave empty to skip"
    )

    assets: VoiceAssets = {"wav": wav_path}
    if pt_path is not None:
        assets["pt"] = pt_path

    metadata: VoiceEntryMetadata = {}
    if cfg_scale is not None:
        metadata["cfg_scale"] = cfg_scale
    if reference_text is not None:
        metadata["reference_text"] = reference_text

    return voice_name, assets, metadata


def collect_voice_order(voice_names: list[str]) -> Optional[list[str]]:
    """Collect an optional explicit voice order."""
    order_text = ask_optional_text(
        "Preferred voice order as comma-separated names, leave empty to keep entry order"
    )
    if order_text is None:
        return None

    requested = [item.strip() for item in order_text.split(",") if item.strip()]
    if not requested:
        print("Voice order was empty after parsing, using entry order instead.")
        return None
    duplicates = {name for name in requested if requested.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Voice order contains duplicates: {', '.join(sorted(duplicates))}"
        )
    unknown = [name for name in requested if name not in voice_names]
    if unknown:
        raise ValueError(f"Voice order names unknown voices: {', '.join(unknown)}")
    return requested


def collect_default_voice(voice_names: list[str]) -> Optional[str]:
    """Collect an optional default voice and validate that it exists."""
    default_voice = ask_optional_text("Default voice name, leave empty to skip")
    if default_voice is None:
        return None
    if default_voice not in voice_names:
        raise ValueError(
            f"Default voice '{default_voice}' is not one of: {', '.join(voice_names)}"
        )
    return default_voice


def collect_character_data() -> CharacterData:
    """Run the CEVOICE wizard and collect validated bundle data."""
    bundle_name = ask_required_text(
        "What name is your character or voice pack?",
        "A character or voice-pack name is required.",
    )
    description = ask_optional_text("Bundle description, leave empty to skip")
    voice_count = ask_positive_int("How many voice tones do they have?")

    voices: dict[str, VoiceAssets] = {}
    voice_metadata: dict[str, VoiceEntryMetadata] = {}
    for index in range(1, voice_count + 1):
        voice_name, assets, entry_metadata = collect_voice_assets(index)
        if voice_name in voices:
            raise ValueError(f"Duplicate voice name: {voice_name}")
        voices[voice_name] = assets
        if entry_metadata:
            voice_metadata[voice_name] = entry_metadata

    voice_names = list(voices)
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


def create_cevoice(data: CharacterData) -> Path:
    """Create a CEVOICE bundle with the collected wizard data."""
    voices = cast(
        "Mapping[str, Mapping[str, Union[bytes, str, Path]]]",
        data["voices"],
    )
    metadata = cast("Mapping[str, ManifestValue]", data["metadata"])
    voice_metadata = cast(
        "Mapping[str, Mapping[str, ManifestValue]]",
        data["voice_metadata"],
    )
    return write_cevoice(
        data["output_path"],
        voices,
        metadata,
        voice_metadata,
    )


def wizard() -> None:
    """Run the CEVOICE creation wizard."""
    try:
        char_data = collect_character_data()
        print("Saving package...")
        output_path = create_cevoice(char_data)
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error

    print(f"Saved CEVOICE pack to {output_path}")


if __name__ == "__main__":
    wizard()
