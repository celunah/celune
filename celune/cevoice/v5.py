# SPDX-License-Identifier: Apache-2.0
"""CECHAR v5 singing metadata and writer."""

from __future__ import annotations

from typing import Final, Union, Optional, cast
from pathlib import Path
from collections.abc import Mapping

from .v4 import V4_COMPRESSION_NONE
from .base import CEVoice
from ..exceptions import CEVoiceError
from ..typing.cevoice import Manifest, ManifestValue

V5_VERSION: Final[int] = 5
SINGING_VOICE_NAME: Final[str] = "sing"


def write_cechar_v5(
    path: Union[str, Path],
    voices: Mapping[str, Mapping[str, Union[bytes, str, Path]]],
    metadata: Optional[Mapping[str, ManifestValue]] = None,
    voice_metadata: Optional[Mapping[str, Mapping[str, ManifestValue]]] = None,
    bundle_assets: Optional[Mapping[str, Union[bytes, str, Path]]] = None,
    compression: int = V4_COMPRESSION_NONE,
) -> Path:
    """Write a CECHAR v5 archive with a reserved singing voice role.

    Args:
        path: The CECHAR v5 package to save as.
        voices: The voice files to bundle into the package.
        metadata: The semantic package metadata to bundle into the package.
        voice_metadata: Extra metadata stored beside each voice's assets.
        bundle_assets: Persona Markdown files stored as top-level assets.
        compression: One of the indexed CECHAR compression mode constants.

    Returns:
        Path: The path to the created CECHAR v5 package.

    Raises:
        CEVoiceError: The package data cannot be represented as CECHAR v5.
    """
    from .v4 import _write_indexed_cechar

    return _write_indexed_cechar(
        path,
        voices,
        metadata,
        voice_metadata,
        bundle_assets,
        V5_VERSION,
        compression,
        None,
        None,
        None,
    )


def voice_order(metadata: Manifest) -> tuple[str, ...]:
    """Return the public speaking-voice order, hiding the v5 singing role."""
    raw_order = metadata.get("voice_order")
    if isinstance(raw_order, list) and all(
        isinstance(voice, str) for voice in raw_order
    ):
        order = cast(list[str], raw_order)
    else:
        raw_voices = metadata.get("voices")
        order = list(raw_voices) if isinstance(raw_voices, dict) else []
    return tuple(voice for voice in order if voice != SINGING_VOICE_NAME)


def singing_voice(bundle: CEVoice) -> Optional[str]:
    """Return the v5 singing role when the bundle defines it."""
    voices = bundle.metadata.get("voices")
    if (
        bundle.metadata.get("version") == V5_VERSION
        and isinstance(voices, dict)
        and SINGING_VOICE_NAME in voices
    ):
        return SINGING_VOICE_NAME
    return None


def singing_pitch_range(bundle: CEVoice) -> Optional[tuple[int, int]]:
    """Return the v5 singing role's inclusive MIDI range."""
    if singing_voice(bundle) is None:
        return None
    voices = bundle.metadata.get("voices")
    if not isinstance(voices, dict):
        return None
    voice = voices.get(SINGING_VOICE_NAME)
    if not isinstance(voice, dict):
        return None
    pitch_range = voice.get("range")
    if pitch_range is None:
        return (0, 127)
    if not isinstance(pitch_range, dict):
        return None
    minimum = pitch_range.get("min")
    maximum = pitch_range.get("max")
    if isinstance(minimum, int) and isinstance(maximum, int):
        return minimum, maximum
    return None


def normalize_singing_metadata(voice_metadata: dict[str, Manifest]) -> None:
    """Restore the historical default range before rewriting an older v5 pack."""
    singing_metadata = voice_metadata.get(SINGING_VOICE_NAME)
    if singing_metadata is not None and "range" not in singing_metadata:
        singing_metadata["range"] = {"min": 0, "max": 127}


def complete_voice_order(order: list[str], voices: Mapping[str, Manifest]) -> None:
    """Append omitted speaking voices without exposing the singing role."""
    if SINGING_VOICE_NAME in order:
        raise CEVoiceError("metadata voice_order must not include the singing voice")
    order.extend(
        voice for voice in voices if voice not in order and voice != SINGING_VOICE_NAME
    )


def prepare_voice_metadata(voice: str, voice_entry: Manifest) -> None:
    """Validate and normalize one v5 voice's singing-specific manifest fields."""
    if voice == SINGING_VOICE_NAME:
        for field_name in ("cfg_scale", "persona"):
            if voice_entry.get(field_name) is not None:
                raise CEVoiceError(
                    f"singing voice '{SINGING_VOICE_NAME}' {field_name} must be null"
                )
            voice_entry[field_name] = None
        cesn = voice_entry.get("cesn")
        if not isinstance(cesn, str) or not cesn.strip():
            raise CEVoiceError("singing voice cesn is required")
        if "range" not in voice_entry:
            raise CEVoiceError("singing voice range is required")
        validate_singing_pitch_range(voice_entry["range"])
        return

    if voice_entry.get("cesn") is not None:
        raise CEVoiceError(f"voice '{voice}' cesn must be null")
    if "range" in voice_entry:
        raise CEVoiceError("singing voice range is only valid for sing")
    voice_entry["cesn"] = None


def validate_global_semantics(
    metadata: Manifest, voices: Mapping[str, Manifest]
) -> None:
    """Validate v5 selection rules that differ from the shared indexed schema."""
    if metadata.get("default_voice") == SINGING_VOICE_NAME:
        raise CEVoiceError("metadata default_voice must not name the singing voice")
    raw_voice_order = metadata.get("voice_order")
    if isinstance(raw_voice_order, list) and SINGING_VOICE_NAME in raw_voice_order:
        raise CEVoiceError("metadata voice_order must not include the singing voice")
    for voice, voice_data in voices.items():
        if not isinstance(voice_data, dict):
            raise CEVoiceError("invalid voice entry")
        if voice == SINGING_VOICE_NAME:
            if "cfg_scale" not in voice_data or voice_data["cfg_scale"] is not None:
                raise CEVoiceError("singing voice cfg_scale must be null")
            if "persona" not in voice_data or voice_data["persona"] is not None:
                raise CEVoiceError("singing voice persona must be null")
            cesn = voice_data.get("cesn")
            if not isinstance(cesn, str) or not cesn.strip():
                raise CEVoiceError("singing voice cesn is required")
            if "range" in voice_data:
                validate_singing_pitch_range(voice_data["range"])
            assets = voice_data.get("assets")
            if not isinstance(assets, dict) or "wav" not in assets:
                raise CEVoiceError("singing voice must provide a wav asset")
            reference_text = voice_data.get("reference_text")
            if not isinstance(reference_text, str) or not reference_text.strip():
                raise CEVoiceError("singing voice reference_text is required")
        else:
            if "cesn" not in voice_data or voice_data["cesn"] is not None:
                raise CEVoiceError(f"voice '{voice}' cesn must be null")
            if "range" in voice_data:
                raise CEVoiceError("singing voice range is only valid for sing")


def validate_singing_pitch_range(value: ManifestValue) -> None:
    """Validate an inclusive MIDI note range from 0 through 127."""
    if not isinstance(value, dict):
        raise CEVoiceError("singing voice range must be an object")
    minimum = value.get("min")
    maximum = value.get("max")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not 0 <= minimum <= maximum <= 127
    ):
        raise CEVoiceError(
            "singing voice range must contain ordered MIDI pitches from 0 to 127"
        )
