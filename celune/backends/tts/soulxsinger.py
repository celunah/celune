# SPDX-License-Identifier: Apache-2.0
"""Score- and F0-conditioned SoulX-Singer integration for CECHAR v5 packs."""

from __future__ import annotations

import re
import sys
import json
import time
import shutil
import tempfile
import importlib
import unicodedata
from pathlib import Path
from types import ModuleType
from dataclasses import field, dataclass
from typing import Union, Optional, Protocol, cast
from collections.abc import Mapping, Callable, Iterator

import torch
import numpy as np
from huggingface_hub import snapshot_download

from ...cesn import (
    CESNSection,
    CESNNotePair,
    CESNRestEvent,
    parse_cesn,
    resolve_active_singer,
)
from ...paths import (
    huggingface_progress,
    backend_environments_dir,
    huggingface_hub_cache_dir,
)
from ...cevoice import (
    V5_VERSION,
    SINGING_VOICE_NAME,
    CEVoice,
    CEVoiceLoader,
    default_loader,
    bundle_character_name,
)
from ...exceptions import (
    BackendError,
    CESNIPAError,
    CEVoiceError,
    CESNRangeError,
    CESNValidationError,
    InvalidCharacterError,
    CESNBackendFormatError,
    CESNUnsupportedFeatureError,
)
from ...registry.tts import (
    TTS_DEFAULT_VOICE,
    SOULX_SINGER_MODEL_ID,
    SOULX_SINGER_REVISION,
    SOULX_SINGER_CODE_REVISION,
    SOULX_SINGER_CODE_REPOSITORY,
)
from ...typing.common import JSON
from ..environment import exclusive_lock
from ...typing.aliases import AudioChunk
from ...singing import contains_cesn_note_event
from ...registry.consent import require_model_consent
from ...registry.other import SOULX_SINGER_RMVPE_MODEL_ID
from .base import CeluneBackend, to_numpy_audio, cached_hf_snapshot_path

__all__ = [
    "SoulXSinger",
    "decode_score",
    "load_upstream_source",
    "pitch_to_midi",
]

_MODEL_FILES = ["config.yaml", "model.pt"]
_MODEL_LOAD_STEPS = 8
_MODEL_ONSET_PREROLL_SECONDS = 2.0
_ARPABET_VOWELS = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)
_IPA_TO_ARPABET = {
    "t͡ʃ": "CH",
    "t͜ʃ": "CH",
    "d͡ʒ": "JH",
    "d͜ʒ": "JH",
    "eɪ": "EY",
    "aɪ": "AY",
    "aʊ": "AW",
    "ɔɪ": "OY",
    "oʊ": "OW",
    "əʊ": "OW",
    "tʃ": "CH",
    "dʒ": "JH",
    "p": "P",
    "b": "B",
    "t": "T",
    "d": "D",
    "k": "K",
    "g": "G",
    "ɡ": "G",
    "f": "F",
    "v": "V",
    "θ": "TH",
    "ð": "DH",
    "s": "S",
    "z": "Z",
    "ʃ": "SH",
    "ʒ": "ZH",
    "h": "HH",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "l": "L",
    "ɫ": "L",
    "ɹ": "R",
    "ɻ": "R",
    "r": "R",
    "j": "Y",
    "w": "W",
    "ɾ": "T",
    "i": "IY",
    "ɪ": "IH",
    "ɛ": "EH",
    "æ": "AE",
    "ɑ": "AA",
    "ɒ": "AA",
    "ɔ": "AO",
    "ʊ": "UH",
    "u": "UW",
    "ʌ": "AH",
    "ə": "AH",
    "ɐ": "AH",
    "ɜ": "ER",
    "ɝ": "ER",
    "ɚ": "ER",
}
_IPA_TO_ARPABET_SYMBOLS = tuple(sorted(_IPA_TO_ARPABET, key=len, reverse=True))
_PITCH_PATTERN = re.compile(r"(?P<step>[a-g])(?P<accidental>[#b]?)(?P<octave>-?\d+)")
_PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_RMVPE_FILES = ["rmvpe/rmvpe.pt"]


class _SoulXInferenceModel(Protocol):
    """Inference surface exposed by the pinned upstream model."""

    def infer(
        self,
        metadata: dict[str, dict[str, Optional[torch.Tensor]]],
        *,
        auto_shift: bool,
        pitch_shift: int,
        n_steps: int,
        cfg: float,
        control: str,
        use_fp16: bool,
    ) -> torch.Tensor: ...


class _SoulXDataProcessor(Protocol):
    """Metadata processor used by the pinned upstream model."""

    phone2idx: Mapping[str, int]
    hop_size: int

    def process(
        self,
        metadata: dict[str, str],
        wav_path: Optional[str] = None,
    ) -> dict[str, Optional[torch.Tensor]]: ...


class _SoulXF0Extractor(Protocol):
    """Framewise F0 extractor from the pinned SoulX preprocessing source."""

    def process(
        self,
        audio_path: str,
        *,
        verbose: Optional[bool] = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class _ScoreEvent:
    """One event converted from an aligned CESN note pair or rest."""

    pitch: Optional[str]
    duration_numerator: int
    duration_denominator: int
    rest: bool
    ipa: Optional[str] = None
    slur_from_previous: bool = False
    note_source_start: Optional[int] = None
    note_source_end: Optional[int] = None
    ipa_source_start: Optional[int] = None
    ipa_source_end: Optional[int] = None


@dataclass(frozen=True)
class _ScoreSection:
    """One fixed singer, language, and tempo section of a CESN request."""

    bpm: int
    key: str
    scale: str
    language: str
    singer_path: str
    events: tuple[_ScoreEvent, ...]
    display_text: str = ""


@dataclass
class _SoulXRuntime:
    """Loaded model and the preprocessing helpers required for inference."""

    model: _SoulXInferenceModel
    processor: _SoulXDataProcessor
    sample_rate: int
    inference_steps: int
    cfg: float
    f0_extractor: Optional[_SoulXF0Extractor] = None
    prompt_cache: dict[str, dict[str, Optional[torch.Tensor]]] = field(
        default_factory=dict
    )


@dataclass
class _SoulXSourceModules:
    """Pinned source modules loaded before CEDTS request threads start."""

    root: Path
    model: ModuleType
    processor: ModuleType
    f0: Optional[ModuleType] = None


class _UnrepresentableIPASymbol(ValueError):
    """Identify one IPA symbol that cannot be encoded as an ARPAbet phone."""

    def __init__(self, character: str, offset: int) -> None:
        self.character = character
        self.offset = offset
        super().__init__("unsupported English IPA symbol")


def _ipa_to_arpabet(ipa: str) -> tuple[str, ...]:
    """Convert supported English IPA symbols into SoulX's ARPAbet phones."""
    return tuple(token for token, _start, _end in _ipa_to_arpabet_symbols(ipa))


def _ipa_to_arpabet_symbols(ipa: str) -> tuple[tuple[str, int, int], ...]:
    """Return ARPAbet phones with the IPA spans that produced each phone."""
    source = unicodedata.normalize("NFC", ipa)
    phones: list[tuple[str, int, int]] = []
    pending_stress: Optional[str] = None
    pending_stress_offset: Optional[int] = None
    has_vowel = False
    offset = 0

    while offset < len(source):
        character = source[offset]
        if character.isspace() or character in {"ː", "ˑ", "ʰ"}:
            offset += 1
            continue
        if character in {"ˈ", "ˌ"}:
            if pending_stress is not None:
                raise _UnrepresentableIPASymbol(character, offset)
            pending_stress = "1" if character == "ˈ" else "2"
            pending_stress_offset = offset
            offset += 1
            continue

        symbol = next(
            (
                candidate
                for candidate in _IPA_TO_ARPABET_SYMBOLS
                if source.startswith(candidate, offset)
            ),
            None,
        )
        if symbol is None:
            raise _UnrepresentableIPASymbol(character, offset)

        phone = _IPA_TO_ARPABET[symbol]
        if phone in _ARPABET_VOWELS:
            if symbol == "ɝ":
                stress = pending_stress or "1"
            elif symbol == "ɚ":
                stress = pending_stress or "0"
            else:
                stress = pending_stress or "0"
            phones.append((f"{phone}{stress}", offset, offset + len(symbol)))
            pending_stress = None
            pending_stress_offset = None
            has_vowel = True
        else:
            phones.append((phone, offset, offset + len(symbol)))
        offset += len(symbol)

    if pending_stress is not None:
        stress_offset = pending_stress_offset or 0
        raise _UnrepresentableIPASymbol(source[stress_offset], stress_offset)
    if not phones or not has_vowel:
        raise _UnrepresentableIPASymbol(source, 0)
    return tuple(phones)


def _raise_unrepresentable_ipa(
    event: _ScoreEvent,
    error: _UnrepresentableIPASymbol,
) -> None:
    """Raise a localized CESN error and attach the source IPA location."""
    error_character = error.character
    failure = CESNIPAError(error_character)
    if event.ipa_source_start is not None and event.ipa_source_end is not None:
        start = event.ipa_source_start + error.offset
        end = start + max(1, len(error_character))
        if end <= event.ipa_source_end:
            failure.source_start = start
            failure.source_end = end
        else:
            failure.source_start = event.ipa_source_start
            failure.source_end = event.ipa_source_end
    raise failure


def _is_vowel_continuation(
    previous_phone_tokens: Optional[tuple[str, ...]],
    phone_tokens: tuple[str, ...],
) -> bool:
    """Check whether a slurred vowel continues the preceding syllable nucleus."""
    if previous_phone_tokens is None or len(phone_tokens) != 1:
        return False
    current_vowel = phone_tokens[0].rstrip("012")
    if current_vowel not in _ARPABET_VOWELS or not previous_phone_tokens:
        return False
    previous_vowel = previous_phone_tokens[-1].rstrip("012")
    return previous_vowel == current_vowel


def _score_event(
    event: Union[CESNNotePair, CESNRestEvent],
    *,
    note_source_span: Optional[tuple[int, int]] = None,
    ipa_source_span: Optional[tuple[int, int]] = None,
) -> _ScoreEvent:
    """Convert one parsed CESN event into the backend's stable event shape."""
    if isinstance(event, CESNRestEvent):
        duration = event.duration_beats
        return _ScoreEvent(
            pitch=None,
            duration_numerator=duration.numerator,
            duration_denominator=duration.denominator,
            rest=True,
        )
    if isinstance(event, CESNNotePair):
        duration = event.event.duration_beats
        return _ScoreEvent(
            pitch=event.event.pitch,
            duration_numerator=duration.numerator,
            duration_denominator=duration.denominator,
            rest=False,
            ipa=event.ipa,
            slur_from_previous=event.slur_from_previous,
            note_source_start=(note_source_span[0] if note_source_span else None),
            note_source_end=(note_source_span[1] if note_source_span else None),
            ipa_source_start=(ipa_source_span[0] if ipa_source_span else None),
            ipa_source_end=(ipa_source_span[1] if ipa_source_span else None),
        )
    raise TypeError(type(event).__name__)


def _serialize_section(section: CESNSection) -> _ScoreSection:
    """Convert one CESN section while retaining its exact beat fractions."""
    singer = section.singer or resolve_active_singer()
    events_list: list[_ScoreEvent] = []
    for block in section.blocks:
        for index, event in enumerate(block.events):
            note_spans = block._source_note_spans
            ipa_spans = block._source_ipa_spans
            note_span = note_spans[index] if index < len(note_spans) else None
            ipa_span = ipa_spans[index] if index < len(ipa_spans) else None
            score_event = _score_event(
                event,
                note_source_span=note_span,
                ipa_source_span=ipa_span,
            )
            if score_event.ipa is not None:
                try:
                    _ipa_to_arpabet_symbols(score_event.ipa)
                except _UnrepresentableIPASymbol as error:
                    _raise_unrepresentable_ipa(score_event, error)
                except ValueError:
                    _raise_unrepresentable_ipa(
                        score_event,
                        _UnrepresentableIPASymbol(score_event.ipa, 0),
                    )
            events_list.append(score_event)
    events = tuple(events_list)
    for event in events:
        if not event.rest:
            try:
                _pitch_to_midi_for_singer(
                    event.pitch or "",
                    singer.name,
                    singer.pitch_range,
                )
            except CESNRangeError as error:
                error.source_start = event.note_source_start
                error.source_end = event.note_source_end
                raise
    return _ScoreSection(
        bpm=section.bpm,
        key=section.key,
        scale=section.scale,
        language=section.language,
        singer_path=str(singer.bundle_path),
        events=events,
        display_text=" ".join(
            "".join(block.lyrics for block in section.blocks).split()
        ),
    )


def _score_to_dict(section: _ScoreSection) -> dict[str, object]:
    """Return a JSON-safe representation of one prepared score section."""
    return {
        "bpm": section.bpm,
        "key": section.key,
        "scale": section.scale,
        "language": section.language,
        "singer_path": section.singer_path,
        "display_text": section.display_text,
        "events": [
            {
                "pitch": event.pitch,
                "duration": [
                    event.duration_numerator,
                    event.duration_denominator,
                ],
                "rest": event.rest,
                **({"ipa": event.ipa} if event.ipa is not None else {}),
                "slur_from_previous": event.slur_from_previous,
                **(
                    {
                        "note_source_start": event.note_source_start,
                        "note_source_end": event.note_source_end,
                        "ipa_source_start": event.ipa_source_start,
                        "ipa_source_end": event.ipa_source_end,
                    }
                    if event.ipa_source_start is not None
                    and event.ipa_source_end is not None
                    else {}
                ),
            }
            for event in section.events
        ],
    }


def decode_score(text: str) -> tuple[_ScoreSection, ...]:
    """Decode the backend payload produced by ``cesn_to_model``.

    Raises:
        CESNBackendFormatError: If the payload is invalid or violates its schema.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise CESNBackendFormatError(
            "SoulX-Singer expected a prepared CESN score payload"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("backend") != "soulxsinger"
        or payload.get("version") != 2
    ):
        raise CESNBackendFormatError("prepared CESN score has an invalid header")
    raw_sections = payload.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise CESNBackendFormatError(
            "prepared CESN score requires at least one section"
        )

    sections: list[_ScoreSection] = []
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            raise CESNBackendFormatError("prepared CESN section must be an object")
        bpm = raw_section.get("bpm")
        key = raw_section.get("key")
        scale = raw_section.get("scale")
        language = raw_section.get("language")
        singer_path = raw_section.get("singer_path")
        display_text = raw_section.get("display_text", "")
        raw_events = raw_section.get("events")
        if (
            not isinstance(bpm, int)
            or isinstance(bpm, bool)
            or bpm <= 0
            or not isinstance(key, str)
            or not key
            or not isinstance(scale, str)
            or not scale
            or not isinstance(language, str)
            or not language
            or not isinstance(singer_path, str)
            or not singer_path
            or not isinstance(display_text, str)
            or not isinstance(raw_events, list)
            or not raw_events
        ):
            raise CESNBackendFormatError(
                "prepared CESN section has invalid or missing fields"
            )

        events: list[_ScoreEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise CESNBackendFormatError("prepared CESN event must be an object")
            duration = raw_event.get("duration")
            pitch = raw_event.get("pitch")
            rest = raw_event.get("rest")
            ipa = raw_event.get("ipa")
            slur_from_previous = raw_event.get("slur_from_previous", False)
            note_source_start = raw_event.get("note_source_start")
            note_source_end = raw_event.get("note_source_end")
            ipa_source_start = raw_event.get("ipa_source_start")
            ipa_source_end = raw_event.get("ipa_source_end")
            has_source_spans = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    note_source_start,
                    note_source_end,
                    ipa_source_start,
                    ipa_source_end,
                )
            )
            missing_source_spans = all(
                value is None
                for value in (
                    note_source_start,
                    note_source_end,
                    ipa_source_start,
                    ipa_source_end,
                )
            )
            if (
                not isinstance(duration, list)
                or len(duration) != 2
                or not all(
                    isinstance(part, int) and not isinstance(part, bool)
                    for part in duration
                )
                or duration[0] <= 0
                or duration[1] <= 0
                or not isinstance(rest, bool)
                or not isinstance(slur_from_previous, bool)
                or not (has_source_spans or missing_source_spans)
                or (
                    has_source_spans
                    and (
                        cast(int, note_source_end) < cast(int, note_source_start)
                        or cast(int, ipa_source_end) < cast(int, ipa_source_start)
                    )
                )
                or (
                    rest
                    and (
                        not missing_source_spans or pitch is not None or ipa is not None
                    )
                )
                or (ipa is not None and (not isinstance(ipa, str) or not ipa.strip()))
                or (
                    rest
                    and (pitch is not None or ipa is not None or slur_from_previous)
                )
                or (
                    not rest
                    and (
                        not isinstance(pitch, str)
                        or not isinstance(ipa, str)
                        or not ipa.strip()
                    )
                )
            ):
                raise CESNBackendFormatError(
                    "prepared CESN event has invalid or missing fields"
                )
            events.append(
                _ScoreEvent(
                    pitch=cast(Optional[str], pitch),
                    duration_numerator=duration[0],
                    duration_denominator=duration[1],
                    rest=rest,
                    ipa=cast(Optional[str], ipa),
                    slur_from_previous=slur_from_previous,
                    note_source_start=(
                        cast(int, note_source_start) if has_source_spans else None
                    ),
                    note_source_end=(
                        cast(int, note_source_end) if has_source_spans else None
                    ),
                    ipa_source_start=(
                        cast(int, ipa_source_start) if has_source_spans else None
                    ),
                    ipa_source_end=(
                        cast(int, ipa_source_end) if has_source_spans else None
                    ),
                )
            )
        sections.append(
            _ScoreSection(
                bpm=bpm,
                key=key,
                scale=scale,
                language=language,
                singer_path=singer_path,
                events=tuple(events),
                display_text=display_text,
            )
        )
    return tuple(sections)


def pitch_to_midi(pitch: str) -> int:
    """Convert a validated CESN scientific pitch name to a MIDI note number."""
    match = _PITCH_PATTERN.fullmatch(pitch.casefold())
    if match is None:
        raise CESNBackendFormatError("prepared CESN score contains an invalid pitch")
    step = match.group("step")
    accidental = match.group("accidental")
    octave = int(match.group("octave"))
    semitone = _PITCH_CLASSES[step]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi_note = (octave + 1) * 12 + semitone
    return midi_note


def _pitch_to_midi_for_singer(
    pitch: str,
    character_name: str,
    pitch_range: tuple[int, int],
) -> int:
    """Convert a CESN pitch and enforce its CECHAR singing range."""
    midi_note = pitch_to_midi(pitch)
    minimum, maximum = pitch_range
    if not minimum <= midi_note <= maximum:
        raise CESNRangeError(
            pitch,
            midi_note,
            character_name,
            minimum,
            maximum,
        )
    return midi_note


def _natural_f0_contour(
    events: tuple[_ScoreEvent, ...],
    durations: list[str],
    pitches: list[str],
    extracted_f0: np.ndarray,
    sample_rate: int,
    hop_size: int,
) -> str:
    """Recenter a score-guided performance contour on each written note.

    Args:
        events: Ordered score notes and rests.
        durations: Event durations in seconds, formatted for SoulX metadata.
        pitches: Event pitches as MIDI values, with rests represented as zero.
        extracted_f0: Framewise F0 extracted from a score-conditioned teacher pass.
        sample_rate: Sample rate used by the model processor.
        hop_size: Number of samples between model frames.

    Returns:
        str: Space-separated F0 values in Hz, with zero for rests and unvoiced frames.
    """
    frame_rate = sample_rate / hop_size
    total_frames = int(sum(float(duration) for duration in durations) * frame_rate)
    source_f0 = np.asarray(extracted_f0, dtype=np.float64).reshape(-1)
    source_f0 = np.where(np.isfinite(source_f0) & (source_f0 > 0), source_f0, 0.0)
    if len(source_f0) < total_frames:
        source_f0 = np.pad(source_f0, (0, total_frames - len(source_f0)))
    else:
        source_f0 = source_f0[:total_frames]

    contour = np.zeros(total_frames, dtype=np.float64)
    elapsed = 0.0
    frame_cursor = 0

    for index, event in enumerate(events):
        elapsed += float(durations[index])
        if index == len(events) - 1:
            frame_end = total_frames
        else:
            frame_end = min(total_frames, int(np.round(elapsed * frame_rate)))
        frame_end = max(frame_cursor, frame_end)

        if not event.rest and frame_end > frame_cursor:
            midi_pitch = int(pitches[index])
            target_frequency = 440.0 * 2.0 ** ((midi_pitch - 69) / 12.0)
            event_f0 = source_f0[frame_cursor:frame_end]
            voiced = event_f0 > 0
            if np.any(voiced):
                cents = 1200.0 * np.log2(event_f0[voiced] / target_frequency)
                centered_cents = cents - np.median(cents)
                event_contour = contour[frame_cursor:frame_end]
                event_contour[voiced] = target_frequency * np.exp2(
                    centered_cents / 1200.0
                )
            else:
                contour[frame_cursor:frame_end] = target_frequency
        frame_cursor = frame_end

    return " ".join(f"{frequency:.10g}" for frequency in contour)


def _prepend_onset_rest(
    metadata: dict[str, str],
    duration_seconds: float,
) -> dict[str, str]:
    """Add silent model context before a target without changing its score."""
    return {
        **metadata,
        "duration": f"{duration_seconds:.10g} {metadata['duration']}",
        "phoneme": f"<SP> {metadata['phoneme']}",
        "note_pitch": f"0 {metadata['note_pitch']}",
        "note_type": f"1 {metadata['note_type']}",
    }


def _serialize_f0(values: np.ndarray) -> str:
    """Serialize finite positive F0 values, retaining zero-valued unvoiced frames."""
    frequencies = np.asarray(values, dtype=np.float64).reshape(-1)
    frequencies = np.where(
        np.isfinite(frequencies) & (frequencies > 0), frequencies, 0.0
    )
    return " ".join(f"{frequency:.10g}" for frequency in frequencies)


def load_upstream_source() -> Path:
    """Return the exact pinned SoulX source checkout, downloading it once if needed."""
    from git import Repo

    source_parent = backend_environments_dir(create=True) / "soulxsinger-source"
    source_parent.mkdir(parents=True, exist_ok=True)
    source_root = source_parent / SOULX_SINGER_CODE_REVISION
    package_path = source_root / "soulxsinger" / "models" / "soulxsinger.py"
    try:
        with exclusive_lock(source_parent / ".install.lock", timeout=300.0):
            if package_path.is_file():
                try:
                    repository = Repo(source_root)
                except Exception as error:
                    raise BackendError(
                        "Unable to prepare the pinned SoulX-Singer source",
                        error_code="soulx_source_revision_mismatch",
                    ) from error
                if (
                    repository.head.commit.hexsha != SOULX_SINGER_CODE_REVISION
                    or repository.is_dirty()
                ):
                    raise BackendError(
                        "Unable to prepare the pinned SoulX-Singer source",
                        error_code="soulx_source_revision_mismatch",
                    )
                return source_root

            if source_root.exists():
                shutil.rmtree(source_root)
            try:
                repository = Repo.clone_from(
                    SOULX_SINGER_CODE_REPOSITORY,
                    source_root,
                    no_checkout=True,
                    multi_options=["--filter=blob:none"],
                )
                repository.git.checkout(SOULX_SINGER_CODE_REVISION)
                if repository.head.commit.hexsha != SOULX_SINGER_CODE_REVISION:
                    raise BackendError(
                        "Unable to prepare the pinned SoulX-Singer source",
                        error_code="soulx_source_revision_mismatch",
                    )
            except Exception as error:
                if source_root.exists():
                    shutil.rmtree(source_root, ignore_errors=True)
                if isinstance(error, BackendError):
                    raise
                raise BackendError(
                    "Unable to prepare the pinned SoulX-Singer source",
                    error_code="soulx_source_unavailable",
                ) from error
    except Exception as error:
        if isinstance(error, BackendError):
            raise
        raise BackendError(
            "Unable to prepare the pinned SoulX-Singer source",
            error_code="soulx_source_unavailable",
        ) from error
    return source_root


def _load_soulx_reference_wav(
    wav_path: str,
    sample_rate: int,
) -> torch.Tensor:
    """Decode a prompt WAV without requiring TorchCodec or FFmpeg."""
    import soundfile as sf
    import torchaudio

    audio, source_sample_rate = sf.read(
        wav_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            source_sample_rate,
            sample_rate,
        )
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


class SoulXSinger(CeluneBackend[_SoulXRuntime]):
    """Score- or F0-controlled singing backend using a CECHAR v5 ``sing`` voice."""

    name = "soulxsinger"
    chunk_rate = 1.0
    singing = True
    uses_voice_bundles = True
    supported_languages = ("en",)
    default_voice = TTS_DEFAULT_VOICE

    def __init__(
        self,
        log: Callable[[str, str], None],
        fatal: Optional[Callable[[], None]] = None,
        quantize: bool = True,
        quantize_kv_cache: bool = True,
        high_precision: bool = False,
    ) -> None:
        """Initialize SoulX-Singer with its selected pitch-conditioning mode.

        Args:
            log: Backend logging callback.
            fatal: Optional callback for unrecoverable worker failures.
            quantize: Shared quantization request, ignored by this backend.
            quantize_kv_cache: Shared cache policy, ignored by this backend.
            high_precision: Use frame-aligned F0 conditioning instead of score
                conditioning.

        Raises:
            TypeError: If ``high_precision`` is not a boolean.
        """
        del quantize, quantize_kv_cache
        if not isinstance(high_precision, bool):
            raise TypeError("high_precision must be a boolean")
        super().__init__(
            log=log,
            model_name=SOULX_SINGER_MODEL_ID,
            fatal=fatal,
            quantize=False,
            quantize_kv_cache=False,
            singing=True,
        )
        self.high_precision = high_precision
        self._source_modules: Optional[_SoulXSourceModules] = None
        self._validate_refs()

    def _prepare_source_modules(self) -> _SoulXSourceModules:
        """Load the pinned model and enabled preprocessing modules once."""
        source_modules = getattr(self, "_source_modules", None)
        if isinstance(source_modules, _SoulXSourceModules):
            return source_modules

        source_root = load_upstream_source()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        model_module = importlib.import_module("soulxsinger.models.soulxsinger")
        processor_module = importlib.import_module("soulxsinger.utils.data_processor")
        processor_module.load_wav = _load_soulx_reference_wav
        f0_module = (
            importlib.import_module("preprocess.tools.f0_extraction")
            if getattr(self, "high_precision", False)
            else None
        )

        source_modules = _SoulXSourceModules(
            root=source_root,
            model=model_module,
            processor=processor_module,
            f0=f0_module,
        )
        self._source_modules = source_modules
        return source_modules

    def prepare_model_loading(self) -> None:
        """Prime pinned SoulX imports before CEDTS starts request threads."""
        self._prepare_source_modules()

    @staticmethod
    def _get_default_loader() -> Optional[CEVoiceLoader]:
        """Return the active CEVOICE loader for SoulX-Singer."""
        return default_loader()

    @staticmethod
    def _validate_singing_voice(loader: CEVoiceLoader) -> tuple[str, Path]:
        """Return the CECHAR v5 singing score and WAV path for one pack."""
        bundle = loader.bundle
        voice = bundle.voices.get(SINGING_VOICE_NAME)
        character = bundle_character_name(bundle) or bundle.path.stem
        if (
            bundle.metadata.get("version") != V5_VERSION
            or bundle.singing_voice != SINGING_VOICE_NAME
            or not isinstance(voice, Mapping)
            or not isinstance(voice.get("cesn"), str)
            or not str(voice.get("cesn", "")).strip()
        ):
            raise InvalidCharacterError(character)
        return str(voice["cesn"]), loader.materialize(SINGING_VOICE_NAME, "wav")

    def _validate_refs(self) -> None:
        """Validate the active pack's hidden CECHAR v5 singing reference."""
        loader = self._get_default_loader()
        if loader is not None:
            self._validate_singing_voice(loader)

    @property
    def voices(self) -> list[str]:
        """Return visible TTS voices while keeping the hidden ``sing`` voice private."""
        compatible_bundle = self._require_compatible_bundle()
        if compatible_bundle is None:
            return []
        return list(compatible_bundle[1])

    def model_is_available_locally(
        self,
        model: str,
        lang: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check the pinned model snapshot without contacting Hugging Face."""
        del lang
        if model != SOULX_SINGER_MODEL_ID:
            return False, None
        available, snapshot_path = cached_hf_snapshot_path(
            model,
            _MODEL_FILES,
            SOULX_SINGER_REVISION,
        )
        if available and getattr(self, "high_precision", False):
            f0_available, _ = cached_hf_snapshot_path(
                SOULX_SINGER_RMVPE_MODEL_ID,
                _RMVPE_FILES,
            )
            return f0_available, snapshot_path
        return available, snapshot_path

    def _ensure_f0_checkpoint(self) -> Path:
        """Return the reviewed RMVPE checkpoint, downloading it when required."""
        metadata = require_model_consent(SOULX_SINGER_RMVPE_MODEL_ID)
        available, snapshot_path = cached_hf_snapshot_path(
            SOULX_SINGER_RMVPE_MODEL_ID,
            _RMVPE_FILES,
            metadata.revision,
        )
        if not available or snapshot_path is None:
            with huggingface_progress(self.report_progress):
                snapshot_path = snapshot_download(
                    repo_id=SOULX_SINGER_RMVPE_MODEL_ID,
                    revision=metadata.revision,
                    allow_patterns=_RMVPE_FILES,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )
        checkpoint = Path(snapshot_path) / _RMVPE_FILES[0]
        if not checkpoint.is_file():
            raise BackendError(
                "SoulX-Singer RMVPE checkpoint is unavailable",
                error_code="soulx_f0_checkpoint_unavailable",
            )
        return checkpoint

    def preload_models(self) -> None:
        """Prepare the reviewed model weights and pinned SoulX source."""
        metadata = require_model_consent(SOULX_SINGER_MODEL_ID)
        available, _ = self.model_is_available_locally(SOULX_SINGER_MODEL_ID)
        if not available:
            with huggingface_progress(self.report_progress):
                snapshot_download(
                    repo_id=SOULX_SINGER_MODEL_ID,
                    revision=metadata.revision,
                    allow_patterns=_MODEL_FILES,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )
        if getattr(self, "high_precision", False):
            self._ensure_f0_checkpoint()
        load_upstream_source()

    def load_model(self, model_id: str, **kwargs: object) -> _SoulXRuntime:
        """Load the pinned checkpoint and prepare the score processor."""
        del kwargs
        if model_id != SOULX_SINGER_MODEL_ID:
            raise BackendError(
                f"SoulX-Singer does not support model {model_id!r}",
                error_code="soulx_unsupported_model",
            )
        metadata = require_model_consent(model_id)
        if not torch.cuda.is_available():
            raise BackendError(
                "SoulX-Singer requires a CUDA-capable GPU",
                error_code="soulx_cuda_required",
            )

        def report_stage(step: int) -> None:
            """Report numeric progress for one model-loading stage."""
            self.report_progress(step, _MODEL_LOAD_STEPS)

        report_stage(0)
        source_modules = self._prepare_source_modules()
        source_root = source_modules.root
        module = source_modules.model
        processor_module = source_modules.processor
        try:
            from omegaconf import OmegaConf
        except ImportError as error:
            raise BackendError(
                "SoulX-Singer requires the OmegaConf package",
                error_code="soulx_dependencies_missing",
            ) from error

        available, snapshot_path = self.model_is_available_locally(model_id)
        if not available or snapshot_path is None:
            with huggingface_progress(self.report_progress):
                snapshot_path = snapshot_download(
                    repo_id=model_id,
                    revision=metadata.revision,
                    allow_patterns=_MODEL_FILES,
                    cache_dir=str(huggingface_hub_cache_dir(create=True)),
                )

        snapshot = Path(snapshot_path)
        report_stage(1)
        config = OmegaConf.load(str(snapshot / "config.yaml"))
        config_values = cast(Mapping[str, object], config)
        audio_config = cast(Mapping[str, object], config_values["audio"])
        inference_config = cast(Mapping[str, object], config_values["infer"])
        device = torch.device("cuda")
        model_type = cast(Callable[..., torch.nn.Module], module.SoulXSinger)
        report_stage(2)
        model = model_type(config)
        report_stage(3)
        checkpoint = cast(
            Mapping[str, object],
            torch.load(
                snapshot / "model.pt",
                map_location="cpu",
                weights_only=False,
            ),
        )
        state_dict = cast(Mapping[str, torch.Tensor], checkpoint["state_dict"])
        report_stage(4)
        model.load_state_dict(state_dict, strict=True)
        report_stage(5)
        model.half()
        mel = getattr(model, "mel", None)
        if isinstance(mel, torch.nn.Module):
            mel.float()
        model.eval()
        model.to(device)

        processor_type = cast(
            Callable[..., _SoulXDataProcessor], processor_module.DataProcessor
        )
        report_stage(6)
        processor = processor_type(
            hop_size=int(audio_config["hop_size"]),
            sample_rate=int(audio_config["sample_rate"]),
            phoneset_path=str(
                source_root / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
            ),
            device=str(device),
        )
        f0_extractor: Optional[_SoulXF0Extractor] = None
        if getattr(self, "high_precision", False):
            if source_modules.f0 is None:
                raise BackendError(
                    "SoulX-Singer F0 extraction source is unavailable",
                    error_code="soulx_f0_extractor_unavailable",
                )
            f0_type = cast(
                Callable[..., _SoulXF0Extractor], source_modules.f0.F0Extractor
            )
            f0_extractor = f0_type(
                str(self._ensure_f0_checkpoint()),
                device=str(device),
                is_half=True,
                target_sr=int(audio_config["sample_rate"]),
                hop_size=int(audio_config["hop_size"]),
                verbose=False,
            )
        report_stage(7)
        runtime = _SoulXRuntime(
            model=cast(_SoulXInferenceModel, model),
            processor=processor,
            sample_rate=int(audio_config["sample_rate"]),
            inference_steps=int(inference_config["n_steps"]),
            cfg=float(inference_config["cfg"]),
            f0_extractor=f0_extractor,
        )
        self.model = runtime
        self.report_progress(_MODEL_LOAD_STEPS, _MODEL_LOAD_STEPS)
        return runtime

    def cesn_to_model(self, text: str) -> str:
        """Convert validated CESN into the score payload consumed by SoulX-Singer."""
        if not contains_cesn_note_event(text):
            raise CESNValidationError("No CESN notation found")
        sections = parse_cesn(text, supported_languages=self.supported_languages)
        prepared = tuple(_serialize_section(section) for section in sections)
        return json.dumps(
            {
                "backend": self.name,
                "version": 2,
                "sections": [_score_to_dict(section) for section in prepared],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _singer_loader(
        singer_path: str,
        owned_loaders: dict[str, CEVoiceLoader],
    ) -> CEVoiceLoader:
        """Resolve a singer pack through the active loader or a request-local loader."""
        path = Path(singer_path).resolve()
        active_loader = default_loader()
        if active_loader is not None and active_loader.bundle.path.resolve() == path:
            SoulXSinger._validate_singing_voice(active_loader)
            return active_loader
        path_key = str(path)
        cached_loader = owned_loaders.get(path_key)
        if cached_loader is not None:
            return cached_loader
        try:
            loader = CEVoiceLoader(CEVoice.open(path))
        except (OSError, ValueError, CEVoiceError) as error:
            raise InvalidCharacterError(path.stem) from error
        try:
            SoulXSinger._validate_singing_voice(loader)
        except Exception:
            loader.close()
            raise
        owned_loaders[path_key] = loader
        return loader

    @staticmethod
    def _prompt_data(
        runtime: _SoulXRuntime,
        loader: CEVoiceLoader,
        score_text: str,
        reference_wav: Path,
        high_precision: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Build or retrieve prompt tensors for the selected conditioning mode."""
        stat = loader.bundle.path.stat()
        cache_key = f"{loader.bundle.path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
        if high_precision:
            cache_key = f"{cache_key}:melody"
        prompt_data = runtime.prompt_cache.get(cache_key)
        if prompt_data is not None:
            return prompt_data
        sections = parse_cesn(
            score_text,
            supported_languages=SoulXSinger.supported_languages,
        )
        if any(
            section.language.split("-")[0].casefold() != "en" for section in sections
        ):
            raise CESNUnsupportedFeatureError(
                "SoulX-Singer reference scores must use English"
            )
        metadata_sections = tuple(
            SoulXSinger._metadata(
                tuple(
                    _score_event(event)
                    for block in section.blocks
                    for event in block.events
                ),
                section.bpm,
                runtime,
                character_name=(
                    bundle_character_name(loader.bundle) or loader.bundle.path.stem
                ),
                pitch_range=loader.bundle.singing_pitch_range or (0, 127),
            )
            for section in sections
        )
        prompt_metadata = {
            key: " ".join(section[key] for section in metadata_sections)
            for key in metadata_sections[0]
        }
        if high_precision:
            if runtime.f0_extractor is None:
                raise BackendError(
                    "SoulX-Singer F0 extraction is unavailable",
                    error_code="soulx_f0_extractor_unavailable",
                )
            prompt_metadata["f0"] = _serialize_f0(
                runtime.f0_extractor.process(str(reference_wav), verbose=False)
            )
        prompt_data = runtime.processor.process(prompt_metadata, str(reference_wav))
        runtime.prompt_cache[cache_key] = prompt_data
        return prompt_data

    @staticmethod
    def _metadata(
        events: tuple[_ScoreEvent, ...],
        bpm: int,
        runtime: _SoulXRuntime,
        character_name: Optional[str] = None,
        pitch_range: Optional[tuple[int, int]] = None,
    ) -> dict[str, str]:
        """Translate CESN events into score fields and optional framewise F0."""
        durations: list[str] = []
        phonemes: list[str] = []
        pitches: list[str] = []
        note_types: list[str] = []
        previous_phone_tokens: Optional[tuple[str, ...]] = None
        for event in events:
            beats = event.duration_numerator / event.duration_denominator
            durations.append(f"{beats * 60 / bpm:.10g}")
            if event.rest:
                phonemes.append("<SP>")
                pitches.append("0")
                note_types.append("1")
                previous_phone_tokens = None
                continue

            if event.ipa is None:
                raise CESNBackendFormatError(
                    "prepared CESN note is missing its IPA syllable"
                )
            try:
                phone_spans = list(_ipa_to_arpabet_symbols(event.ipa))
            except _UnrepresentableIPASymbol as error:
                _raise_unrepresentable_ipa(event, error)
            except ValueError:
                _raise_unrepresentable_ipa(
                    event,
                    _UnrepresentableIPASymbol(event.ipa, 0),
                )
            phone_tokens = [phone for phone, _start, _end in phone_spans]
            phone_symbols = [f"en_{token}" for token in phone_tokens]
            if not phone_symbols:
                _raise_unrepresentable_ipa(
                    event,
                    _UnrepresentableIPASymbol(event.ipa, 0),
                )
            for symbol, (_phone, start, end) in zip(phone_symbols, phone_spans):
                if symbol not in runtime.processor.phone2idx:
                    _raise_unrepresentable_ipa(
                        event,
                        _UnrepresentableIPASymbol(event.ipa[start:end], start),
                    )
            phonemes.append("en_" + "-".join(phone_tokens))
            if character_name is not None and pitch_range is not None:
                try:
                    midi_pitch = _pitch_to_midi_for_singer(
                        event.pitch or "",
                        character_name,
                        pitch_range,
                    )
                except CESNRangeError as error:
                    error.source_start = event.note_source_start
                    error.source_end = event.note_source_end
                    raise
            else:
                midi_pitch = pitch_to_midi(event.pitch or "")
            pitches.append(str(midi_pitch))
            current_phone_tokens = tuple(phone_tokens)
            is_repeated_slur = event.slur_from_previous and (
                current_phone_tokens == previous_phone_tokens
                or _is_vowel_continuation(previous_phone_tokens, current_phone_tokens)
            )
            note_types.append("3" if is_repeated_slur else "2")
            previous_phone_tokens = current_phone_tokens

        metadata = {
            "duration": " ".join(durations),
            "phoneme": " ".join(phonemes),
            "note_pitch": " ".join(pitches),
            "note_type": " ".join(note_types),
        }
        return metadata

    @staticmethod
    def _extract_performance_f0(
        runtime: _SoulXRuntime,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Extract a score-conditioned performance F0 track from temporary audio."""
        if runtime.f0_extractor is None:
            raise BackendError(
                "SoulX-Singer F0 extraction is unavailable",
                error_code="soulx_f0_extractor_unavailable",
            )
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="celune-soulx-f0-") as directory:
            audio_path = Path(directory) / "performance.wav"
            sf.write(audio_path, audio, runtime.sample_rate, subtype="FLOAT")
            return runtime.f0_extractor.process(str(audio_path), verbose=False)

    def generate_stream(
        self,
        model: _SoulXRuntime,
        **kwargs: object,
    ) -> Iterator[tuple[AudioChunk, int, Optional[JSON]]]:
        """Generate one waveform using score or framewise F0 conditioning."""
        high_precision = getattr(self, "high_precision", False) is True
        text = kwargs.pop("text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not text.lstrip().startswith("{"):
            if not contains_cesn_note_event(text):
                raise CESNValidationError("No CESN notation found")
            text = self.cesn_to_model(text)
        sections = decode_score(text)
        section_count = len(sections)
        owned_loaders: dict[str, CEVoiceLoader] = {}
        self._apply_seed()
        try:
            for section_index, section in enumerate(sections):
                if section.language.split("-")[0].casefold() != "en":
                    raise CESNUnsupportedFeatureError(
                        "SoulX-Singer supports English CESN only"
                    )
                loader = self._singer_loader(section.singer_path, owned_loaders)
                score_text, reference_wav = self._validate_singing_voice(loader)
                prompt_data = self._prompt_data(
                    model,
                    loader,
                    score_text,
                    reference_wav,
                    high_precision=high_precision,
                )
                character_name = (
                    bundle_character_name(loader.bundle) or loader.bundle.path.stem
                )
                target_metadata = self._metadata(
                    section.events,
                    section.bpm,
                    model,
                    character_name=character_name,
                    pitch_range=loader.bundle.singing_pitch_range or (0, 127),
                )
                onset_preroll_frames = round(
                    _MODEL_ONSET_PREROLL_SECONDS
                    * model.sample_rate
                    / model.processor.hop_size
                )
                onset_preroll_samples = onset_preroll_frames * model.processor.hop_size
                onset_preroll_seconds = onset_preroll_samples / model.sample_rate
                model_metadata = _prepend_onset_rest(
                    target_metadata,
                    onset_preroll_seconds,
                )
                with torch.inference_mode():
                    if high_precision:
                        score_target_data = model.processor.process(
                            dict(model_metadata)
                        )
                        performance = model.model.infer(
                            {"prompt": prompt_data, "target": score_target_data},
                            auto_shift=False,
                            pitch_shift=0,
                            n_steps=model.inference_steps,
                            cfg=model.cfg,
                            control="score",
                            use_fp16=True,
                        )
                        performance_audio = np.ascontiguousarray(
                            np.clip(to_numpy_audio(performance), -1.0, 1.0),
                            dtype=np.float32,
                        )
                        extracted_f0 = self._extract_performance_f0(
                            model,
                            performance_audio,
                        )[onset_preroll_frames:]
                        target_f0 = _natural_f0_contour(
                            section.events,
                            target_metadata["duration"].split(),
                            target_metadata["note_pitch"].split(),
                            extracted_f0,
                            model.sample_rate,
                            model.processor.hop_size,
                        )
                        model_metadata["f0"] = " ".join(
                            ["0"] * onset_preroll_frames + target_f0.split()
                        )
                        target_data = model.processor.process(dict(model_metadata))
                        audio = model.model.infer(
                            {"prompt": prompt_data, "target": target_data},
                            auto_shift=False,
                            pitch_shift=0,
                            n_steps=model.inference_steps,
                            cfg=model.cfg,
                            control="melody",
                            use_fp16=True,
                        )
                    else:
                        target_data = model.processor.process(dict(model_metadata))
                        audio = model.model.infer(
                            {"prompt": prompt_data, "target": target_data},
                            auto_shift=False,
                            pitch_shift=0,
                            n_steps=model.inference_steps,
                            cfg=model.cfg,
                            control="score",
                            use_fp16=True,
                        )
                generated_audio = np.ascontiguousarray(
                    np.clip(to_numpy_audio(audio), -1.0, 1.0),
                    dtype=np.float32,
                )
                if generated_audio.size <= onset_preroll_samples:
                    raise BackendError(
                        "SoulX-Singer generated no audio after onset conditioning",
                        error_code="soulx_empty_audio",
                    )
                section_audio = generated_audio[onset_preroll_samples:]
                yield (
                    section_audio,
                    model.sample_rate,
                    {
                        "backend": self.name,
                        "chunk_index": section_index,
                        "chunk_steps": section_count,
                        "total_steps_so_far": section_index + 1,
                        "section_caption": section.display_text,
                        "section_final": True,
                        "first_chunk_time": time.monotonic(),
                        "is_final": section_index + 1 == section_count,
                    },
                )
        finally:
            for loader in owned_loaders.values():
                loader.close()
