# SPDX-License-Identifier: Apache-2.0
"""Parse and render section-based Celune Singing Notation."""

import re
import warnings
import unicodedata
from typing import Union, Optional, NamedTuple, cast
from pathlib import Path
from fractions import Fraction
from dataclasses import field, dataclass

from iso639 import Lang, exceptions

from .xsampa import to_ipa as xsampa_to_ipa
from .exceptions import (
    CESNError,
    CEVoiceError,
    CESNMatchError,
    CESNAlignmentError,
    CESNNotationWarning,
    CESNValidationError,
    InvalidCharacterError,
    CESNUnsupportedFeatureError,
)


class CESNTuplet(NamedTuple):
    """A tuplet grouping actual events into a number of normal durations."""

    actual_count: int
    normal_count: int


class CESNNoteEvent(NamedTuple):
    """A pitched note, duration, and optional tuplet timing."""

    pitch: str
    note_type: int
    dotted: bool
    tuplet: Optional[CESNTuplet] = None
    tuplet_start: bool = False

    @property
    def duration_beats(self) -> Fraction:
        """Return this note's exact duration in quarter-note beats."""
        return _event_duration_beats(self.note_type, self.dotted, self.tuplet)


class CESNRestEvent(NamedTuple):
    """A timed rest that remains in notation without a lyric pair."""

    note_type: int
    dotted: bool
    tuplet: Optional[CESNTuplet] = None
    tuplet_start: bool = False

    @property
    def duration_beats(self) -> Fraction:
        """Return this rest's exact duration in quarter-note beats."""
        return _event_duration_beats(self.note_type, self.dotted, self.tuplet)


class CESNNotePair(NamedTuple):
    """One pitched note aligned with an explicit IPA syllable."""

    event: CESNNoteEvent
    ipa: str
    slur_from_previous: bool = False


class CESNSinger(NamedTuple):
    """A resolved CECHAR v5 pack selected by its character name."""

    name: str
    bundle_path: Path
    pitch_range: tuple[int, int] = (0, 127)


@dataclass(frozen=True)
class CESNBlock:
    """One event-and-lyric pair inside a section."""

    lyrics: str
    events: tuple[Union[CESNNotePair, CESNRestEvent], ...]
    _source_note_spans: tuple[Optional[tuple[int, int]], ...] = field(
        default=(), compare=False, repr=False
    )
    _source_ipa_spans: tuple[Optional[tuple[int, int]], ...] = field(
        default=(), compare=False, repr=False
    )

    @property
    def pairs(self) -> tuple[CESNNotePair, ...]:
        """Return pitched event/IPA pairs, excluding rests."""
        return tuple(event for event in self.events if isinstance(event, CESNNotePair))


@dataclass(frozen=True)
class CESNSection:
    """A continuous inference section with fixed singer and musical settings."""

    bpm: int
    key: str
    scale: str
    language: str
    singer: Optional[CESNSinger]
    blocks: tuple[CESNBlock, ...]


_CESNEvent = Union[CESNNoteEvent, CESNRestEvent]

NOTE_EVENT_PATTERN = re.compile(
    r"(?:(?P<pitch>[a-g](?:[#b])?-?\d+)|(?P<rest>r))\s*:\s*"
    r"(?P<note_type>32|16|8|4|2|1)\s*(?P<dotted>d?)",
    re.IGNORECASE,
)
_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")
_KEY_PATTERN = re.compile(r"[a-g](?:[#b])?\Z", re.IGNORECASE)
_TUPLET_PATTERN = re.compile(r"(?P<actual>\d+)\s*,\s*(?P<normal>\d+)\Z")
_SCALE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural-minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic-minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic-minor": (0, 2, 3, 5, 7, 9, 11),
}
_PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_DEFAULT_BPM = 120
_DEFAULT_KEY = "c"
_DEFAULT_SCALE = "major"


@dataclass(frozen=True)
class _CESNState:
    bpm: int = _DEFAULT_BPM
    key: str = _DEFAULT_KEY
    scale: str = _DEFAULT_SCALE
    language: Optional[str] = None
    singer: Optional[CESNSinger] = None


def _parse_cesn(
    text: str,
    *,
    supported_languages: Optional[tuple[str, ...]] = None,
    location: list[int],
) -> tuple[CESNSection, ...]:
    """Parse CESN directives, align pronunciation, and return fixed-state sections.

    Each section begins when its singer, language, BPM, key, or scale changes.
    Every pitched note requires one dot-separated pronunciation syllable. Raw
    IPA is retained and marked X-SAMPA is converted to IPA. The original lyrics
    remain intact on each block; no spelling-based syllabification is performed.
    Rests remain in each block's event stream but consume no pronunciation.
    A lyric block ending in a period or comma gains an automatic half-note or
    quarter-note rest unless an explicit trailing rest already follows it.

    Args:
        text: Complete CESN notation containing directive and note tags.
        supported_languages: Optional language list from the selected TTS model.

    Returns:
        tuple[CESNSection, ...]: Ordered sections with parsed note blocks.

    Raises:
        CESNValidationError: Notation is malformed or violates a score invariant.
        CESNUnsupportedFeatureError: A feature or language is unavailable.
        CESNAlignmentError: The IPA syllable and pitched note counts differ.
        CESNMatchError: A pitched note has no corresponding IPA syllable.
    """
    if not isinstance(text, str):
        raise CESNValidationError("cesn input must be a string")

    state = _CESNState()
    pending_events: Optional[tuple[_CESNEvent, ...]] = None
    pending_event_spans: tuple[Optional[tuple[int, int]], ...] = ()
    pending_lyrics: list[str] = []
    pending_ipa: Optional[tuple[str, ...]] = None
    pending_ipa_spans: tuple[tuple[int, int], ...] = ()
    pending_slur: Optional[int] = None
    pending_note_span = (0, 1)
    pending_ipa_span: Optional[tuple[int, int]] = None
    pending_slur_span: Optional[tuple[int, int]] = None
    active_tuplet_span: Optional[tuple[int, int]] = None
    pending_state: Optional[_CESNState] = None
    section_state: Optional[_CESNState] = None
    section_blocks: list[CESNBlock] = []
    sections: list[CESNSection] = []
    resolved_singers: dict[str, CESNSinger] = {}
    active_tuplet: Optional[CESNTuplet] = None
    tuplet_remaining = 0

    def close_section() -> None:
        nonlocal section_state, section_blocks
        if active_tuplet is not None:
            raise CESNValidationError(
                "CESN tuplet group cannot cross a section boundary"
            )
        if section_state is None or not section_blocks:
            section_state = None
            section_blocks = []
            return
        language = section_state.language
        if language is None:
            raise CESNValidationError("CESN language must be specified")
        sections.append(
            CESNSection(
                bpm=section_state.bpm,
                key=section_state.key,
                scale=section_state.scale,
                language=language,
                singer=section_state.singer,
                blocks=tuple(section_blocks),
            )
        )
        section_state = None
        section_blocks = []

    def apply_tuplet(events: tuple[_CESNEvent, ...]) -> tuple[_CESNEvent, ...]:
        nonlocal active_tuplet, tuplet_remaining
        annotated: list[_CESNEvent] = []
        for event in events:
            if active_tuplet is None:
                annotated.append(event)
                continue
            annotated.append(
                event._replace(
                    tuplet=active_tuplet,
                    tuplet_start=tuplet_remaining == active_tuplet.actual_count,
                )
            )
            tuplet_remaining -= 1
            if tuplet_remaining == 0:
                active_tuplet = None
        return tuple(annotated)

    def append_block(block: CESNBlock, block_state: _CESNState) -> None:
        nonlocal section_state, section_blocks
        if section_state is not None and section_state != block_state:
            close_section()
        if section_state is None:
            section_state = block_state
        section_blocks.append(block)

    def finish_pending(*, following_rest_only: bool = False) -> None:
        nonlocal pending_events, pending_lyrics, pending_ipa
        nonlocal pending_event_spans, pending_ipa_spans
        nonlocal pending_slur
        nonlocal pending_state
        nonlocal pending_note_span, pending_ipa_span
        nonlocal pending_slur_span
        if pending_events is None:
            return
        if pending_state is None:
            raise CESNValidationError("CESN note list has no musical state")

        lyrics = "".join(pending_lyrics).lstrip()
        note_events = tuple(
            event for event in pending_events if isinstance(event, CESNNoteEvent)
        )
        rests = tuple(
            event for event in pending_events if isinstance(event, CESNRestEvent)
        )

        if pending_state.language is None:
            raise CESNValidationError("CESN language must be specified before notes")
        if not lyrics and note_events:
            raise CESNMatchError(0, _format_note_event(note_events[0]), 0)
        if lyrics and not note_events:
            raise CESNValidationError("CESN lyrics require pitched note events")
        if note_events and pending_ipa is None:
            raise CESNValidationError(
                "CESN IPA syllables are required for pitched notes"
            )
        if pending_slur is not None:
            if len(note_events) != pending_slur:
                raise CESNValidationError(
                    "CESN slur count must match its pitched note count"
                )
            if any(
                isinstance(event, CESNRestEvent)
                for event in pending_events[:pending_slur]
            ):
                raise CESNValidationError("CESN slur cannot include a rest")
        if supported_languages is not None and not _model_supports_language(
            pending_state.language, supported_languages
        ):
            raise CESNUnsupportedFeatureError(
                f"language {pending_state.language!r} is not supported by the selected model"
            )

        if not lyrics and not rests:
            raise CESNValidationError("CESN note list cannot be empty")

        ipa_syllables = pending_ipa or ()
        if len(note_events) > len(ipa_syllables):
            index = len(ipa_syllables)
            raise CESNMatchError(
                index,
                _format_note_event(note_events[index]),
                len(ipa_syllables),
            )
        if len(note_events) < len(ipa_syllables):
            location[:] = pending_ipa_span or location
            raise CESNAlignmentError(len(note_events), len(ipa_syllables))

        event_items: list[Union[CESNNotePair, CESNRestEvent]] = []
        source_note_spans: list[Optional[tuple[int, int]]] = []
        source_ipa_spans: list[Optional[tuple[int, int]]] = []
        note_index = 0
        event_index = 0
        for event in pending_events:
            if isinstance(event, CESNRestEvent):
                event_items.append(event)
                source_note_spans.append(None)
                source_ipa_spans.append(None)
                event_index += 1
                continue
            pair = CESNNotePair(
                event,
                ipa_syllables[note_index],
                slur_from_previous=(
                    pending_slur is not None and 0 < note_index < pending_slur
                ),
            )
            event_items.append(pair)
            source_note_spans.append(pending_event_spans[event_index])
            source_ipa_spans.append(pending_ipa_spans[note_index])
            note_index += 1
            event_index += 1
            if not _note_is_in_scale(
                event.pitch,
                pending_state.key,
                pending_state.scale,
            ):
                warnings.warn(
                    CESNNotationWarning(
                        _format_note_event(event),
                        pending_state.key,
                        pending_state.scale,
                    ),
                    stacklevel=2,
                )

        append_block(
            CESNBlock(
                lyrics=lyrics,
                events=tuple(event_items),
                _source_note_spans=tuple(source_note_spans),
                _source_ipa_spans=tuple(source_ipa_spans),
            ),
            pending_state,
        )
        trailing_text = lyrics.rstrip()
        if (
            trailing_text
            and trailing_text[-1] in {".", ","}
            and not following_rest_only
            and not (event_items and isinstance(event_items[-1], CESNRestEvent))
        ):
            note_type = 2 if trailing_text.endswith(".") else 4
            rest = cast(
                CESNRestEvent,
                apply_tuplet((CESNRestEvent(note_type, False),))[0],
            )
            append_block(CESNBlock(lyrics="", events=(rest,)), pending_state)
        pending_events = None
        pending_event_spans = ()
        pending_lyrics = []
        pending_ipa = None
        pending_ipa_spans = ()
        pending_slur = None
        pending_state = None
        pending_note_span = (0, 1)
        pending_ipa_span = None
        pending_slur_span = None

    def finish_pending_at(
        current_start: int,
        current_end: int,
        *,
        following_tag: Optional[str] = None,
    ) -> None:
        if pending_events is not None:
            location[:] = pending_note_span
        following_rest_only = False
        if following_tag is not None:
            try:
                following_events = _parse_note_events(following_tag)
            except CESNValidationError:
                pass
            else:
                following_rest_only = bool(following_events) and all(
                    isinstance(event, CESNRestEvent) for event in following_events
                )
        finish_pending(following_rest_only=following_rest_only)
        location[:] = [current_start, current_end]

    for is_tag, value, start, end in _tokens_with_spans(text):
        location[:] = [start, end]
        if not is_tag:
            if pending_events is not None:
                pending_lyrics.append(value)
            elif value.strip():
                raise CESNValidationError("CESN lyrics must follow a note list")
            continue

        tag = value.strip()
        if tag.startswith("#"):
            continue

        tag_name, separator, tag_value = tag.partition(":")
        directive = tag_name.strip().casefold()
        argument = tag_value.strip()
        pending_has_no_lyrics = (
            pending_events is not None and not "".join(pending_lyrics).strip()
        )
        pending_event_items: tuple[_CESNEvent, ...] = ()
        if pending_events is not None:
            pending_event_items = pending_events
        pending_has_pitched_notes = any(
            isinstance(event, CESNNoteEvent) for event in pending_event_items
        )
        if (
            pending_has_no_lyrics
            and pending_has_pitched_notes
            and directive == "tuplet"
        ):
            raise CESNValidationError("CESN tuplet must precede the events it affects")
        if (
            pending_has_no_lyrics
            and pending_has_pitched_notes
            and directive in {"bpm", "lang", "singer", "slur"}
        ):
            raise CESNValidationError(
                "CESN settings and feature tags must precede the note list"
            )
        is_raw_ipa_syllable_list = (
            pending_has_no_lyrics
            and pending_has_pitched_notes
            and not separator
            and directive not in {"slur", "tuplet"}
        )
        is_xsampa_syllable_list = (
            pending_has_no_lyrics
            and pending_has_pitched_notes
            and separator
            and directive == "x-sampa"
        )
        is_pronunciation_syllable_list = (
            is_raw_ipa_syllable_list or is_xsampa_syllable_list
        )
        if not is_pronunciation_syllable_list:
            finish_pending_at(start, end, following_tag=tag)

        if separator and directive == "bpm":
            changes = _parse_tempo_key_scale(tag)
            updated = _CESNState(
                bpm=changes[0],
                key=changes[1],
                scale=changes[2],
                language=state.language,
                singer=state.singer,
            )
            if updated != state:
                close_section()
                state = updated
                if pending_has_no_lyrics:
                    pending_state = updated
            continue

        if separator and directive == "lang":
            language = _parse_language(argument)
            updated = _CESNState(
                bpm=state.bpm,
                key=state.key,
                scale=state.scale,
                language=language,
                singer=state.singer,
            )
            if updated != state:
                close_section()
                state = updated
                if pending_has_no_lyrics:
                    pending_state = updated
            continue

        if separator and directive == "singer":
            name = _parse_singer_name(argument)
            identity = unicodedata.normalize("NFC", name).casefold()
            singer = resolved_singers.get(identity)
            if singer is None:
                try:
                    singer = resolve_singer(name)
                except InvalidCharacterError as error:
                    raise CESNUnsupportedFeatureError(
                        f"CESN singer {name!r} does not provide a singing voice"
                    ) from error
                resolved_singers[identity] = singer
            updated = _CESNState(
                bpm=state.bpm,
                key=state.key,
                scale=state.scale,
                language=state.language,
                singer=singer,
            )
            if updated != state:
                close_section()
                state = updated
                if pending_has_no_lyrics:
                    pending_state = updated
            continue

        tag_content_start = start + 1 + (len(value) - len(value.lstrip()))
        if is_pronunciation_syllable_list:
            if pending_ipa is not None:
                raise CESNValidationError(
                    "CESN pronunciation syllables may appear only once"
                )
            pronunciation = argument if is_xsampa_syllable_list else tag
            pronunciation_offset = tag_content_start + tag.find(pronunciation)
            pending_ipa = (
                _parse_xsampa(argument) if is_xsampa_syllable_list else _parse_ipa(tag)
            )
            pending_ipa_spans = _ipa_source_spans(
                pronunciation,
                pronunciation_offset,
            )
            pending_ipa_span = (start, end)
            continue

        if directive == "slur":
            if pending_slur is not None:
                raise CESNValidationError(
                    "CESN slur must be used only once per note group"
                )
            if not separator:
                raise CESNValidationError("CESN slur requires a syllable count")
            pending_slur = _parse_slur_count(argument)
            pending_slur_span = (start, end)
            continue

        if directive == "tuplet":
            if not separator:
                raise CESNValidationError("CESN tuplet requires an actual,normal ratio")
            if active_tuplet is not None:
                raise CESNValidationError("CESN tuplets cannot overlap")
            active_tuplet = _parse_tuplet(argument)
            tuplet_remaining = active_tuplet.actual_count
            active_tuplet_span = (start, end)
            continue

        events = _parse_note_events(tag)
        if not events:
            raise CESNValidationError("CESN note list cannot be empty")
        pending_event_spans = _note_event_source_spans(tag, tag_content_start)
        events = apply_tuplet(events)
        pending_events = events
        pending_lyrics = []
        pending_state = state
        pending_note_span = (start, end)

    finish_pending_at(location[0], location[1])
    if pending_ipa is not None:
        location[:] = pending_ipa_span or location
        raise CESNValidationError("CESN IPA syllables require notes and lyrics")
    if pending_slur is not None:
        location[:] = pending_slur_span or location
        raise CESNValidationError("CESN slur requires pitched notes and IPA syllables")
    if active_tuplet is not None:
        location[:] = active_tuplet_span or location
        raise CESNValidationError(f"CESN tuplet is missing {tuplet_remaining} event(s)")
    close_section()
    if not sections:
        raise CESNValidationError("CESN notation must contain at least one note list")
    return tuple(sections)


def parse_cesn(
    text: str,
    *,
    supported_languages: Optional[tuple[str, ...]] = None,
) -> tuple[CESNSection, ...]:
    """Parse CESN, convert marked X-SAMPA, and locate notation errors.

    Args:
        text: Complete CESN notation containing directive and note tags.
        supported_languages: Optional language list from the selected TTS model.

    Returns:
        tuple[CESNSection, ...]: Ordered sections with parsed note blocks.

    Raises:
        CESNValidationError: If the notation is malformed or violates an invariant.
        CESNUnsupportedFeatureError: If a requested language or feature is unsupported.
        CESNAlignmentError: If IPA syllables outnumber pitched notes.
        CESNMatchError: If a pitched event has no corresponding IPA syllable.
    """
    location = [0, 1]
    try:
        return _parse_cesn(
            text,
            supported_languages=supported_languages,
            location=location,
        )
    except CESNError as error:
        if error.source_start is None:
            error.source_start, error.source_end = location
        raise


def render_cesn(sections: tuple[CESNSection, ...]) -> str:
    """Render parsed sections as one canonical, section-marked CESN string.

    Each block keeps its original lyric text and groups its notes with their
    explicitly segmented IPA syllables. Slurs, explicit rests, and automatic
    punctuation rests retain their event order.

    Args:
        sections: Parsed CESN sections in inference order.

    Returns:
        str: Canonical notation with repeated settings at every section boundary.
    """
    rendered: list[str] = []
    for section in sections:
        rendered.append(
            f"[bpm:{section.bpm},key:{section.key},scale:{section.scale}]"
            f"[lang:{section.language}]"
        )
        if section.singer is not None:
            rendered.append(f"[singer:{section.singer.name}]")
        for block in section.blocks:
            items = block.events
            if not items:
                rendered.append(block.lyrics)
                continue
            note_events = tuple(
                item.event if isinstance(item, CESNNotePair) else item for item in items
            )
            if any(event.tuplet_start for event in note_events[1:]):
                raise CESNValidationError(
                    "CESN tuplet must begin at the start of a note list"
                )

            rendered.append(_format_tuplet_start(note_events[0]))
            pairs = tuple(item for item in items if isinstance(item, CESNNotePair))
            slurred_positions = tuple(
                index for index, pair in enumerate(pairs) if pair.slur_from_previous
            )
            if slurred_positions:
                if slurred_positions != tuple(range(1, len(pairs))):
                    raise CESNValidationError(
                        "CESN slur must cover every note in its group"
                    )
                pair_positions = tuple(
                    index
                    for index, item in enumerate(items)
                    if isinstance(item, CESNNotePair)
                )
                if any(
                    isinstance(item, CESNRestEvent)
                    for item in items[pair_positions[0] : pair_positions[-1] + 1]
                ):
                    raise CESNValidationError("CESN slur cannot include a rest")
                rendered.append(f"[slur:{len(pairs)}]")

            event_text = ",".join(
                _format_event_contents(event) for event in note_events
            )
            rendered.append(f"[{event_text}]")
            if pairs:
                if any(not pair.ipa for pair in pairs):
                    raise CESNValidationError(
                        "CESN pitched notes require explicit IPA syllables"
                    )
                rendered.append(f"[{'.'.join(pair.ipa for pair in pairs)}]")
            rendered.append(block.lyrics)
    return "".join(rendered)


def tokens(text: str) -> tuple[tuple[bool, str], ...]:
    """Split notation into literal text and bracketed tags."""
    parts: list[tuple[bool, str]] = []
    for is_tag, value, _start, _end in _tokens_with_spans(text):
        parts.append((is_tag, value))
    return tuple(parts)


def _tokens_with_spans(text: str) -> tuple[tuple[bool, str, int, int], ...]:
    """Split notation into tokens while retaining their source offsets."""
    parts: list[tuple[bool, str, int, int]] = []
    cursor = 0
    while cursor < len(text):
        opening = text.find("[", cursor)
        if opening < 0:
            closing = text.find("]", cursor)
            if closing >= 0:
                error = CESNValidationError("CESN closing bracket has no opening tag")
                error.source_start = closing
                error.source_end = closing + 1
                raise error
            parts.append((False, text[cursor:], cursor, len(text)))
            break
        if opening > cursor:
            literal = text[cursor:opening]
            closing = literal.find("]")
            if closing >= 0:
                position = cursor + closing
                error = CESNValidationError("CESN closing bracket has no opening tag")
                error.source_start = position
                error.source_end = position + 1
                raise error
            parts.append((False, literal, cursor, opening))
        closing = text.find("]", opening + 1)
        if closing < 0 or "[" in text[opening + 1 : closing]:
            error = CESNValidationError("CESN tag is not closed")
            error.source_start = opening
            error.source_end = len(text) if closing < 0 else closing + 1
            raise error
        parts.append((True, text[opening + 1 : closing], opening, closing + 1))
        cursor = closing + 1
    return tuple(parts)


def _parse_tempo_key_scale(tag: str) -> tuple[int, str, str]:
    """Parse one complete BPM, key, and scale directive."""
    fields: dict[str, str] = {}
    for field_text in tag.split(","):
        name, separator, value = field_text.partition(":")
        field_name = name.strip().casefold()
        if not separator or field_name not in {"bpm", "key", "scale"}:
            raise CESNValidationError(
                "CESN settings require bpm, key, and scale fields"
            )
        if field_name in fields or not value.strip():
            raise CESNValidationError(
                "CESN settings contain a duplicate or empty field"
            )
        fields[field_name] = value.strip()
    if set(fields) != {"bpm", "key", "scale"}:
        raise CESNValidationError("CESN settings require bpm, key, and scale fields")

    try:
        bpm = int(fields["bpm"])
    except ValueError as error:
        raise CESNValidationError("CESN BPM must be a positive integer") from error
    if bpm <= 0:
        raise CESNValidationError("CESN BPM must be a positive integer")

    key = fields["key"].casefold()
    if _KEY_PATTERN.fullmatch(key) is None:
        raise CESNValidationError("CESN key must be a note name: A-G, # or b")

    scale = fields["scale"].casefold().replace("_", "-")
    if scale not in _SCALE_INTERVALS:
        raise CESNUnsupportedFeatureError(f"unsupported CESN scale {scale!r}")
    return bpm, key, scale


def _parse_language(value: str) -> str:
    """Validate and normalize an explicit CESN language identifier."""
    language = value.strip().casefold().replace("_", "-")
    if _LANGUAGE_PATTERN.fullmatch(language) is None:
        raise CESNValidationError("CESN language must be an ISO language code")
    return language


def _parse_tuplet(value: str) -> CESNTuplet:
    """Validate an actual-event count and normal-duration count."""
    match = _TUPLET_PATTERN.fullmatch(value)
    if match is None:
        raise CESNValidationError("CESN tuplet must contain two positive integers")
    actual_count, normal_count = (int(count) for count in match.groups())
    if actual_count < 2 or normal_count < 1:
        raise CESNValidationError(
            "CESN tuplet counts must be positive and actual must exceed one"
        )
    return CESNTuplet(actual_count, normal_count)


def _event_duration_beats(
    note_type: int,
    dotted: bool,
    tuplet: Optional[CESNTuplet],
) -> Fraction:
    """Calculate an exact event duration using quarter-note beats."""
    duration = Fraction(4, note_type)
    if dotted:
        duration *= Fraction(3, 2)
    if tuplet is not None:
        duration *= Fraction(tuplet.normal_count, tuplet.actual_count)
    return duration


def _model_supports_language(
    language: str,
    supported_languages: tuple[str, ...],
) -> bool:
    """Return whether the model accepts an explicit CESN language tag."""
    language_keys = _language_equivalents(language)
    model_keys: set[str] = set()
    for supported in supported_languages:
        model_keys.update(_language_equivalents(supported.casefold()))
    return bool(language_keys.intersection(model_keys))


def _language_equivalents(language: str) -> set[str]:
    """Return short and three-letter forms for a language tag."""
    normalized = language.strip().casefold().replace("_", "-")
    pieces = normalized.split("-")
    primary = pieces[0]
    equivalents = {normalized, primary}
    try:
        language_info = Lang(primary)
    except (exceptions.DeprecatedLanguageValue, exceptions.InvalidLanguageValue):
        return equivalents
    if language_info.pt1:
        equivalents.add(language_info.pt1.casefold())
    if language_info.pt3:
        equivalents.add(language_info.pt3.casefold())
        if len(pieces) > 1:
            script = next(
                (piece for piece in pieces[1:] if len(piece) == 4 and piece.isalpha()),
                None,
            )
            if script is not None:
                equivalents.add(f"{language_info.pt3.casefold()}-{script}")
    return equivalents


def _parse_singer_name(value: str) -> str:
    """Validate a character name from a singer directive."""
    name = value.strip()
    if not name or "[" in name or "]" in name:
        raise CESNValidationError("CESN singer name is empty or malformed")
    return name


def resolve_singer(name: str) -> CESNSinger:
    """Find a matching CECHAR v5 pack that contains the hidden singing voice."""
    from .cevoice import (
        V5_VERSION,
        SINGING_VOICE_NAME,
        CEVoice,
        active_bundle_path,
        bundled_voices_dir,
        bundle_character_name,
    )

    normalized_name = unicodedata.normalize("NFC", name).casefold()
    active_path = active_bundle_path().resolve()
    candidate_paths = (active_path, *sorted(bundled_voices_dir().glob("*.cevoice")))
    seen_paths: set[Path] = set()
    for path in candidate_paths:
        path = path.resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            bundle = CEVoice.open(path)
        except (OSError, CEVoiceError):
            continue
        bundle_name = bundle_character_name(bundle) or path.stem
        if unicodedata.normalize("NFC", bundle_name).casefold() != normalized_name:
            continue
        if (
            bundle.metadata.get("version") == V5_VERSION
            and bundle.singing_voice == SINGING_VOICE_NAME
        ):
            return CESNSinger(
                bundle_name,
                path.resolve(),
                bundle.singing_pitch_range or (0, 127),
            )

    raise InvalidCharacterError(name)


def resolve_active_singer() -> CESNSinger:
    """Return the active pack when it provides a CECHAR v5 singing voice."""
    from .cevoice import (
        V5_VERSION,
        SINGING_VOICE_NAME,
        CEVoice,
        active_bundle_path,
        bundle_character_name,
    )

    path = active_bundle_path()
    try:
        bundle = CEVoice.open(path)
    except (OSError, CEVoiceError) as error:
        raise InvalidCharacterError(path.stem) from error
    if (
        bundle.metadata.get("version") != V5_VERSION
        or bundle.singing_voice != SINGING_VOICE_NAME
    ):
        raise InvalidCharacterError(bundle_character_name(bundle) or path.stem)
    return CESNSinger(
        bundle_character_name(bundle) or path.stem,
        path.resolve(),
        bundle.singing_pitch_range or (0, 127),
    )


def _parse_ipa(value: str) -> tuple[str, ...]:
    """Parse dot-separated IPA syllables without a tag prefix or slashes."""
    source = unicodedata.normalize("NFC", value.strip())
    if not source or "/" in source or ":" in source:
        raise CESNValidationError(
            "CESN IPA syllables must be written without labels or slash delimiters"
        )
    syllables = tuple(part.strip() for part in source.split("."))
    if any(not syllable for syllable in syllables):
        raise CESNValidationError("CESN IPA syllables cannot be empty")
    return syllables


def _parse_xsampa(value: str) -> tuple[str, ...]:
    """Convert explicit X-SAMPA syllables to the parser's canonical IPA."""
    if "/" in value:
        raise CESNValidationError(
            "CESN X-SAMPA syllables must be written without slash delimiters"
        )
    try:
        ipa = xsampa_to_ipa(value)
    except ValueError as error:
        raise CESNValidationError(str(error)) from error
    return _parse_ipa(ipa)


def _parse_slur_count(value: str) -> int:
    """Parse the number of notes in a slurred score group."""
    if not value.isdecimal():
        raise CESNValidationError("CESN slur count must be an integer of at least two")
    count = int(value)
    if count < 2:
        raise CESNValidationError("CESN slur count must be an integer of at least two")
    return count


def _parse_note_events(source: str) -> tuple[_CESNEvent, ...]:
    """Validate and parse a comma-separated list of notes and rests."""
    if not source.strip():
        raise CESNValidationError("CESN note list cannot be empty")

    events: list[_CESNEvent] = []
    offset = 0
    for raw_event in source.split(","):
        candidate = raw_event.strip()
        match = NOTE_EVENT_PATTERN.fullmatch(candidate)
        if match is None:
            raise CESNValidationError(f"malformed CESN note event at offset {offset}")
        pitch, rest, note_type, dotted = match.groups()
        if rest is not None:
            events.append(CESNRestEvent(int(note_type), bool(dotted)))
        else:
            events.append(
                CESNNoteEvent(
                    pitch=pitch.casefold(),
                    note_type=int(note_type),
                    dotted=bool(dotted),
                )
            )
        offset += len(raw_event) + 1
    return tuple(events)


def _note_event_source_spans(
    source: str,
    source_start: int,
) -> tuple[Optional[tuple[int, int]], ...]:
    """Locate each pitched event's pitch text inside its original note tag."""
    spans: list[Optional[tuple[int, int]]] = []
    offset = 0
    for raw_event in source.split(","):
        candidate = raw_event.strip()
        match = NOTE_EVENT_PATTERN.fullmatch(candidate)
        if match is None:
            raise CESNValidationError(f"malformed CESN note event at offset {offset}")
        pitch = match.group("pitch")
        if pitch is None:
            spans.append(None)
        else:
            candidate_start = (
                source_start + offset + len(raw_event) - len(raw_event.lstrip())
            )
            start = candidate_start + match.start("pitch")
            spans.append((start, start + len(pitch)))
        offset += len(raw_event) + 1
    return tuple(spans)


def _ipa_source_spans(value: str, source_start: int) -> tuple[tuple[int, int], ...]:
    """Locate each dot-separated IPA or X-SAMPA syllable in its source tag."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for syllable in value.split("."):
        stripped = syllable.strip()
        start = source_start + offset + len(syllable) - len(syllable.lstrip())
        spans.append((start, start + len(stripped)))
        offset += len(syllable) + 1
    return tuple(spans)


def _format_note_event(event: CESNNoteEvent) -> str:
    """Return the canonical textual representation of one pitched event."""
    dotted = "d" if event.dotted else ""
    return f"{event.pitch}:{event.note_type}{dotted}"


def _format_rest_event(event: CESNRestEvent) -> str:
    """Return the canonical textual representation of one rest."""
    dotted = "d" if event.dotted else ""
    return f"r:{event.note_type}{dotted}"


def _format_event_contents(event: _CESNEvent) -> str:
    """Return one note-list event without its surrounding brackets."""
    if isinstance(event, CESNRestEvent):
        return _format_rest_event(event)
    return _format_note_event(event)


def _format_tuplet_start(event: Union[CESNNoteEvent, CESNRestEvent]) -> str:
    """Render a tuplet directive immediately before its first event."""
    if not event.tuplet_start:
        return ""
    if event.tuplet is None:
        raise CESNValidationError("CESN tuplet start has no ratio")
    return f"[tuplet:{event.tuplet.actual_count},{event.tuplet.normal_count}]"


def _note_is_in_scale(pitch: str, key: str, scale: str) -> bool:
    """Return whether one pitch class belongs to the selected scale."""
    note = re.fullmatch(r"([a-g])([#b])?-?\d+", pitch)
    root = re.fullmatch(r"([a-g])([#b])?", key)
    if note is None or root is None:
        return False
    note_value = _pitch_class(note.group(1), note.group(2))
    root_value = _pitch_class(root.group(1), root.group(2))
    return (note_value - root_value) % 12 in _SCALE_INTERVALS[scale]


def _pitch_class(letter: str, accidental: Optional[str]) -> int:
    """Return a pitch-class integer for one note spelling."""
    value = _PITCH_CLASSES[letter]
    if accidental == "#":
        value += 1
    elif accidental == "b":
        value -= 1
    return value % 12
