# SPDX-License-Identifier: Apache-2.0
"""Parse and render section-based Celune Singing Notation."""

import re
import warnings
import unicodedata
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass
from typing import Union, Optional, NamedTuple

from syllabreak import Syllabreak
from iso639 import Lang, exceptions

from .exceptions import (
    CEVoiceError,
    CESNError,
    CESNMatchError,
    CESNAlignmentError,
    CESNNotationWarning,
    CESNUnsupportedFeatureError,
    CESNValidationError,
    InvalidCharacterError,
)


class CESNTuplet(NamedTuple):
    """A tuplet grouping actual events into a number of normal durations."""

    actual_count: int
    normal_count: int


class CESNNoteEvent(NamedTuple):
    """A pitched note and duration assigned to one lyric syllable."""

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
    """One pitched note aligned with a lyric syllable and optional overrides."""

    event: CESNNoteEvent
    syllable: str
    ipa: Optional[str] = None
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

    @property
    def pairs(self) -> tuple[CESNNotePair, ...]:
        """Return pitched event/syllable pairs, excluding rests."""
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
    """Parse CESN directives, align lyrics, and return fixed-state sections.

    Each section begins when its singer, language, BPM, key, or scale changes.
    Lyrics use the explicitly selected language for Syllabreak; automatic
    language detection is never used. Rests remain in each block's event stream
    but do not consume a lyric syllable.

    Args:
        text: Complete CESN notation containing directive and note tags.
        supported_languages: Optional language list from the selected TTS model.

    Returns:
        tuple[CESNSection, ...]: Ordered sections with parsed note blocks.

    Raises:
        CESNValidationError: Notation is malformed or violates a score invariant.
        CESNUnsupportedFeatureError: A feature or language is unavailable.
        CESNAlignmentError: There are more lyric syllables than pitched notes.
        CESNMatchError: A pitched note has no corresponding lyric syllable.
    """
    if not isinstance(text, str):
        raise CESNValidationError("cesn input must be a string")

    state = _CESNState()
    pending_events: Optional[tuple[_CESNEvent, ...]] = None
    pending_lyrics: list[str] = []
    pending_pattern: Optional[tuple[str, ...]] = None
    pending_ipa: Optional[tuple[str, ...]] = None
    pending_slur: Optional[int] = None
    pending_note_span = (0, 1)
    pending_pattern_span: Optional[tuple[int, int]] = None
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

    def finish_pending() -> None:
        nonlocal pending_events, pending_lyrics, pending_pattern, pending_ipa
        nonlocal pending_slur
        nonlocal pending_state
        nonlocal pending_note_span, pending_pattern_span, pending_ipa_span
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
        if pending_ipa is not None:
            if pending_pattern is None:
                raise CESNValidationError(
                    "CESN IPA override requires a syllable pattern"
                )
            if len(pending_ipa) != len(pending_pattern):
                raise CESNValidationError(
                    "CESN IPA syllables must match the syllable pattern"
                )
        if pending_slur is not None:
            if pending_pattern is None:
                raise CESNValidationError("CESN slur requires a syllable pattern")
            if len(pending_pattern) != pending_slur:
                raise CESNValidationError(
                    "CESN slur count must match its syllable pattern"
                )
            if any(
                isinstance(event, CESNRestEvent)
                for event in pending_events[:pending_slur]
            ):
                raise CESNValidationError("CESN slur cannot include a rest")
        syllabification_language = _syllabreak_language(pending_state.language)
        if supported_languages is not None and not _model_supports_language(
            pending_state.language,
            syllabification_language,
            supported_languages,
        ):
            raise CESNUnsupportedFeatureError(
                f"language {pending_state.language!r} is not supported by the selected model"
            )

        if not lyrics and note_events:
            raise CESNMatchError(0, _format_note_event(note_events[0]), 0)
        if not lyrics and not rests:
            raise CESNValidationError("CESN note list cannot be empty")

        syllables = (
            _syllabify(lyrics, syllabification_language, pending_pattern)
            if lyrics
            else ()
        )
        if len(note_events) > len(syllables):
            index = len(syllables)
            raise CESNMatchError(
                index,
                _format_note_event(note_events[index]),
                len(syllables),
            )
        if len(note_events) < len(syllables):
            raise CESNAlignmentError(len(note_events), len(syllables))

        event_items: list[Union[CESNNotePair, CESNRestEvent]] = []
        note_index = 0
        for event in pending_events:
            if isinstance(event, CESNRestEvent):
                event_items.append(event)
                continue
            ipa = (
                pending_ipa[note_index]
                if pending_ipa is not None and note_index < len(pending_ipa)
                else None
            )
            pair = CESNNotePair(
                event,
                syllables[note_index],
                ipa,
                slur_from_previous=(
                    pending_slur is not None and 0 < note_index < pending_slur
                ),
            )
            event_items.append(pair)
            note_index += 1
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
            CESNBlock(lyrics=lyrics, events=tuple(event_items)),
            pending_state,
        )
        pending_events = None
        pending_lyrics = []
        pending_pattern = None
        pending_ipa = None
        pending_slur = None
        pending_state = None
        pending_note_span = (0, 1)
        pending_pattern_span = None
        pending_ipa_span = None
        pending_slur_span = None

    def finish_pending_at(current_start: int, current_end: int) -> None:
        if pending_events is not None:
            location[:] = pending_note_span
        finish_pending()
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
        if pending_has_no_lyrics and directive == "tuplet":
            raise CESNValidationError("CESN tuplet must precede the events it affects")
        if not (
            pending_has_no_lyrics
            and directive in {"bpm", "ipa", "lang", "pattern", "singer", "slur"}
        ):
            finish_pending_at(start, end)

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
            _syllabreak_language(language)
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

        if separator and directive == "pattern":
            if pending_pattern is not None:
                raise CESNValidationError(
                    "CESN pattern must be followed by a lyric word"
                )
            pending_pattern = _parse_pattern(argument)
            pending_pattern_span = (start, end)
            continue

        if separator and directive == "ipa":
            if pending_ipa is not None:
                raise CESNValidationError(
                    "CESN IPA override must be used only once per word"
                )
            pending_ipa = _parse_ipa(argument)
            pending_ipa_span = (start, end)
            continue

        if directive == "slur":
            if pending_slur is not None:
                raise CESNValidationError("CESN slur must be used only once per word")
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

        events = apply_tuplet(_parse_note_events(tag))
        if not events:
            raise CESNValidationError("CESN note list cannot be empty")
        pending_events = events
        pending_lyrics = []
        pending_state = state
        pending_note_span = (start, end)

    finish_pending_at(location[0], location[1])
    if pending_pattern is not None:
        location[:] = pending_pattern_span or location
        raise CESNValidationError("CESN pattern must be followed by a lyric word")
    if pending_ipa is not None:
        location[:] = pending_ipa_span or location
        raise CESNValidationError(
            "CESN ipa override requires a syllable pattern and word"
        )
    if pending_slur is not None:
        location[:] = pending_slur_span or location
        raise CESNValidationError("CESN slur requires a syllable pattern and word")
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
    """Parse CESN and attach source offsets to notation errors.

    Args:
        text: Complete CESN notation containing directive and note tags.
        supported_languages: Optional language list from the selected TTS model.

    Returns:
        tuple[CESNSection, ...]: Ordered sections with parsed note blocks.

    Raises:
        CESNValidationError: If the notation is malformed or violates an invariant.
        CESNUnsupportedFeatureError: If a requested language or feature is unsupported.
        CESNAlignmentError: If syllables outnumber pitched notes.
        CESNMatchError: If a pitched event has no corresponding syllable.
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

    Pitched note/syllable pairs are rendered individually so explicit patterns
    survive serialization. Slurred words stay grouped, and rests remain at their
    original event position.

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
            index = 0
            while index < len(items):
                item = items[index]
                if isinstance(item, CESNRestEvent):
                    rendered.append(_format_tuplet_start(item))
                    rendered.append(_format_rest_event(item))
                    index += 1
                    continue

                if item.slur_from_previous:
                    raise CESNValidationError("cesn slur has no first syllable")
                group_end = index + 1
                next_item = items[group_end] if group_end < len(items) else None
                if (
                    index == 0
                    and isinstance(next_item, CESNNotePair)
                    and next_item.slur_from_previous
                ):
                    while group_end < len(items):
                        next_item = items[group_end]
                        if not isinstance(next_item, CESNNotePair):
                            break
                        if not next_item.slur_from_previous:
                            break
                        group_end += 1
                    _render_slurred_word(items[index:group_end], rendered)
                    index = group_end
                    continue

                rendered.append(_format_tuplet_start(item.event))
                rendered.append(f"[{_format_note_event(item.event)}]")
                if item.ipa is not None:
                    word_ranges = _word_ranges(item.syllable)
                    if len(word_ranges) != 1:
                        raise CESNValidationError(
                            "CESN IPA override must align with one lyric word"
                        )
                    start, end = word_ranges[0]
                    rendered.append(
                        f"[pattern:{item.syllable[start:end]}][ipa:/{item.ipa}/]"
                    )
                rendered.append(item.syllable)
                index += 1
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


def _syllabreak_language(language: str) -> str:
    """Map an explicit language tag to a supported Syllabreak language."""
    supported = _supported_syllabreak_languages()
    if language in supported:
        return language

    pieces = language.split("-")
    primary = pieces[0]
    try:
        language_info = Lang(primary)
    except (
        exceptions.DeprecatedLanguageValue,
        exceptions.InvalidLanguageValue,
    ) as error:
        raise CESNUnsupportedFeatureError(
            f"language {language!r} is not supported by Syllabreak"
        ) from error

    language_code = language_info.pt3 or primary
    script = next(
        (piece for piece in pieces[1:] if len(piece) == 4 and piece.isalpha()),
        None,
    )
    syllabreak_code = (
        f"{language_code}-{script}" if script is not None else language_code
    )
    if syllabreak_code not in supported:
        raise CESNUnsupportedFeatureError(
            f"language {language!r} is not supported by Syllabreak"
        )
    return syllabreak_code


def _model_supports_language(
    language: str,
    syllabreak_language: str,
    supported_languages: tuple[str, ...],
) -> bool:
    """Return whether the model and Syllabreak accept the same language."""
    language_keys = _language_equivalents(language)
    language_keys.update(_language_equivalents(syllabreak_language))
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


def _parse_pattern(value: str) -> tuple[str, ...]:
    """Parse comma-separated syllable fragments for the next lyric word."""
    pieces = tuple(piece.strip() for piece in value.split(","))
    if not pieces or any(not piece or not _is_one_word(piece) for piece in pieces):
        raise CESNValidationError("CESN pattern must contain comma-separated syllables")
    return pieces


def _parse_ipa(value: str) -> tuple[str, ...]:
    """Parse a slash-delimited IPA word into explicitly divided syllables."""
    source = unicodedata.normalize("NFC", value.strip())
    if len(source) < 3 or not source.startswith("/") or not source.endswith("/"):
        raise CESNValidationError("CESN IPA override must be slash-delimited")
    syllables = tuple(part.strip() for part in source[1:-1].split("."))
    if not syllables or any(not syllable or "/" in syllable for syllable in syllables):
        raise CESNValidationError("CESN IPA override must contain IPA syllables")
    return syllables


def _parse_slur_count(value: str) -> int:
    """Parse the number of syllables in a word-scoped slur."""
    if not value.isdecimal():
        raise CESNValidationError("CESN slur count must be an integer of at least two")
    count = int(value)
    if count < 2:
        raise CESNValidationError("CESN slur count must be an integer of at least two")
    return count


def _render_slurred_word(
    items: tuple[Union[CESNNotePair, CESNRestEvent], ...],
    rendered: list[str],
) -> None:
    """Render a word-scoped slur as one aligned note-list/lyric block."""
    pairs = tuple(item for item in items if isinstance(item, CESNNotePair))
    if len(pairs) != len(items) or len(pairs) < 2:
        raise CESNValidationError("CESN slur must contain adjacent pitched notes")
    fragments: list[str] = []
    for pair in pairs:
        word_ranges = _word_ranges(pair.syllable)
        if len(word_ranges) != 1:
            raise CESNValidationError("CESN slur must align with one lyric word")
        start, end = word_ranges[0]
        fragments.append(pair.syllable[start:end])
    lyrics = "".join(pair.syllable for pair in pairs)
    word_ranges = _word_ranges(lyrics)
    if (
        len(word_ranges) != 1
        or "".join(fragments) != lyrics[word_ranges[0][0] : word_ranges[0][1]]
    ):
        raise CESNValidationError("CESN slur must align with one lyric word")
    rendered.append(_format_tuplet_start(pairs[0].event))
    rendered.append(f"[pattern:{','.join(fragments)}]")
    if any(pair.ipa is not None for pair in pairs):
        if any(pair.ipa is None for pair in pairs):
            raise CESNValidationError(
                "CESN slur IPA overrides must cover every syllable"
            )
        rendered.append(f"[ipa:/{'.'.join(pair.ipa or '' for pair in pairs)}/]")
    rendered.append(f"[slur:{len(pairs)}]")
    events = ",".join(_format_note_event(pair.event) for pair in pairs)
    rendered.append(f"[{events}]")
    rendered.append(lyrics)


def _is_one_word(value: str) -> bool:
    """Return whether a pattern fragment contains exactly one word."""
    spans = _word_ranges(value)
    return len(spans) == 1 and spans[0] == (0, len(value))


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


def _format_note_event(event: CESNNoteEvent) -> str:
    """Return the canonical textual representation of one pitched event."""
    dotted = "d" if event.dotted else ""
    return f"{event.pitch}:{event.note_type}{dotted}"


def _format_rest_event(event: CESNRestEvent) -> str:
    """Return the canonical textual representation of one rest."""
    dotted = "d" if event.dotted else ""
    return f"[r:{event.note_type}{dotted}]"


def _format_tuplet_start(event: Union[CESNNoteEvent, CESNRestEvent]) -> str:
    """Render a tuplet directive immediately before its first event."""
    if not event.tuplet_start:
        return ""
    if event.tuplet is None:
        raise CESNValidationError("CESN tuplet start has no ratio")
    return f"[tuplet:{event.tuplet.actual_count},{event.tuplet.normal_count}]"


def _syllabify(
    text: str,
    language: str,
    pattern: Optional[tuple[str, ...]] = None,
) -> tuple[str, ...]:
    """Split lyrics with explicit Syllabreak language and preserve separators."""
    marker = _syllable_marker(text)
    words = _word_ranges(text)
    if not words:
        return ()

    syllables: list[str] = []
    cursor = 0
    for word_index, (start, end) in enumerate(words):
        word = text[start:end]
        leading = ""
        if syllables:
            syllables[-1] += text[cursor:start]
        else:
            leading = text[:start]

        if pattern is not None and word_index == 0:
            word_syllables = _apply_pattern(word, pattern)
        else:
            marked = _syllabifier(marker).syllabify(word, lang=language)
            word_syllables = [part for part in marked.split(marker) if part]
        if not word_syllables:
            raise CESNValidationError("Syllabreak returned no syllables for a word")
        if not syllables and leading:
            word_syllables[0] = leading + word_syllables[0]
        syllables.extend(word_syllables)
        cursor = end

    syllables[-1] += text[cursor:]
    return tuple(syllables)


def _word_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return Unicode letter-word spans, retaining internal apostrophes."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        if not text[cursor].isalpha():
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text):
            character = text[cursor]
            if (
                character.isalpha()
                or unicodedata.category(character).startswith("M")
                or (
                    character in {"'", "’"}
                    and cursor + 1 < len(text)
                    and text[cursor - 1].isalpha()
                    and text[cursor + 1].isalpha()
                )
            ):
                cursor += 1
            else:
                break
        ranges.append((start, cursor))
    return tuple(ranges)


def _apply_pattern(word: str, pattern: tuple[str, ...]) -> list[str]:
    """Apply explicit syllable fragments while preserving lyric spelling."""
    word_clusters = _grapheme_clusters(word)
    pattern_clusters = tuple(_grapheme_clusters(piece) for piece in pattern)

    def normalize(value: str) -> str:
        """Normalize case and Unicode composition before pattern comparison."""
        return unicodedata.normalize("NFC", value).casefold()

    if normalize("".join("".join(group) for group in pattern_clusters)) != normalize(
        "".join(word_clusters)
    ):
        raise CESNValidationError("CESN pattern does not match its lyric word")

    result: list[str] = []
    offset = 0
    for clusters in pattern_clusters:
        result.append("".join(word_clusters[offset : offset + len(clusters)]))
        offset += len(clusters)
    return result


def _grapheme_clusters(text: str) -> list[str]:
    """Group combining marks with their preceding Unicode code point."""
    clusters: list[str] = []
    for character in text:
        if unicodedata.category(character).startswith("M") and clusters:
            clusters[-1] += character
        else:
            clusters.append(character)
    return clusters


def _syllable_marker(text: str) -> str:
    """Choose a private-use delimiter that does not occur in the lyrics."""
    for start, end in ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)):
        for codepoint in range(start, end + 1):
            marker = chr(codepoint)
            if marker not in text:
                return marker
    raise CESNValidationError("CESN lyrics exhaust private-use unicode characters")


@lru_cache(maxsize=1)
def _supported_syllabreak_languages() -> frozenset[str]:
    """Return the language identifiers exposed by the installed Syllabreak."""
    return frozenset(Syllabreak().supported_languages())


@lru_cache(maxsize=16)
def _syllabifier(marker: str) -> Syllabreak:
    """Return a cached Syllabreak engine with the requested boundary marker."""
    return Syllabreak(marker)


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
