# SPDX-License-Identifier: Apache-2.0
"""Parse Celune Singing Notation into note and syllable pairs."""

import re
from typing import NamedTuple
from functools import lru_cache

from syllabreak import Syllabreak

from .exceptions import (
    CESNAlignmentError,
    CESNMatchError,
    CESNSyntaxError,
)


class CESNNoteEvent(NamedTuple):
    """A pitch and duration assigned to one lyric syllable."""

    pitch: str
    note_type: int
    dotted: bool


_NOTE_EVENT_PATTERN = re.compile(
    r"([a-g](?:[#b])?-?\d+)\s*:\s*(32|16|8|4|2|1)\s*(d?)",
    re.IGNORECASE,
)


def parse_cesn(text: str) -> tuple[tuple[CESNNoteEvent, str], ...]:
    """Parse one CESN notation block and align its notes with lyric syllables.

    Lyrics are syllabified by Syllabreak using its automatic language detection.
    Spaces and hyphens mark word boundaries, and are retained with the
    preceding syllable.

    Args:
        text: One ``[notes] lyrics`` CESN request.

    Returns:
        tuple[tuple[CESNNoteEvent, str], ...]: One note-event/syllable pair per
        lyric syllable.

    Raises:
        CESNSyntaxError: The bracketed note list or one of its events is malformed.
        CESNAlignmentError: There are fewer note events than detected syllables.
        CESNMatchError: A note event has no syllable to match.
    """
    if not isinstance(text, str):
        raise CESNSyntaxError("cesn input must be a string")

    source = text.strip()
    opening = source.find("[")
    if opening != 0:
        raise CESNSyntaxError("cesn notation must start with a note list")
    closing = source.find("]")
    if closing < 0 or "[" in source[1:closing]:
        raise CESNSyntaxError("cesn note list is not closed")

    raw_events = source[1:closing]
    lyrics = source[closing + 1 :].lstrip()
    if not lyrics:
        raise CESNSyntaxError("cesn lyrics are required")
    if "[" in lyrics or "]" in lyrics:
        raise CESNSyntaxError("cesn lyrics cannot contain notation brackets")

    note_events = _parse_note_events(raw_events)
    syllables = _syllabify(lyrics)

    if len(note_events) > len(syllables):
        index = len(syllables)
        raise CESNMatchError(
            index,
            _format_note_event(note_events[index]),
            len(syllables),
        )
    if len(note_events) < len(syllables):
        raise CESNAlignmentError(len(note_events), len(syllables))

    return tuple(zip(note_events, syllables, strict=True))


def _parse_note_events(source: str) -> tuple[CESNNoteEvent, ...]:
    """Validate and parse a comma-separated list of note events."""
    if not source.strip():
        raise CESNSyntaxError("cesn note list cannot be empty")

    events: list[CESNNoteEvent] = []
    offset = 0
    for raw_event in source.split(","):
        candidate = raw_event.strip()
        match = _NOTE_EVENT_PATTERN.fullmatch(candidate)
        if match is None:
            raise CESNSyntaxError(f"malformed cesn note event at offset {offset}")
        pitch, note_type, dotted = match.groups()
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
    """Return the canonical textual representation of one note event."""
    dotted = "d" if event.dotted else ""
    return f"{event.pitch}:{event.note_type}{dotted}"


def _syllabify(text: str) -> tuple[str, ...]:
    """Split lyrics with Syllabreak while retaining punctuation and spacing."""
    marker = _syllable_marker(text)
    marked_text = _syllabifier(marker).syllabify(text)
    syllables: list[str] = []
    current: list[str] = []

    for character in marked_text:
        if character == marker:
            _append_syllable(current, syllables)
            current.clear()
        elif character.isspace() or character in "-\u2010\u2011":
            current.append(character)
            _append_syllable(current, syllables)
            current.clear()
        else:
            current.append(character)

    _append_syllable(current, syllables)
    return tuple(syllables)


def _append_syllable(current: list[str], syllables: list[str]) -> None:
    """Add a lyric fragment when it contains a letter-bearing syllable."""
    if any(character.isalpha() for character in current):
        syllables.append("".join(current))


def _syllable_marker(text: str) -> str:
    """Choose a private-use delimiter that does not occur in the lyrics."""
    for start, end in ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD)):
        for codepoint in range(start, end + 1):
            marker = chr(codepoint)
            if marker not in text:
                return marker
    raise CESNSyntaxError("cesn lyrics exhaust private-use unicode characters")


@lru_cache(maxsize=16)
def _syllabifier(marker: str) -> Syllabreak:
    """Return a cached Syllabreak engine with the requested boundary marker."""
    return Syllabreak(marker)
