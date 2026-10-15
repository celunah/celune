# CESN format and parsing

This page is for Celune backend and singing-feature developers. It defines
Celune Singing Notation (CESN) and the core parser's note-to-syllable contract.

## Notation

A CESN request starts with one bracketed, comma-separated list of note events,
followed by the lyrics:

```text
[c3:4d,d3:8,e3:16] Text to sing.
```

Each note event has a pitch, a colon, a note type, and an optional dotted
marker:

| Field | Syntax | Example |
| --- | --- | --- |
| Pitch | A note from `a` through `g`, optional `#` or `b`, and an optional minus sign followed by an octave number | `c3`, `f#4`, `bb2` |
| Note type | `1`, `2`, `4`, `8`, `16`, or `32` | `8` |
| Dotted marker | Optional `d` after the note type | `4d` |

Spaces around events and separators are ignored. Empty events, unsupported
durations, missing brackets, and additional brackets in the lyrics are syntax
errors. The parser accepts case-insensitive note letters and dotted markers;
it normalizes pitches to lowercase.

## Parse and align lyrics

`parse_cesn(text: str) -> tuple[tuple[CESNNoteEvent, str], ...]` returns one
`(note_event, syllable)` pair for every matched lyric syllable. A
`CESNNoteEvent` has `pitch: str`, `note_type: int`, and `dotted: bool` fields.
The parser preserves lyric spelling, punctuation, and whitespace by attaching
separators to neighboring syllables.

```python
from celune.cesn import parse_cesn

parsed = parse_cesn("[c3:4d,d3:8,e3:16] Text to sing.")
assert tuple((event.pitch, syllable) for event, syllable in parsed) == (
    ("c3", "Text "),
    ("d3", "to "),
    ("e3", "sing."),
)
```

The parser uses [Syllabreak](https://github.com/apakabarlabs/syllabreak-python)
to find orthographic syllable boundaries without a fixed word dictionary. It
uses the library's automatic language detection; language coverage and known
limitations are documented in its README. Unsupported writing systems may be
returned without internal syllable boundaries. Syllables are orthographic, not
phonetic, so a singing backend should not treat them as pronunciation truth.

The parser preserves punctuation, whitespace, and hyphens by attaching them to
the preceding syllable. For example, Syllabreak's English rules split
`Moonlight` into `Moon` and `light`, while treating `quiet` as one syllable.
Check note counts against the library's results for the lyrics being used.

## Errors and recovery

| Exception | Meaning | Useful fields |
| --- | --- | --- |
| `CESNSyntaxError` | The note list or CESN structure is malformed. | Exception message includes the malformed field or event offset. |
| `CESNAlignmentError` | Detected syllables outnumber note events. | `note_count`, `syllable_count` |
| `CESNMatchError` | A note event has no remaining syllable. | Zero-based `note_index`, canonical `note_event`, `syllable_count` |

These errors derive from `CESNError`, a `ValueError` subtype. A caller can ask
for corrected notation when syntax is invalid, add or remove note events when
alignment fails, and report the offending zero-based event when a note has no
syllable. The parser does not rewrite or partially return invalid notation.

## See also

- [Python API](../reference/public-api.md#cesn-parsing)
- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Backend integration](../development/backends.md#cesn-singing-hook)
