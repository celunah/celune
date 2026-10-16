# CESN format and parsing

This page defines Celune Singing Notation (CESN) for singing-backend developers
and users who write singing requests. It describes language selection,
syllable alignment, musical sections, singer packs, and parser failures.

## Notation

CESN is a sequence of bracketed directives and note-list/lyric pairs. One
notation may contain many pairs and many sections:

```text
[lang:en]
[bpm:120,key:c,scale:major]
[#Opening phrase]
[c3:4,r:1,d3:4] Moonlight.
[pattern:qui,et][c3:4,d3:4] Quiet.
[singer:Celune]
[bpm:160,key:e,scale:minor]
[e3:4] Word.
```

Each note event contains a pitch or rest, a colon, a note type, and an optional
dotted marker:

| Field | Syntax | Example |
| --- | --- | --- |
| Pitch | A note from `a` through `g`, optional `#` or `b`, and an optional octave | `c3`, `f#4`, `bb2` |
| Rest | `r` instead of a pitch | `r:1` |
| Note type | `1`, `2`, `4`, `8`, `16`, or `32` | `8` |
| Dotted marker | Optional `d` after the note type | `4d` |

The defaults are 120 BPM, C major, and the currently active singer. A language
tag is required before lyric text is syllabified; it can follow a note list if
it appears before that list's lyrics. CESN never detects language automatically.
The language must be supported both by the selected TTS model and by
Syllabreak. The tag accepts an ISO language identifier such as `en`, `eng`, or
`srp-latn`.

`[bpm:120,key:c,scale:major]` sets all three musical settings. A later settings
tag updates them from that point forward. Supported scales are `major`,
`minor`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `locrian`,
`harmonic-minor`, and `melodic-minor`. Keys use a note name with an optional
sharp or flat.

`[#Comment]` is ignored. `[pattern:qui,et]` replaces Syllabreak's automatic
syllable mapping for the next lyric word; its fragments must join to that word.
`[singer:Celune]` selects a matching character pack, which must be a CECHAR v5
pack containing the reserved `sing` voice.

## Parse and render

`parse_cesn(text: str, *, supported_languages: Optional[tuple[str, ...]] =
None) -> tuple[CESNSection, ...]` parses the notation and returns ordered
sections. A section groups note blocks with the same singer, language, BPM,
key, and scale. Any change to those settings begins a new section, which is the
singing backend's boundary for resetting its current inference state.

Each `CESNBlock` contains the original lyric text and an ordered `events`
sequence. Pitched entries are `CESNNotePair(event, syllable)` values; rests are
`CESNRestEvent` values. A rest keeps its timing and position but does not
consume a lyric syllable. `block.pairs` returns only the pitched note/syllable
pairs.

```python
from celune.cesn import parse_cesn, render_cesn

sections = parse_cesn(
    "[lang:en][bpm:120,key:c,scale:major][pattern:qui,et][c3:4,r:1,d3:4]Quiet."
)
rendered = render_cesn(sections)
```

When `supported_languages` is provided, the parser rejects a language the
selected model does not support. Celune supplies that list before invoking a
singing backend's `cesn_to_model()` hook. `render_cesn(sections)` combines all
sections into one canonical notation string, repeats settings at each section
boundary, and writes one pitched event per aligned syllable so explicit
patterns remain intact.

Syllabification uses [Syllabreak](https://github.com/apakabarlabs/syllabreak-python)
with the explicit language tag. It does not use a fixed word dictionary or
automatic language detection. Syllables are orthographic, not phonetic, and
unsupported or ambiguous spellings may still need an explicit `[pattern:...]`
override.

## Warnings, errors, and recovery

Every pitched note is checked against the current key and scale. An out-of-scale
pitch emits `CESNNotationWarning` and remains in the notation so the backend
can decide how to render it.

| Type | Meaning | Recovery |
| --- | --- | --- |
| `CESNSyntaxError` | A tag, setting, language, event, or pattern is malformed or unsupported. | Correct the tag or choose a language supported by both layers. |
| `CESNAlignmentError` | There are more detected syllables than pitched notes. | Add pitched note events or correct the pattern. |
| `CESNMatchError` | A pitched note has no remaining syllable. | Remove the extra pitched event or add lyrics. |
| `InvalidCharacterError` | The selected character has no matching CECHAR v5 pack with a `sing` voice. | Install/select a compatible character pack. |
| `CESNNotationWarning` | A note pitch is outside the section's key and scale. | Review the pitch, key, and scale; the note is retained. |

The three CESN errors derive from `CESNError`, a `ValueError` subtype. A
singing-backend request is parsed and validated before conversion. A parse or
character failure rejects that request before the backend receives it.

## See also

- [Python API](../reference/public-api.md#cesn-parsing)
- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Backend integration](../development/backends.md#cesn-singing-hook)
