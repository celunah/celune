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
[g3:4,r:1,a3:4] Moonlight.
[pattern:qui,et][g3:4,a3:4] Quiet.
[pattern:qui,et][ipa:/ˈkwaɪ.ət/][g3:4,a3:4] Quiet.
[pattern:cel,lo][ipa:/ˈtʃɛl.oʊ/][slur:2][c4:4,d4:4]cello
[tuplet:3,2][g3:4,a3:4,b3:4] Trip-let.
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
`minor` (also named `natural-minor` or `natural_minor`), `dorian`, `phrygian`,
`lydian`, `mixolydian`, `locrian`, `harmonic-minor`, and `melodic-minor`. Keys
use a note name with an optional sharp or flat.

`[#Comment]` is ignored. `[pattern:qui,et]` replaces Syllabreak's automatic
syllable mapping for the next lyric word; its fragments must join to that word.
`[ipa:/ˈkwaɪ.ət/]` overrides G2P for the same word. Write the pronunciation
between slashes and separate its syllables with periods. Use one IPA syllable
for each pattern fragment, in the same order. The pattern controls how the
written word is split across notes; IPA controls the phones used for each
split. The lyric spelling remains unchanged. Both tags must precede that
word's lyric text and can appear before or after its note list. IPA requires a
matching pattern. A rest does not consume either a written or IPA syllable.

`[slur:N]` marks the syllables of the next explicitly patterned word as a slur,
where `N` is the pattern's syllable count and must be at least two. The slur tag
can appear before or after the note list, but before the lyric word. The notes
must be adjacent and cannot include a rest. CESN retains each note's pitch,
phonemes, and duration. SoulX-Singer uses its continuation note type only when a
slurred syllable repeats the preceding syllable's ARPAbet phones. A different
phone sequence stays a normal note event so the new sound is not dropped; this
preserves its pitch and timing, but may leave an audible syllable boundary.

For captions and saved FLAC names, Celune removes bracketed CESN directives and
uses the remaining lyrics. It still sends the complete notation to the singing
backend.

SoulX-Singer converts explicit IPA into its English ARPAbet phone inventory.
The following table shows the supported symbol mappings. A period separates
syllables and is not sent to the model.

| IPA | SoulX ARPAbet |
| --- | --- |
| `p b t d k g ɡ f v θ ð s z ʃ ʒ h m n ŋ l ɫ ɹ ɻ r j w ɾ` | `P B T D K G G F V TH DH S Z SH ZH HH M N NG L L R R R Y W T` |
| `tʃ t͡ʃ t͜ʃ` | `CH` |
| `dʒ d͡ʒ d͜ʒ` | `JH` |
| `i ɪ ɛ æ ɑ ɒ ɔ ʊ u ʌ ə ɐ ɜ ɝ ɚ` | `IY IH EH AE AA AA AO UH UW AH AH AH ER ER ER` |
| `eɪ aɪ aʊ ɔɪ oʊ əʊ` | `EY AY AW OY OW OW` |

Stress marks `ˈ` and `ˌ` set the following vowel's stress to 1 or 2. Vowels
without an explicit stress mark use 0, except `ɝ` (1) and `ɚ` (0). Length
marks `ː` and `ˑ` and aspiration `ʰ` are ignored. IPA sounds without a matching
model phone are rejected during generation. For example, `Quiet` can use
`[pattern:qui,et][ipa:/ˈkwaɪ.ət/][g3:4,a3:4]Quiet.` to force the pronunciation
`K W AY1` followed by `AH0 T` while retaining the original spelling.
`[singer:Celune]` selects a matching character pack, which must be a CECHAR v5
pack containing the reserved `sing` voice.

`[tuplet:a,b]` scales the next `a` note or rest events by `b/a`, then expires.
For example, `[tuplet:3,2]` makes three following quarter notes fit into the
time of two quarter notes. Rests count toward the `a` events. A tuplet can span
successive note/lyric blocks, but it must finish before another tuplet, a
musical section change, or the end of the notation. Tuplets cannot overlap.
The parsed event retains its tuplet ratio and exposes its exact effective
duration as `duration_beats`, a `fractions.Fraction` measured in quarter-note
beats. Dotted notes apply their 3/2 multiplier before the tuplet ratio.

## Parse and render

`parse_cesn(text: str, *, supported_languages: Optional[tuple[str, ...]] =
None) -> tuple[CESNSection, ...]` parses the notation and returns ordered
sections. A section groups note blocks with the same singer, language, BPM,
key, and scale. Any change to those settings begins a new section, which is the
singing backend's boundary for resetting its current inference state.
Raised `CESNError` instances include zero-based `source_start` and `source_end`
offsets for the source span the parser identified; Celune displays the start as
a one-based line and column.

Each `CESNBlock` contains the original lyric text and an ordered `events`
sequence. Pitched entries are `CESNNotePair(event, syllable, ipa=None,
slur_from_previous=False)` values; `ipa` contains the aligned IPA syllable when
an override is present, and `slur_from_previous` marks a note that continues a
slur from the previous syllable. Rests are `CESNRestEvent` values. A rest keeps
its timing and position but does not consume a lyric syllable or IPA syllable.
`block.pairs` returns only the pitched note/syllable pairs. Note and rest events
expose an exact `duration_beats` value and their optional `CESNTuplet`;
`tuplet_start` marks the event where rendering re-emits the tuplet directive.

```python
from celune.cesn import parse_cesn, render_cesn

sections = parse_cesn(
    "[lang:en][bpm:120,key:c,scale:major][pattern:qui,et]"
    "[tuplet:3,2][g3:4,r:4,a3:4]Quiet."
)
rendered = render_cesn(sections)
```

When `supported_languages` is provided, the parser rejects a language the
selected model does not support. Celune supplies that list before invoking a
singing backend's `cesn_to_model()` hook. `render_cesn(sections)` combines all
sections into one canonical notation string, repeats settings at each section
boundary, and writes one pitched event per aligned syllable so explicit
patterns remain intact. Slurred syllables are grouped into a single note list
with their pattern and IPA overrides preserved.

Syllabification uses [Syllabreak](https://github.com/apakabarlabs/syllabreak-python)
with the explicit language tag. It does not use a fixed word dictionary or
automatic language detection. Syllables are orthographic, not phonetic, and
unsupported or ambiguous spellings may still need an explicit `[pattern:...]`
override.

## Warnings, errors, and recovery

Every pitched note is checked against the current key and scale. An out-of-scale
pitch emits `CESNNotationWarning` and remains in the notation so the backend
can decide how to render it.

The selected CECHAR v5 singing voice defines its accepted pitch range in
`voices.sing.range`. A note outside the inclusive MIDI bounds raises
`CESNRangeError` during singing-backend preprocessing. For example, Celune's
default pack accepts F#3–C5 (MIDI 54–72); another character pack can declare a
different range. See [CEVOICE singing metadata](../CEVOICE.md#cechar-v5-singing-voice).

| Type | Meaning | Recovery |
| --- | --- | --- |
| `CESNValidationError` | Notation is malformed or violates a score invariant. | Correct the directive, event, pattern, IPA override, or slur group. |
| `CESNUnsupportedFeatureError` | The notation requests a scale, language, singer, or feature unsupported by the parser or backend. | Choose a supported scale, language, singer, or feature. |
| `CESNRangeError` | A note is outside the selected character's declared MIDI pitch range. | Move the note inside the pack's `sing.range` bounds or select a compatible character. |
| `CESNBackendFormatError` | A serialized backend score has invalid JSON or violates its backend-specific schema. | Recreate the score through CESN preprocessing; report persistent failures as a backend defect. |
| `CESNAlignmentError` | There are more detected syllables than pitched notes. | Add pitched note events or correct the pattern. |
| `CESNMatchError` | A pitched note has no remaining syllable. | Remove the extra pitched event or add lyrics. |
| `CESNSyntaxError` | Compatibility superclass for typed CESN validation, feature, and backend-format errors. | Catch a specific subtype when handling a known failure. |
| `InvalidCharacterError` | The selected character has no matching CECHAR v5 pack with a `sing` voice. | Install/select a compatible character pack. |
| `CESNNotationWarning` | A note pitch is outside the section's key and scale. | Review the pitch, key, and scale; the note is retained. |

All CESN exception types derive from `CESNError`, a `ValueError` subtype. The
legacy `CESNSyntaxError` remains their compatibility superclass. CESN exception
messages are technical diagnostics and are not translated; the user-facing
diagnostic header and missing-notation warning remain localized. Parser errors
carry nullable, zero-based `source_start` and `source_end` offsets when they map
to input notation. A singing-backend request is parsed and validated before
conversion. Plain text submitted to a singing backend is logged as a
missing-notation warning and does not replace Celune's current status. Malformed
notation is logged with its one-based line and column, the complete source line,
and a caret at the error location; surrounding source lines and a traceback are
omitted from this user diagnostic. A parse failure returns the visible status to
Idle. Backend-format errors refer to the internal
serialized score and do not carry source-line offsets.
SoulX-Singer also raises a localized backend error during preprocessing when
an IPA override contains a sound its English ARPAbet inventory cannot encode;
replace that sound with a supported pronunciation before retrying.

## See also

- [Python API](../reference/public-api.md#cesn-parsing)
- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Backend integration](../development/backends.md#cesn-singing-hook)
