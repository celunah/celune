# CESN format and parsing

This page defines Celune Singing Notation (CESN) for users writing singing requests and developers implementing singing backends. It covers explicit pronunciation, score timing, singer packs, and parser errors.

## Notation

CESN is a sequence of musical settings, note groups, explicit pronunciation, and lyrics. Each lyric-bearing score group follows this order: optional extra tags, one note list, one pronunciation tag, then the lyric text. A rest-only group may contain just a note list.

```text
[bpm:120,key:c,scale:major][lang:en]
[#Opening phrase]
[g3:4,r:1,a3:4][ˈmuːn.laɪt] Moonlight.
[g3:4,a3:4][ˈkwaɪ.ət] Quiet.
[slur:2][c4:4,d4:4][ˈtʃɛl.oʊ]cello
[c4:4,d4:4][x-sampa:"tSEl.oU]cello
[tuplet:3,2][g3:4,a3:4,b3:4][ˈmjuː.zɪ.kəl]Musical.
[singer:Celune][bpm:160,key:e,scale:minor][e3:4][wɜːd]Word.
```

The note list and pronunciation tag pair by position. Each pitched note requires exactly one IPA syllable, separated from the next syllable by a period. A rest consumes no pronunciation syllable. The lyric text remains intact and is not split or matched to the pronunciation by an algorithm. For example, `quiet` stays spelled `Quiet.` while `ˈkwaɪ.ət` explicitly assigns its two syllables to two notes.

When a lyric block ends with a comma or period, CESN automatically appends a quarter-note rest (`[r:4]`) or half-note rest (`[r:2]`) after that block. Trailing spaces do not affect the punctuation check. An explicit rest-only group immediately after the lyric block, or a trailing rest already included in its note list, supplies the pause without adding a duplicate automatic rest. Other explicit rests remain additive.

Each note event contains a pitch or rest, a colon, a note type, and an optional dotted marker:

| Field | Syntax | Example |
| --- | --- | --- |
| Pitch | A note from `a` through `g`, optional `#` or `b`, and an optional octave | `c3`, `f#4`, `bb2` |
| Rest | `r` instead of a pitch | `r:1` |
| Note type | `1`, `2`, `4`, `8`, `16`, or `32` | `8` |
| Dotted marker | Optional `d` after the note type | `4d` |

The defaults are 120 BPM, C major, and the currently active singer. A language tag is required before the note list; CESN never detects language automatically. The tag accepts an ISO language identifier such as `en`, `eng`, or `srp-latn`. Each selected singing backend may support a different set of languages and IPA sounds.

`[bpm:120,key:c,scale:major]` sets all three musical settings. A later settings tag updates them from that point forward, before the next note list. Supported scales are `major`, `minor` (also named `natural-minor` or `natural_minor`), `dorian`, `phrygian`, `lydian`, `mixolydian`, `locrian`, `harmonic-minor`, and `melodic-minor`. Keys use a note name with an optional sharp or flat.

`[#Comment]` is ignored. Other settings and feature tags, including `[lang:...]`, `[singer:...]`, `[slur:...]`, and `[tuplet:...]`, precede the note list. A raw bracket after the note list contains Unicode IPA, without an `ipa:` label or slash delimiters. Alternatively, `[x-sampa:...]` marks an ASCII X-SAMPA pronunciation. For example, `[x-sampa:"tSEl.oU]` is converted to IPA `ˈtʃɛl.oʊ`. X-SAMPA symbols are case-sensitive; a period separates syllables, `"` marks primary stress, and `%` marks secondary stress. CESN converts X-SAMPA to canonical Unicode IPA during parsing, so `CESNNotePair.ipa` and rendered notation contain IPA regardless of the input form. Unknown symbols and non-ASCII characters in the X-SAMPA value are rejected rather than guessed. A rest does not consume a syllable.

`[slur:N]` marks all `N` pitched notes in the following group as one slur, where `N` must be at least two and match the IPA syllable count. Put it before the note list. Slurred notes must be adjacent and cannot include a rest. CESN retains each note's pitch, IPA, and duration. SoulX-Singer uses its continuation note type only when a slurred syllable repeats the preceding syllable's ARPAbet phones. A different phone sequence stays a normal note event so the new sound is not dropped; this preserves its pitch and timing, but may leave an audible syllable boundary.

For captions and saved FLAC names, Celune removes bracketed CESN directives and uses the remaining lyrics. Spoken output uses the `<app-slug>_speech_...flac` filename pattern; sung output uses `<app-slug>_singing_...flac`. Celune still sends the complete notation to the singing backend.

SoulX-Singer converts explicit IPA into its English ARPAbet phone inventory. The following table shows the supported symbol mappings. A period separates syllables and is not sent to the model. X-SAMPA conversion only changes the transcription encoding; it does not extend a backend's supported languages or sounds.

| IPA | SoulX ARPAbet |
| --- | --- |
| `p b t d k g ɡ f v θ ð s z ʃ ʒ h m n ŋ l ɫ ɹ ɻ r j w ɾ` | `P B T D K G G F V TH DH S Z SH ZH HH M N NG L L R R R Y W T` |
| `tʃ t͡ʃ t͜ʃ` | `CH` |
| `dʒ d͡ʒ d͜ʒ` | `JH` |
| `i ɪ ɛ æ ɑ ɒ ɔ ʊ u ʌ ə ɐ ɜ ɝ ɚ` | `IY IH EH AE AA AA AO UH UW AH AH AH ER ER ER` |
| `eɪ aɪ aʊ ɔɪ oʊ əʊ` | `EY AY AW OY OW OW` |

Stress marks `ˈ` and `ˌ` set the following vowel's stress to 1 or 2. Vowels without an explicit stress mark use 0, except `ɝ` (1) and `ɚ` (0). Length marks `ː` and `ˑ` and aspiration `ʰ` are ignored. IPA sounds without a matching model phone are rejected during generation. For example, `Quiet` can use `[g3:4,a3:4][ˈkwaɪ.ət]Quiet.` to force `K W AY1` followed by `AH0 T` while retaining the original spelling. `[singer:Celune]` selects a matching character pack, which must be a CECHAR v5 pack containing the reserved `sing` voice.

`[tuplet:a,b]` scales the next `a` note or rest events by `b/a`, then expires. For example, `[tuplet:3,2]` makes three following quarter notes fit into the time of two quarter notes. Rests count toward the `a` events. A tuplet can span successive note/lyric groups, but it must finish before another tuplet, a musical section change, or the end of the notation. Tuplets cannot overlap. The parsed event retains its tuplet ratio and exposes its exact effective duration as `duration_beats`, a `fractions.Fraction` measured in quarter-note beats. Dotted notes apply their 3/2 multiplier before the tuplet ratio.

## Parse and render

`parse_cesn(text: str, *, supported_languages: Optional[tuple[str, ...]] = None) -> tuple[CESNSection, ...]` parses the notation and returns ordered sections. A section groups note blocks with the same singer, language, BPM, key, and scale. Any change to those settings begins a new section, which is the singing backend's boundary for resetting its current inference state. Raised `CESNError` instances include zero-based `source_start` and `source_end` offsets for the source span the parser identified; Celune displays the start as a one-based line and column.

Each `CESNBlock` contains the original lyric text and an ordered `events` sequence. Pitched entries are `CESNNotePair(event, ipa, slur_from_previous=False)` values; `ipa` contains the canonical Unicode IPA syllable, converted from X-SAMPA when that input form was used. `slur_from_previous` marks a note that continues a slur from the previous note. Rests are `CESNRestEvent` values. A rest keeps its timing and position but does not consume a pronunciation syllable. `block.pairs` returns only the pitched note/IPA pairs. Note and rest events expose an exact `duration_beats` value and their optional `CESNTuplet`; `tuplet_start` marks the event where rendering re-emits the tuplet directive.

```python
from celune.cesn import parse_cesn, render_cesn

sections = parse_cesn(
    "[lang:en][bpm:120,key:c,scale:major][tuplet:3,2][g3:4,r:4,a3:4][ˈkwaɪ.ət]Quiet."
)
rendered = render_cesn(sections)
```

When `supported_languages` is provided, the parser rejects a language the selected model does not support. Celune supplies that list before invoking a singing backend's `cesn_to_model()` hook. `render_cesn(sections)` combines all sections into one canonical notation string, repeats settings at each section boundary, and writes one IPA syllable per pitched note, including when the original notation used X-SAMPA. Slur and tuplet tags precede their note list; rests remain in event order.

There is no automatic syllabification or spelling-to-IPA conversion. The score author supplies the pronunciation and syllable boundaries explicitly, so words with unusual pronunciation or note splits can be corrected without changing the lyric text. Automatic punctuation rests are attached to the end of the lyric block's event stream and rendered as rest-only groups after the lyric text.

## Warnings, errors, and recovery

Every pitched note is checked against the current key and scale. An out-of-scale pitch emits `CESNNotationWarning` and remains in the notation so the backend can decide how to render it.

The selected CECHAR v5 singing voice defines its accepted pitch range in `voices.sing.range`. A note outside the inclusive MIDI bounds raises `CESNRangeError` during singing-backend preprocessing. For example, Celune's default pack accepts F#3–C5 (MIDI 54–72); another character pack can declare a different range. See [CEVOICE singing metadata](../CEVOICE.md#cechar-v5-singing-voice).

| Type | Meaning | Recovery |
| --- | --- | --- |
| `CESNValidationError` | Notation is malformed or violates a score invariant. | Correct the directive, note/IPA count, or slur group. |
| `CESNUnsupportedFeatureError` | The notation requests a scale, language, singer, or feature unsupported by the parser or backend. | Choose a supported scale, language, singer, or feature. |
| `CESNRangeError` | A note is outside the selected character's declared MIDI pitch range. | Move the note inside the pack's `sing.range` bounds or select a compatible character. |
| `CESNBackendFormatError` | A serialized backend score has invalid JSON or violates its backend-specific schema. | Recreate the score through CESN preprocessing; report persistent failures as a backend defect. |
| `CESNAlignmentError` | There are more IPA syllables than pitched notes. | Remove IPA syllables or add pitched note events. |
| `CESNMatchError` | A pitched note has no corresponding IPA syllable. | Add an IPA syllable or remove the extra pitched event. |
| `CESNSyntaxError` | Compatibility superclass for typed CESN validation, feature, and backend-format errors. | Catch a specific subtype when handling a known failure. |
| `InvalidCharacterError` | The selected character has no matching CECHAR v5 pack with a `sing` voice. | Install/select a compatible character pack. |
| `CESNNotationWarning` | A note pitch is outside the section's key and scale. | Review the pitch, key, and scale; the note is retained. |

All CESN exception types derive from `CESNError`, a `ValueError` subtype. The legacy `CESNSyntaxError` remains their compatibility superclass. CESN exception messages are technical diagnostics and are not translated; the user-facing diagnostic header and missing-notation warning remain localized. Parser errors carry nullable, zero-based `source_start` and `source_end` offsets when they map to input notation. A singing-backend request is parsed and validated before conversion. Plain text submitted to a singing backend is logged as a missing-notation warning and does not replace Celune's current status. Malformed notation is logged with its one-based line and column, the complete source line, and a caret at the error location; surrounding source lines and a traceback are omitted from this user diagnostic. A parse failure returns the visible status to Idle. Backend-format errors refer to the internal serialized score and do not carry source-line offsets.

SoulX-Singer raises an English backend error during preprocessing when an IPA syllable contains a sound its English ARPAbet inventory cannot encode; replace that sound with a supported pronunciation before retrying.

## See also

- [Python API](../reference/public-api.md#cesn-parsing)
- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Backend integration](../development/backends.md#cesn-singing-hook)
