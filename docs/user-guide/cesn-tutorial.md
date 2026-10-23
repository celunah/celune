# CESN singing tutorial

This tutorial helps singers and songwriters create Celune Singing Notation
(CESN) inputs for a singing backend. It shows how to set a melody, guide
pronunciation, shape note connections, and paste short example songs.

## Before you start

Select a TTS backend that supports singing and a CECHAR v5 pack with a `sing`
voice. Enter a CESN score in Celune's input box and submit it with `CTRL+ENTER`.
The examples here use English and stay within Celune's default singing range,
F#3–C5 (MIDI 54–72); a different character pack can declare another range.

CESN keeps the lyric text you write and sends its explicit notes and
pronunciation to the backend. It does not infer syllables or pronunciation
from spelling, so the notation controls both the words shown in captions and
the sounds passed to the singer.

## Write a first phrase

Start with a tempo, key, scale, and language. A lyric-bearing group follows the
order `[extra tags][notes][pronunciation] lyrics`. Every pitched note needs
one dot-separated pronunciation syllable; rests take time but do not take a
syllable.

```text
[bpm:120,key:c,scale:major][lang:en]
[g3:4,a3:4][hə.ˈloʊ]Hello,
[c4:2][sɪŋ]sing.
```

The first group assigns the two syllables of “Hello” to G3 and A3. The second
group holds “sing” for a half note. A comma automatically adds a quarter-note
rest after its lyric group, and a period adds a half-note rest. These rests
also make phrases easier to distinguish in playback.

## Set notes and rhythm

Note names use A–G, an optional sharp or flat, and an optional octave, such as
`g3`, `c4`, or `f#4`. Note types are whole (`1`), half (`2`), quarter (`4`),
eighth (`8`), sixteenth (`16`), or thirty-second (`32`); add `d` for a dotted
note. Tempo is in beats per minute. Key and scale describe the musical
context; an out-of-scale note is retained with a warning, while a note outside
the singer's declared pitch range is rejected.

This example gives the first two syllables a quicker pickup and holds the
third longer:

```text
[bpm:112,key:g,scale:major][lang:en]
[g3:8,a3:8,b3:2][ˈgʊd.mɔɹ.nɪŋ]Good morning.
```

The spelling remains `Good morning.` while the three IPA syllables determine
how the singer pronounces it. Use the singer pack's accepted range and the
chosen scale when selecting pitches.

## Guide pronunciation with IPA or X-SAMPA

Write a raw IPA value inside brackets directly after the note list. Do not add
an `ipa:` label or slash delimiters. A period in the pronunciation marks a
syllable boundary, not a pause. Stress marks such as `ˈ` can guide the model's
pronunciation. For example, this spelling uses the explicit two-syllable
pronunciation `ˈkwaɪ.ət`:

```text
[bpm:120,key:c,scale:major][lang:en]
[c4:4,a3:4][ˈkwaɪ.ət]Quiet.
```

If typing IPA is inconvenient, enter the ASCII X-SAMPA equivalent after the
`x-sampa:` marker. X-SAMPA is case-sensitive, and periods still separate
syllables:

```text
[bpm:120,key:c,scale:major][lang:en]
[c4:4,a3:4][x-sampa:"kwaI.@t]Quiet.
```

The spelling, punctuation, and caption stay `Quiet.` in both scores. X-SAMPA
changes only the pronunciation encoding; it does not add sounds or languages
that the selected backend does not support.

## Shape note connections

Use `[slur:N]` before a note list to connect `N` adjacent pitched notes. The
number must match the number of IPA syllables in the group, and a slur cannot
include a rest. Note pitches and durations remain in the score; a slur guides
the connection between their sounds.

```text
[bpm:104,key:c,scale:major][lang:en]
[slur:2][c4:4,d4:4][ˈtʃɛl.oʊ]cello
```

Use `[slide:<onset-and-vowel>,<notes>]` when a consonant should be articulated
once and one vowel should continue over several notes. In X-SAMPA, `lA` means
an `l` onset followed by the `A` vowel. This three-note example expands to
`lA.A.A`: it does not repeat the consonant on every note.

```text
[bpm:96,key:c,scale:major][lang:en]
[slide:lA,g3:2,a3:2,b3:2]la
```

The slide form needs at least two pitched notes, one vowel nucleus, and no
rests. Use a slur for multiple syllables; use a slide for one onset followed
by a sustained vowel.

## Add vibrato to one note

Use `[v:<delay-note-type>,<speed-hz>,<cents>]` before a pitched note to add a
sine-wave vibrato to that note only. The delay uses a CESN note value, speed is
in hertz, and depth is 20, 40, 60, or 80 cents. Use `0` for vibrato from the
note onset. The tag always looks forward to the next pitched event, so a tag
before a note list applies to its first note.

```text
[bpm:120,key:c,scale:major][lang:en]
[v:8,5,20][c4:2,d4:2][ˈkɑː.lm]calm
```

Here the first half note starts vibrating after an eighth-note delay; the
second half note is unchanged. Vibrato is applied to the existing framewise F0
track only in SoulX-Singer's high-precision mode. Standard score mode does not
apply it. A delay as long as the target note leaves no time for vibrato.

## Use rests, tuplets, and sections

Write an explicit rest as `r:<note-type>`, such as `[r:4]`, or include it in a
note list. Rests consume time and do not consume IPA syllables. Punctuation
adds a rest automatically, so avoid adding a trailing rest when the
comma- or period-rest already gives the pause you want.

`[tuplet:a,b]` scales the next `a` note or rest events by `b/a`, then expires.
This example fits three quarter notes into two beats:

```text
[bpm:120,key:c,scale:major][lang:en]
[tuplet:3,2][g3:4,a3:4,b3:4][ˈbjuː.tɪ.fəl]Beautiful.
```

A change to singer, language, tempo, key, or scale starts a new inference
section. Use `[singer:name]` only for a CECHAR v5 pack whose name matches and
which contains a `sing` voice.

## Paste-ready songs

These short scores use different tempos and note shapes. Paste one complete
score at a time. The first uses long notes and punctuation rests for a gentle
feel:

```text
[bpm:76,key:c,scale:major][lang:en]
[g3:2,a3:2][ˈmɔɹ.nɪŋ]Morning
[c4:2][laɪt] light,
[a3:2,g3:2][ˈdɹɪft.ɪŋ]drifting
[g3:1][hoʊm]home.
```

This one moves more quickly with shorter notes and a bright major scale:

```text
[bpm:132,key:g,scale:major][lang:en]
[g3:8,a3:8,b3:4][ˈlɪt.əl.bɝdz]Little birds
[d4:4,c4:8,b3:8][ɑɹ.ˈɹeɪ.sɪŋ] are racing,
[g3:4][ʌp]up
[a3:4,b3:4,c4:2][ˈɪn.tuː.ðə] into the
[d4:1][deɪ]day.
```

The final example combines two vowel slides with a two-note slur:

```text
[bpm:96,key:g,scale:major][lang:en]
[slide:lA,g3:2,a3:2,b3:2]La
[slide:lA,c4:2,b3:2,a3:2] la,
[slur:2][g3:2,a3:2][ˈsɪŋ.ɪŋ] singing
[g3:1][hoʊm] home.
```

The examples validate notation structure and syllable counts. They do not
guarantee identical sound across random seeds, model versions, or voice packs.

## Check a score

If a score fails, check that a `[lang:...]` tag appears before its note list,
that each lyric group's pitched-note count matches its pronunciation
syllables, and that all notes are inside the selected singer's declared range.
`CESNMatchError` and `CESNAlignmentError` identify note/IPA count mismatches;
`CESNRangeError` identifies an out-of-range note. Celune reports notation
errors against the input line and column. See the CESN format reference for
all tags, constraints, and error types.

## See also

- [CESN format and parsing](../formats/cesn.md)
- [TUI commands and controls](../interfaces/tui.md#slash-commands)
- [CEVOICE and CECHAR formats](../CEVOICE.md)
