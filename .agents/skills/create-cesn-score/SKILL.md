---
name: create-cesn-score
description: Turn exact lyrics into a Celune Singing Notation score or write original lyrics from an idea and score them. Use when a user asks for a singing score or CESN notation.
---

# Create a CESN singing score

Create a usable Celune Singing Notation (CESN) score from either supplied lyrics or a creative idea. This skill writes notation; it does not generate or audition audio.

## Choose the mode

- **Verbatim text:** The user supplies words to sing and wants them preserved. Keep every lyric word, spelling, and punctuation mark unchanged. Add musical settings, note events, and explicit pronunciation only. Do not add a title, refrain, filler, or new lyric.
- **Idea:** The user gives a theme, scene, message, or other concept rather than fixed lyrics. Write original, singable lyrics that fit the idea, then score those lyrics. Keep the language and sentiment appropriate to the request; do not present generated lyrics as the user's verbatim text.

If the mode is unclear, ask whether the user wants their exact wording preserved or wants new lyrics written from the idea. Do not ask for details already provided.

## Resolve language and target

CESN never detects language automatically. Require the user to explicitly name the lyric language, even when the supplied text makes it seem obvious. Ask for it if missing; do not guess or silently add a language tag based on the text.

Verify that the target singing backend supports the selected language and the pronunciation symbols it needs. When the backend or model is known, use the repository's backend documentation or implementation. If the target is unknown or support cannot be verified, ask which singing backend/model will use the score or clearly state what compatibility remains unverified. Never claim a model supports a language without evidence.

## Compose the notation

1. Choose a tempo, key, and scale that fit the requested style. Include musical settings and an explicit language tag before score groups:

   ```text
   [bpm:120,key:c,scale:major][lang:en]
   ```

   Use the user's stated musical preferences when provided. Supported scales include `major`, `minor`/`natural_minor`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `locrian`, `harmonic-minor`, and `melodic-minor`. Keys use a note name from A through G with an optional sharp or flat.

2. Write each lyric-bearing score group in this exact order: optional extra tags, one note list, one pronunciation tag, then the lyric text. Use a raw IPA tag by default, such as `[slur:2][c3:4,d3:4][ˈsɪŋ.ɪŋ] Singing.` If the user cannot type IPA, use the explicit ASCII form `[x-sampa:"sIN.IN] Singing.` A rest-only group may contain just a note list. Do not use `[pattern:...]`, `[ipa:...]`, or Syllabreak.

3. Supply one dot-separated IPA or X-SAMPA syllable for each pitched note. In either form, periods mark syllable boundaries; do not split a single syllable's vowels or diphthongs with a period. X-SAMPA must follow the `[x-sampa:...]` marker, use ASCII characters, and preserve case because symbols are case-sensitive. Its `"` and `%` symbols mark primary and secondary stress. The parser converts X-SAMPA to canonical IPA before backend preprocessing. Rests remain in the note list and consume no pronunciation syllable. Keep lyric spelling and punctuation unchanged in verbatim mode. Use the pronunciation and note boundaries you intend rather than inferring them from the written spelling.

   When a lyric-bearing group ends in a comma or period, CESN automatically appends `[r:4]` or `[r:2]` to its event stream. Group score events at punctuation boundaries when you want those rests, and do not add another trailing rest unless you want a longer pause.

4. Use pitch names such as `c3` or `f#4`; note types are `1`, `2`, `4`, `8`, `16`, or `32`, with an optional `d` for a dotted note. Keep pitches inside the selected singer's declared range and in the selected scale unless an intentional chromatic note is musically useful. Use `r` for a timed rest.

5. Use `[slur:N]` before the note list to tie `N` adjacent pitched notes. `N` must be at least two and match the number of IPA syllables in that group. A slur cannot include a rest.

6. Use `[tuplet:a,b]` before the first note list it affects. It scales the next `a` note or rest events by `b/a`, then expires. Rests count toward those events. Do not overlap tuplets or let one cross a section boundary.

7. Keep one singer and one set of musical settings per section where practical. A change to singer, language, BPM, key, or scale begins a new inference section. Use `[singer:name]` only when the user asks to switch to a specific character; that character needs a CECHAR v5 pack with a `sing` voice.

Example shape:

```text
[bpm:120,key:c,scale:major][lang:en]
[c3:4][ɑ] Ah.
[c3:4,d3:4][ˈsɪŋ.ɪŋ] Singing.
```

## Check and return the score

- Check that each lyric-bearing group follows `[extra tags][notes][pronunciation] lyrics`, has exactly one IPA or X-SAMPA syllable for each pitched note, and excludes rests from that count. A rest-only group can omit pronunciation and lyrics.
- Check that each dot represents a syllable boundary and that the supplied IPA or X-SAMPA explicitly expresses the intended pronunciation. Preserve all supplied lyrics exactly in verbatim mode.
- When practical, validate the completed notation with `celune.cesn.parse_cesn()` and pass the selected model's `supported_languages`. Resolve parser alignment or syntax errors before returning the score. Do not invoke model inference or generate audio just to validate notation.
- Return the complete score in a fenced `text` block. Briefly identify the mode, explicit language, tempo, key, and scale. Mention parser validation only if it was actually run; distinguish syntax/alignment validation from audible quality or model compatibility.
- In verbatim mode, do not provide a rewritten lyric version alongside the score. In idea mode, the generated lyrics belong in the CESN score and may be summarized separately only if useful.

## CESN reference

For exact tag syntax, parser behavior, supported scales, tuplets, and errors, read [the CESN format documentation](../../../docs/formats/cesn.md) from the repository root when needed.
