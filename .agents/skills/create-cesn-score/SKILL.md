---
name: create-cesn-score
description: Turn exact lyrics into a Celune Singing Notation score or write original lyrics from an idea and score them. Use when a user asks for a singing score or CESN notation.
---

# Create a CESN singing score

Create a usable Celune Singing Notation (CESN) score from either supplied lyrics or a creative idea. This skill writes notation; it does not generate or audition audio.

## Choose the mode

- **Verbatim text:** The user supplies words to sing and wants them preserved. Keep every lyric word, spelling, and punctuation mark unchanged. Add musical settings and note events only. A `[pattern:...]` tag is allowed when needed to align a word's syllables without changing its lyric text. Do not add a title, refrain, filler, or new lyric.
- **Idea:** The user gives a theme, scene, message, or other concept rather than fixed lyrics. Write original, singable lyrics that fit the idea, then score those lyrics. Keep the language and sentiment appropriate to the request; do not present generated lyrics as the user's verbatim text.

If the mode is unclear, ask whether the user wants their exact wording preserved or wants new lyrics written from the idea. Do not ask for details already provided.

## Resolve language and target

CESN never detects language automatically. Require the user to explicitly name the lyric language, even when the supplied text makes it seem obvious. Ask for it if missing; do not guess or silently add a language tag based on the text.

The language must be supported by both Syllabreak and the singing model that will consume the score. When the target backend or model is known, verify its supported languages from the repository's backend documentation or implementation. Confirm Syllabreak support through Celune's parser where practical. If the target is unknown or support cannot be verified, ask which singing backend/model will use the score or clearly state what compatibility remains unverified. Never claim a model supports a language without evidence.

## Compose the notation

1. Choose a tempo, key, and scale that fit the requested style. Include an explicit header and language tag before the lyrics:

   ```text
   [bpm:120,key:c,scale:major]
   [lang:en]
   ```

   Use the user's stated musical preferences when provided. Supported scales include `major`, `minor`/`natural_minor`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `locrian`, `harmonic-minor`, and `melodic-minor`. Keys use a note name from A through G with an optional sharp or flat.

2. Write note-list/lyric pairs with one pitched note per lyric syllable. Use pitch names such as `c3` or `f#4`; note types are `1`, `2`, `4`, `8`, `16`, or `32`, with an optional `d` for a dotted note. Use `r` for a timed rest; rests do not take a syllable. Keep pitches in the selected scale unless an intentional chromatic note is musically useful.

3. Use Syllabreak with the explicit language rather than a hand-written syllable dictionary. Add `[pattern:frag,ments]` immediately before a word only when automatic syllabification needs correction or the user requests a particular split. Pattern fragments must concatenate to the unchanged lyric word. Preserve all supplied text in verbatim mode.

4. Use `[tuplet:a,b]` only when the rhythm needs it. It scales the next `a` note or rest events by `b/a`, then expires. Rests count toward those events. Do not overlap tuplets or let one cross a section boundary.

5. Keep one singer and one set of musical settings per section where practical. A change to singer, language, BPM, key, or scale begins a new inference section. Use `[singer:name]` only when the user asks to switch to a specific character; that character needs a CECHAR v5 pack with a `sing` voice.

Example shape:

```text
[bpm:120,key:c,scale:major]
[lang:en]
[c3:4] Ah.
```

## Check and return the score

- Check that each note-list/lyric pair has exactly one pitched note per syllable, that rests are excluded from the syllable count, and that patterns spell the original word exactly.
- When practical, validate the completed notation with `celune.cesn.parse_cesn()` and pass the selected model's `supported_languages`. Resolve parser alignment or syntax errors before returning the score. Do not invoke model inference or generate audio just to validate notation.
- Return the complete score in a fenced `text` block. Briefly identify the mode, explicit language, tempo, key, and scale. Mention parser validation only if it was actually run; distinguish syntax/alignment validation from audible quality or model compatibility.
- In verbatim mode, do not provide a rewritten lyric version alongside the score. In idea mode, the generated lyrics belong in the CESN score and may be summarized separately only if useful.

## CESN reference

For exact tag syntax, parser behavior, supported scales, tuplets, and errors, read [the CESN format documentation](../../../docs/formats/cesn.md) from the repository root when needed. Syllabification is provided by [Syllabreak](https://github.com/apakabarlabs/syllabreak-python); do not replace it with a fixed dictionary.
