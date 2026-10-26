# Create a character pack

Use `scripts/cac.py` to build a CECHAR pack from existing voice references and
optional Persona metadata. This guide covers format selection, the interactive
wizard, and validation before installing a character.

## Choose a format

Run commands from the repository root after completing the normal
[installation](../getting-started/installation.md).

| Option | Output contract |
| --- | --- |
| `--format-version 3` | Legacy CECHAR v3; the default for compatibility. |
| `--format-version 4` | Indexed CECHAR v4 with speaking voices and Persona metadata. |
| `--format-version 5` | Indexed CECHAR v5 with optional reserved `sing` reference. |

All modes write `<bundle_name>.cevoice` in the current directory. The extension
does not determine the CECHAR generation. See the canonical
[format specification](../CEVOICE.md) for layout, invariants, and compatibility.

## Build a full character

1. Prepare a WAV for each voice and its exact spoken transcript.
2. Start the wizard with `uv run python scripts/cac.py --format-version 5`.
3. Enter the pack name, description, voice count, and each voice's name and WAV.
   Embeddings, `cfg_scale`, and display names are optional. Enter each reference
   transcript and any voice-specific Persona style overrides. Voice keys remain
   backend identifiers; display names control selector labels.
4. Choose a speaking default and voice order, optional theme colors, and the
   shared identity, speaking style, boundaries, prompt rules, and examples.
5. Inspect the saved pack before copying it into `voices/`. Avoid overwriting
   an existing character without retaining its original pack.

CAC normalizes references to 24 kHz mono when resampling or remixing is needed.
For v4/v5 it always encodes canonical PCM16 WAV headers, including when the
input already reports the correct rate and channel count.
It uses the repository's version-specific writers and their validation rules.
It does not generate voices, embeddings, or transcripts.

For a v5 singing entry, use the exact name `sing`. The wizard then asks for its
transcript, matching CESN score, and inclusive minimum and maximum MIDI pitches
from 0 to 127. It omits speaking-only settings for that entry. The singing voice
cannot be the speaking default or appear in the speaking voice order.

Speech-only v5 packs are valid: omit `sing`. Such packs are non-sing-compliant
and cannot be selected by singing backends. State this in the pack description
when users need a visible status; no separate compliance flag is required by
the format.

## Build a single voice

Supply a pack name, reference WAV path, and transcript as positional arguments;
omit the transcript to be prompted. Add `--format-version 4` or
`--format-version 5` to select an indexed format. Without the option, CAC keeps
its legacy v3 behavior. Simple mode names the single speaking voice `Standard`
and sets it as the default.

## Verify and recover

Use `uv run python scripts/cac.py --help` to inspect accepted arguments. Invalid
format choices are argparse errors. Missing files, invalid metadata, and
writer validation failures prevent successful creation; correct the reported
input and retry. MIDI prompts repeat until the bounds are valid and ordered.

Use `celune ls voices/<bundle_name>.cevoice` to inspect an installed pack.
Developer validation can open the result with `CEVoice.open()` and read every
asset to verify its digest. Confirm the manifest version, voice order,
transcripts, and Persona before synthesis. Loading alone does not verify the
perceived voice quality.

The creator is owned by `scripts/cac.py`; regression tests live in
`tests/test_scripts_cac.py`. Validate code changes with
`python scripts/run_ci.py`.

## Nol

`voices/nol.cevoice` is a speech-only CECHAR v5 pack with **Balanced** as its
default, followed by **Calm** and **Spirited**. It is marked non-sing-compliant
in its description and contains no `sing` entry.

Nol is a Black American man from New York City, Celune's partner, and a fully
human observer and support character on the Moon. His Persona preserves his
open, conversational, creative, expressive temperament, polished American
English with subtle NYC/Bronx cadence, and faint yellow lunar-protection field.
The supplied laughter cue in Spirited identifies a nonverbal part of the audio;
its reference transcript contains only the spoken words.

## See also

- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Persona](persona.md)
- [Persona Markdown](../formats/persona-markdown.md)
- [Singing tutorial](cesn-tutorial.md)
