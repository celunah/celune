# CEVOICE

`CEVOICE` is Celune's voice-pack container format. The bundled default pack uses
the `CECHAR` v4 schema. Older `CECHAR` v2/v3 and legacy `CEVOICE` v1 bundles
remain readable, and `write_cevoice()` remains available for producing those
legacy-compatible bundles. Use `write_cechar_v4()` for new v4 archives. A
`.cevoice` file stores:

- a small fixed-size binary header
- UTF-8 JSON metadata
- a contiguous binary payload containing the voice assets

Celune currently accepts two asset kinds:

- `wav`: reference audio used by the voice-cloning backends
- `pt`: optional PyTorch embedding data used by Celune's analysis tools

The canonical implementation lives in `celune/cevoice.py`.

## File layout

CECHAR v4 bundles are written as:

| Region | Size | Description |
| --- | ---: | --- |
| Header | 9 bytes | `struct.Struct("<7sBB")` |
| Stored payload | variable | uncompressed or one complete compressed stream |

The decompressed v4 payload is:

| Region | Size | Description |
| --- | ---: | --- |
| Metadata length | 4 bytes | Little-endian unsigned length |
| Metadata | variable | UTF-8 JSON with a name-only `files` list |
| File count | 4 bytes | Little-endian unsigned count |
| Jump table | 8 bytes per file | Little-endian offset and length pairs |
| File data | variable | Assets in manifest order |

Legacy CECHAR v2/v3 bundles use the older layout:

| Region | Size | Description |
| --- | ---: | --- |
| Header | 14 bytes | `struct.Struct("<8sHI")` |
| Metadata | variable | UTF-8 JSON |
| Payload | variable | concatenated asset bytes |

### CECHAR v4 header

The v4 header fields are:

| Field | Type | Value |
| --- | --- | --- |
| `magic` | `7s` | `b"CECHAR\0"` |
| `version` | `B` | `4` |
| `compression` | `B` | `0` none, `1` gzip, `2` Zstandard, `3` XZ, `4` OpenZL |

Celune also accepts legacy bundles with:

| Field | Type | Value |
| --- | --- | --- |
| `magic` | `8s` | `b"CEVOICE\0"` |
| `version` | `H` | `1` |

V4 asset offsets are relative to the start of the decompressed logical payload,
not the start of the file. Physical offsets and lengths are never stored in v4
JSON metadata.

## Metadata schema

`write_cevoice()` creates the required metadata automatically:

```json
{
  "format": "CECHAR",
  "version": 3,
  "name": "My Pack",
  "default_voice": "balanced",
  "voice_order": ["balanced"],
  "theme": {
    "background": "#1d1826",
    "accent": "#cebaff",
    "glow_color": "#cebaff",
    "faded_accent": "#9c88ce"
  },
  "persona": {
    "identity": {
      "name": "Celune",
      "profile": "A measured contralto presence."
    },
    "speaking_style": "Calm, clipped, and observant.",
    "boundaries": ["Do not break character."],
    "prompt_rules": ["Prefer concise answers."],
    "example_dialogue": ["User: hi", "Celune: Hello."],
    "style": {
      "warmth": "mid",
      "directness": "high",
      "humor": "low",
      "detail": "mid",
      "formality": "mid",
      "enthusiasm": "low"
    }
  },
  "voices": {
    "balanced": {
      "cfg_scale": 2.4,
      "reference_text": "My name is Celune...",
      "persona": {
        "speaking_style": "Gentle and measured.",
        "style": {"enthusiasm": "low"}
      },
      "assets": {
        "wav": {
          "offset": 0,
          "length": 123456,
          "sha256": "..."
        },
        "pt": {
          "offset": 123456,
          "length": 9876,
          "sha256": "..."
        }
      }
    }
  }
}
```

Supported optional metadata fields are:

| Field | Meaning |
| --- | --- |
| `name` | Display name logged when the bundle is loaded |
| `description` | Free-form descriptive text |
| `default_voice` | Initial voice to select when present |
| `voice_order` | Preferred UI order for voices |
| `theme` | Optional UI colors: `background`, `accent`, and optional `glow_color` / `faded_accent` |
| `persona` | Optional character metadata used for naming and Persona-facing behavior |

Each voice entry may also include:

| Field | Meaning |
| --- | --- |
| `cfg_scale` | Optional positive VoxCPM2 classifier-free guidance scale for that voice |
| `reference_text` | Optional non-empty transcript for the voice's reference audio |
| `persona` | Optional style/rule additions layered on the shared Persona metadata |

Supported `persona` fields are:

| Field | Meaning |
| --- | --- |
| `persona.identity.name` | Character name; takes precedence over top-level `name` for character naming |
| `persona.identity.age` | Optional age text |
| `persona.identity.gender` | Optional gender text |
| `persona.identity.profile` | Optional profile text |
| `persona.profile` | Optional profile text fallback |
| `persona.speaking_style` | Optional one-string speaking style summary |
| `persona.boundaries` | Optional string or list of strings describing constraints |
| `persona.prompt_rules` | Optional string or list of strings with behavioral rules |
| `persona.example_dialogue` | Optional string or list of strings with example dialogue |
| `persona.style.*` | Optional style values for `warmth`, `directness`, `humor`, `detail`, `formality`, `enthusiasm` |

Each voice may provide its own `persona` object using `speaking_style`,
`boundaries`, `prompt_rules`, `example_dialogue`, and `style.*`. These values
are layered on top of the shared top-level `persona`: the shared character
identity remains active, text rules and examples are combined, and voice style
values refine the shared trait values. Voices without this block use the base
Persona unchanged.

Supported top-level bundle Markdown assets are:

- `identity.md`
- `soul.md`
- `personality.md`
- `speech_style.md`
- `boundaries.md`
- `examples.md`

In legacy CECHAR v3, those `.md` files live in the payload alongside `wav` and
`pt` data and are indexed from the top-level `assets` manifest table. In v4,
they are entries in the shared name-only `files` manifest and point into the
binary jump table.

Validation rules enforced by Celune:

- legacy `format`/`version` must be either `"CECHAR"` / `2` or `3`, or `"CEVOICE"` / `1`
- v4 `format`/`version` must be `"CECHAR"` / `4`
- v4 compression must be one of `0`, `1`, `2`, `3`, or `4`
- v4 metadata must contain a `files` list whose length equals the binary file count
- v4 metadata JSON must not begin with a UTF-8 byte-order mark
- v4 `files` entries must not contain `offset` or `length` fields
- v4 file names must be unique, safe logical names with `.wav`, `.pt`, or supported `.md` extensions
- v4 jump-table ranges must be inside the decompressed payload and must not overlap
- v4 WAV files must be mono, signed 16-bit PCM at 24 kHz
- v4 PT files must be restricted tensor-only 2048-element float32 embeddings
- v4 Persona Markdown must be UTF-8 inert character text without unsupported capability declarations or executable code
- `voices` must be an object
- `default_voice`, when present, must name a defined voice
- `voice_order`, when present, must be a duplicate-free list of defined voice names
- if `voice_order` omits valid voices, Celune appends the missing ones when loading
- `theme` must be an object when present
- `theme.background` and `theme.accent` must be `#RRGGBB` hex colors
- `theme.glow_color` and `theme.faded_accent` must be `#RRGGBB` hex colors when present
- legacy `theme.sleeping_color` is still accepted and is normalized into `theme.faded_accent`
- `persona` must be an object when present
- `assets` must be an object when present
- `persona.identity` and `persona.style` must be objects when present
- Persona text fields must be strings, and list-capable Persona fields must be either a string or a list of strings
- `voices.<name>.cfg_scale`, when present, must be a positive number
- `voices.<name>.reference_text`, when present, must be a non-empty string
- voice names and asset kinds may not contain path separators and may not be `""`, `"."`, or `".."`
- only `wav` and `pt` asset kinds are supported
- only supported persona `.md` filenames may appear in top-level `assets`
- legacy asset entries need a non-negative integer `offset`, a non-negative integer `length`, and a 64-character SHA-256 digest
- legacy assets must fit inside the payload region

## How Celune uses a bundle

At startup, Celune resolves `voice_bundle` from config:

- `default` becomes `voices/default.cevoice`
- a bare name such as `my_pack` becomes `voices/my_pack.cevoice`
- an explicit path is used as-is

The loader parses and validates the bundle, then lazily materializes assets into a temporary directory only when a backend needs a filesystem path.

- Qwen3 clone mode reads `wav` assets as reference audio and requires per-voice `reference_text`.
- VoxCPM2 reads `wav` assets as reference audio and uses per-voice `cfg_scale` when present.
- Voice names are local to the pack; they are not required to match a backend's built-in voice names.
- CEDTS synchronizes the active pack with its worker and resolves each voice to the shared backend model.
- Analysis helpers read optional `pt` assets directly from the bundle.
- `default_voice` controls the initial selected voice.
- `voice_order` controls the user-facing order.
- `theme.accent` or `theme.glow_color` can affect Celune's UI glow color.
- `persona.identity.name`, when present, becomes the bundle's character name ahead of top-level `name`.

## Persona debug overrides

Persona Markdown can be overridden during local character development without
rebuilding the CECHAR bundle. Enable the opt-in setting in the active config:

```yaml
persona:
  debug_overrides: true
```

Celune then checks the app-data directory for a character-specific folder. On
Windows, the default location is:

```text
C:\Users\<user>\AppData\Local\Celune\persona\<character-slug>\
```

Supported non-empty UTF-8 files with the same names as CECHAR assets replace the
matching files from the active bundle:

- `identity.md`
- `soul.md`
- `personality.md`
- `speech_style.md`
- `boundaries.md`
- `examples.md`

Persona memory is always stored separately for each character at
`<character-slug>\memory\records.json` under the same app-data directory.
The setting only controls whether local Markdown files replace the embedded
CECHAR Markdown.

If the configured bundle is missing or malformed, Celune simply has no compatible voice pack to load.

## Recommended way to make one

Use `write_cevoice()` instead of hand-building bytes. It computes offsets, lengths, hashes, required metadata, and the final header for you.

```python
from pathlib import Path

from celune.cevoice import write_cevoice

write_cevoice(
    "my_pack.cevoice",
    {
        "balanced": {
            "wav": Path("source-assets/balanced.wav"),
            "pt": Path("source-assets/balanced.pt"),
        },
        "calm": {
            "wav": Path("source-assets/calm.wav"),
            "pt": Path("source-assets/calm.pt"),
        },
        "bold": {
            "wav": Path("source-assets/bold.wav"),
        },
    },
    {
        "name": "My Pack",
        "description": "My custom Celune voice assets",
        "default_voice": "balanced",
        "voice_order": ["balanced", "calm", "bold"],
        "theme": {
            "background": "#1d1826",
            "accent": "#cebaff",
            "glow_color": "#cebaff",
            "faded_accent": "#9c88ce",
        },
        "persona": {
            "identity": {
                "name": "My Character",
                "profile": "A direct but kind archivist.",
            },
            "speaking_style": "Measured, observant, and brief.",
            "boundaries": [
                "Do not break character.",
            ],
            "prompt_rules": [
                "Prefer concrete observations.",
            ],
            "example_dialogue": [
                "User: is it fixed?",
                "My Character: The noisy part is gone. Let's verify the rest.",
            ],
            "style": {
                "warmth": "mid",
                "directness": "high",
                "humor": "low",
                "detail": "mid",
                "formality": "mid",
                "enthusiasm": "low",
            },
        },
    },
    {
        "balanced": {
            "cfg_scale": 2.4,
            "reference_text": "My reference transcript.",
        },
        "calm": {
            "cfg_scale": 3.0,
            "reference_text": "My calm reference transcript.",
        },
        "bold": {
            "cfg_scale": 2.4,
            "reference_text": "My bold reference transcript.",
        },
    },
)
```

The values in the `voices` mapping can be:

- `bytes`
- a string path
- a `Path`

`wav` is the important runtime asset. Include `pt` only if you also want Celune's analysis features to have embeddings for that voice.

## Minimal bundle recipe

For the smallest practical bundle:

1. Prepare at least one reference `.wav` file.
2. Give the voice a safe name such as `balanced`, `calm`, or `my_voice`.
3. Call `write_cevoice()` with one `wav` asset.
4. Optionally set `name`, `default_voice`, `voice_order`, and `theme`.
5. Point `voice_bundle` in config to the file path or place the file under `voices/`.

Example:

```python
from celune.cevoice import write_cevoice

write_cevoice(
    "single_voice.cevoice",
    {"my_voice": {"wav": "my_voice.wav"}},
    {
        "name": "Single Voice",
        "default_voice": "my_voice",
        "voice_order": ["my_voice"],
    },
)
```

## If you need to write a generator in another language

The writer algorithm is simple:

1. Start an empty payload buffer.
2. For every voice and asset:
   - read the raw asset bytes
   - record `offset = current payload length`
   - record `length = asset byte length`
   - record `sha256 = SHA-256(asset bytes)`
   - append the bytes to the payload
3. Build the metadata object with `format`, `version`, `voices`, and any optional fields.
4. Serialize metadata as compact JSON using UTF-8.
5. Write:
   - `b"CECHAR\0\0"`
   - little-endian `uint16(2)`
   - little-endian `uint32(len(metadata_bytes))`
   - `metadata_bytes`
   - `payload_bytes`

If you want byte-for-byte output compatible with Celune's current writer, serialize JSON with:

- ASCII escapes enabled
- sorted keys
- compact separators `(",", ":")`

That last part is not required for validity, but it matches the built-in writer.

## Reading and checking a bundle

You can validate a finished file through Celune's parser:

```python
from celune.cevoice import CEVoice

bundle = CEVoice.open("my_pack.cevoice")
print(bundle.voice_order)
print(bundle.read_asset("balanced", "wav")[:16])
```

`CEVoice.open()` validates:

- the header
- metadata structure
- allowed asset kinds
- offsets and lengths against the payload size

`read_asset()` additionally validates the SHA-256 digest of the asset bytes before returning them.

## Notes from Celune's default pack

Celune's bundled `default.cevoice` uses:

- `name`: `Celune`
- `default_voice`: `balanced`
- `voice_order`: `balanced`, `calm`, `bold`, `upbeat`
- `theme`: `background`, `accent`, `glow_color`, and `faded_accent`
- top-level `assets`: CECHAR v4-style prompt source material for the bundled character
- `soul.md`: shared character continuity and relationship guidance
- per-voice `persona`: response-style refinements layered onto the shared character
- `cfg_scale`: `2.4` for `balanced`, `bold`, and `upbeat`; `3.0` for `calm`
- `reference_text`: the transcript matching each bundled reference `wav`
- both `wav` and `pt` assets for each voice

That pack is a good real-world model if you want your own bundle to behave like the stock one.
