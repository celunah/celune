# CEVOICE

`CEVOICE` is Celune's voice-pack container format. A `.cevoice` file stores:

- a small fixed-size binary header
- UTF-8 JSON metadata
- a contiguous binary payload containing the voice assets

Celune currently accepts two asset kinds:

- `wav`: reference audio used by the voice-cloning backends
- `pt`: optional PyTorch embedding data used by Celune's analysis tools

The canonical implementation lives in `celune/cevoice.py`.

## File layout

Each bundle is written as:

| Region | Size | Description |
| --- | ---: | --- |
| Header | 14 bytes | `struct.Struct("<8sHI")` |
| Metadata | variable | UTF-8 JSON |
| Payload | variable | concatenated asset bytes |

The header fields are:

| Field | Type | Value |
| --- | --- | --- |
| `magic` | `8s` | `b"CEVOICE\0"` |
| `version` | `H` | `1` |
| `metadata_length` | `I` | byte length of the JSON metadata |

Asset offsets are relative to the start of the payload, not the start of the file.

## Metadata schema

`write_cevoice()` creates the required metadata automatically:

```json
{
  "format": "CEVOICE",
  "version": 1,
  "voices": {
    "balanced": {
      "cfg_scale": 2.4,
      "reference_text": "My name is Celune...",
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
| `theme` | Optional UI colors: `background`, `accent`, and optional `glow_color` |

Each voice entry may also include:

| Field | Meaning |
| --- | --- |
| `cfg_scale` | Optional positive VoxCPM2 classifier-free guidance scale for that voice |
| `reference_text` | Optional non-empty transcript for the voice's reference audio |

Validation rules enforced by Celune:

- `format` must be `"CEVOICE"` and `version` must be `1`
- `voices` must be an object
- `default_voice`, when present, must name a defined voice
- `voice_order`, when present, must be a duplicate-free list of defined voice names
- if `voice_order` omits valid voices, Celune appends the missing ones when loading
- `theme.background`, `theme.accent`, and `theme.glow_color` must be `#RRGGBB` hex colors when present
- `voices.<name>.cfg_scale`, when present, must be a positive number
- `voices.<name>.reference_text`, when present, must be a non-empty string
- voice names and asset kinds may not contain path separators and may not be `""`, `"."`, or `".."`
- only `wav` and `pt` asset kinds are supported
- every asset entry needs a non-negative integer `offset`, a non-negative integer `length`, and a 64-character SHA-256 digest
- each asset must fit inside the payload region

## How Celune uses a bundle

At startup, Celune resolves `voice_bundle` from config:

- `default` becomes `celune/voices/default.cevoice`
- a bare name such as `my_pack` becomes `celune/voices/my_pack.cevoice`
- an explicit path is used as-is

The loader parses and validates the bundle, then lazily materializes assets into a temporary directory only when a backend needs a filesystem path.

- Qwen3 clone mode reads `wav` assets as reference audio and uses per-voice `reference_text` when present.
- VoxCPM2 reads `wav` assets as reference audio and uses per-voice `cfg_scale` when present.
- Analysis helpers read optional `pt` assets directly from the bundle.
- `default_voice` controls the initial selected voice.
- `voice_order` controls the user-facing order.
- `theme.accent` or `theme.glow_color` can affect Celune's UI glow color.

If the configured bundle is missing, Celune simply has no bundle to load. If a configured bundle exists but is malformed, Celune falls back to the legacy loose reference files when possible.

## Recommended way to make one

Use `write_cevoice()` instead of hand-building bytes. It computes offsets, lengths, hashes, required metadata, and the final header for you.

```python
from pathlib import Path

from celune.cevoice import write_cevoice

write_cevoice(
    "my_pack.cevoice",
    {
        "balanced": {
            "wav": Path("refs/balanced.wav"),
            "pt": Path("refs/balanced.pt"),
        },
        "calm": {
            "wav": Path("refs/calm.wav"),
            "pt": Path("refs/calm.pt"),
        },
        "bold": {
            "wav": Path("refs/bold.wav"),
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
5. Point `voice_bundle` in config to the file path or place the file under `celune/voices/`.

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
   - `b"CEVOICE\0"`
   - little-endian `uint16(1)`
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
- `cfg_scale`: `2.4` for `balanced`, `bold`, and `upbeat`; `3.0` for `calm`
- `reference_text`: the transcript matching each bundled reference `wav`
- both `wav` and `pt` assets for each voice

That pack is a good real-world model if you want your own bundle to behave like the stock one.
