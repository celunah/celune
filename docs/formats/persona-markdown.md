# Persona Markdown

CECHAR v4 can carry inert Markdown prompt assets shared by the character and
selected by Persona. These files are data, not executable extensions. The
loader validates their names, encoding, and contents before exposing them to a
model.

## Supported files

Only these top-level files are recognized:

| File | Intended content |
| --- | --- |
| `identity.md` | Name, profile, age/gender text, and identity continuity. |
| `soul.md` | Shared continuity, values, and relationship guidance. |
| `personality.md` | Personality traits and general behavior. |
| `speech_style.md` | Sentence rhythm, hesitation, brevity, and delivery style. |
| `boundaries.md` | Safety, role, and character boundaries. |
| `examples.md` | Example dialogue and response patterns. |

An empty optional file can be omitted. The v4 manifest stores only safe file
names; byte offsets and lengths live in the binary jump table.

## Content rules

Persona Markdown must be UTF-8 text without a byte-order mark. It is inert
character data: it must not declare unsupported capabilities, executable code,
or an instruction to bypass the runtime's model/revision policy. Keep prompt
files focused on character behavior rather than operational secrets, API keys,
filesystem paths, or tool permissions.

The CEVOICE metadata fields `boundaries`, `prompt_rules`, and
`example_dialogue` follow the same conceptual layering even when the pack does
not include Markdown assets.

## Layering and precedence

Persona combines the sources in this order:

1. Embedded shared character Markdown and metadata.
2. Embedded per-voice metadata refinements.
3. Debug override Markdown when `persona.debug_overrides` is enabled.
4. The current turn, attachments, memory retrieval, and bounded conversation
   context.

The shared identity remains active when a voice changes. Voice-level speaking
style, boundaries, examples, and style values refine or extend the shared
character; they do not silently create a second character.

## Debug override directory

For a character slug such as `celune`, put matching files under:

```text
<Celune app-data>/persona/celune/
```

The override is selected by the active character name and is read only when
the opt-in flag is true. Persona memory remains under the separate
`memory/records.json` child directory. This makes it possible to edit prompt
material without modifying a validated archive while keeping memories out of
the pack itself.

## Security boundary

The Markdown loader does not treat Markdown as a plugin mechanism. It rejects
path traversal, unsupported filenames, malformed text types, and unsupported
declarations. Do not use Persona Markdown as a way to grant an agent new
tools; tool availability is defined by the typed agent catalog and configuration.
