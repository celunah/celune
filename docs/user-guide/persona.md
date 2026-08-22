# Persona

Persona is Celune's local character runtime. It turns text, speech, and vision
inputs into a response that is spoken by the active TTS pipeline. Persona is
available in `converse` and `agent` modes; `speak` mode deliberately bypasses
it.

## Model and capabilities

The default model is `Qwen/Qwen3-VL-4B-Instruct`. The model registry in
`celune.constants` also contains the supported Qwen3-VL, Qwen3.5, and Gemma
variants, each with an allowlisted revision. The selected `vram` tier controls
quantization and loading choices. Persona can report these capabilities:

```python
capabilities = celune.vision.capabilities() if celune.vision is not None else None
```

`PersonaCapabilities` distinguishes text, vision, image uploads, and emotion
probes. A speech-only model has no attachment path, even if the UI command is
present.

## Character context

The active CEVOICE/CECHAR pack supplies the character identity, speaking style,
boundaries, prompt rules, examples, and per-voice refinements. The shared
character layer is combined with the selected voice layer; changing voice does
not discard the character identity.

The default short-term history keeps up to 20 messages. When compaction is
enabled, the runtime retains the configured recent messages and asks a neutral
summarizer to produce a bounded summary. Compaction prevents an ever-growing
conversation from consuming the Persona context window.

## Text and speech input

Type directly in the UI and press `CTRL+ENTER`, call `Celune.think()` or
`think_async()`, or post to `/v1/think`. In the Textual UI, `CTRL+R` toggles
Persona speech capture when talkback is enabled. Whisper uses the configured
speech model, automatic language detection by default, a five-second no-input
timeout, and a 1.5-second speech-end delay.

The transcriber exposes segment and word timestamps so the UI can display
captions and word-level progress. Input is resampled to the runtime's audio
invariants before transcription.

## Attachments

`/attach` accepts local images (`.jpeg`, `.jpg`, `.png`, `.webp`) and videos
(`.mp4`, `.webm`). HTTP and HTTPS image/video URLs are accepted when their path
extension identifies one of those types. Use `/attach clear` to remove all
pending attachments. Attachments are sent on the next vision-capable Persona
turn; they are not interpreted by a speech-only backend.

`/say <text>` is useful after an attachment: it sends the text as a direct
vision prompt. In a vision-capable setup, Celune also keeps an internal display
form when IPA replacement or other normalization changes the text that is
synthesized.

## Emotion probes

When enabled by the active Persona runtime, the `lunahr/emotispace-128` model
produces GoEmotions-style labels. The implementation weights the user's
emotion more strongly than the assistant's previous emotion and turns the
result into a response-style cue. It is an input to response shaping, not a
diagnostic claim about the user's mental state.

## Long-term memory

Memory is character-scoped. The default store is:

```text
<Celune app-data>/persona/<character-slug>/memory/records.json
```

Each record contains an ID, normalized content, importance from 1 to 3,
explicitness, creation/update timestamps, and last-use timestamp. Use explicit
phrases such as “remember this”, “don't forget”, “make a note of this”, “save
this”, “on record”, or “going forward” when a detail should be persisted.

The classifier may suggest additional memories when its confidence is above
the configured threshold and the candidate count is within the configured
limit. Sensitive-looking secrets, passwords, API keys, payment numbers, and
similar credentials are rejected rather than stored. Retrieval uses a local
sentence-transformer embedding when available and falls back to token overlap;
the configured thresholds control both paths.

Agent tools expose `remember`, `recall`, `forget`, `clear_recent_context`, and
`summarize_context`. The same memory store is used by ordinary Persona turns.

## Debug Markdown overrides

Set `persona.debug_overrides: true` to develop a character without rebuilding
the pack. Persona looks in the character-slug directory for the supported
Markdown files described in [Persona Markdown](../formats/persona-markdown.md).
Overrides replace the matching embedded file only; memory remains in its
separate `memory/` directory.

Malformed or missing character packs do not silently become a different
character. The loader reports that no compatible bundle is available.
