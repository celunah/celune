# Persona

Persona is Celune's local character runtime. It turns text, speech, and vision
inputs into a response that is spoken by the active TTS pipeline. Persona is
available in `converse` and `agent` modes. `performance` mode deliberately bypasses
it.

## Model and capabilities

The default model is `Qwen/Qwen3-VL-4B-Instruct`. Celune also supports
registered Qwen3-VL, Qwen3.5, Gemma, and KRAFTON Raon-1-nano models. The
selected `vram` tier controls loading and quantization. The `high` preset
supports standard registered models, including the 4B tier. Registered smart
8B-tier models require `xhigh`.

Raon-1-nano is text-only and does not accept attachments. Its native
Transformers loader requires Transformers 5.16.1 or newer within the supported
5.x range. Native 4-bit and 8-bit quantization of its embedding table is
approximate and may affect output quality. Persona's GGUF runtime does not yet
support the Raon architecture.

Persona context is capped at 2,048 tokens outside agent mode. Agent requests use
a separate 8,192-token cap.

## Character context

The active CEVOICE/CECHAR pack supplies the character identity, speaking style,
boundaries, prompt rules, examples, and per-voice refinements. The shared
character layer is combined with the selected voice layer. Changing voice does
not discard the character identity.

The default short-term history keeps up to 20 messages. The shared
`persona.compact_at` setting controls when conversation history is compacted in
both `converse` and `agent` modes. When context summarization is enabled, the
runtime retains the configured recent messages and asks a neutral summarizer to
produce a bounded summary. In agent mode, the same percentage is applied to the
larger agent context window.

## Text and speech input

Type directly in the UI and press `^ENTER`, call `Celune.think()` or
`think_async()`, or post to `/v1/think`. In the Textual UI, `^R` toggles
Persona speech capture. Whisper uses the configured
speech model, automatic language detection by default, a five-second no-input
timeout, and a 1.5-second speech-end delay.

Input audio is resampled before transcription. Speech captions use the separate
IPA aligner documented in [Speech](speech.md).

## Attachments

`/attach` accepts local images (`.jpeg`, `.jpg`, `.png`, `.webp`) and videos
(`.mp4`, `.webm`). HTTP and HTTPS image/video URLs are accepted when their path
extension identifies one of those types. Use `/attach clear` to remove all
pending attachments. Attachments are sent on the next vision-capable Persona
turn. They are not interpreted by a speech-only backend.

`/say <text>` is useful after an attachment: it sends the text as a direct
vision prompt.

## Emotion probes

When enabled, the `lunahr/emotispace-128` model can provide emotion labels as a
response-style cue. These labels are not a diagnostic claim about the user's
mental state.

## Long-term memory

Memory is scoped to the active character and stored under
`<Celune app-data>/persona/<character-slug>/memory/records.json`. Use explicit
phrases such as "remember this" or "save this" to persist a detail. Automatic
suggestions use user-authored messages only. Credentials and similar sensitive
secrets are rejected rather than stored.

Explicit memories remain until forgotten. Inferred memories expire 60 days
after their last update by default. Set
`persona.memory.automatic_max_age_days` to `null` to disable that expiry.
Retrieval combines semantic similarity with literal text matching when local
embeddings are available, and uses text matching otherwise.

Agent mode provides `remember`, `recall`, `forget`, `clear_recent_context`, and
`summarize_context` tools. Ordinary Persona turns use the same memory store.

## Debug Markdown overrides

Set `persona.debug_overrides: true` to develop a character without rebuilding
the pack. Persona looks in the character-slug directory for the supported
Markdown files described in [Persona Markdown](../formats/persona-markdown.md).
Overrides replace the matching embedded file only. Memory remains in its
separate `memory/` directory.

Malformed or missing character packs do not silently become a different
character. The loader reports that no compatible bundle is available.
