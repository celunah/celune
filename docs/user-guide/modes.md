# Modes

Celune has three operation modes and a separate legacy input-mode layer. The
operation mode decides how text is routed; the input mode decides whether the
engine accepts text or audio.

## Modes

| Mode | Text route | Typical use |
| --- | --- | --- |
| `speak` | Text goes directly to the TTS queue. | Speech-only applications and deterministic playback. |
| `converse` | Text goes to Persona, whose response is spoken through the same queue. | Character conversation, memory, vision, and talkback. |
| `agent` | Persona plans, selects a registered tool, executes it under policy, and responds. | Bounded local tasks with typed tool calls and approvals. |

Set the mode in YAML:

```yaml
mode: converse
```

The resolver accepts the explicit names `speak`, `converse`, and `agent`. The
older input names `text_to_speech`, `tts`, `voice_conversion`, and `revoice`
remain accepted when older integrations construct the runtime. They are
normalized internally and should not be used as new public configuration.

## Text-to-speech flow

In `speak` mode, `say()` and `/say` queue the supplied text literally. Celune
normalizes special characters, segments long input, asks the active backend for
streaming chunks, applies the smart buffer and DSP, then sends normalized audio
to the playback worker. A request can save the final result, expose chunks to a
Python queue, or stream the result through the REST API.

## Conversation flow

In `converse` mode, typed text is handled by Persona. The response may use the
loaded CEVOICE character metadata, short-term history, long-term memory, visual
attachments, and emotion cues. The final response is passed to the normal
speech pipeline; Persona does not own a separate playback implementation.

`/think` is the direct UI command for this route. The REST `/v1/think` endpoint
accepts a request asynchronously and returns `202` when the turn is accepted.

## Agent flow

Agent mode extends Persona with a typed, allowlisted tool catalog. A task can
plan, call read-only tools, pause for approval, execute an approved mutating
tool, and answer through the speech pipeline. The production limits are 20
loops, a 32,768-token agent context size, and compaction at 75 percent. See
[Agent mode](agent.md) for every built-in tool family and its permissions.

## Voice-conversion input

Voice conversion is an audio-input mode. Text commands that require TTS are
not accepted while VC is active. Use `/vc <file>` for a file or `CTRL+R` to
toggle live microphone capture. The converted audio still passes through the
shared playback and save path, but it does not load a TTS model.

## Mode changes

The stable Python calls are `set_backend*`, `set_cevoice*`, `set_voice*`, and
the `mode`/input-mode configuration. Slash commands provide the interactive
surface. A backend or character change may reload models and therefore should
be awaited with the `*_and_wait()` or async form when the caller needs the new
state immediately.
