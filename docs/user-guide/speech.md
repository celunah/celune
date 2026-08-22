# Speech and voices

This page explains TTS backend differences, voice selection, reference
conditioning, playback controls, and Celune's audio invariants.

## TTS backends

Celune's TTS backends all consume the active CEVOICE/CECHAR pack when they
support cloning. The main differences are model family, language coverage,
latency, and reference conditioning.

| Backend | Strength | Chunk rate | Notes |
| --- | --- | ---: | --- |
| `qwen3` | Fast expressive cloning | 12.5 chunks/s | Supports Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian. |
| `mini` | Small and CPU-friendly | 12.5 chunks/s | Pocket TTS; English, French, German, Italian, Portuguese, and Spanish. |
| `voxcpm2` | High-fidelity multilingual generation | 6.25 chunks/s | Uses reference WAV plus per-voice `cfg_scale`; needs a compiler in some installs. |
| `dotstts` | Speaker similarity and diffusion quality | 6.25 chunks/s | Uses Celune's forked `dots.tts` package. |
| `gpt-sovits` | GPT-SoVITS family compatibility | 6.25 chunks/s | Supports Chinese, English, Japanese, Korean, and Cantonese variants; may exhibit accent drift. |

The backend manager installs backend-only requirements separately when
`isolated_backends` is enabled. Core modules do not import these packages at
startup.

## Voice selection

The bundled character has `balanced`, `calm`, `bold`, and `upbeat` voices. A
pack can define any safe voice names and an explicit `voice_order`. The Textual
style button cycles voices; Python callers select one with:

```python
celune.set_voice_and_wait("balanced")
```

`set_voice()` requests the change and returns quickly. `set_voice_and_wait()`
waits for the synchronized reload; `set_voice_async()` is the non-blocking
counterpart. The same pattern is available for backends and CEVOICE packs.

## Reference conditioning

Qwen3 and the other cloning backends use the active voice's `wav` and, when
required, its exact `reference_text`. A full reference recording is preferred
to a very short fragment because it preserves timbre more consistently. The
`qwen3_x_vector_only` option can lock speaker identity while reducing expressive
conditioning; use it when identity stability matters more than style transfer.

VoxCPM2 reads `cfg_scale` from per-voice metadata, with the bundled defaults at
2.4 for balanced/bold/upbeat and 3.0 for calm. GPT-SoVITS uses longer reference
audio and has family-specific preprocessing requirements.

## Playback controls

- `/speed <percent>` sets the interactive playback speed from 80% to 120%.
- `/reverb <percent>` sets the interactive reverb strength from 0% to 100%.
- `/seed <number>` selects a backend seed from 0 through `2^32 - 1`;
  `/seed random` chooses a new seed.
- `/stop` interrupts active speech through the shared cancellation path.
- `/play <path> [volume]` queues a sound effect, with volume from 0 to 1.
- `/consumebuffer <true|false>` controls whether the UI consumes buffered
  audio at a boundary.

The Python `Celune.say()`, `say_async()`, `say_stream()`, and
`say_stream_async()` calls expose the same pipeline. Saved speech and API audio
are FLAC. Effects can be marked `keep=true`/`keep=True` to prepend them to the
next saved utterance.

## Audio invariants

Inside Celune, audio is normalized `numpy.float32` in the range -1.0 to 1.0.
The canonical output is 48 kHz stereo. Input helpers accept mono or stereo
arrays and resample as needed. Code that crosses a backend or file boundary
must preserve the declared sample rate, channels, dtype, and normalization;
int16 payloads are converted using `/32768` when decoded by CEDTS.

## Long-form speech

Long text is segmented before generation. Smart buffering protects already
played audio and throttles playback speed only within its configured limits.
For a caller that needs every chunk, use `say_stream()` and drain the returned
queue until its terminal sentinel/condition; do not read the internal playback
queue directly.
