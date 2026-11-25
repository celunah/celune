# Speech

This page explains TTS backend differences, voice selection, reference
conditioning, playback controls, and Celune's audio invariants.

## TTS backends

Celune's TTS backends all consume the active CEVOICE/CECHAR pack when they
support cloning. The main differences are model family, language coverage,
latency, and reference conditioning.

| Backend | Strength | Chunk rate | Notes |
| --- | --- | ---: | --- |
| `qwen3` | Fast expressive cloning | 12.5 chunks/s | Supports Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian. |
| `mini` | Small and CPU-friendly | 12.5 chunks/s | Pocket TTS supports English, French, German, Italian, Portuguese, and Spanish. |
| `fireredtts3` | Multilingual zero-shot cloning | 1 complete chunk/request | Supports 24 languages and 21 Chinese dialect tags. It uses the active reference WAV and transcript and loads its transformer path in BF16. |
| `voxcpm2` | High-fidelity multilingual generation | 6.25 chunks/s | Uses the reference WAV plus per-voice `cfg_scale` and may need a compiler for some installs. |
| `dotstts` | Speaker similarity and diffusion quality | 6.25 chunks/s | Uses Celune's forked `dots.tts` package. |
| `ming3` | Multilingual streaming voice cloning | Native model-time stream | Streams Ming 3.0 TTS through audio.cpp, supports Chinese, English, Japanese, Korean, German, French, and Spanish, and uses Q8_0 when quantization is enabled or BF16 otherwise. |

Backend-specific packages are resolved from Celune's configured application
environment and imported lazily when the selected backend is needed. Normal
startup does not create or manage per-backend environments. Explicit CEDTS
worker callers may still use the manifest-backed setup path.

## Voice selection

The bundled character has `balanced`, `calm`, `bold`, and `upbeat` voices. A
pack can define any safe voice names and an explicit `voice_order`. The Textual
style button cycles voices. Python callers select one with:

```python
celune.set_voice_and_wait("balanced")
```

`set_voice()` requests the change and returns quickly. `set_voice_and_wait()`
waits for the synchronized reload. `set_voice_async()` is the non-blocking
counterpart. The same pattern is available for backends and CEVOICE packs.

## Reference conditioning

Qwen3 and the other cloning backends use the active voice's `wav` and, when
required, its exact `reference_text`. A full reference recording is preferred
to a very short fragment because it preserves timbre more consistently. The
`backend_options.qwen3.x_vector_only` can lock speaker identity while reducing
expressive conditioning. Set it when identity stability matters more than style
transfer.

FireRedTTS3 uses the active reference WAV and its exact `reference_text` for
zero-shot cloning and keeps those two prompt spans aligned to prevent reference
text from leaking into the requested speech. It returns a complete 24 kHz
waveform after inference, which the application resamples at the common playback
boundary. Its Qwen backbone, stop head, and RedAE encoder load in BF16. The
flow head and decoder remain F32.

VoxCPM2 reads `cfg_scale` from per-voice metadata, with the bundled defaults at
2.4 for balanced/bold/upbeat and 3.0 for calm. Ming3 uses the active voice's
reference WAV and exact transcript, and requires a CUDA-capable runtime.

## Playback controls

- `/speed <percent>` sets the interactive playback speed from 80% to 120%.
- `/reverb <percent>` sets the interactive reverb strength from 0% to 100%.
- `/seed <number>` selects a backend seed from 0 through `2^32 - 1`.
  `/seed random` chooses a new seed.
- `/stop` interrupts active speech through the shared cancellation path.
- `/play <path> [volume]` queues a sound effect, with volume from 0 to 1.
- `/consumebuffer <true|false>` controls whether the UI consumes buffered
  audio at a boundary.

The Python `Celune.say()`, `say_async()`, `say_stream()`, and
`say_stream_async()` calls expose the same pipeline. Saved speech and API audio
are FLAC. Effects can be marked `keep=true`/`keep=True` to prepend them to the
next saved utterance.

## Saved audio metadata

Saved speech FLAC Vorbis comments use the active character for `artist` and the
character portion of `album`. The album tag follows `<character> via <backend>`.
When no character is active, Celune's name is used.

## Audio invariants

Inside Celune, audio is normalized `numpy.float32` in the range -1.0 to 1.0.
The canonical output is 48 kHz stereo. Input helpers accept mono or stereo
arrays and resample as needed. Code that crosses a backend or file boundary
must preserve the declared sample rate, channels, dtype, and normalization.
Int16 payloads are converted using `/32768` when decoded by CEDTS.

## Long-form speech

Celune segments long text before generation and uses smart buffering to keep
playback smooth. Captions are optional. Set `captions: true` to align words
with speech. This requires the `espeak-ng` executable and the optional
`sadda-speech/wav2vec2-espeak-ctc` model. Install caption support on Windows
with:

```powershell
uv sync --dev --extra api --extra captions --extra cu132
```

Celune prefers the ONNX Runtime CUDA provider and falls back to CPU. If caption
dependencies are unavailable or alignment fails, speech continues without
captions. With captions disabled, Celune does not download the alignment model.

Python callers that need generated chunks can use `say_stream()` and drain its
returned queue. Do not read the internal playback queue directly.
