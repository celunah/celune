# Voice conversion

Voice conversion accepts recorded or live audio and renders it in the active
voice. It is an input mode, not a TTS mode: text speech commands are disabled
while the engine is in VC mode.

## Backends

The Seed-VC backend is the production conversion path. It supports
reference-based conversion, pitch adjustment, optional F0 conditioning, and a
low-latency live path.

Seed-VC's primary controls are:

| Control | Default | Meaning |
| --- | ---: | --- |
| diffusion steps | 30 | Quality/latency tradeoff. |
| length adjust | 1 | Timing adjustment. |
| inference CFG rate | 0.5 | Diffusion guidance. |
| F0 condition | `false` | Talk mode by default; singing/intonation path when enabled. |
| automatic F0 adjust | `true` | Aligns pitch for conversion. |
| pitch shift | 0 | Semitones; UI range -12 through +12. |

The active CEVOICE/CECHAR pack supplies the target reference WAV. A pack with
no compatible WAV cannot be used as a VC target.

## File conversion

Use:

```text
/vc path/to/input.wav
```

The command decodes the file as float32, submits it through the shared audio
pipeline, and plays the converted output. Python callers can use
`submit_audio()` for fire-and-play behavior or `convert_audio()` when they need
an `AudioOutput` value back.

The REST API exposes the same operation at `POST /v1/convert` as a multipart
upload. Optional `pitch_shift` and `f0_condition` form fields override the
runtime defaults for that request.

## Live microphone conversion

Press `CTRL+R` in VC mode to start or stop microphone capture. The capture path
uses preroll, hangover, voice-activity checks, and a bounded submission queue
so the first phonemes are not cut off and stale audio cannot grow without
limit. With the optional `live-vc-ai` extra, Celune uses Silero VAD; otherwise
it uses the built-in energy detector. VAD is always active during live capture.

Use `/vcmode talk` for ordinary speech and `/vcmode sing` for F0-conditioned
singing. Use `/vcpitch <semitones>` or `/vcpitch clear` to adjust pitch. The UI
also exposes buttons for F0 mode and pitch cycling.

The live backend keeps overlap state between blocks. Call
`convert_live_audio()` for each low-latency block and `stop_live_audio()` when
the stream ends so that backend state is flushed.

## Audio and rate behavior

Input arrays may be mono or stereo and are normalized to float32. Seed-VC's
non-F0 conversion path returns 22,050 Hz audio; its F0 path returns 44,100 Hz
audio before Celune resamples for the shared 48 kHz playback path. Do not infer
the sample rate from the number of samples; use the returned metadata.

Feedback protection stops a live capture when playback leakage creates a large
RMS spike. If live capture stops unexpectedly, move the microphone away from
speakers, lower output volume, or use headphones before changing VAD thresholds.
