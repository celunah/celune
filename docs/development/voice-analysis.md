# Voice analysis and pack creation

## Analysis API

`celune.analysis` is used by the generation pipeline and developer workflows.
Its public functions are:

| Call | Purpose |
| --- | --- |
| `load_audio(path) -> (audio, sample_rate)` | Read an audio file as a Celune analysis chunk. |
| `compute_raw_metrics(audio, sample_rate)` | Measure duration, loudness, pitch, pauses, and related metrics. |
| `add_reference_similarity_metrics(metrics, voice)` | Compare an analyzed voice with a compatible embedded reference. |
| `compute_traits(metrics)` | Convert metrics into calm/energy/presence/style traits. |
| `generate_assessment(metrics, traits)` | Produce human-readable assessment lines. |
| `plot_radar(metrics, traits, path)` | Write a trait radar plot. |
| `write_report(metrics, traits, assessment, voice, path)` | Write a plain-text report. |
| `analyze_voice_audio(audio, sample_rate, ...)` | Run the in-memory analysis path used by pipeline markers. |

Reference similarity requires a `.pt` asset with a 2,048-element float32
speaker embedding in the active pack. Missing or malformed embeddings produce
a clear analysis error; the rest of the raw analysis can still be useful.

## `scripts/cac.py`

The Character and Audio Creator script creates a CEVOICE pack through either a
simple command or an interactive wizard.

### Simple mode

```bash
python scripts/cac.py Nova reference.wav "Hello, my name is Nova."
```

The positional arguments are `NAME`, `WAV`, and optional `REFERENCE_TEXT`. If
the transcript is omitted, the script prompts for it. The WAV is normalized to
mono, float32, and the pack's required PCM representation before it is passed
to `write_cevoice()`.

### Wizard mode

```bash
python scripts/cac.py
```

The wizard collects a pack/character name, number of voices, each voice name,
reference WAV, transcript, optional CFG scale, optional per-voice Persona style,
theme, shared Persona metadata, default voice, and voice order. It rejects
duplicate/unknown names and writes a validated bundle. It is intentionally
interactive; automation should call `write_cevoice()` or `write_cechar_v4()`
directly.

## Reference design

Use a full, clean reference recording with an exact transcript. Very short
fragments can drift in timbre. Keep the output sample rate and channels
explicit, remove accidental leading/trailing noise, and test the pack on the
backend(s) that will consume it. Include `.pt` embeddings only when the
analysis/reference-similarity path is needed.

The existing [Voice design](../VOICES.md) page records Celune's reference text,
audition prompts, voice candidates, pitch edits, reverb/compressor settings,
and final output target.
