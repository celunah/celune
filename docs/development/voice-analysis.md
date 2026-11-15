# Voice and pack design

This page documents the voice-analysis calls and the `scripts/cac.py` workflows
used to evaluate audio and create CEVOICE/CECHAR packs.

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
| `analyze_voice_audio(audio, sample_rate, ..., reference_text, language, backend_id, latency_ms, realtime_speed)` | Run the in-memory analysis path used by pipeline markers and write CVS results. |

`analyze_voice_audio()` returns `None` and writes a radar PNG and text report
under `out_dir`. Its CVS-specific arguments are the TTS reference text,
language tag, backend identifier, request-to-first-audio latency in
milliseconds, and synthesis realtime speed. Audio analysis failures propagate
to the caller; ASR failures make WER unavailable, and database write failures
leave the per-utterance score in the report while the aggregate is `N/A`.

```python
from pathlib import Path

from celune.analysis import analyze_voice_audio, load_audio

audio, sample_rate = load_audio(Path("speech.wav"))
report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)
analyze_voice_audio(
    audio,
    sample_rate,
    "speech.wav",
    report_dir,
    "speech",
    reference_voice="Celune",
    reference_text="Hello, this is Celune.",
    language="en",
    backend_id="pocket_tts",
    latency_ms=420.0,
    realtime_speed=1.1,
)
```

Reference similarity requires a `.pt` asset with a 2,048-element float32
speaker embedding in the active pack. Missing or malformed embeddings produce
a clear analysis error; the rest of the raw analysis can still be useful.

## Automatic generation analysis

When verbose or debug logging is enabled and Celune saves generated audio, the
pipeline schedules its radar chart and report analysis after playback. A daemon
launcher thread starts the detached analysis process, so Windows process
startup and audio transfer do not pause the playback event loop or subsequent
speech generation. A process-start failure is reported as a warning.

## Cel Voice Score

The generated report replaces the standalone cosine similarity summary with a
Cel Voice Score (CVS) report. It shows the current utterance as `CVS[<language>]`,
the backend mean as `CVS[all]`, and the four component scores with their raw
measurements. The backend aggregate is the arithmetic mean of complete CVS
scores recorded for that backend.

| Component | Score | Input |
| --- | --- | --- |
| Voice match (40%) | 100 at 97% similarity; 10 points per similarity percentage point below 97%; 0 at 87% or lower | Cosine similarity to the selected CEVOICE reference embedding |
| WER (30%) | `max(0, 100 - 5 × WER_percent)` | Whisper Tiny transcription compared with the exact text sent to TTS; WER is in percentage points and is an ASR estimate |
| Latency (15%) | 100 through 500 ms; then 5 points per additional 250 ms; 0 at 5,500 ms | Request-to-first-usable-audio latency |
| Realtime speed (15%) | Linear from 0 at 0× to 100 at 1.2×; capped at 100 | Generated audio duration divided by generation time |

The weighted total is `0.40V + 0.30W + 0.15L + 0.15S`. A missing input
component leaves that utterance's total unavailable; the report still shows
available component measurements. WER compares case-folded,
punctuation-normalized whitespace-separated words and reports the edit count
and reference word count. Whisper recognition errors can raise WER even when
the speech sounds correct, so treat it as an ASR estimate rather than a direct
measure of audible intelligibility. WER is unavailable for Japanese, Chinese,
Thai, Lao, Myanmar, and Khmer text because this analysis path does not segment
those scripts into words. The Whisper Tiny analysis adapter can run on CUDA or
CPU; on CPU, report generation may take longer.

Utterance measurements and the backend aggregate are stored in
`cvs_aggregates.db` under Celune's application data directory. The database
stores hashed utterance identifiers, backend and language tags, CVS component
scores, and numeric source measurements. It does not store transcripts or audio
paths. Re-analyzing the same output path updates its row rather than counting
the utterance twice. Only complete CVS scores contribute to `CVS[all]`.

## Verification and troubleshooting

- Run `python scripts/run_ci.py` after changes to the analysis or ASR code.
- A report with `CVS[<language>]: N/A` has at least one unavailable component;
  inspect the four component rows to identify it.
- `CVS[all]: N/A` means this backend has no complete utterance scores in the
  aggregate database yet, or the database could not be written.
- WER and voice match require the exact generation transcript and a compatible
  CEVOICE reference embedding, respectively. Language tags are reduced to
  their base language for score labels and ASR prompting.

## See also

- [Voice design](../VOICES.md)
- [Model registry and consent](model-registry.md)

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
interactive; automation should call `write_cevoice()` for legacy-compatible
packs or `write_cechar_v5()` for new indexed character packs. Use
`write_cechar_v4()` only when v4 compatibility is required.

## Reference design

Use a full, clean reference recording with an exact transcript. Very short
fragments can drift in timbre. Keep the output sample rate and channels
explicit, remove accidental leading/trailing noise, and test the pack on the
backend(s) that will consume it. Include `.pt` embeddings only when the
analysis/reference-similarity path is needed.

The existing [Voice design](../VOICES.md) page records Celune's reference text,
audition prompts, voice candidates, pitch edits, reverb/compressor settings,
and final output target.
