# Speech samples

This page describes the README's backend demonstrations for listeners and
contributors comparing the current default voices.

## Recordings and references

The samples use the balanced, calm, bold, and upbeat recordings embedded in
`voices/default.cevoice`, refreshed on 30 September 2026. Each backend reads
the same voice references and synthesizes the same introduction and narration.
The reference transcripts come from the pack's `reference_text` metadata.

Generation calls the installed model libraries directly, using snapshots in
Celune's local Hugging Face cache and the existing isolated backend environments.
It does not start the engine, UI, playback pipeline, or Persona. Consequently,
these clips demonstrate the model's voice cloning, without Celune's playback DSP.

Files under `demos/` use the name `{voice}_{length}_{backend}.flac`, where `sc`
means introduction and `lc` means narration. Every new recording is mono,
24-bit PCM FLAC at 48 kHz. Lower-rate model outputs are resampled after inference;
outputs exceeding a peak amplitude of 0.98 are scaled down before encoding.
Earlier WAV recordings remain available as historical samples.

The generation record in `demos/samples.json` contains the exact input passages,
voice-pack and reference hashes, model snapshot revisions, inference settings,
and output measurements.

## Comparing backends

Use the README's introduction and narration links to compare delivery for one
voice across Qwen3-TTS, VoxCPM2, Celune Mini, dots.tts, FireRedTTS3, and LuxTTS.
All generation uses seed 1234. Backend-specific sampling and conditioning remain
different, so matching text and references do not guarantee identical timing.

## Verification and limitations

Before installation, validate the FLAC encoding, sample rate, finite waveform,
duration, and nonzero signal. Transcribe the samples with the cached Whisper
model to check for missing passages and reference leakage. Recognition errors
can differ from speech errors; phonetic substitutions for "Celune" are accepted.
Transcription is not a listening-quality score.
Model mistakes can remain even when the input text is reproduced correctly.

README playback URLs point to the repository's `main` branch. New recordings
become available through those links after the assets and README are pushed.

## See also

- [Speech](../user-guide/speech.md)
- [Backends](backends.md)
- [CEVOICE/CECHAR](../CEVOICE.md)
- [Voice and pack design](voice-analysis.md)
