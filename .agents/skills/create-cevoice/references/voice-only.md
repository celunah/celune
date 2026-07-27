# Voice-only mode

Use this mode when the user wants a reusable voice without a character identity or personality pack.

## Brief and candidates

Collect only missing details: age range, gender presentation, pitch, tone, texture, pace, accent, language, emotional range, and traits to avoid. Pick one short preview sentence and keep it identical for A/B/C. Create a VoiceDesign instruction that describes the sound, not an identifiable person.

Generate A/B/C with the exact model and cache workflow in `qwen-voice-design.md`. Candidates must share the same instruction, preview, language, snapshot, and generation settings; vary only the recorded seed or equivalent sampling seed. Keep their WAVs and JSON metadata in Celune temporary storage.

Play all candidates and ask exactly:

> Say A, B, or C to select a candidate, or state a new direction to retry.

Do not create a pack until the user selects one. A descriptive response is a revision request, not a selection; retain the original brief, add the new direction, and regenerate a bounded number of batches.

## CAC creation

After explicit selection, stage the selected WAV and invoke the repository CAC simple mode from a controlled working directory:

```text
python scripts/cac.py <bundle-name> <selected-wav> <reference-text>
```

Use the actual argument quoting and working directory required by the current `scripts/cac.py`; inspect `--help` first. CAC's simple mode creates one voice with its default voice name and stores the reference transcript. Let CAC perform its required 24 kHz/PCM normalization. Do not manually call `write_cevoice()` or assemble the archive.

Validate the output with `celune.cevoice.CEVoice.open()` and a minimal load/synthesis smoke test. Ask for confirmation before installing the validated selected pack at the current Celune `voices/` location. Never overwrite a same-name pack without confirmation.
