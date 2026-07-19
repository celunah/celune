---
name: create-cevoice
description: Design, generate, validate, and install Celune CEVOICE packs from a voice brief. Use when the user asks to create a new synthetic voice, a voice-only pack, or a full CEVOICE/CECHAR personality pack in this repository.
---

# Create CEVOICE Pack

Use this skill from the Celune repository. It orchestrates Qwen3 VoiceDesign candidate generation and the repository's existing CAC workflow; it does not reimplement CEVOICE serialization, validation, or pack creation.

## Non-negotiable rules

- Use exactly `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` for voice generation. Never substitute another model, including an installed Celune TTS backend model.
- Use the Celune-managed Hugging Face cache obtained from `celune.paths.huggingface_hub_cache_dir()`. Never fall back to the host `~/.cache/huggingface` cache.
- Keep candidate WAV files and metadata below a temporary directory returned by `celune.paths.temp_data_dir(create=True)`, never inside the Hugging Face cache.
- Use `scripts/cac.py` as the authoritative CEVOICE/CECHAR creator. Do not manually construct a pack or duplicate CAC serialization logic.
- Do not overwrite an existing installed pack without explicit confirmation.
- Do not install a candidate before the user explicitly selects it.
- A real-person impersonation request requires permission; otherwise decline that aspect and offer a fictional or generalized voice.

## Workflow

1. Resolve the repository root and inspect the current CAC entry points before acting. Run `python scripts/cac.py --help` and read `scripts/cac.py` if the invocation or wizard fields are unclear. Preserve unrelated worktree changes.
2. Infer the mode only when unambiguous:
   - voice-only: create the voice with CAC simple mode and no personality;
   - full personality: establish identity/personality first, then use CAC's complete interactive wizard for the resulting pack.
   Ask one concise mode question when the request is genuinely ambiguous. Never silently switch modes.
3. Gather the voice brief without repeating information already supplied: age range, gender presentation, pitch, tone, texture, pace, accent, language, emotional range, and traits to avoid. In full-personality mode, gather identity/personality before finalizing the voice prompt. Use the reference guides in `references/`.
4. Choose a short preview sentence suitable for all candidates. Build one natural-language VoiceDesign instruction from the brief. Keep the original brief and instruction in the run metadata.
5. Prepare the exact model in an isolated subprocess. First use Celune's cache helpers and `local_hf_offline_mode()` to locate a complete local snapshot. Only if it is missing, download the exact model into Celune's managed cache with `huggingface_hub.snapshot_download(..., cache_dir=...)`. Set `HF_HOME`, `HF_HUB_CACHE`, and related variables only for that subprocess, restore any changed environment variables, and record the resolved snapshot/revision. See `references/qwen-voice-design.md`.
6. Verify the installed Qwen package exposes a VoiceDesign API before loading the model. The supported API currently exposes `Qwen3TTSModel.from_pretrained(...).generate_voice_design(text=..., instruct=..., language=...)`; the faster wrapper may expose the same. Inspect signatures at runtime and stop with a clear error if the installed package cannot load the exact VoiceDesign model. Do not substitute clone/custom-voice generation.
7. Generate three comparable candidates, A/B/C, with the same preview text, language, instruction, model snapshot, and generation settings. Vary only recorded random seeds or equivalent sampling seeds. Save playable WAVs and JSON metadata containing model ID, snapshot/revision, prompt, seed, preview text, language, settings, and output path.
8. Present or play all three candidates, then ask exactly: **"Say A, B, or C to select a candidate, or state a new direction to retry."** Accept A/B/C and clear equivalents. Ask again for ambiguous choices. Treat descriptive feedback as a revision request: preserve the original brief, append the new direction, and generate another A/B/C batch. Allow at most three revision batches by default and ask before exceeding that limit.
9. On cancellation, stop without creating or installing a pack. On generation failure, preserve the temporary metadata/logs needed to diagnose it and report the failed stage. After successful completion or an intentional cancel, clean up candidate files when doing so will not destroy diagnostics.
10. Create the pack only from the selected candidate. Let CAC normalize/resample the WAV when its simple mode requires it; retain the original candidate until CAC succeeds. For voice-only, invoke CAC simple mode with the selected WAV and its exact preview transcript. For full personality, pass the approved identity, description, voices, reference text, and persona fields through CAC's complete wizard; do not bypass its prompts. See `references/voice-only.md` and `references/full-personality.md`.
11. Validate before installation: open the generated pack with `celune.cevoice.CEVoice`, verify its digest/assets, voice order, required reference transcript, and expected CECHAR metadata where applicable. Run a minimal load/synthesis smoke test using the existing Celune path resolution. If repository Python files were modified, run the repository's canonical formatting and CI commands from `AGENTS.md`.
12. Show the exact staged pack path and validation result, then request confirmation before placing it at the existing Celune install location (`voices/<bundle>.cevoice`, unless the repository's current CAC/install path says otherwise). Move/copy only the validated selected pack. Re-check the installed path with `CEVoice.open()` and report success only after that check passes.

## CAC boundary

CAC currently writes the pack at the path implied by its working directory and command arguments. Stage output in a Celune temporary/staging directory, then use the repository's existing `voices/` convention or any newer CAC installer entry point discovered during inspection. Never reproduce `write_cevoice()` in this skill. If CAC's current behavior does not support a requested field or install operation, stop and explain the limitation instead of modifying CAC or silently creating a different format.

## Dry-run behavior

For a dry run, resolve the mode, collect missing brief fields, show the exact model/cache/CAC plan, and list the expected candidate metadata and install path. Do not download the model, generate audio, invoke the CAC writer, or install anything. Exercise these cases mentally or in a no-write plan:

- `Create a soft, mature female voice. Voice-only.`
- `Create a full personality called Nyra. She is quiet, direct, and observant.`
- `Make candidate B deeper and less breathy.`

Load a reference only when the concrete request requires it; do not load unrelated project files or download unrelated models.
