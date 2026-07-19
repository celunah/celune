# Full personality mode

Use this mode when the user asks for a named character, identity, persona, or a combined CEVOICE/CECHAR pack.

## Establish the character first

Confirm the character name and gather the identity/personality before locking the voice: description, speaking style, temperament, boundaries, language, emotional range, and traits to avoid. Do not ask again for facts already supplied. Keep the character clearly fictional or generalized when the request resembles an identifiable real person without permission.

Then gather the voice brief and generate A/B/C using the exact model and candidate protocol in `qwen-voice-design.md`. Candidate feedback may revise the voice direction, but must not erase the approved personality. Require explicit A/B/C selection before opening CAC.

## CAC wizard boundary

Use CAC's complete interactive wizard for the selected voice and character. Inspect the current `collect_character_data()` prompt order before supplying answers; it is the source of truth. Populate the required name, description, voice count, voice name, selected WAV, reference transcript, per-voice persona, default voice, voice order, theme, and shared persona fields from the approved brief. Use a controlled stdin sequence only when every answer is known and validated; otherwise run the wizard interactively.

Do not create a CECHAR/CEVOICE archive yourself and do not add fields that CAC does not support. If the wizard cannot represent a requested personality detail, report that limitation and preserve the user-approved data in the run metadata rather than silently dropping it.

## Validate and install

Stage the wizard output, open it with `celune.cevoice.CEVoice.open()`, verify all selected assets/digests and CECHAR metadata, and perform a minimal load/synthesis smoke test. Ask for confirmation before installing to the repository's current `voices/` location. Reopen the installed file after placement and report its exact path. Preserve diagnostics on failure and clean temporary candidate material only after success or cancellation.
