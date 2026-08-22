# Troubleshooting

## Startup or import failures

Run `celune doctor` first. Confirm Python is 3.12–3.14, the active `uv`
environment is the expected one, and the CUDA/PyTorch architecture is supported.
If a heavy library fails before the loading UI, check that the source-tree
entrypoint is being run through the supported environment and that a stale
compiled artifact is not shadowing the checkout.

## Backend environment failures

Check the backend name and manifest. Isolated environments are backend-specific;
the core environment being healthy does not prove a TTS worker is installed.
Read the environment's manifest and worker logs, then retry the backend-specific
setup after Celune exits. Do not mix the core CUDA 13.0 lock with the worker
CUDA 12.8 compatibility stack by manually installing packages into both.

## No audio or a stale device

Use `/restartaudio`, verify `audio_api`, `input_device`, and `output_device`,
and check PortAudio permissions. On Windows, disconnect/reconnect invalidated
devices before restarting the stream. A backend readiness signal distinguishes
an audio-server problem from a model-load problem.

## Distorted audio

Verify that every boundary declares sample rate, channels, and dtype. Celune
expects normalized float32 internally. Do not feed signed int16 values directly
to a playback helper. For VC, remember that Seed-VC's F0 and non-F0 paths have
different intermediate sample rates and are resampled before playback.

## Voice pack rejected

Use `CEVoice.open()` or the `celune.cevoice` loader to validate the archive.
Check safe asset names, supported extensions, UTF-8 metadata, WAV format,
embedding shape, checksum, `default_voice`, and `voice_order`. A valid archive
may still be incompatible with a backend if it lacks the reference kind or
transcript that backend requires.

## Persona does not talk back

Confirm `persona.enabled` and `persona.talkback`, use a vision/text-capable
model, and check that the active operation mode is `converse` or `agent`.
`CTRL+R` first wakes Celune when sleeping; press again after the engine is ready
to capture speech. Whisper has a five-second no-input timeout and a configured
speech-end delay, so silence does not create an infinite recording.

## Agent is waiting

`awaiting_approval` and `awaiting_choice` are intentional task states. Review
the typed operation and danger level in the UI, then approve, deny, answer, or
cancel. If local management is unexpectedly available, inspect
`agent_local_management` and `CELUNE_LOCAL_MANAGEMENT`; keep them disabled in
normal deployments.

## CI or build issues

Use `python scripts/run_ci.py` for the canonical gate. Native DLL collection,
GPU drivers, PortAudio, and cache permissions are environment failures, not
proof that the code assertions passed. For documentation-only work, run the
strict MkDocs build and report Python CI as not required rather than implying it
was executed.
