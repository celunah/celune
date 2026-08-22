# First run

This page walks new users through the first safe launch, backend and voice
selection, and the checks that confirm Celune is ready to speak.

## Start with a safe baseline

1. Install the core and optional dependencies with `uv sync --all-extras --dev`.
2. Run `uv run python main.py doctor` and resolve failed prerequisites.
3. Start Celune once. It creates the user configuration by copying
   `default_config.yaml` into the platform-specific application-data directory.
4. Leave `backend: null`, `voice_bundle: default`, `mode: converse`, and
   `vram: medium` until the basic engine is working.
5. Use `/help` in the Textual UI to see commands supported by the active
   backend and mode.

The default configuration enables the REST API on loopback port 2060 and keeps
isolated backend environments enabled. A first launch can therefore spend time
creating a backend environment and downloading models before the ready state.

## Select a backend

Set `backend` in the active config or pass `CELUNE_BACKEND` for a one-process
override. The backend name is one of `mini`, `qwen3`, `dotstts`, `voxcpm2`,
`gpt-sovits`, or a voice-conversion backend such as `seed-vc`. See
[Backends and environments](../development/backends.md) for capability and
dependency details.

The backend controls language support, reference-audio requirements, model
size, chunk rate, and whether the active voice pack can be used. An unknown
voice or a missing compatible pack is reported as a load error rather than
silently falling back to a different identity.

## Select a voice pack

Celune resolves `voice_bundle` as follows:

- `default` selects `voices/default.cevoice` from the repository or package.
- A bare name such as `my_pack` selects `voices/my_pack.cevoice` in the
  user-local voice directory.
- An explicit relative or absolute path is used as written.

Use `/cevoice <name|path>` or the Python `set_cevoice_and_wait()` call to load a
different pack. Then use `/voiceprompt`, `/backend`, and `/help` to inspect the
active capabilities.

## Verify speech

In speak mode, type text and press `CTRL+ENTER`. In converse mode, the same
action sends the text through Persona; use `CTRL+R` for speech input when
Persona talkback is enabled. Generated files are placed in the repository
`outputs/` directory when saving is enabled. Celune writes 48 kHz, 24-bit FLAC
outputs; the runtime keeps audio arrays as normalized float32.

For a headless first check, set `headless: true` and use the REST API or an
extension. The headless UI intentionally does not provide an interactive
terminal editor.

## If startup fails

Run:

```bash
uv run python main.py doctor
```

Then inspect the application log and traceback path reported by the runtime.
Do not delete a backend environment while it is running. If an environment is
corrupt, use `celune doctor --fix` where the doctor offers a repair, or remove
only the named backend environment after Celune has fully exited.
