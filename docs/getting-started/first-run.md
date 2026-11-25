# First run

This page walks new users through the first safe launch, backend and voice
selection, and the checks that confirm Celune is ready to speak.

## Start with a safe baseline

1. Run `python configure.py` from the repository root. It installs the core
   environment, creates AppData, and copies the default configuration and voice
   pack there.
2. Run `uv run python main.py doctor` and resolve failed prerequisites.
3. Start Celune once. Existing configuration and voice data are preserved.
4. Leave `backend: null`, `voice_bundle: default`, `mode: performance`, and
   `vram: medium` until the basic engine is working.
5. Use `/help` in the Textual UI to see commands supported by the active
   backend and mode.

The default configuration enables the REST API on loopback port 2060. The
first launch may take longer while Celune prepares the selected backend and
downloads its model.

## Select a backend

When `backend` is `null`, an interactive launch asks you to choose a TTS
backend and saves your selection. Headless launches use the configured preset
default. Set `backend` in the config or use `CELUNE_BACKEND` for a one-run
override. Compare TTS options in [Speech](../user-guide/speech.md), and see the
[backend guide](../development/backends.md) for setup details.

A backend may require specific languages or reference audio. If the active voice
pack is incompatible, Celune reports a load error rather than switching to a
different voice.

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

In performance mode, type text and press `^ENTER` to speak. In converse mode,
the same action sends the text to Persona. Use `^R` for Persona speech input.
When saving is enabled, speech is written as 48 kHz, 24-bit FLAC in
`outputs/`.

For a headless first check, set `headless: true` and use the REST API or an
extension. To let Celune choose based on the environment, set `headless: null`.
This selects Textual on Windows and in detected Linux desktop sessions, and
headless mode in Linux TTY or non-interactive sessions. The headless UI
intentionally does not provide an interactive terminal editor.

## If startup fails

Run:

```bash
uv run python main.py doctor
```

If the loading screen shows `Failed to start`, read its diagnostic and check
the application log for details. Press `^Q` to close the error screen. A missing
required dependency returns exit code `4` to the launcher, which can report or
repair the environment.
Do not delete a backend environment while it is running. If an environment is
corrupt, use `celune doctor --fix` where the doctor offers a repair, or remove
only the named backend environment after Celune has fully exited.

### Recover a corrupted Windows runtime

If the Windows launcher reports `Celune is corrupted.`, it received Windows
error `1392` (`ERROR_FILE_CORRUPT`) while starting the compiled runtime.
Rebuild the binaries from a source checkout using the [CMake release
build](../development/packaging.md#cmake-release-builds), or download and
extract Celune's latest release again. Then launch Celune from the repaired
build or release folder.

## See also

- [Installation](installation.md)
- [CMake release builds](../development/packaging.md#cmake-release-builds)
