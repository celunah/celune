# TUI

The Textual UI is the primary interactive surface. It has a loading phase,
status header, log panel, caption/progress area, input editor, voice/style
button, optional VC controls, and rotating resource information. The
application registers its default dark/light palette and mounts the loading screen before
importing or initializing the engine, Persona, agent, audio, backend, or model
runtime, so the first frame is both themed and responsive. Pack-derived colors
are applied later when the runtime is available. Those dependencies load in the
post-frame worker and report failures on the loading overlay.

## Keyboard controls

| Key | Action |
| --- | --- |
| `^ENTER` | Submit the input. During the tutorial it advances/skips the current step. |
| `^J` | Submit the input in terminals that map Enter differently. It also cancels the tutorial. |
| `^Q` | Fade out the UI, then shut down through the graceful teardown path. |
| `^T` | Toggle dark/light themes and persist the selection. |
| `^R` | Wake from sleep, start/stop Persona speech capture, or start/stop live VC capture depending on mode. |
| `^B` | Open or close the debug inspector when `log_level` is `debug`. |
| `ESC` during startup | Reveal the main interface when the loading footer offers this action. |

Pressing `^C` displays a temporary toast at the bottom of the interface:
`You pressed ^C. Press ^Q to exit.`

Core code can request a transient message with
`Celune.toast(msg: str, severity: str = "info", *, duration: Optional[float] = None) -> None`.
The TUI maps `info` to lavender, `warning` to yellow, and `error` to red. The
message appears on one line, centered near the bottom, then fades after the
duration. Omitting it uses three seconds. Each call replaces the visible toast.
The method returns immediately. For example:

```python
celune.toast("Settings saved", duration=2.0)
celune.toast("Check the selected path", severity="warning")
celune.toast("The operation failed", severity="error")
```

UI integrations can use
`CeluneUI.show_toast(msg, duration=None, *, severity="info")` directly. It
schedules work on Textual's event loop, so callbacks from worker threads do not
block input.

## Debug inspector

Set `log_level: debug` in `config.yaml`, then press `^B` to open the inspector.
Press `^B` or `ESC` to close it. The inspector shows runtime state and
component details, with sensitive values such as API keys and passwords
redacted. Its Live Calls pane can filter, pause, resume, and clear recent calls.

Invoke Mode can run expressions against imported functions and the active
engine. Invoked code retains normal Python permissions and may have side
effects, so use it only with trusted expressions. Tracing may be unavailable for
native code or isolated backend workers, but component snapshots remain
available. See the [UI components reference](../development/ui-components.md)
for implementation details.

## Log panel

The main log panel retains the current session's log history while the loading
screen transitions to the ready state and while the UI repaints. Background
runtime messages are appended on the Textual application thread, so a backend
worker cannot directly corrupt the panel. Each background notification
reconciles retained history with the rendered entries, so messages are not lost
if a repaint or loading transition overlaps delivery. Switching themes repaints
existing entries with the selected severity colors. The same entries are also
appended to Celune's persisted `celune.log` file for troubleshooting.
Captured stdout and stderr lines are classified from explicit severity tags,
warning or exception markers, and deprecation notices before falling back to
the stream's default severity.

## Startup failures

If initialization fails before the core is attached, the loading overlay remains
visible as an error report. Its heading reads `Early initialization failed`, the
initialization error remains in the diagnostic area, the spinner is removed, and
the overlay says `Celune can't continue.` A compact instruction appears directly
below it for missing dependencies: use the project `.venv` with `uv run`, run
`python configure.py` when the environment is missing or incomplete, or start
the user-facing launcher when a compiled entrypoint was opened directly. The
lower-left status identifies the cause, such as `Missing dependency: dateutil`.
No further startup progress is pending. The UI switches to the red error theme
and catches early startup exceptions, including missing dependencies, without
leaving a worker traceback on the terminal. The terminal title changes to an
error state with a concise cause.
Press `^Q` on this screen to close the application. A missing-dependency
failure then returns exit code `4` to the launcher, while other early startup
failures return the general failure code. After core attachment, press `ESC` to
reveal the main UI and inspect its log if a later initialization step fails.
The GPU preflight runs after core attachment and before backend model loading.
Its GPU check diagnostics appear in the main log and in the startup error
report.
Fatal Textual callback failures also set a nonzero UI return code, which the
entrypoint passes to the outer launcher so its failure diagnostics remain
visible instead of returning to the shell over the traceback.

## Slash commands

Commands are entered in the input box and start with `/`.
Fatal and stopped states still allow text editors to be used for composing a
draft, but reject submitted text and commands.

Built-in commands register their handler and localized help entry in the UI
command registry. The dispatcher looks up the command by name, and `/help`
builds an alphabetized list from the same registry, showing only commands
available for the active backend, mode, and Persona capabilities. Each handler
continues to validate its own arguments and report command-specific errors.

| Command | Arguments and behavior |
| --- | --- |
| `/help` | Show commands available for the current backend, mode, and Persona capability. |
| `/savings` | Compare session and all-time generated speech with equivalent per-minute API pricing. |
| `/restartaudio` | Recreate the audio server, test each detected speaker channel, then play a readiness signal. |
| <code>/consumebuffer true&#124;false</code> | Toggle boundary consumption of live input. |
| `/invoke NAME [ARGS...]` | Invoke a registered extension. |
| `/extensions` | List loaded extensions. |
| `/voiceprompt TEXT` | Set a backend voice prompt. Use `clear` to remove it. Unavailable on Mini. |
| `/speed VALUE` | Set 80-120% playback speed. Rubber Band is required. |
| `/reverb VALUE` | Set 0-100% reverb strength. |
| `/backend NAME` | Switch TTS/VC backend and wait for the reload worker. |
| <code>/cevoice NAME&#124;PATH</code> | Load a CEVOICE/CECHAR pack. |
| `/settings` | Open the configuration manager in normal operation or after a recoverable configuration failure. During configuration recovery it is the only command available. ENTER saves nested YAML values and requests a silent launcher restart. ESC discards changes. |
| `/vc FILE` | Submit a file for voice conversion. VC mode is required. |
| <code>/vcmode talk&#124;sing</code> | Select ordinary speech or F0-conditioned singing. |
| <code>/vcdebug on&#124;off</code> | Save raw microphone and converted VC audio to temporary AppData files. |
| <code>/xvectoronly true&#124;false</code> | Toggle Qwen3 identity-only cloning. Qwen3 only. |
| `/play PATH [VOLUME]` | Play a local/remote sound effect. Remote playback reports its playing status as soon as the audio is ready to queue. |
| `/attach FILE...` | Add vision attachments. Use `/attach clear` to remove them. |
| `/say TEXT` | Send a direct Persona/vision prompt when vision is available. |
| <code>/seed NUMBER&#124;random</code> | Set a backend seed or restore random seeds. |
| `/tutorial` | Speak short localized tutorial passages in the active voice and demonstrate `/help`. |
| `/stop` | Stop current speech. |
| `/vram` | Report allocated, reserved, and peak memory for each active component. |
| `/exit` | Exit the application. |

`/help` hides commands that the current backend cannot use, so the list is a
capability report rather than a fixed promise. Unknown commands are logged as a
warning and do not terminate the runtime.

`/vram` reports each currently loaded component as
`allocated/reserved/peak (device)`. It includes TTS, voice conversion, Persona,
speech input, the normalizer, and the agent whether their tensors are on CUDA or
the CPU. Persona recording and playback captions share one Whisper model, so
speech input appears at most once in the report.
When all three values are identical, the reserved and peak positions are shown
as `-` because those per-component allocator statistics are not applicable.
For CUDA components, allocated memory is the component's resident tensor
storage. Reserved and peak memory are the allocator totals from the owning
process because PyTorch does not attribute those values to individual model
objects. For CPU components, all three values are the component's resident
tensor storage because the CPU allocator has no corresponding per-component
reserved or peak counters. Worker-backed TTS and VC backends report from their
own CEDTS processes.

`/savings` reports estimated session and all-time API costs for each model in
the configured comparison list. The base price is shown per minute of speech.
These values use public API pricing for comparison, can differ from actual
usage costs, and are not fees charged by Celune.

## Themes and lighting

The built-in dark and light themes can be overridden by the active pack's
`theme` metadata. If OpenRGB is available, the `AudioRGBGlow` integration can
connect, start, sleep, wake, react to audio, and enter a fatal state. Lighting
failure does not make speech unavailable.

The glow controller also selects calendar effects using the computer's local
date. These effects scale the existing lunar-adjusted idle brightness:

- June 2 keeps the glow at `#cebaff` and slowly cycles brightness between 50%
  and 100% of its base level over 12 seconds.
- October 31 uses `#9d1017`, stays at 50% of base brightness, and pulses to
  100% every 4.12 seconds before fading back over one second.
- December 24-26 cycles through the pastel blue `#9fcfff`, pink `#ff9fcc`, and
  green `#b0f5b4` colors. Each fades in over 0.5-2 seconds and stays for 1-3
  seconds. Brightness is unchanged.

When speech starts, the controller resets brightness to the active effect's
baseline and pauses its idle brightness animation while audio-reactive speech
brightness runs. Idle modulation resumes after playback finishes.

## Headless mode

Set `headless: true` or `CELUNE_HEADLESS=1` for service/API/extension use. Set
`headless: null` to auto-detect the mode: Windows uses the full Textual UI.
Linux uses Textual when a desktop session is detected, and uses headless mode
for TTY-only or non-interactive sessions. If detection fails, the application
attempts the full Textual UI and retains its existing unsupported-terminal handling.
The headless UI forwards logs and lifecycle callbacks but has no editable input
box. Use the REST API, Python calls, or extensions to submit work. Press Ctrl+C
to request an orderly shutdown. Celune stops isolated backend workers before
exiting, and the launcher treats a clean shutdown as success.
