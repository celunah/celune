# Textual UI

The Textual UI is the primary interactive surface. It has a loading phase,
status header, log panel, caption/progress area, input editor, voice/style
button, optional VC controls, and rotating resource information. Heavy model
imports are deferred until the loading screen is mounted so the first frame is
responsive.

## Keyboard controls

| Key | Action |
| --- | --- |
| `CTRL+ENTER` | Submit the input. During the tutorial it advances/skips the current step. |
| `CTRL+J` | Submit the input in terminals that map Enter differently; it also cancels the tutorial. |
| `CTRL+Q` | Shut down immediately through the graceful teardown path. |
| `CTRL+T` | Toggle dark/light themes and persist the selection. |
| `CTRL+R` | Wake from sleep, start/stop Persona speech capture, or start/stop live VC capture depending on mode. |

The style button cycles the active voice. In VC mode the mode button toggles
talk/sing F0 conditioning and the pitch button cycles the pitch-shift value.
Touch users can use these visible buttons rather than keyboard shortcuts.

## Slash commands

Commands are entered in the input box and start with `/`.

| Command | Arguments and behavior |
| --- | --- |
| `/help` | Show commands available for the current backend, mode, and Persona capability. |
| `/restartaudio` | Recreate the audio server and play a readiness signal. |
| `/consumebuffer true\|false` | Toggle boundary consumption of live input. |
| `/invoke NAME [ARGS...]` | Invoke a registered extension. |
| `/extensions` | List loaded extensions. |
| `/voiceprompt TEXT` | Set a backend voice prompt; use `clear` to remove it. Unavailable on Mini. |
| `/speed VALUE` | Set 80–120% playback speed. Rubber Band is required. |
| `/reverb VALUE` | Set 0–100% reverb strength. |
| `/backend NAME` | Switch TTS/VC backend and wait for the reload worker. |
| `/cevoice NAME\|PATH` | Load a CEVOICE/CECHAR pack. |
| `/vc FILE` | Submit a file for voice conversion; VC mode is required. |
| `/vcmode talk\|sing` | Select ordinary speech or F0-conditioned singing. |
| `/vcpitch SEMITONES\|clear` | Set -12 through +12 semitones or reset to 0. |
| `/xvectoronly true\|false` | Toggle Qwen3 identity-only cloning. Qwen3 only. |
| `/play PATH [VOLUME]` | Play a local/remote sound effect. |
| `/attach FILE...` | Add vision attachments; `/attach clear` removes them. |
| `/say TEXT` | Send a direct Persona/vision prompt when vision is available. |
| `/seed NUMBER\|random` | Set a backend seed or restore random seeds. |
| `/tutorial` | Play the four bundled tutorial clips and demonstrate `/help`. |
| `/stop` | Stop current speech. |
| `/exit` | Exit the application. |

`/help` hides commands that the current backend cannot use, so the list is a
capability report rather than a fixed promise. Unknown commands are logged as a
warning and do not terminate the runtime.

## Themes and lighting

The built-in dark and light themes can be overridden by the active pack's
`theme` metadata. If OpenRGB is available, the `AudioRGBGlow` integration can
connect, start, sleep, wake, react to audio, and enter a fatal state. Lighting
failure does not make speech unavailable.

## Headless mode

Set `headless: true` or `CELUNE_HEADLESS=1` for service/API/extension use. The
headless UI forwards logs and lifecycle callbacks but has no editable input
box; use the REST API, Python calls, or extensions to submit work.
