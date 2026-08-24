# TUI

The Textual UI is the primary interactive surface. It has a loading phase,
status header, log panel, caption/progress area, input editor, voice/style
button, optional VC controls, and rotating resource information. Celune
registers its default dark/light palette and mounts the loading screen before
importing or initializing the engine, Persona, agent, audio, backend, or model
runtime, so the first frame is both themed and responsive. Pack-derived colors
are applied later when the runtime is available. Those dependencies load in the
post-frame worker and report failures on the loading overlay.

## Keyboard controls

| Key | Action |
| --- | --- |
| `CTRL+ENTER` | Submit the input. During the tutorial it advances/skips the current step. |
| `CTRL+J` | Submit the input in terminals that map Enter differently; it also cancels the tutorial. |
| `CTRL+Q` | Shut down immediately through the graceful teardown path. |
| `CTRL+T` | Toggle dark/light themes and persist the selection. |
| `CTRL+R` | Wake from sleep, start/stop Persona speech capture, or start/stop live VC capture depending on mode. |

The style button cycles the active voice on a normal click. Hold it to open
`Select voice`, which lists every readable `.cevoice` and `.cechar` CEVOICE/CECHAR
pack as one character row. Left/Right cycles the voice entry when a pack has
multiple entries; single-entry packs omit the bracketed value. Confirming a row
hot-loads that pack and voice. In VC mode
the mode button toggles talk/sing F0 conditioning and the pitch button cycles
the pitch-shift value. Touch users can use these visible buttons rather than
keyboard shortcuts.

## Value-aware selection menus

`celune.ui.terminal.SelectMenuWidget` is a reusable Textual widget for compact
configuration and voice lists. Create each row with
`SelectMenuOption(label, value, editable, keybind, autocomplete_values,
display_value, show_value, confirm_value)`, then mount the widget in a screen
or app. A `footer_builder` can produce context-sensitive keybind hints for the
highlighted row:

```python
from celune.ui.terminal import SelectMenuOption, SelectMenuWidget

menu = SelectMenuWidget(
    "Configuration manager",
    [
        SelectMenuOption("Backend", "mini", keybind="b"),
        SelectMenuOption(
            "Voice",
            "calm",
            autocomplete_values=("calm", "soft", "distorted"),
        ),
        SelectMenuOption("VRAM target", "medium"),
        SelectMenuOption("Theme", "dark"),
        SelectMenuOption("Information", editable=False),
    ],
    footer="↑/↓ select   ENTER confirm   ESC back",
)
```

Labels begin at the same column as the `-> ` selection marker. Every displayed
value begins two cells after the longest label, including when labels contain
wide Unicode characters. Set `value_display="current"` to show a bracketed
value only on the highlighted editable row; the default `"all"` shows values on
every editable row. Use `display_value` when a row returns structured data but
should show a shorter human-readable value in brackets. `set_value(index,
value)` updates an editable row and raises `ValueError` for a non-editable row.
Values that exceed the available row width are rendered with an ellipsis while
their full value remains available through the confirmation message.
Set `SelectMenuOption.explanation` to render a selected-row explanation above
the footer hints. The configuration manager converts dotted YAML keys to
human-readable labels, such as `api.enabled` to `API enabled`, while retaining
names such as API, T2S, GPT-SoVITS, and Persona.
The menu is a centered overlay with the themed rounded border and sizes itself
to its content, up to the available viewport. Its surrounding layer is
transparent but modal: mouse and keyboard input outside the menu is consumed.
When the option list is taller than the overlay, the visible rows form a
moving window around the highlighted row so the footer bindings remain visible.
Application menus show only applicable navigation/value hints; hints are
separated with `・`.

Up/Down wrap through the rows. Left/Right cycle an editable row's
`autocomplete_values`; printable keys search those candidates. When a row has
declared candidates, text that matches none of them snaps back to the current
valid value. Rows without candidates still accept free-form strings, while
boolean, integer, float, and `None` values reject invalid text and retain their
previous valid type and value. Backspace removes the last search character.
Autocomplete candidates retain their original JSON-compatible return types,
including strings, numbers, booleans, and `None` when explicitly supplied.
Non-editable rows ignore value controls. An option's `keybind` or the
constructor's `keybinds={"key": index}` mapping selects a row directly;
navigation and value-editing keys cannot be used for these shortcuts.

ENTER posts `SelectMenuWidget.Confirmed` with the selected `option_index`,
`option`, and `value`. Set `return_value=False` to make `value` be `None` while
retaining the selected row in the message. A non-editable row may provide
`confirm_value` when it should still return a fixed result. ESC posts
`SelectMenuWidget.Cancelled` with the selected row and `value=None`; the host
screen can handle that message by popping or hiding the menu. The widget does
not automatically change screens.

## Log panel

The main log panel retains the current session's log history while the loading
screen transitions to the ready state and while the UI repaints. Background
runtime messages are appended on the Textual application thread, so a backend
worker cannot directly corrupt the panel. Switching themes repaints existing
entries with the selected severity colors. The same entries are also appended
to Celune's persisted `celune.log` file for troubleshooting.

## Startup failures

If initialization fails, the loading overlay remains visible as an error report.
Its state reads `Failed to start`, the initialization error remains in the
diagnostic area, the spinner is removed, and the overlay says `Celune can't
continue.` The lower-left status changes to `Celune could not start`; no further
startup progress is pending. Fatal Textual callback failures also set a nonzero
UI return code, which the entrypoint passes to the outer launcher so its failure
diagnostics remain visible instead of returning to the shell over the traceback.

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
| `/settings` | Open the configuration manager. ENTER saves nested YAML values and requests a silent launcher restart; ESC discards changes. |
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
