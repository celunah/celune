# Extensions

Extensions are small Python modules loaded from the repository's `extensions/`
directory. They run inside the Celune process and use the same core engine;
they are not isolated backend workers. Treat third-party extensions as trusted
code.

## Minimal extension

```python
from celune import CeluneExtension


class Greeting(CeluneExtension):
    EXTENSION_NAME = "greeting"

    def invoke(self, *args: str) -> None:
        self.say("Hello from an extension.")
```

The base class exposes `name`, state, and these helpers:

| Helper | Purpose |
| --- | --- |
| `log(message, severity)` | Write to Celune's log stream. |
| `say(text, save=True, display_text=None)` | Queue speech through the normal pipeline. |
| `think(text)` | Submit a Persona turn. |
| `play(path, keep=False, volume=1.0)` | Queue a sound effect. |
| `status(message, severity)` | Update UI/runtime status. |
| `set_voice(name)` | Change the active voice. |
| `with_backend(name)` | Temporarily use a backend in a context manager. |
| `with_cevoice(bundle)` | Temporarily use a character/voice pack. |

## Context and registration

`CeluneContext` is the narrow context object passed to extension integrations.
It provides `log`, `say`, `think`, `play`, `status`, `set_voice`,
`get_state`, `wait_until_ready`, optional backend/CEVOICE overrides, the
extension name/version, a shared mapping, and the active log level. Its
`expose(name, value)` and `get(name, default)` methods are intended for
cooperating extensions without reaching into Celune internals.

The root package exports `subscribe`:

```python
from celune import subscribe


@subscribe("generation_end")
def after_generation(event) -> None:
    print(event.saved_path)
```

The manager supports `register`, `unregister`, `invoke`, `list_extensions`,
and automatic loading from `extensions`. `/invoke NAME [ARGS...]` calls an
extension by name and `/extensions` lists what loaded successfully.

## Events

The dispatcher accepts these event names:

`ready`, `shutdown`, `fatal`, `error`, `voice_changed`, `state_changed`,
`generation_start`, `generation_end`, `generation_error`, `audio_start`,
`audio_end`, `character_changed`, `character_loaded`,
`character_unloaded`, `agent_task_state_changed`,
`agent_approval_requested`, `agent_choice_requested`, and
`agent_task_finished`.

Payloads are typed dataclasses. The important fields are:

- `ReadyEvent` and `ShutdownEvent`: the engine reference.
- `FatalEvent` and `ErrorEvent`: engine, exception, and source.
- `VoiceChangedEvent`: old and new voice.
- `StateChangedEvent`: old and new runtime state.
- `GenerationStartEvent`: synthesis text, display text, save flag, and language.
- `GenerationEndEvent`: generation fields plus the saved path, when any.
- `GenerationErrorEvent`: generation fields plus the exception.
- `AudioStartEvent` and `AudioEndEvent`: source ID, label, kind, and saved path.
- Character events: character name, bundle path, default flag, and old/new
  character fields where applicable.
- Agent events: task ID/state and approval/choice/task-finished payloads.

An exception in one callback is logged and does not stop dispatch to the other
callbacks. Always unsubscribe callbacks owned by a short-lived integration.
