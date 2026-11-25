# UI components

This page documents the Textual debug inspector and selection-menu components for contributors extending Celune's terminal interface.

## Debug inspector

The inspector opens with `^B` when `log_level` is `debug`. It presents runtime state, component details, callback information, and a live-call trace. Closing it stops tracing; reopening it resumes tracing while retaining the current buffer. Sensitive values such as API keys, tokens, passwords, and credentials are redacted from component details.

Live Calls uses Python's `sys.monitoring` API to observe Python functions and exposed C-callable boundaries in the current process. It retains up to 100 calls and displays the latest 25. Source filters narrow monitoring to a module and its submodules. The trace cannot inspect native-library internals or isolated backend worker processes. Monitoring requires Python 3.12 or later and a free monitoring tool slot. If monitoring is unavailable, component snapshots remain usable.

Invoke Mode parses a restricted expression tree rather than using `eval`. It supports imported dotted names, `engine`, built-ins, Python literals, containers, and nested calls. Invoked code runs with its normal Python permissions and may have side effects. Calls run on a worker thread and continue if Invoke Mode is turned off.

The inspector is owned by `celune/ui/debug.py`, launched from `celune/ui/interaction.py`, and uses the invocation parser in `celune/ui/invoke.py`. Keep tracing scoped to the inspector lifecycle and preserve redaction when adding displayed state.

## Selection menus

`celune.ui.terminal.SelectMenuWidget` and `SelectMenuOption` provide modal, value-aware menus for Celune's Textual screens. Each option has a label and return value, with optional editability, keybind, autocomplete candidates, display value, and confirmation value. For example:

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
        SelectMenuOption("Information", editable=False),
    ],
    footer="Up/Down select   Enter confirm   Esc back",
)
```

The widget posts `Confirmed` with `option_index`, the selected option, and its value. `Cancelled` includes the selected option and no value. It does not change screens. Non-editable rows ignore value controls. Autocomplete candidates retain their JSON-compatible types; rows without candidates accept free-form strings. The widget keeps the full value available when its display is truncated and keeps navigation hints visible when the menu exceeds the viewport.

The inspector and invocation parser are covered by `tests/ui/test_debug.py` and `tests/ui/test_invoke.py`. Selection menus are covered by `tests/ui/test_terminal.py`. Preserve typed return values and existing message behavior when changing presentation or input handling.

## Verification and troubleshooting

Run the relevant UI tests when changing these components. If inspector tracing is unavailable, check the Python version and whether another feature has occupied the `sys.monitoring` tool slot. A missing trace does not prevent component-state inspection.

The canonical repository validation command is `python scripts/run_ci.py`.

## Compatibility

The inspector's function tracing depends on Python 3.12 or later. Native libraries and isolated worker processes remain outside its tracing boundary. Selection-menu event payloads and option values are consumed by host screens, so changes to those contracts require updating their callers and tests.

## See also

- [TUI guide](../interfaces/tui.md) for inspector access and user-facing behavior.
- [Architecture](architecture.md) for runtime and UI ownership boundaries.
- [Testing](testing.md) for test selection and repository validation.
