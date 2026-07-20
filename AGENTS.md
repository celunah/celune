# AGENTS.md

## Project Overview

Celune is a real-time local AI TTS character engine focused on expressive voice delivery, fast buffered speech generation, and a polished user experience.

Celune supports multiple voice styles, configurable voice packs, frontend/API/extension modes, long-form narration, built-in DSP/audio controls, GPU inference, character responses, a Textual TUI, a FastAPI REST API, and a Gradio-based WebUI.

The project targets Windows and Linux, supports Python 3.12 and 3.13, and is designed for consumer GPU hardware with VRAM presets from roughly 6 GB to 16 GB+.

## Development Principles

* Keep changes focused on the requested task.
* Avoid unrelated refactors.
* Prefer simple, maintainable code over clever code.
* Avoid unnecessary dependencies.
* Do not add placeholder implementations.
* Do not add TODO comments.
* Do not silently disable features to make tests pass.
* Preserve Celune's local-first, polished, anti-slop project identity.
* Reuse existing architecture instead of creating parallel systems.

## Typing Style

Prefer classic unions like `Union[str, int]` or `Optional[str]`, rather than using PEP 604 unions like `str | int` or `str | None`.

Other typing features from e.g. PEP 585 or PEP 695 may be used normally.

Avoid using broad types like `Any`, `object` or `T`, unless the function explicitly requires a broad type.

Prefer concrete, meaningful types.

## Reuse Existing Code

Prefer reusable variables, constants, helpers, and project abstractions already present in the repository.

Do not hardcode strings, colors, ports, paths, app names, status labels, or repeated values when the repository already defines them.

Only hardcode or redefine values when importing the existing value would create a circular import, break architecture, create excessive coupling, or otherwise be impractical.

## CI and Validation

The canonical CI command is:

```bash
python scripts/run_ci.py
```

On Windows, the path may appear as:

```powershell
python scripts\run_ci.py
```

Prefixing it with `uv run` is not required, as it runs the CI commands with it already.

Always use the CI script for validation unless explicitly instructed otherwise.

Do not use:

```text
- .\.venv\Scripts\python.exe
- python -m pytest
- pytest
- uv cache overrides
- UV_CACHE_DIR
- etc.
```

If for any reason any `uv` command exits with `Access is denied.` or `Permission denied` errors, apply `--no-cache` to `uv`, and try again.

Do not modify the execution environment to work around failures.

Before CI, format the repository with `uv run ruff format .`.

Expected CI runtime is 5 minutes or less.

If CI runtime exceeds 5 minutes:

* Assume it may have stalled.
* Stop it from running any further.
* Report that the CI has taken too long.
* Extend the timeout one time to 10 minutes.
* Do not extend the timeout again if the one-time extension fails.
* Attempt to run again only the relevant CI steps directly, not using a sandbox.
* If non-sandboxed CI attempts also fail or time out, report it back.

After each task, run `scripts/update_docstrings.py` and then replace placeholders in docstrings like:

```text
Describe this function.

Args:
    value: Value for `value`.
    
Raises:
    RuntimeError: If `RuntimeError` needs to be raised.
    
Returns:
    type: Result of this function.
```

with proper documentation, while preserving the docstring format.

If this process updates typing or dataclass related docstrings, remove the placeholders instead of completing them.

This process may leave some formatting inaccuracies, run `uv run ruff format .` again after completing docstrings.

Additionally, perform all actions listed in the `Import Ordering` section below.

Make sure to remove all `__pycache__` directories. Celune code is compiled and does not use said cache files.

## Import Ordering

Celune code follows a specific import ordering strategy. Always order all imports after finishing a task, according to this example:

```text
import stdlib
from stdlib import function

import third_party
from third_party import function
from third_party import (
    many,
    functions,
)

from .local import function
from ..local2 import function
from ...local3 import function

from .local import (
    many,
    functions,
)
```

At the end of every task, always sort and verify imports in every modified Python source file. Imports must follow this order: standard-library imports, a blank line, third-party imports, a blank line, then local relative imports. Within each group, sort import statements by line length from shortest to longest. Preserve multiline import formatting, and prefer `.file` over `celune.file` for local imports.

Code reviews should state mismatches in the import ordering.

## Exceptions

All Celune related exceptions follow a Python-style format. Adhere to the below example when writing exceptions:

```text
# Do not use
Error: Error description.

# Use
Error: error description
```

Use reusable exception classes from `celune.exceptions`, if any match. General exceptions should use Python exception classes rather than Celune's own ones.

If a new Celune specific exception category needs to be created, create it in `celune.exceptions`, associating all related exceptions with it.

Document it according to the usual CI rules. If the exception type would be too broad, do not add it, using Python exceptions instead.

## Localization

Celune does not use hardcoded strings in English. Define each new string you add into Celune's localization string database.

Do not use raw string literals in the code. Always use `string("key_name", **kwargs)` in string literals to populate them from the global localization string database.

If you find any raw strings in the code, add them to the localization string database, and remove the hardcoded string.

Make sure to only modify user-facing strings (both normal and dev mode strings), don't change anything internal.

## Python and Environment

* Supported Python versions are 3.12 and 3.13.
* Use `uv` for environment management.
* Ensure the environment was set up with `--all-extras --dev` to prevent any missing packages from causing issues later on.
* Do not use `pip` directly unless explicitly required. If you need to run `pip` alone, do it so with `uv pip` instead.
* Do not assume CPU-only mode supports all features. CPU-only execution is only supported with Celune Mini.
* Be aware that many features require an RTX 30 series GPU or newer.

## Audio Format

Celune only works with normalized `np.float32` audio arrays `-1.0` to `1.0`. When dealing with audio-related code that returns other audio formats, such as signed 16-bit PCM `-32768` to `32767`, normalize it to Celune's expected audio format.

Not normalizing such audio may result in extreme audio distortions.

Keep audio related computations in `np.float32`, using `np.float64` only if precision would be insufficient to represent said audio.

Always output audio files in 24-bit 48 kHz FLAC. Do not output other formats.

## UI and WebUI

Celune has a Textual terminal UI and a Gradio WebUI mounted through FastAPI.

When modifying UI code:

* Preserve Celune's visual identity.
* The WebUI should feel like a high-resolution counterpart to the TUI.
* Avoid generic Hugging Face Space-style design.
* Do not assume Gradio examples for older versions still apply.
* FastAPI is the application server; Gradio is mounted as the WebUI.
* Keep mobile/touch support in mind.
* Do not rely only on screen width for mobile behavior. Prefer pointer/hover media queries when the issue is input method.
* Desktop keyboard shortcuts must have visible button alternatives for touch devices.
* Try to write CSS, override page variables, etc. to keep Celune's canonical page colors.

## API

Celune exposes a REST API for programmatic use.

When modifying API code:

* Preserve existing endpoint behavior where practical.
* Reuse existing request and response models.
* Keep API behavior consistent with the TUI/WebUI runtime behavior.
* Do not make the WebUI depend on raw REST calls unless explicitly requested.

## Audio and TTS

Celune includes multiple TTS backends, voice styles, configurable voice packs, long-form narration support, built-in DSP, and native audio controls.

When modifying audio code:

* Prefer existing audio abstractions.
* Avoid adding large audio/game frameworks for small playback tasks.
* Do not bypass the existing playback, buffering, stream, or DSP infrastructure without a clear reason.
* Keep long-form narration stability in mind.
* Do not add markup/control tags to generated speech unless the backend explicitly supports them.

## System Dependencies

Celune may depend on external system tools such as SoX, Rubber Band, OpenRGB, CUDA Toolkit 12.8, symbolic link support on Windows, and C/C++ build tools for some backends.

Do not remove checks, documentation, or fallback behavior for these dependencies without understanding the runtime impact.

## Documentation

Keep documentation concise, direct, and technically accurate.

When documenting licensing, distinguish between:

* Celune source code, licensed under MIT.
* Third-party models and assets, which may use their own licenses.

Do not claim third-party models are covered by Celune's MIT license.

When documenting commands, use the canonical project commands from the README.

## Testing Behavior

* Run relevant tests when practical.
* Prefer the full CI script for final validation.
* Do not silently narrow validation scope after a failure.
* If a test cannot be run, say why.
* If a command times out, report it as a timeout, not as a pass.
* Do not hide infrastructure failures behind vague wording.

## If Unsure

When unsure, prefer this order:

1. Reuse existing project code.
2. Preserve current behavior.
3. Avoid new dependencies.
4. Keep the TUI, WebUI, API, and runtime consistent.
5. Run `python scripts/run_ci.py`.
6. Report failures honestly.
