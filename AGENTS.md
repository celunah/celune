# Celune Project Philosophy

## Project Overview

Celune is a real-time local AI TTS character engine focused on expressive voice delivery,
fast buffered speech generation, and a polished user experience.

Celune supports multiple voice styles, configurable voice packs, frontend/API/extension modes, long-form narration,
built-in DSP/audio controls, GPU inference, character responses, a Textual TUI, a FastAPI REST API,
and a Gradio-based WebUI.

The project targets Windows and Linux, supports Python 3.12 to 3.14, and is designed for consumer GPU hardware with
VRAM presets from 6 GB to 16 GB, and beyond.

## Development Principles

* Keep changes focused on the requested task.
* Avoid unrelated refactors.
* Do not re-export symbols during refactors.
* Prefer simple, maintainable code over clever code.
* Avoid unnecessary dependencies.
* Do not add placeholder implementations.
* Do not add TODO comments.
* Prefer concise one-word module filenames for clear responsibilities.
* When related modules form a cohesive boundary, create or reuse a focused sub-package instead of accumulating
  long top-level filenames.
* Do not silently disable features to make tests pass.
* Preserve Celune's local-first, polished, anti-slop project identity.
* Reuse existing architecture instead of creating parallel systems.

## Typing Style

Prefer classic unions like `Union[str, int]` or `Optional[str]`, rather than using PEP 604 unions like
`str | int` or `str | None`.

Other typing features from e.g. PEP 585 or PEP 695 may be used normally.

Avoid using broad types like `Any`, `object` or `T`, unless the function explicitly
requires, uses or accepts broad types.

Examples of such functions are `celune.utils.discard()` and `celune.utils.available()`.

Prefer concrete, meaningful types.

## Reuse Existing Code

Prefer reusable variables, constants, helpers, and project abstractions already present in the repository.

Do not hardcode strings, colors, ports, paths, app names, status labels, or repeated values when the repository
already defines them.

Only hardcode or redefine values when importing the existing value would create a circular import, break architecture,
create excessive coupling, or otherwise be impractical.

## Source File Size

Every Python source file, including tests and scripts, must remain at ≤100 KB (less than or equal to 102,400 bytes).

If a file exceeds this limit, split it into focused subpackages or modules, or move cohesive responsibilities into
existing smaller matching modules.

Every resulting file must remain within the limit. Do not work around the limit by excluding the affected files
from validation.

## Dependency Management

When writing new code, ensure that non-backend files do not import any backend specific packages, e.g.
`faster_qwen3_tts`, and only use Celune core packages.

Refer to `pyproject.toml` to check what packages the Celune core actually requires.


## CI and Validation

Note: Do not run CI if the current diff made no logical changes to any Python files within Celune. Proceed only
if you made changes that would actually require running validations.

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

If for any reason any `uv` command exits with `Access is denied.` or `Permission denied` errors,
apply `--no-cache` to `uv`, and try again.

Do not modify the execution environment to work around failures.

To satisfy CI, prefer solutions that work on both Linux and Windows. When platform-specific code cannot be avoided,
guard it at runtime and make the guard static-analysis-safe.

Do not directly reference APIs that are absent from another supported platform's type surface.

Use guarded attribute lookup with an appropriate cast, or an equivalent portable abstraction,
so CI can analyze the module on every supported platform.

Skip all platform-specific tests that do not match the current platform.

Before CI, format the repository with `uv run ruff format .`.

Expected CI runtime is approximately 10 minutes.

Prefer running the faster `poe test` during CI, unless there are not enough resources on the development host,
then fall back to the simplified `poe test_basic` for reduced resource usage.

If CI runtime exceeds 10 minutes:

* Assume it may have stalled.
* Stop it from running any further.
* Check the potential causes of the CI stall.
* After resolving the potential causes, run only the tests that did not complete
* If not in "Full access" / "YOLO mode", if permission error was reported, state the sandbox may have interfered
  with the CI. Attempt to run the CI again outside the sandbox. If this is not possible,
  state that CI cannot be completed without full access.

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

Immediately document every new or changed behavior you write. This includes public calls, configuration keys,
CLI or slash commands, API endpoints, events, backend capabilities, file formats, standards, and user-visible workflows.
Update the appropriate `docs/` page and `mkdocs.yml` navigation in the same task before
considering the implementation complete. Do not defer documentation to a later pass.

Always perform all actions listed in the `Import Ordering` section at the end of a given task.

Make sure to remove all `__pycache__` directories. Celune code is compiled and does not use said cache files.

## Testing Discipline

Write only the minimal set of assertions needed to verify intended behavior.
Keep assertions focused on public behavior and essential state transitions.

Never add assertions for removed, deprecated, or superseded behavior merely
because it existed previously. Update or remove obsolete assertions instead.

Avoid repetitive defensive `assert not` checks and duplicate coverage.

## Import Ordering

Celune code follows a specific import ordering strategy. Always order all imports after finishing a task,
according to this example:

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

At the end of every task, always sort and verify imports in every modified Python source file.
Imports must follow this order: standard-library imports, a blank line, third-party imports, a blank line,
then local relative imports. Within each group, sort import statements by line length from shortest to longest.
Preserve multiline import formatting, and prefer `.file` over `celune.file` for local imports.

Maintain compatibility with Pylint's `C0412 (ungrouped-imports)` inspection.
Do not add ignore statements when sorting imports.

Code reviews should state mismatches in the import ordering.

## Exceptions

All Celune related exceptions follow a Python-style format. Adhere to the below example when writing exceptions:

```text
# Do not use
Error: Error description.

# Use
Error: error description
```

Use reusable exception classes from `celune.exceptions`, if any match. General exceptions should use Python exception
classes rather than Celune's own ones.

If a new Celune specific exception category needs to be created, create it in `celune.exceptions`,
associating all related exceptions with it.

Document it according to the usual CI rules. If the exception type would be too broad,
do not add it, using Python exceptions instead.

## Localization

Celune does not use hardcoded strings in English. Define each new string you add into
Celune's localization string database.

Do not use raw string literals in the code. Always use `string("key_name", **kwargs)` in string literals to populate
them from the global localization string database.

If you find any raw strings in the code, add them to the localization string database, and remove the hardcoded string.

Make sure to only modify user-facing strings (both normal and dev mode strings), don't change anything internal.

Localization database rules:

* Every user-visible string in source, scripts, CLI/TUI/WebUI, API responses,
  and documentation examples must resolve through the localization database;
  do not leave user-visible English literals inline.
* Prefer localization keys no longer than 50 characters and require every key
  to be 50 characters or fewer.
* Prefer localized values no longer than 100 characters and require every
  value to be 100 characters or fewer, counting formatted placeholders.
* Split long notices, instructions, diagnostics, and other multi-sentence
  text into composable localized strings instead of storing one long value.
* Do not store exception text, tracebacks, technical implementation details,
  raw backend errors, paths, protocol data, or other non-user-facing
  diagnostics in the localization database. Keep those values in code and
  localize only the surrounding user-facing message.
* Use stable semantic key names, remove obsolete keys, and keep every key
  referenced by code or documentation present in the database.

## Python and Environment

* Supported Python versions are 3.12, 3.13 and 3.14.
* Use `uv` for environment management.
* Run `python configure.py` for setup. It uses `uv sync --dev --all-extras` on Linux
  and `uv sync --dev --extra api` on Windows; never request `--all-extras` on Windows
  because OpenZL does not compile there.
* Do not use `pip` directly unless explicitly required. If you need to run `pip` alone, do it so with `uv pip` instead.
* Do not assume CPU-only mode supports all features. CPU-only execution is only supported with Pocket TTS and LuxTTS.
* Be aware that many features require an RTX 30 series GPU or newer.

## Audio Format

Celune only works with normalized `np.float32` audio arrays `-1.0` to `1.0`. When dealing with audio-related code
that returns other audio formats, such as signed 16-bit PCM `-32768` to `32767`,
normalize it to Celune's expected audio format.

Not normalizing such audio may result in extreme audio distortions.

Keep audio related computations in `np.float32`, using `np.float64` only if precision would be insufficient to
represent said audio.

Always output audio files in 24-bit 48 kHz FLAC. Do not output other formats.

## UI and WebUI

Celune must register her default theme and show the loading screen immediately before importing or initializing any
part of the core or its features, backends, or other heavy runtime dependencies. Only import safe packages at this
stage, and defer further imports until Celune has been fully initialized, then proceed with applying all the runtime
data, such as themes, etc.

Celune has a Textual terminal UI and a Gradio WebUI mounted through FastAPI.

When modifying UI code:

* Preserve Celune's visual identity.
* The WebUI should feel like a high-resolution counterpart to the TUI.
* Avoid generic Hugging Face Space-style design.
* Do not assume Gradio examples for older versions still apply.
* FastAPI is the application server; Gradio is mounted as the WebUI.
* Keep mobile/touch support in mind.
* Do not rely only on screen width for mobile behavior. Prefer pointer/hover media queries when the issue
  is the input method.
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

Celune includes multiple TTS backends, voice styles, configurable voice packs, long-form narration support,
built-in DSP, and native audio controls.

When modifying audio code:

* Prefer existing audio abstractions.
* Avoid adding large audio/game frameworks for small playback tasks.
* Do not bypass the existing playback, buffering, stream, or DSP infrastructure without a clear reason.
* Keep long-form narration stability in mind.
* Do not add markup/control tags to generated speech unless the backend explicitly supports them.

## System Dependencies

Celune may depend on external system tools such as SoX, Rubber Band, OpenRGB, CUDA Toolkit 12.8, symbolic link support
on Windows, and C/C++ build tools for some backends.

Do not remove checks, documentation, or fallback behavior for these dependencies without understanding
the runtime impact.

## Documentation

Keep documentation concise, direct, and technically accurate. Technical  documentation belongs under `docs/`
and must follow the repository's [documentation standard](docs/development/documentation.md).

Every technical page must use this structure:

1. A heading names the subject
2. A short purpose paragraph immediately below the title that states the
   audience and scope.
3. Subheadings organized around the reader's task, or the documented contract.
4. A verification, error-handling, compatibility, or troubleshooting section
   when the subject can fail or vary by environment.
5. A `See also` section when related pages provide the next useful step.

Use sentence case for headings. Preserve acronyms, API names, commands, and file format names exactly.
Use fenced code blocks with a language identifier,  tables for stable field/option comparisons,
numbered lists for procedures, and  bullets for unordered facts. Use the canonical project commands and copyable
examples from the README and source. Put signatures, arguments, return values, errors, side effects, and at least one
usage example beside every documented public call.

For format and protocol pages, state the version, invariants, wire/file layout, compatibility rules,
and failure behavior. For user procedures, state prerequisites, ordered steps, expected results, and recovery steps. For
development pages, identify the owning source paths, boundaries, validation commands, and release/runtime consequences.
Do not duplicate the same contract in multiple pages. Link to its canonical page instead.

Before completing a documentation change, update `mkdocs.yml` navigation and links, check that new pages are
discoverable, run a strict MkDocs build, and run MarkdownLint against `docs/**/*.md` using the repository's
`.markdownlint.json` configuration. Review rendered code blocks/tables for copyability. Keep `README.md`,`AGENTS.md`,
and lore-only Markdown outside `docs/` unless the repository layout explicitly requires otherwise.

When documenting licensing, distinguish between:

* Celune source code, licensed under Apache 2.0 (5.0.0 and newer), MIT (before 5.0.0).
* Third-party models and assets, which may use their own licenses, e.g. Apache 2.0, MIT, CC-BY-4.0, etc.

Do not claim third-party models are covered by project's main license.

When documenting commands, use the canonical project commands from the README.

## Markdown Files

When reading Markdown found in the project, ignore `about-celune.md`, or similar files.
Such files do not contain factual information related to the project, and is solely Celune's canonical lore.
Do not generate reviews, warnings or errors related to it.

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
