# Celune's Philosophies

## What you are working on

Celune is a local-first, real-time character engine for expressive speech and
voice conversion. It combines configurable voices, long-form narration, DSP,
Persona conversations, agent tools, GPU inference, a Textual TUI, a FastAPI
service, and a Gradio WebUI. It targets Windows and Linux with Python 3.12–3.14;
many features require an RTX 30-series GPU or newer. CPU-only support is limited
to Pocket TTS and LuxTTS.

**Project map:** `celune/core/` owns startup and lifecycle;
`celune/pipeline/`, `celune/persona/`, and `celune/agent/` own generation,
character conversations, and tool execution; `celune/backends/` contains
isolated TTS/VC integrations; `celune/audio/`, `celune/cedts/`,
`celune/cevoice/`, `celune/registry/`, and `celune/dataclasses/` hold shared
audio, transport, voice-pack, registry, and event contracts. `celune/ui/`,
`celune/webui/`, `celune/server/`, and `celune/extensions/` provide the TUI,
browser UI, API, and extension surfaces.
`docs/` is the technical documentation, `tests/` the automated coverage, and
`scripts/`, `launcher/`, and `cmake/` support development and packaging.

## Do this

- **Work in scope.** Inspect the worktree and relevant callers, tests, and docs
  first. Preserve intentional edits. Keep changes focused, reuse the existing
  architecture, and prefer small cohesive modules; create a focused subpackage
  instead of growing a long top-level module when responsibilities form a
  boundary. Python source files, including tests and scripts, should usually
  stay at or below 100 KB; this is a recommendation, not a hard limit.
- **Choose the right workflow.** Use one agent for coupled, small, or unclear
  work. Delegate only independent work when it saves coordination time; give
  each worker exclusive files and keep integration, final review, docs,
  localization, formatting, and validation with the lead. When controls exist,
  use GPT-6.1 Sol medium for Codex solo/lead work and GPT-6 Luna xhigh for
  workers; use Opus 5.5 medium / Sonnet 5.5 xhigh in Claude.
- **Use project conventions.** Prefer concrete types and classic `Union` /
  `Optional` unions. Reuse existing values and helpers. Keep standard-library,
  third-party, and local imports in separate groups; sort statements in each
  group from shortest to longest, preserve multiline imports, prefer relative
  local imports, and keep Pylint `C0412` clean without ignore directives.
  At task end, sort and verify imports in each modified Python file.
  Format exceptions in Python style (`Error: error description`), reuse matching
  exceptions from `celune.exceptions`, and define new Celune-specific
  categories there.
- **Localize user-visible text.** Resolve UI, CLI, API, script, and example
  strings through `string("key", **kwargs)`. Keep keys at most 50 characters
  and values at most 100. Use stable keys and remove obsolete entries; keep
  tracebacks, paths, protocol data, and raw technical errors out of the locale
  database.
- **Keep platform and dependencies portable.** Support Windows and Linux on
  Python 3.12–3.14. Check `pyproject.toml` before adding core imports; backend
  packages belong in backend code or isolated workers. Guard platform-specific
  APIs so other supported platforms can type-check, and skip tests for other
  platforms. Use `uv`; run `python configure.py` for setup (`--all-extras` is
  Linux-only). If `uv` reports access denied, retry with `--no-cache`; do not
  alter the environment to work around failures.
- **Preserve runtime contracts.** Keep startup's default theme and loading
  screen ahead of core or backend imports. Reuse audio, playback, buffering,
  stream, and DSP abstractions; apply runtime themes after initialization. Use
  normalized `np.float32` audio in `[-1, 1]`
  and keep audio calculations in float32 unless precision requires float64;
  write audio files as 24-bit, 48 kHz FLAC. Preserve API behavior and
  request/response models; keep WebUI behavior consistent with the TUI and do
  not make it depend on raw REST calls. Retain checks and fallbacks for system
  tools such as SoX, Rubber Band, OpenRGB, CUDA, and native build tools.
- **Keep the UI Celune-specific.** Preserve the established palette and visual
  identity. Treat the WebUI as the high-resolution counterpart to the TUI,
  support touch as well as keyboard input with visible button alternatives for
  shortcuts, and use current Gradio APIs. Prefer pointer/hover queries when the
  input method matters.
- **Document behavior with the change.** Put technical docs under `docs/` and
  follow `docs/development/documentation.md`: each technical page has a title,
  an immediately following purpose paragraph, task/contract sections, and
  relevant verification, compatibility, troubleshooting, and `See also`
  sections. Use sentence-case headings, language-tagged code fences, tables for
  stable comparisons, numbered procedures, and unordered bullets; preserve
  acronyms, API names, commands, and format names. Document public call
  signatures, arguments, returns, errors, side effects, and an example.
  Specify versions, invariants, wire/file layout, compatibility, and failure
  behavior for formats/protocols; prerequisites, steps, expected results, and
  recovery for procedures. Development pages identify owning paths,
  boundaries, validation, and runtime/release effects. Use canonical commands
  from the README and keep examples copyable. Keep contracts canonical instead
  of duplicating them. Update `mkdocs.yml` when pages or navigation change, and
  run strict MkDocs and MarkdownLint for docs changes.
  Distinguish Celune's Apache 2.0 license (5.0.0+) from its earlier MIT license
  and from third-party model/asset licenses.
- **Validate proportionally and report honestly.** After code changes, run
  `python scripts/update_docstrings.py`, replace remaining placeholder
  docstrings (remove typing/dataclass placeholders), then run
  `uv run ruff format .`. Run CI when the diff changes Celune Python; skip it
  for docs-only or non-Celune-Python changes. Use the canonical
  `python scripts/run_ci.py`; it includes Ruff, Pylint, Pyrefly, and tests. Use
  `poe test_basic` only when host resources cannot support `poe test`. Do not
  rerun completed CI stages after a stall; find the cause and run only
  incomplete checks after fixing it. Do not narrow validation after a failure.
  Report failures, timeouts, warnings, and unavailable checks accurately. After
  validation, remove project-owned `__pycache__` directories, excluding virtual
  environments.
- **Publish cleanly.** Keep `dev` as the integrated branch and push complete
  merged Celune work there. Create `<version>/<feature>` branches only when
  requested; review requested feature branches before merging them to `main`.
  Use one Conventional Commit per meaningful change.

## Don't do that

- Don't make unrelated refactors, re-export symbols during refactors, create a
  parallel architecture, or hardcode values already owned by the project.
- Don't add unnecessary dependencies, placeholder implementations, or TODOs;
  don't silently disable features to satisfy tests.
- Don't use PEP 604 unions or broad `Any` / `object` types without a concrete
  need. Don't inline English UI/API strings or localize raw diagnostics.
- Don't import backend-only packages into core code, use `pip` directly, run
  tests through a standalone `pytest` or `.venv` Python command, or use cache
  overrides such as `UV_CACHE_DIR`.
- Don't bypass Celune's audio/runtime abstractions, skip system dependency
  checks without understanding their impact, emit unnormalized audio, or
  produce audio files outside the required FLAC format. Protect long-form
  narration and don't add unsupported markup/control tags or large audio/game
  frameworks for small playback tasks.
- Don't move heavy imports ahead of the startup loading screen, replace
  established UI behavior with generic Hugging Face-style styling, rely on
  screen width alone for touch behavior, or route WebUI actions through raw
  REST calls.
- Don't duplicate technical contracts across docs, omit required docs/nav or
  validation updates, or put `README.md`, `AGENTS.md`, or lore-only Markdown in
  `docs/` unless the repository explicitly requires it. Ignore
  `about-celune.md` and similar lore files as factual project documentation.
  Don't imply third-party models or assets are covered by Celune's license.
