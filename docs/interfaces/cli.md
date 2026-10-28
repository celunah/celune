# CLI

`main.py` is the lightweight launcher. It rejects unsupported Python versions
before importing the heavy runtime, then dispatches to `celune.entrypoint`.

Commands that do not start the core keep that boundary: help, version, the
doctor report, and the test-mode listing use lightweight command dependencies.
Configuration commands load only the configuration-path and system-browser
helpers. Voice-pack commands use the CEVOICE reader and writer without starting
the runtime. Normal starts load the full Celune dependency contract. Agent test
mode loads the engine and configured integrations; UI test mode uses the Textual
interface with a lightweight runtime state object.

## Commands

| Command | Behavior |
| --- | --- |
| `celune` | Start Celune with the active configuration. |
| packaged `celune --root <path>` | Start from a validated Celune root, bypassing automatic lookup. |
| `celune start` / `celune run` | Start explicitly. |
| `celune start --verbose` / `-v` | Start with verbose startup diagnostics. |
| `celune start --debug` | Start with debug diagnostics. |
| <code>celune start --log-level=info&#124;verbose&#124;debug</code> | Choose one startup log level. |
| `celune start --test` / `-t` | Run the lightweight UI test runtime. |
| `celune test` | Print available explicit test modes. |
| `celune test ui` | Open the UI without starting the engine or audio runtime. |
| `celune test agent` | Run the agent test mode. |
| `celune config view` | Print the active YAML configuration. |
| `celune config edit` | Open the active configuration in the system editor. |
| `celune ls <pack.cevoice>` | List a pack's ordered entries and byte sizes. |
| `celune tar --help` | Show CEVOICE listing, extraction, and edit operations. |
| `celune doctor` | Run environment checks without starting the app. |
| `celune doctor --fix` | Run checks and ask the setup helper to repair supported issues. |
| `celune help` / `--help` / `-h` | Print command help. |
| `celune version` / `--version` | Print version, revision, and tagline. |

The agent test runs the configured Persona, Needle, and TTS path through one
read-only task. It then runs every active production tool handler with isolated
test state and speaks a short worked/didn't work status for each tool; detailed
results are recorded in the test log. Celune speaks a multiple-choice question
and an approval request before displaying them and accepting typed answers. The
answers are applied directly to the paused diagnostic runtime, without another
Persona response. Either listed choice passes. Approving or declining the
simulated mutation passes when the response is processed. The test also checks task
lifecycle, schema validation, permissions, context limits, and terminal
outcomes. Each typed response prompt waits up to two minutes; a missing answer
fails that check. When local-management tools are enabled, filesystem operations stay
inside a temporary directory; process and application mutations use test
doubles and never launch or close real resources. Independent checks continue
after a failure.

UI test mode opens the live Textual interface without importing the production
engine, TTS backend, model, GPU, or audio integrations. The screen remains
available for visual inspection, while its runtime actions stay disabled. It
reports that test mode is active, then marks the runtime stopped after startup.
Use agent test mode when checking the configured engine and its integrations.

The stopped test UI log shows one `PASS`, `FAIL`, or `SKIP` entry per check and
a total for each status. Local-management tools are skipped when
`agent.fs_tools` is disabled. When enabled, filesystem mutations stay inside a
temporary directory; process and application launch or termination use test
doubles and do not affect running programs. A failed active check makes
`celune test agent` return the standard failure exit code after the UI closes.

The command help aligns descriptions with spaces, so terminal tab-stop settings
do not shift the text.

## Voice pack operations

`celune ls <pack.cevoice>` and `celune tar -tf <pack.cevoice>` validate the
container structure and entry checksums, then list its filename, optional
character name, and entries in stored order. They do not load PyTorch to
inspect `.pt` tensor contents. Each entry has a 1-based number and size in
bytes.

`celune tar` accepts one operation per invocation:

| Syntax | Behavior |
| --- | --- |
| `celune tar -tf <pack.cevoice>` | List entries. |
| `celune tar -xf <pack.cevoice> [destination]` | Extract to the current directory or the optional destination. |
| `celune tar -rf <pack.cevoice> <file...>` | Remove entries by exact filename. |
| `celune tar -df <pack.cevoice> <entry...>` | Remove entries by exact filename or 1-based index. |
| `celune tar -af<N> <pack.cevoice> <file...>` | Insert files before entry N; use entry count + 1 to append. |

Extraction creates the optional destination and any subdirectories used by
entry names. It checks every output path before writing and fails if an output
already exists. Files are never overwritten.

Add only `.wav`, `.pt`, and the supported Persona Markdown filenames described
in [CEVOICE and CECHAR](../CEVOICE.md). A WAV entry needs matching voice
metadata with a non-empty `reference_text`. New names must be unique without
regard to letter case. `-af<N>` accepts a positive decimal position from 1
through the current entry count plus one.

For `-df`, Celune first looks for an exact case-sensitive filename. If there is
no exact match and the selector is decimal, it is treated as a 1-based index.
Repeated selectors, missing entries, and out-of-range indexes are argument
errors. `-rf` accepts filenames only. Removing a voice's final asset also
removes its voice metadata references; an edit that would leave an invalid
pack fails without replacing the original.

Successful edits preserve the pack generation, metadata, compression, and
unaffected entry order. The new pack is written to a temporary sibling,
validated, and then moved over the original.

```bash
celune ls voices/default.cevoice
celune tar -xf voices/default.cevoice ./unpacked
celune tar -af4 voices/custom.cevoice ./assets/identity.md
celune tar -df voices/custom.cevoice identity.md 2
```

Syntax errors and invalid selectors return exit code `5`. Invalid packs,
unsupported files, metadata violations, duplicate additions, existing
extraction targets, unavailable compression dependencies, and filesystem
failures return exit code `1`. When a pack uses an unavailable codec, Celune
names the required dependency instead of reporting the pack as invalid.
Adding a new indexed `.pt` entry requires PyTorch to validate the tensor.

The `-v` spelling is accepted both as the verbose start flag and as the version
alias by the dispatcher; use `--verbose` and `--version` in scripts to avoid
ambiguity.

## Environment overrides

The launcher reads `CELUNE_LOG_LEVEL`, `CELUNE_BACKEND`, `CELUNE_HEADLESS`,
`CELUNE_LAUNCHER`, and `CELUNE_ROOT`. `CELUNE_ROOT` and the packaged launcher's
`--root <path>` option select a validated installation when automatic lookup
should be bypassed. Persistent settings belong in `config.yaml`; environment
overrides are useful for one process, a service wrapper, or a packaged launcher.

## Doctor checks

Doctor checks Python version, repository/default-config paths, version metadata,
the active interpreter/venv, core Python imports, system binaries, runtime
configuration, PyTorch build and CUDA backend, GPU architecture, and a compute
smoke test where possible. TTS backend dependencies belong to isolated CEDTS
worker environments, so doctor does not look for them in Celune's core Python
environment. Source-tree runs also check for a committed Git
checkout with a commit and show the upstream clone command when Git metadata is missing.
Compiled releases use their bundled update manifest and do not need a local Git
checkout. The report distinguishes a missing prerequisite from an accelerator
that is present but unusable. `--fix` does not promise to repair third-party
GPU drivers or arbitrary backend environments. The optional eSpeak NG check is
a warning: without eSpeak NG, Celune still runs, but word-timed captions are
unavailable.

## CPU compatibility

The compiled Windows and Linux launchers check for AVX before starting Python.
On x86-64, AVX is a consistent startup requirement on both platforms. The
Python-side `doctor` report also checks the x86-64-v2 baseline used by Celune's
native dependencies: SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B, and
LAHF/SAHF. ARM64 uses its native instruction-set baseline because AVX is an
x86 instruction set.

If AVX is unavailable, the launcher exits with code `9` and does not start the
Python runtime. Run `celune doctor` to inspect the complete CPU compatibility
report on a system where the feature probe is available.

## Exit behavior

The launcher uses typed exit codes for success, generic failure, unknown
arguments, already-running instances, no ANSI-capable terminal, launcher loss,
pending updates, and an unsupported CPU. A normal interactive start should use
the launcher so process-loss, CPU compatibility, and update handoff behavior
are preserved.

## See also

- [Agent mode](../user-guide/agent.md)
