# CLI

`main.py` is the lightweight launcher. It rejects unsupported Python versions
before importing the heavy runtime, then dispatches to `celune.entrypoint`.

Commands that do not start the core keep that boundary: help, version, the
doctor report, and the test-mode listing use lightweight command dependencies.
Configuration commands load only the configuration-path and system-browser
helpers. The full Celune dependency contract is loaded by the default start,
explicit start/run commands, and the UI or agent test modes.

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
| `celune test ui` | Run the UI test mode. |
| `celune test agent` | Run the agent test mode. |
| `celune config view` | Print the active YAML configuration. |
| `celune config edit` | Open the active configuration in the system editor. |
| `celune doctor` | Run environment checks without starting the app. |
| `celune doctor --fix` | Run checks and ask the setup helper to repair supported issues. |
| `celune help` / `--help` / `-h` | Print command help. |
| `celune version` / `--version` | Print version, revision, and tagline. |

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
smoke test where possible. It distinguishes a missing prerequisite from an
accelerator that is present but unusable. `--fix` does not promise to repair
third-party GPU drivers or arbitrary backend environments.

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
