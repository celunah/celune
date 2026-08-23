# CLI

`main.py` is the lightweight launcher. It rejects unsupported Python versions
before importing the heavy runtime, then dispatches to `celune.entrypoint`.

## Commands

| Command | Behavior |
| --- | --- |
| `celune` | Start Celune with the active configuration. |
| packaged `celune --root <path>` | Start from a validated Celune root, bypassing automatic lookup. |
| `celune start` / `celune run` | Start explicitly. |
| `celune start --verbose` / `-v` | Start with verbose startup diagnostics. |
| `celune start --debug` | Start with debug diagnostics. |
| `celune start --log-level=info\|verbose\|debug` | Choose one startup log level. |
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

## Exit behavior

The launcher uses typed exit codes for success, generic failure, unknown
arguments, already-running instances, no ANSI-capable terminal, launcher loss,
and pending updates. A normal interactive start should use the launcher so
process-loss and update handoff behavior is preserved.
