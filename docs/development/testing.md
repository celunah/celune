# Testing

This page is for Celune contributors who run or extend the pytest suite. It
defines Celune's friendly pytest output, the parallel worker contract, the
local test commands, and the limits of incremental test selection.

## Read the test output

The test suite loads `tests/celtest.py` as a pytest plugin through
`tests/conftest.py`. It keeps pytest's collection, fixtures, warning capture,
reporting hooks, and exit codes, while replacing pytest's terminal presentation.
Its result-status hook takes precedence over other pytest plugins, so foreign
categories such as an `u`/unknown marker cannot leak into Celtest's status row
or add a second percentage progress display.
Without `-v`, it prints a compact status row as tests finish, followed by the
totals:

```text
testing [app name and version]

....................

passed 20/20 time 0:10 warnings 0
```

Non-verbose output does not include test descriptions or failure tracebacks. If
tests fail or emit warnings, concise details appear directly below the totals.
With `-v`, ANSI-capable terminals show a processing line followed by the final
result and detailed display:

```text
testing [app name and version]

? friendly test description
. friendly test description
F friendly test description
[pytest's exact failure representation]
W friendly test description
E friendly test description

passed 2/4 time 0:01 warnings 1

W warnings
[warnings reported during tests]

test failure hint
[a concise assertion or exception explanation]
```

When ANSI markup is unavailable, verbose mode omits the processing `?` line and
emits each final result directly as a plain line terminated by `\n`; it does not
use carriage-return replacement.

`F` marks assertion-style failures, while `E` marks execution, setup, teardown,
collection, and other errors. Configured skips use the blue `S` marker. Tests
that were collected but never started use `N`.

If CTRL+C interrupts a run after collection, Celtest keeps any completed test
results already shown and ends with a simple `interrupted` message. It omits
the interruption hint, totals, and all incomplete or skipped test entries:

```text
interrupted
```

The header uses `project.version` and `tool.celtest.display_name` from the
active project's `pyproject.toml`, falling back to `project.name` when no
display name is configured. Non-verbose output appends each completed status
symbol once, avoiding carriage-return redraws in interactive terminals and
redirected output. When ANSI markup is unavailable, these symbols and summary
details are emitted without color. ANSI-capable verbose terminals use a raw
carriage return and fixed-width padding for processing lines, including
controller output from parallel runs. Non-ANSI verbose output emits only final
results as plain newline-terminated lines.
Assertion-style failures are marked `F`; execution, setup, teardown, collection,
and other errors are marked `E`, including when a failed test also produces
warnings. Skipped tests are omitted from the final list, count, and summary.
Fallback descriptions preserve common Celune acronyms such as `UI`, `TTS`, and
`VRAM`. Parallel runs label worker startup as `setting up parallel test harness`.
Collection and other fatal errors use a separate failure block with module names
and a concise hint. Configured skips appear as blue `S` markers in the
non-verbose status row, while tests prevented from starting use `N` and are
reported as `N not run N`. The `test collection failed` block is reserved for
actual collection failures; interruptions use `interrupted`.

Decorate tests with a friendly description, and optionally provide a stable
failure explanation:

```python
from tests.celtest import celtest


@celtest("loads the default voice")
def test_default_voice() -> None: ...


@celtest(
    "rejects an unknown voice",
    hint="The selected voice is not present in the active voice pack.",
)
def test_unknown_voice() -> None: ...
```

The decorator stores metadata without wrapping the test callable and preserves
the supplied description exactly, including its capitalization and punctuation.
Existing tests without metadata use their first non-empty docstring line as the
friendly description after lowercasing its sentence opener and removing one
terminal period; names and acronyms elsewhere in that line are preserved.

## Run the suite

The default development and CI test task uses two xdist workers and keeps each
test module on one worker:

```bash
uv run poe test
```

Use the basic serial task when debugging a failure or investigating
process-global state:

```bash
uv run poe test_basic
```

The basic task runs the complete `tests/` collection without xdist. Both
tasks return pytest's exit status and include all selected tests.

## Keep tests parallel-safe

Each xdist worker receives a unique Celune data root before test modules are
collected. Hugging Face and Numba caches are placed below that root, so tests
must resolve application data through Celune's path helpers instead of writing
to a fixed user directory.

Tests should follow these rules:

- use pytest's `tmp_path` or `tempfile` for generated files;
- mock audio devices, GPU queries, network ports, and external subprocesses;
- restore environment variables, singleton guards, background threads, and
  process handles during teardown;
- bound asynchronous synchronization waits so a broken test fails instead of
  waiting indefinitely;
- pass fake TTS and voice-conversion backends, or patch backend resolution, in
  orchestration tests so named backends cannot provision model environments;
- keep tests within one module when they intentionally share class or module
  fixtures, because `--dist loadfile` does not split a module across workers.

The worker roots isolate filesystem caches, but they do not virtualize physical
audio devices, GPUs, operating-system ports, or external services. Tests that
need those resources must provide fakes or use an explicitly serial task.

## Use incremental local selection

For local iteration, testmon records coverage-based test dependencies in the
ignored `.testmondata` file:

```bash
uv run poe test_changed
```

The first run establishes dependency data. Later runs select tests affected by
changed Python code and retain failing tests. Testmon can observe Python code
executed in the pytest process, but it cannot reliably infer changes to
generated assets, runtime configuration, external tools, or code executed only
inside CEDTS worker subprocesses. Run `uv run poe test` before handing off a
change and in CI.

## Verify failures

The plugin retains setup, call, and teardown failures and writes pytest's
original failure representation immediately after the failed test's `F` or `E`
line. It derives the final hint from pytest's assertion or exception message,
without stack frames; a decorator `hint=` overrides that derived text.

Collection failures, fatal pytest errors, and interruptions retain pytest's
non-zero exit status and are reported with a concise explanation instead of a
second terminal traceback. Interruptions omit incomplete and skipped test
entries as well as the normal totals summary.

When parallel execution fails:

1. Re-run the affected node with `uv run poe test_basic` or a focused pytest
   invocation.
2. Check for fixed paths, environment mutations, open worker processes,
   singleton state, physical-device access, or shared cache writes.
3. Add isolation or teardown to the test before restoring it to the parallel
   task.

The canonical project validation command remains:

```bash
python scripts/run_ci.py
```

## See also

- [Validation standards](validation.md)
- [Architecture](architecture.md)
