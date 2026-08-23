# Test execution

This page is for Celune contributors who run or extend the pytest suite. It
defines the parallel worker contract, the local test commands, and the limits
of incremental test selection.

## Run the suite

The default development and CI test task uses two xdist workers and keeps each
test module on one worker:

```bash
uv run poe test
```

Use the serial task when debugging a failure or investigating process-global
state:

```bash
uv run poe test_serial
```

The serial task runs the complete `tests/` collection without xdist. Both
tasks return pytest's exit status and include all collected tests.

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

When parallel execution fails:

1. Re-run the affected node with `uv run poe test_serial` or a focused pytest
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
