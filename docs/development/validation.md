# Validation standards

This page defines the validation workflow for Celune code, documentation,
packaging, and cross-platform changes.

## Canonical checks

For logical Python changes, the repository's canonical command is:

```bash
python scripts/run_ci.py
```

Before CI, format the repository:

```bash
uv run ruff format .
```

The CI workflow then checks formatting, Pylint, Pyrefly, docstring coverage,
unit tests, import smoke tests, and Windows/Linux launcher builds. It installs
core dependencies with `uv sync --dev --all-extras` on Linux and
`uv sync --dev --extra api` on Windows, then uses the project's warning wrapper
for CI annotations.

Do not infer success from a reduced test collection. If a native DLL, CUDA,
PortAudio, SciPy, safetensors, or cache permission failure prevents collection,
report it as an environment failure and distinguish it from test assertions.

## Docstrings and imports

After changing Python, run `scripts/update_docstrings.py`, inspect every
mechanical edit, replace placeholders with real documentation, and run Ruff
format again. It is allowed to update many files, so review the diff and keep
only changes belonging to the task. Sort imports in every modified Python file
as standard library, third party, then local relative imports, with the
repository's length-sorted convention.

Delete generated `__pycache__` directories before handing off. They are not
part of Celune's compiled-artifact boundary.

## Documentation checks

MarkdownLint checks every technical page under `docs/` using the repository
root `.markdownlint.json` configuration:

```bash
npx --yes markdownlint-cli2 "docs/**/*.md"
```

This command requires Node.js 22 or newer. The GitHub Action supplies Node.js
24 automatically.

ReadTheDocs builds the repository with `.readthedocs.yaml`, Python 3.13, and
`docs/requirements.txt`, which installs `mkdocs-material`. Validate locally
with:

```bash
uv run --with mkdocs-material mkdocs build --strict
```

Strict mode catches missing navigation pages and broken relative links. Check
that gallery assets are under `docs/`, because ReadTheDocs only publishes the
configured MkDocs documentation tree.

## Focused checks

For a docs-only change, a strict MkDocs build, link/path inspection, and Git
diff review are the relevant gates; do not claim the full Python CI passed when
no Python validation was needed or when native prerequisites prevent it. For
format/protocol/API changes, add the corresponding focused tests and then run
the canonical CI when the environment allows it.
