# Packaging

This page describes how contributors build, package, update, and deploy
Celune's compiled artifacts and documentation.

## Nuitka builds

The build scripts produce a compiled core executable plus a small launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_nuitka.ps1
```

```bash
bash scripts/build_nuitka.sh
```

Windows builds use Nuitka, Visual Studio C++ tools, the launcher C sources,
the project icon/resource, and `scripts/write_update_manifest.py`. Linux builds
need GCC, `appimagetool`, and `zip`, then produce the executable, AppImage, and
archive. The scripts stop existing Celune processes before replacing build
artifacts.

Typical output files are:

| Platform | Artifacts |
| --- | --- |
| Windows | `bin/celune.exe`, `bin/celune-bin.exe`, `bin/celune-update.json`, `bin/Celune-win-x64.zip` |
| Linux | `bin/celune`, `bin/celune-bin`, `bin/celune.AppImage`, `bin/celune-update.json`, `bin/Celune-linux-x64.zip` |

The launcher sets the compiled environment, owns update handoff, and preserves
process-loss behavior. Its terminal cleanup restores modes without scrolling
the cursor through the console buffer, so returning to the shell does not leave
a large blank region. If startup fails, the launcher preserves the visible
console and scans the rendered buffer for the last non-blank row before placing
diagnostics. This avoids trusting a cursor coordinate that Rich or Textual may
leave at the start of an earlier traceback row, so stale loading-screen
characters cannot overprint the error. `celune-bin` is not the user-facing
command. When the runtime is moved, the Windows and Linux launchers search
nearby user/system roots and fixed filesystem roots for a validated runtime for
up to sixty seconds or five hundred thousand folders before reporting that
Celune cannot be found. Each directory enumeration has its own five-second
budget. Before traversal, the launcher checks the directory from which it was
invoked and uses a valid local runtime immediately when one is present. Fatal
Textual UI return codes are propagated to this launcher, so callback failures
use the same preserved-output failure path as startup errors. Search traversal
is breadth-first, so progress levels are processed in ascending order. Level
transitions repaint immediately; repeated updates within one level are
throttled. The loading screen's terminal-title and early-shutdown helpers stay
lightweight, so exiting before runtime initialization cannot enter an undefined
or heavy runtime-import path.

The repository root is identified by a `.celune-root` marker containing a
version, abbreviated commit, and date in the form `v5.0.0 (d870260),
23/08/2026`. The native search ignores unreadable entries and all symbolic-link
or reparse-point entries; a marker stops traversal only when its expected
`celune-bin` runtime is also present. Configure the tracked post-commit hook once with
`git config core.hooksPath .githooks`; on Linux, make the hook executable with
`chmod +x .githooks/post-commit`. Builds also run `python scripts/root.py` as a
fallback when the hook is not installed.

## Relocatable data

Compiled launches resolve the project root beside the executable and use the
Celune application-data directory for configuration, model caches, runtime
downloads, voice packs, Persona memory, and backend environments. The
`configure.py` setup helper seeds AppData with `config.yaml` and the default
voice pack from the bundled `default_config.yaml` and `voices/` files. Package
assets remain alongside the installed Python package. Source-tree launches use the repository root, but Hugging Face
downloads still use the same Celune application-data cache as compiled
launches. Explicit `HF_HOME` and `HF_HUB_CACHE` values remain authoritative.

## Updates

The update manifest records the project version, Git revision, artifact name,
and files. The launcher can request a pending update and restart through the
internal `__apply_update` command. Do not call that internal command directly;
it is a launcher protocol.

## Documentation deployment

ReadTheDocs is configured by `.readthedocs.yaml`:

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"
mkdocs:
  configuration: mkdocs.yml
python:
  install:
    - requirements: docs/requirements.txt
```

The documentation source is `docs/`, the navigation is `mkdocs.yml`, and
`mkdocs-material` is installed from `docs/requirements.txt`. Pushing the
configured repository branch triggers the ReadTheDocs webhook when the
project's GitHub integration is enabled; the published URL is
`https://celune.readthedocs.io/en/latest/`. The repository's
`.github/workflows/docs.yml` workflow builds the same MkDocs site in GitHub
Actions for documentation changes, but it does not replace the ReadTheDocs
incoming webhook. The site loads the canonical palette from `celune/theme/colors.py`
into `docs/assets/stylesheets/celune.css` and uses Michroma for display text
and Outfit for body text.
