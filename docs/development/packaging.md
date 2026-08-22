# Packaging and release

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
process-loss behavior. `celune-bin` is not the user-facing command.

## Relocatable data

Compiled launches resolve the project root beside the executable and use the
Celune application-data directory for configuration, model caches, runtime
downloads, voice packs, Persona memory, and backend environments. The bundled
`default_config.yaml`, `voices/`, and package assets are copied/located as
release data. Source-tree launches use the repository root and host caches
unless the runtime explicitly configures portable cache behavior.

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
configured repository branch triggers the ReadTheDocs webhook; the published
URL is `https://celune.readthedocs.io/en/latest/`.
