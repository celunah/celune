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

Pass `signed` to either build script to require a signed update manifest and
write a signed ZIP with a `-signed` suffix:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_nuitka.ps1 signed
```

```bash
bash scripts/build_nuitka.sh signed
```

## Python build target

Celune source installations support Python 3.12 through 3.14. Compiled
artifacts use Python 3.13 deliberately so v4 and v5 releases share one
CPython ABI. Both build scripts pin the Nuitka invocation to Python 3.13.

Build from a Git checkout with at least one commit. Each script checks for a valid `HEAD`
before stopping a running Celune process or changing build artifacts. A GitHub
**Download ZIP** has no commit history and cannot supply the revision recorded
in `.celune-root` and the update manifest; the script exits with the clone
command when Git metadata is missing.

The compiled build remains a deployment build rather than a fully standalone
bundle, so its `.venv` must use the matching Python 3.13 runtime. Python 3.14
remains available for source development and source-based launches; a Python
3.14 `.venv` cannot run the compiled 3.13 executable.

Windows builds use Nuitka, Visual Studio C++ tools, the launcher C sources,
static libsodium from the `x64-windows-static-md` vcpkg triplet, the project
icon/resource, and `scripts/write_update_manifest.py`. Linux builds use GCC,
`libsodium-dev`, `pkg-config`, `appimagetool`, and `zip`. Both release launchers
link libsodium statically. Install the native crypto dependency before building:

```powershell
vcpkg install libsodium:x64-windows-static-md
```

```bash
sudo apt-get install -y libsodium-dev pkg-config
```

The Windows script refreshes `resources/vcruntime140.dll` from the configured
Visual Studio 14.44.35112 CRT installation, includes Windows runtime DLLs during
Nuitka compilation, places that exact DLL beside both release executables, and
adds it to the update manifest and archive. Linux builds produce the executable,
AppImage, and archive. Before replacing
Windows build artifacts, the script force-stops existing Celune processes,
waits up to thirty seconds for their process handles to exit, and retries
transient locked-file cleanup for up to five seconds. A persistent lock still
fails the build with the original cleanup error instead of producing a partial
package.

Typical output files are:

| Platform | Artifacts |
| --- | --- |
| Windows | `bin/celune.exe`, `bin/celune-bin.exe`, `bin/vcruntime140.dll`, `bin/celune-update.json`, optional `bin/celune-update.sig`, `bin/Celune-win-x64.zip` or `bin/Celune-win-x64-signed.zip` |
| Linux | `bin/celune`, `bin/celune-bin`, `bin/celune.AppImage`, `bin/celune-update.json`, optional `bin/celune-update.sig`, `bin/Celune-linux-x64.zip` or `bin/Celune-linux-x64-signed.zip` |

The Windows archive contains `celune.exe`, `celune-bin.exe`,
`vcruntime140.dll`, and `celune-update.json`; signed builds also include
`celune-update.sig`. It does not contain the legacy
`celune-bin.cmd` compatibility wrapper; the native launcher owns the runtime
environment setup. The launcher checks for AVX in native C before starting
Python on x86-64,
consistently on Windows and Linux. The `doctor`, help, and version commands
remain available so a missing AVX capability can be diagnosed. The launcher
sets the compiled environment, owns update handoff, and preserves
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
or heavy runtime-import path. Ctrl+C stops an active runtime search, and the
launcher failure screen handles errors from the default, `start`, and `run`
operations; other CLI commands return their own exit status directly.
When the automatic lookup should be bypassed, the compiled launchers accept
`CELUNE_ROOT=/path/to/celune`, `--root /path/to/celune`, or
`--root=/path/to/celune`. The selected path must identify a valid Celune root;
the launcher removes `--root` and its value before forwarding the remaining
arguments to the runtime.

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
and files. Exit code 7 is `EXIT_PENDING_RESTART`: the launcher silently restarts
Celune when no manifest is present, and uses the existing update helper and
internal `__apply_update` command when the compiled bundle contains a manifest.
The UI terminal title changes to `Restarting` before the handoff. Do not call
that internal command directly; it is a launcher protocol.

### Release signatures

Compatible release ZIPs carry `celune-update.json` and its detached
`celune-update.sig` Ed25519 signature. The signature covers the exact manifest
bytes; the manifest contains SHA-256 digests for the shipped executables and
runtime files. The C launcher verifies the signature and the selected runtime
digest before starting that runtime. The Python entrypoint independently
verifies the signature and every listed file during startup. A present but
invalid signature or a mismatched runtime digest rejects startup. When no
signature is present, the C launcher allows startup and Python warns that the
copy is unsigned; this keeps legacy JSON-manifest bundles usable. The updater
still rejects unsigned remote release ZIPs.

Create a release key once from the checkout that will build the release:

```powershell
uv run python scripts\make_signing_key.py
```

```bash
uv run python scripts/make_signing_key.py
```

The script creates an unencrypted Ed25519 PEM key at
`%LOCALAPPDATA%\Celune\keys\release-ed25519-private.pem` on Windows and
`~/.celune/release-signing-key.pem` on Linux. It refuses to overwrite an
existing file, restricts file access to the current account, rejects
destinations inside the checkout, and does not print private-key contents. It
also adds the public key to
`celune/release_keys.py` and `launcher/release_keys.h`, while preserving
previously trusted public keys. Include those public-key changes in the source
used to build the signed package.
Clients built before a new public key was added do not trust that key. For key
rotation, distribute a transition build signed by a key those clients already
trust before signing later releases with the new key.

The build scripts use their platform's default private-key path when called
with `signed`. On Windows, that is the `%LOCALAPPDATA%` path above. For another
key location, pass it to `--output` and set
`CELUNE_RELEASE_SIGNING_KEY_FILE` to the same path before building. The existing
`CELUNE_REQUIRE_RELEASE_SIGNATURE=1` setting also continues to require signing:

```powershell
uv run python scripts\make_signing_key.py --output "C:\path\to\release-ed25519-private.pem"
$env:CELUNE_RELEASE_SIGNING_KEY_FILE = "C:\path\to\release-ed25519-private.pem"
powershell -ExecutionPolicy Bypass -File scripts\build_nuitka.ps1 signed
```

```bash
uv run python scripts/make_signing_key.py --output "$HOME/keys/release-ed25519-private.pem"
export CELUNE_RELEASE_SIGNING_KEY_FILE="$HOME/keys/release-ed25519-private.pem"
bash scripts/build_nuitka.sh signed
```

The native C signature tests run as part of both platform launcher builds. The
build fails before compilation when signed mode has no key. Publishing compiled
release assets must use signed ZIPs and preserve the detached
`celune-update.sig` file. Unsigned CI build artifacts are for build validation
and cannot be installed through the signed updater path. Historical ZIP releases
with compatible update metadata must retain their version and file contents and
gain a signed manifest and sidecar when migrated; precompiled legacy clients do
not verify signatures themselves.

For each compatible GitHub release, keep the original ZIP under its existing
name as the unsigned archive and publish a second archive named
`Celune-{platform}-x64-{tag}-signed.zip`, for example
`Celune-win-x64-v5.0.0-signed.zip`. Signed builds now produce a
`Celune-win-x64-signed.zip` or `Celune-linux-x64-signed.zip` package; rename it
to the versioned release name before publishing. The updater selects only the
signed archive whose name matches the release tag; it ignores the original
unsigned archive.

The `signed` suffix means that Celune's custom Ed25519 bundle anti-tamper
signature verifies against the public key embedded in Celune. It does not mean
that the executable has a CA-issued or Authenticode code-signing certificate.
The original ZIPs, which keep their existing names without the `-signed` suffix,
are unsigned because they have no Celune bundle signature; legacy copies remain
allowed with a startup warning.

The published Windows and Linux ZIPs from v4.1.0 through v5.0.0 have signed
counterparts. Their original ZIP assets remain unchanged as unsigned copies.
Earlier source-only releases did not publish compatible update ZIPs.

### Signature troubleshooting

If Celune reports invalid update metadata, confirm that the archive contains
both update files, that the signature was made with the matching release key,
and that every manifest-listed file still matches its SHA-256 digest. Rebuild
the archive from the matching release output rather than editing a signed ZIP.

## See also

- [Source map](../reference/source-map.md)

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
