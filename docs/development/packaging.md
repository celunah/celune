# Packaging

This page describes how contributors build, package, update, and deploy
Celune's compiled artifacts and documentation.

## CMake release builds

CMake builds a pinned static libsodium, compiles the C launcher and runs its
signature verification, compiles the Python core with Nuitka, then packages the
release. Use the `unsigned` preset for a validation archive or the `signed`
preset for a release archive:

```powershell
cmake --preset unsigned
cmake --build --preset unsigned
```

```bash
cmake --preset unsigned
cmake --build --preset unsigned
```

The signed preset keeps the same two-step workflow:

```powershell
cmake --preset signed
cmake --build --preset signed
```

```bash
cmake --preset signed
cmake --build --preset signed
```

Both presets build the `celune_release` target with at most two build jobs;
Nuitka's C compiler is also capped at two jobs.
The dependency chain enforces libsodium → C launcher and native signature
verification → Python core (Nuitka) → release packaging. `uv run poe build`
builds the unsigned preset; `uv run poe build_signed` builds the signed preset.

## Python build target

Celune source installations support Python 3.12 through 3.14. Compiled
artifacts use Python 3.13 deliberately so v4 and v5 releases share one
CPython ABI. The CMake build pins the Nuitka invocation to Python 3.13.

Build from a Git checkout with at least one commit. CMake checks for a valid
`HEAD` before stopping a running Celune process or changing build artifacts. A GitHub
**Download ZIP** has no commit history and cannot supply the revision recorded
in `.celune-root` and the update manifest; use a Git clone to build release
artifacts.

The compiled build remains a deployment build rather than a fully standalone
bundle, so its `.venv` must use the matching Python 3.13 runtime. Python 3.14
remains available for source development and source-based launches; a Python
3.14 `.venv` cannot run the compiled 3.13 executable.

Windows builds use CMake, Visual Studio C++ tools, the launcher C sources, the
project icon/resource, and `scripts/write_update_manifest.py`. Linux builds
use CMake, GCC, Make, and `appimagetool`. CMake downloads the pinned libsodium
1.0.22 source archive, verifies its SHA-256 digest, and builds a static library
as part of the release target. Linux compiles libsodium without LTO so the
launcher does not depend on a host archive or its LTO plugin configuration.
The Windows build uses libsodium's `StaticRelease` MSVC project, whose static
CRT matches the launcher's `/MT` runtime. Install CMake 3.25 or newer, `uv`, and
the platform C compiler and build tools before configuring. Windows requires
the Visual Studio 2022 C++ tools. Linux also needs `appimagetool`.

CMake configures `resources/celune.rc.in` with the project version from
`pyproject.toml`, then compiles and links that resource with the Windows C
launcher. It supplies the Celune icon and the `Celune Launcher` file identity.

On Windows, Nuitka includes Windows runtime DLLs, and the tracked
`resources/vcruntime140.dll` is placed beside the release executables and added
to the update manifest and archive. Linux builds produce the launcher,
Nuitka executable, AppImage, and archive. CMake stops running Celune processes
before preparing output artifacts.

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
On Windows, a `CreateProcessA` failure with `ERROR_FILE_CORRUPT` (1392) while
starting the compiled runtime displays the recovery guidance in
[First run](../getting-started/first-run.md#recover-a-corrupted-windows-runtime).
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
runtime files. Signed launchers carry a root-signed public-key catalog in a
trailer after the executable image. The C launcher reads that keyring from its
own file, verifies the manifest signature and selected runtime digest, then
starts the runtime. The Python entrypoint independently reads the same launcher
trailer and verifies the signature and every listed file during startup. A present but
invalid signature or a mismatched runtime digest rejects startup. When no
signature is present, the C launcher allows startup and Python warns that the
copy is unsigned; this keeps legacy JSON-manifest bundles usable. The updater
still rejects unsigned remote release ZIPs.

The public keys that sign manifests live in `celune/release_keyring.json`.
Only the Ed25519 keyring root remains pinned in the C and Python verifier
sources; the root signs the catalog before the builder appends it to the
launcher. A missing or invalid trailer rejects a signed bundle. This lets a
future release add a signing key without recompiling a list of keys into both
verifiers. The root private key is required to create the trailer. Previously
published launchers keep their original key lists; releases that must remain
verifiable by those launchers must use a key they already trust.

Prepare the signing keys and configure the signed build from the checkout that
will build the release:

```powershell
uv run python scripts\make_signing_key.py
```

```bash
uv run python scripts/make_signing_key.py
```

The script creates an unencrypted Ed25519 release-key PEM at
`%LOCALAPPDATA%\Celune\keys\release-ed25519-private.pem` on Windows and
`~/.celune/release-signing-key.pem` on Linux. A valid existing release key is
reused; a missing one is generated. The script also prepares the separate root
key at
`%LOCALAPPDATA%\Celune\keys\release-keyring-root.pem` on Windows or
`~/.celune/release-keyring-root.pem` on Linux. If the root key is missing, the
script generates it and updates the public root pin in `celune/constants.py`
and `launcher/release_keys.h`, along with the root entry in
`celune/release_keyring.json`. Existing root keys that match both verifier pins
are reused. If an existing key does not match, the script stops unless
`--trust-existing-root` is supplied; that option updates the checkout's trust
pins to the existing key. A root-key change means launchers compiled with the
previous pin will reject catalogs signed by the new root, so review and commit
the updated public files with the release.

The script restricts generated key-file access to the current account, rejects
destinations inside the checkout, never overwrites an existing file, and does
not print private-key contents. It adds both public keys to
`celune/release_keyring.json`, preserving other catalogued release keys. It then
configures the signed CMake preset with both private-key paths. `--no-configure`
prepares the keys and trust data without running CMake. A cached root path left
over from older configurations is reset to the separate default path when it
matches the release-key path. The release key signs manifests, while the pinned
root signs the public-key catalog trailer.

`--output` selects the release key path and `--root-key` selects the root key
path. Use `--trust-existing-root` only when intentionally changing which root
key the checkout trusts. `CELUNE_REQUIRE_RELEASE_SIGNATURE=1` also requires
signing with the unsigned preset. The CMake cache variables and matching
environment variables select key paths when configuring outside this script.
For example, with keys created using `--no-configure`:

```powershell
cmake --preset signed `
  -DCELUNE_RELEASE_SIGNING_KEY_FILE="C:\path\to\release-ed25519-private.pem" `
  -DCELUNE_RELEASE_ROOT_KEY_FILE="C:\secure\release-keyring-root.pem"
cmake --build --preset signed
```

When using an existing root key at a custom path:

```powershell
uv run python scripts\make_signing_key.py `
  --output "C:\path\to\release-ed25519-private.pem" `
  --root-key "C:\secure\release-keyring-root.pem"
cmake --build --preset signed
```

```bash
uv run python scripts/make_signing_key.py --output "$HOME/keys/release-ed25519-private.pem"
cmake --build --preset signed
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
signature verifies against the root-authenticated keyring appended to the
launcher. It does not mean
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
