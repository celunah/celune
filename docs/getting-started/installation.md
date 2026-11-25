# Installation

This page lists supported environments and system dependencies, then shows how
to install and launch Celune from the source repository.

## Supported environments

Celune supports Python 3.12, 3.13, and 3.14 on Windows and Linux. The core
environment is GPU-oriented: most Persona and TTS features expect an NVIDIA
RTX 30-series-or-newer GPU, while CPU-only operation is supported for the Mini
backend. VRAM presets name the minimum total GPU capacity and reserve 2 GiB for
the system and other GPU users:

| Preset | Total VRAM | Celune budget |
| --- | ---: | ---: |
| `low` | 6 GB | 4 GiB |
| `medium` | 8 GB | 6 GiB |
| `high` | 12 GB | 10 GiB |
| `xhigh` | 16 GB or more | 14 GiB or more |

An unprofiled hardware/model combination is allowed to start with a warning that
it may not work on that hardware. A confirmed profile that exceeds the budget
or current GPU headroom is rejected. Persona and standard agent models require
`high`. Registered smart 8B-tier models require `xhigh`.

The `doctor` command reports the active Python, PyTorch, CUDA, GPU, and system
dependency state. It is the first diagnostic to run when a feature is missing.

## System tools

Install these before using the corresponding features:

- SoX for audio inspection and conversion helpers.
- Rubber Band for high-quality time stretching and pitch shifting. Celune
  falls back to a simpler speed path when it is unavailable.
- OpenRGB, plus a compatible device, for audio-reactive lighting.
- A C/C++ compiler for VoxCPM2 and any backend that builds native extensions.
- A driver that supports the installed PyTorch CUDA build. Celune's runtime
  check accepts CUDA 12.8 to 13.4. The setup helper defaults to CUDA 13.2,
  and CEDTS workers use a separate CUDA 12.8 stack.
- Symbolic-link support on Windows when developing or using workflows that
  create links.

The setup helper can install system tools with `apt` on Debian, Ubuntu, Mint,
and Pop!_OS, `pacman` on Arch, Manjaro, and EndeavourOS, `dnf` on Fedora,
Rocky Linux, and AlmaLinux, and `zypper` on openSUSE. Alpine Linux is supported
experimentally through `apk`, where some required packages may be unavailable.
On Debian-like Linux distributions, `sounddevice` commonly requires PortAudio
development headers such as `portaudio19-dev`. Install the equivalent package
on other distributions.

## Source installation

From a clean checkout:

GitHub's **Download ZIP** archive does not include Git metadata. Source builds
and in-place source updates require a Git checkout with a commit. Celune reports
when that metadata is missing and prints the upstream clone command. Clone the
repository with:

```bash
git clone https://github.com/celunah/celune
```

The project uses `uv` for dependency resolution. Run the setup helper from the
repository root:

```bash
python configure.py
```

The helper installs required system tools, synchronizes the Python environment,
creates Celune's AppData directories, and seeds the default configuration and
voice pack. It selects the recommended `cu132` PyTorch extra on both platforms.
Linux also installs the API, captions, and OpenZL extras. Windows installs the
API extra and leaves captions optional. OpenZL is skipped there because it is
not currently buildable. Choose another PyTorch CUDA index with
`--cuda-index cu128`, `--cuda-index cu130`, or `--cuda-index cu134`.
If all setup criteria are already satisfied, it asks, `You have already
configured Celune. Repair Celune?` Answer `yes` to remove and recreate the
project `.venv` before synchronizing dependencies, or press Enter to leave the
existing installation unchanged. Run the helper with a system Python
interpreter rather than the `.venv` interpreter when repairing, because the
active environment cannot safely remove itself.
The selected CUDA extra includes Silero VAD for AI-assisted live
voice-conversion capture. `cechar4-openzl` adds OpenZL-compressed CECHAR v4/v5
support on Linux.

For a dependency-only manual setup, select one CUDA extra. The `cu132` examples
below use the recommended index:

```bash
uv sync --dev --extra api --extra captions --extra cechar4-openzl --extra cu132
```

```powershell
uv sync --dev --extra api --extra cu132
```

Do not use `--all-extras`, because the CUDA extras are mutually exclusive. Keep
the selected CUDA extra on every `uv sync` command so an exact sync does not
replace the CUDA build with a default PyPI build. A normal `uv run` preserves
the installed CUDA packages because they are not part of the default dependency
set.

Backend-specific dependencies are normally installed by Celune's isolated
backend environment manager. Do not add backend packages to core application
modules merely to make one backend importable.

## Launching

For source development, run the supported entrypoint through the project
environment:

```bash
uv run python main.py
```

The packaged launchers are preferred for compiled builds:

```powershell
.\bin\celune.exe
```

```bash
./bin/celune.AppImage
```

`celune-bin` is the implementation executable used by the launcher. Run the
user-facing `celune` launcher so environment setup, update checks, process
ownership, and shutdown behavior remain intact.

## First model download

The first load downloads the selected model and backend assets from their
configured sources. All launch modes use a Celune-owned Hugging Face cache
under the application data directory by default, including isolated backend
workers. On Windows this is typically
`C:\Users\<user>\AppData\Local\Celune\huggingface\hub`. Explicit
`HF_HOME` or `HF_HUB_CACHE` environment variables override these defaults.
Downloads are reported through Celune's startup progress path rather than raw
model progress bars.

## Optional environment variables

These variables are useful for diagnostics and deployment:

| Variable | Effect |
| --- | --- |
| `CELUNE_LOG_LEVEL` | Initial log level: `info`, `verbose`, or `debug`. |
| `CELUNE_BACKEND` | Overrides the configured backend name. |
| `CELUNE_HEADLESS` | Overrides `headless`. Enabled values start without Textual. |
| `CELUNE_API_TOKEN` | Supplies the API key as a bearer token when config does not contain one. |
| `CELUNE_LAUNCHER` | Marks a launch as owned by the Celune launcher. |
| `CELUNE_OVERRIDE_CELINE_DAY` | Bypasses the name-day startup pause used by the app. |
| `CELUNE_THEME` | Overrides the selected UI theme. |
| `CELUNE_AGENT_FS_TOOLS` | Overrides `agent.fs_tools` for the agent's unsandboxed filesystem and process tools. |

Environment flags are parsed as booleans by the runtime. Configuration-file
values remain the normal way to set persistent behavior.
