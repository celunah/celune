# Installation

This page lists supported environments and system dependencies, then shows how
to install and launch Celune from the source repository.

## Supported environments

Celune supports Python 3.12, 3.13, and 3.14 on Windows and Linux. The core
environment is GPU-oriented: most Persona and TTS features expect an NVIDIA
RTX 30-series-or-newer GPU, while CPU-only operation is supported for the Mini
backend. VRAM presets are approximately:

| Preset | Target VRAM |
| --- | ---: |
| `low` | 6 GB |
| `medium` | 8 GB |
| `high` | 12 GB |
| `xhigh` | 16 GB or more |

The `doctor` command reports the active Python, PyTorch, CUDA, GPU, and system
dependency state. It is the first diagnostic to run when a feature is missing.

## System tools

Install these before using the corresponding features:

- SoX for audio inspection and conversion helpers.
- Rubber Band for high-quality time stretching and pitch shifting. Celune
  falls back to a simpler speed path when it is unavailable.
- OpenRGB, plus a compatible device, for audio-reactive lighting.
- A C/C++ compiler for VoxCPM2 and any backend that builds native extensions.
- CUDA Toolkit 12.8-compatible runtime components for the supported PyTorch
  CUDA build. The exact driver must still support the installed CUDA runtime.
- Symbolic-link support on Windows when developing or using workflows that
  create links.

The setup helper can install system tools with `apt` on Debian, Ubuntu, Mint,
and Pop!_OS; `pacman` on Arch, Manjaro, and EndeavourOS; `dnf` on Fedora,
Rocky Linux, and AlmaLinux; and `zypper` on openSUSE. Alpine Linux is supported
experimentally through `apk`; some packages may potentially be unsupported
there. On Debian-like Linux distributions, the project commonly needs
PortAudio development headers for `sounddevice`. The CI workflow installs
`portaudio19-dev`; other distributions should use their equivalent package.

## Source installation

From a clean checkout:

The project uses `uv` for dependency resolution. Run the setup helper from the
repository root:

```bash
python configure.py
```

The helper installs required system tools, synchronizes the Python environment,
creates Celune's AppData directories, and seeds the default configuration and
voice pack. On Linux it uses `uv sync --dev --all-extras`; on Windows it uses
`uv sync --dev --extra api` because OpenZL is not currently buildable there.
If all setup criteria are already satisfied, it asks, `You have already
configured Celune. Repair Celune?`; answer `yes` to remove and recreate the
project `.venv` before synchronizing dependencies, or press Enter to leave the
existing installation unchanged. Run the helper with a system Python
interpreter rather than the `.venv` interpreter when repairing, because the
active environment cannot safely remove itself.
The optional `live-vc-ai` extra adds Silero VAD for AI-assisted live
voice-conversion capture; `openzl` adds OpenZL-compressed CECHAR v4 support on
Linux.

For a dependency-only manual setup, use `uv sync --dev --all-extras` on Linux
or `uv sync --dev --extra api` on Windows.

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
| `CELUNE_HEADLESS` | Overrides `headless`; enabled values start without Textual. |
| `CELUNE_API_TOKEN` | Supplies the API bearer token when config does not contain one. |
| `CELUNE_LAUNCHER` | Marks a launch as owned by the Celune launcher. |
| `CELUNE_OVERRIDE_CELINE_DAY` | Bypasses the name-day startup pause used by the app. |
| `CELUNE_THEME` | Overrides the selected UI theme. |
| `CELUNE_AGENT_FS_TOOLS` | Overrides `agent.fs_tools` for the agent's unsandboxed filesystem and process tools. |

Environment flags are parsed as booleans by the runtime. Configuration-file
values remain the normal way to set persistent behavior.
