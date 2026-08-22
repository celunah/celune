# Requirements and installation

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

On Debian-like Linux distributions, the project commonly needs PortAudio
development headers for `sounddevice`. The CI workflow installs
`portaudio19-dev`; other distributions should use their equivalent package.

## Source installation

From a clean checkout:

```bash
uv sync --all-extras --dev
```

The project uses `uv` for dependency resolution. The optional `api` extra adds
FastAPI, Uvicorn, multipart upload support, Pydantic, and Gradio. The optional
`live-vc-ai` extra adds Silero VAD for AI-assisted live voice-conversion capture;
`openzl` adds OpenZL-compressed CECHAR v4 support.

The interactive setup helper is also available:

```bash
python setup.py
```

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
configured sources. Compiled launches use a Celune-owned Hugging Face cache
under the application data directory; source-tree launches preserve the host
Hugging Face cache unless the environment explicitly overrides it. Downloads
are reported through Celune's startup progress path rather than raw model
progress bars.

## Optional environment variables

These variables are useful for diagnostics and deployment:

| Variable | Effect |
| --- | --- |
| `CELUNE_LOG_LEVEL` | Initial log level: `info`, `verbose`, or `debug`. |
| `CELUNE_BACKEND` | Overrides the configured backend name. |
| `CELUNE_HEADLESS` | Starts without the Textual UI. |
| `CELUNE_API_TOKEN` | Supplies the API bearer token when config does not contain one. |
| `CELUNE_LAUNCHER` | Marks a launch as owned by the Celune launcher. |
| `CELUNE_OVERRIDE_CELINE_DAY` | Bypasses the name-day startup pause used by the app. |
| `CELUNE_THEME` | Overrides the selected UI theme. |
| `CELUNE_LOCAL_MANAGEMENT` | Enables the agent's unsandboxed local-management tools; keep disabled unless explicitly needed. |

Environment flags are parsed as booleans by the runtime. Configuration-file
values remain the normal way to set persistent behavior.
