# Celune
Celune is a real-time AI TTS engine focused on natural voice delivery, low-latency playback, and distinct voice styles.

It has been designed for real-time performance on consumer GPUs.

## Features

- Real-time speech generation pipeline
- Distinct voice styles (Calm, Balanced, Bold, Upbeat)
- Stable long-form narration without drift
- Source-level audio control (no post-processing)
- GPU-accelerated inference

## Voices & samples
Each voice is demonstrated using a short introduction and a longer narration sample to showcase consistency, pacing, and expressiveness.

| Voice        | Intro | Narration |
|--------------|-------|-----------|
| Balanced     | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/balanced_sc.wav) | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/balanced_lc.wav) |
| Calm         | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/calm_sc.wav)    | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/calm_lc.wav)    |
| Bold         | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/bold_sc.wav) | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/bold_lc.wav) |

> [!CAUTION]
> Do not use markup or tags (e.g. `<...>`).  
> They may be interpreted as control sequences and break speech output.
>
> Do not mix multiple languages in one sentence.  
> Keep language boundaries clear and explicit.
>
> **Good:**
> ```
> This is a sentence. This is another sentence.
> ```
>
> **Bad:**
> ```
> <think>Thinking text.</think>
> This is a sentence, 中文, 日本語, 한국어.
> ```

Samples were captured from Celune's output directory. No extra post-processing was applied.

For details on voice production, check the VOICES.md file.

## System Requirements

Celune requires [Python](https://python.org) 3.12 or 3.13.

Celune also depends on external system tools that are not installed via `pip`:

- **NVIDIA GPU with CUDA support**
- **CUDA Toolkit 12.8**
- **SoX (Sound eXchange)** - required for audio processing
- **Rubber Band library** - required to control Celune's speed
- **Symbolic link support** (recommended on Windows)

Celune requires CUDA for GPU acceleration. CPU-only execution is not supported.

If Rubber Band is not installed, Celune will speak at normal speed, and speed controls will be unavailable.

## GPU requirements

**GPU (CUDA):**
- Minimum: 6 GB VRAM (e.g. GTX 1660 / RTX 2060)
- Recommended: 8 GB+ VRAM (e.g. RTX 3060 or better)

Celune’s core model fits within ~4 GB VRAM, but additional memory is required for runtime overhead, buffering, stable real-time playback, and input normalization.

Tested on: RTX 5070 (12 GB VRAM)

## Installation

```bash
# Download Celune
git clone https://github.com/celunah/celune
cd celune

# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or on Unix systems:
curl -Ls https://astral.sh/uv/install.sh | sh

# Validate uv works
uv --version

# Expected output:
# uv 0.11.2 (02036a8ba 2026-03-26 x86_64-pc-windows-msvc) (or similar version)

# Create environment
# Celune expects Python 3.12 or 3.13
uv sync

# Run
celune

# Or on Unix systems:
./celune.AppImage
```

You can also open Celune from within your desktop by running the aforementioned executables. They are usable as an entry point.

### SoX & Rubber Band installation
If SoX & Rubber Band are already installed, you can skip this section.

**Windows (Scoop)**
```powershell
# Install Scoop if you don't already have it
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm https://get.scoop.sh | iex

# Install SoX
scoop install sox

# Install Rubber Band
scoop install rubberband
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt install sox rubberband-cli
```

**Linux (Arch Linux)**
```bash
sudo pacman -S sox rubberband
```

**macOS (Homebrew)**
```bash
# Install Homebrew if you don't already have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SoX
brew install sox rubberband
```

**Validate SoX & Rubber Band are installed**
```bash
sox --version

# Expected output:
# sox:      SoX v14.4.2 (or similar version)

rubberband --version

# Expected output:
# 4.0.0 (or similar version)
```

### CUDA 12.8 installation
Download and install CUDA Toolkit 12.8 from NVIDIA:

https://developer.nvidia.com/cuda-12-8-0-download-archive

Make sure to:
- Select the correct OS and version
- Install both **CUDA Toolkit** and **NVIDIA drivers** (if not already installed)

After installation, verify CUDA:

```bash
nvidia-smi
```

You should see your GPU listed along with driver information.

### Symbolic links (Windows)

Symbolic links are recommended for best performance and compatibility.

To enable them:
- Enable **Developer Mode** in Windows settings  
  (Settings → Privacy & Security → For Developers)

Without this, Celune may require elevated permissions or fall back to slower behavior.

# Screenshots
These screenshots show Celune's user interface.

### Before init
[![Before init](./demos/state_before_init.png)](./demos/state_before_init.png)

### Ready
[![Ready](./demos/state_ready.png)](./demos/state_ready.png)

### Talking
[![Talking](./demos/state_talking.png)](./demos/state_talking.png)

### Change voice
[![Change voice](./demos/state_change_voice.png)](./demos/state_change_voice.png)

### Commands
[![Commands](./demos/state_commands.png)](./demos/state_commands.png)

### Extension autostart
[![Extension autostart](./demos/state_extension_autostart.png)](./demos/state_extension_autostart.png)

### Extension invoke
[![Extension invoke](./demos/state_extension_invoke.png)](./demos/state_extension_invoke.png)

