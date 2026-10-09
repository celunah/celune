<!-- markdownlint-disable-next-line MD041 -->
![Celune](./resources/branding/celune_wordmark.png "Celune wordmark")

---

![Python](https://img.shields.io/badge/Python-3.12–3.14-cebaff)
![License](https://img.shields.io/badge/License-Apache%202.0-cebaff)
![Platform](https://img.shields.io/badge/Platform-Windows%2FLinux-cebaff)
![VRAM](https://img.shields.io/badge/VRAM-6%20GB–16%20GB+-cebaff)

Celune is a conversational character engine with agentic features.

You can use it to speak, use it to talk, or even use it to manage.

No matter what you want to do, it is there to help you.

It was proudly made in 🇵🇱 for the finest of uses.

## Features

- Real-time buffered speech generation pipeline
- Distinct default voice styles: calm, balanced, bold, upbeat
- Multiple operation modes: speak, converse and agent
- Stable long-form narration with low risk of drifting
- Native audio controls & effects via built-in DSP
- Optimized GPU inference where possible
- Configurable character voices via CEVOICE voice packs
- Characters can respond back to you
- Can voice change into character voices
- Automatic memory saving
- Agentic capabilities with tools

## Operation modes

Celune can operate in three major modes, each meant for different uses.

Set the following value in Celune's configuration to change the operation mode:

```yaml
mode: converse  # speak|converse|agent
```

- `speak` uses only Celune's speech features. It is best suited for performance use.
  It does not use any extra features.
- `converse` uses Persona, allowing you to talk with any characters you've set up with Celune.
- `agent` turns Celune into a conversational local agent, allowing her to perform actions
  on your computer while speaking as needed.

## Note on development

Celune is against the stance of "vibe coding" used in development.

None of the 100,000+ lines of code in Celune were created solely using AI. AI tools (e.g. Codex)
were only used to assist in faster development, iteration and solving issues.

All decisions and implementations were reviewed, validated, and approved by human developers.

Celune never was, and will never become an "AI slop" project.

## Documentation

Visit <https://celune.readthedocs.io/en/latest> to view Celune documentation. They will help you learn to use Celune.

## License note

Celune is licensed under the Apache 2.0 License, but the software may download certain models
from [Hugging Face](https://huggingface.co) that are of varying licenses,
such as [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) (Qwen, etc.), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/deed.en) (Pocket TTS),
[GPL-3.0](https://opensource.org/license/gpl-3.0) (SeedVC),
and MIT (Whisper, Silero VAD, etc.). Users of Celune are expected to read and comply with any applicable
license terms for the models they intend to use.

Please note that Celune 4.3.2 and older are still licensed under the MIT license.

## Voices and samples

Listen to the complete [voice sample gallery](https://celune.readthedocs.io/en/latest/development/samples/), with native audio players for the current and historical recordings. The page also documents the sample text, generation metadata, and model settings.

The current FLAC recordings use the Classic Remastered voice pack. Historical WAV recordings remain available for comparison with earlier voices and backend revisions.

## Voice packs

Celune comes with three default packs, containing both newer and older revisions of her mainline voice.

The packs are:

| Pack               | File                  | Released      | Description                                                                         |
|--------------------|-----------------------|---------------|-------------------------------------------------------------------------------------|
| Classic            | classic.cevoice       | Mar 19, 2026* | Classic Celune voices from her initial inception.                                   |
| Natural            | default_sep19.cevoice | Sep 19, 2026  | Celune's 6-month anniversary voices.                                                |
| Classic Remastered | default.cevoice       | Sep 30, 2026  | New and improved voices that sound natural, and maintain Celune's classic identity. |

<sub><sup>*Based on Celune's voice debut date. Pack support was released May 16, 2026.</sup></sub>

Browse the `demos` directory for demonstration content from the current version of Celune, as well as any past releases.

> [!CAUTION]
> Do not use markup or unknown tags (e.g. `<...>`).
> They may be interpreted as control sequences and cause unexpected vocalizations.
> Refer to the model's known tags before including them.
>
> Do not mix multiple languages in one sentence.
> Keep language boundaries clear and explicit.
>
> Some models may tolerate badly formatted inputs differently, but it is advised to correctly format your text inputs.
>
> **Good:**
>
> ```text
> This is a sentence. This is another sentence. [laughter]
> ```
>
> **Bad:**
>
> ```text
> <think>Thinking text.</think>
> This is a sentence, 中文, 日本語, 한국어.
> ```

Samples were captured directly from Celune's output directory. No extra post-processing was applied.

For details on voice production, check [VOICES.md](./docs/VOICES.md).

> [!NOTE]
> AI-generated voices may occasionally mispronounce words. 
> Minor pronunciation errors are expected, and do not necessarily indicate a software defect.
>
> If a word is mispronounced, try spelling it phonetically to guide the voice.

## System Requirements

Celune requires [Python](https://python.org) 3.12, 3.13 or 3.14.

Celune also depends on external system dependencies that are not available in `pip`:

- **CUDA Toolkit 12.8 (or 13.0)** - only if not using pre-built PyTorch wheels
- **SoX (Sound eXchange)** - required for audio processing
- **Rubber Band library** - required to control Celune's speed
- **OpenRGB** - required to glow compatible devices
- **Symbolic link support** - recommended on Windows for optimal operation
- **C/C++ compiler** - to compile required dependencies for VoxCPM2
- **eSpeak-NG** - for IPA-based forced alignment of captions

Celune requires an RTX 30 series GPU or newer to use most features.

CPU-only execution is supported with Celune Mini and LuxTTS.

Usage of Celune's UI requires an ANSI-capable terminal. Non-compliant terminals can only use the headless (CEF) mode.

The terminal should support True Color, especially when using voice packs that declare new app themes.

Terminals not supporting True Color may look incorrect, as Textual will fall back to a lower color mode.

If Celune looks incorrect while your terminal supports True Color on Linux, run Celune with the following command:

```bash
COLORTERM=truecolor celune
```

If Rubber Band is not installed or fails to run, Celune will speak at normal speed, and speed controls
will be unavailable.

## VRAM presets & requirements

Celune reserves 2 GiB for the system and other GPU users. The presets name the
minimum total GPU capacity; the remaining budget is available to Celune:

| Preset   | Total VRAM    | Allocated budget |
| -------- | ------------: | ---------------: |
| `low`    | 6 GB          | 4 GiB            |
| `medium` | 8 GB          | 6 GiB            |
| `high`   | 12 GB         | 10 GiB           |
| `xhigh`  | 16 GB or more | 14 GiB or more   |

Use of Persona and related features (such as the agent mode) requires the `high` preset, or above.

Large Persona models (~8B parameters) will only run on `xhigh`. 
Persona context is variable depending on Celune's operation mode: 2,048 tokens in basic conversation
mode (`mode: converse`), and 8,192 tokens in agent mode (`mode: agent`).

Combinations of presets and model selections were validated where possible, unchecked configurations
display clear warnings and may fail with OOM errors, use then with caution.

The desired preset may be set in Celune's configuration file. Refer to `default_config.yaml` for details.

---
Performance may be reduced when running GPU intensive applications along with Celune.

Tested on: RTX 5070 (12 GB VRAM)

## Installation

Download and extract the [latest SemVer binary release](https://github.com/celunah/celune/releases/latest) prior to running the below commands in an already
cloned copy of Celune.

Copies obtained via `Download ZIP` will not work.

Alternatively, run `scripts/build_nuitka.ps1` or `scripts/build_nuitka.sh` to build Celune binaries by yourself,
depending on your platform.

Don't have a copy yet? Run the following commands:

```bash
git clone https://github.com/celunah/celune
cd celune
```

Then, follow these steps:

```bash
# Quick setup
python configure.py

# Manual setup
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or on Unix systems:
curl -Ls https://astral.sh/uv/install.sh | sh

# Validate uv works
uv --version

# Expected output:
# uv 0.11.2 (02036a8ba 2026-03-26 x86_64-pc-windows-msvc) (or similar version)

# Manual environment setup
# Linux: uv sync --dev --all-extras
# Windows: uv sync --dev --extra api --extra captions
# The selected backend is installed automatically into Celune's AppData
# environment the first time it is used.

# Run
# Command Prompt users
bin\celune

# PowerShell users
.\bin\celune.exe

# Or on Unix systems:
./bin/celune.AppImage
```

Don't run `celune-bin` manually. The `celune` binary is Celune's main entrypoint.

Both binaries are required for correct operation, `celune-bin` contains core code, while `celune` is the outer launcher.

Celune can also run from other working directories, provided the main binary is installed correctly. The binary must
always be located as part of the cloned repository, as it depends on files contained within it.

### SoX & Rubber Band installation

If SoX & Rubber Band are already installed, you can skip this section.

#### Windows (Scoop)

```bat
REM Install Scoop if you don't already have it
powershell -ExecutionPolicy RemoteSigned -c "irm https://get.scoop.sh | iex"

REM Install SoX
scoop install sox

REM Install Rubber Band
scoop install rubberband
```

#### Linux (Debian/Ubuntu)

```bash
sudo apt install sox rubberband-cli
```

#### Linux (Arch Linux)

```bash
sudo pacman -S sox rubberband
```

#### Validate SoX & Rubber Band are installed

```bash
sox --version

# Expected output:
# sox:      SoX v14.4.2 (or similar version)

rubberband --version

# Expected output:
# 4.0.0 (or similar version)
```

### OpenRGB installation

To install OpenRGB, go to <https://openrgb.org/>, download and install a package appropriate for your platform.

This will allow Celune to glow up your PC as she speaks.

### eSpeak-NG installation

Install a [Windows version](https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi) on your machine,
or use your package manager to get an appropriate Linux version for your distribution.

This will enable captioning and caption display.

### C/C++ compiler setup

Celune's VoxCPM2 backend may require a C/C++ compiler to compile dependencies. To install a suitable compiler,
run one of the following commands:

This is not required to use other backends, but you may need to install dependencies manually.

```bash
# Windows
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Linux (Ubuntu)
sudo apt install build-essential

# Linux (Arch Linux)
sudo pacman -S base-devel
```

### CUDA Toolkit installation

This step can be skipped if you are using pre-built PyTorch wheels.

Download and install CUDA Toolkit 12.8 or 13.0 from NVIDIA:

- <https://developer.nvidia.com/cuda-12-8-0-download-archive>
- <https://developer.nvidia.com/cuda-13-0-0-download-archive>

Make sure to:

- Select the correct OS and version
- Install both **CUDA Toolkit** and **NVIDIA drivers** (if not already installed)

Celune is expected to work with CUDA Toolkit 12.8 or 13.0. Not all backends are compatible with newer versions.

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

## REST API

See [API.md](./docs/API.md) for REST API configuration, authentication, endpoints, and cURL examples.
The API allows programmatic usage of all Celune features. It can be used both as a public and local interface.

## Extensions

Celune comes with extension support, allowing custom code to run with the engine.

Extensions may subscribe at will to certain core events with `@celune.subscribe(...)`.

Celune exposes the following core events to extensions:

- `ready`
- `shutdown`
- `fatal`
- `error`
- `voice_changed`
- `state_changed`
- `generation_start`
- `generation_end`
- `generation_error`
- `audio_start`
- `audio_end`
- `character_changed`
- `character_loaded`
- `character_unloaded`

For example extension usage, check the [example extension](./extensions/test.py).

The aforementioned extension defines a basic usage case for Celune extensions.

## Web UI

Celune exposes a web interface for remote access to Celune. It reuses the Celune API commands to provide
an interface for control.
It can be accessed via `/ui` on Celune's exposed API URL.

## Shortlink

Need a quick and easy shortlink to spread Celune to the public? [Copy link](https://go.lunah.site/celune)

---

<p align="right">
  <i>"Your voice, your way."&#x2000;</i>
  <img src="./resources/branding/celune_88x31_206.png" alt="Celune 88x31 badge" title="enlightened by Celune" width="88" height="31">
</p>

<img src="https://lunah.site/common/signature.png" alt="- l u n a h - signature footer" title="signature project of - l u n a h -" width="2048" height="128">
