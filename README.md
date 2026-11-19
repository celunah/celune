<!-- markdownlint-disable-next-line MD041 -->
![Celune](./resources/branding/celune_wordmark.png "Celune wordmark")

---

![Python](https://img.shields.io/badge/Python-3.12%E2%80%913.14-cebaff)
![License](https://img.shields.io/badge/License-Apache%202.0-cebaff)
![Platform](https://img.shields.io/badge/Platform-Windows%2FLinux-cebaff)
![VRAM](https://img.shields.io/badge/VRAM-6%20GB%E2%80%9116%20GB+-cebaff)

Celune is a conversational character engine with agentic features.

She can speak.

She can sing.

She can talk.

She can work.

She can also become anyone.

Proudly made in :poland: Poland, for the finest of use cases.

## Features

- Low-latency, real-time audio pipeline
- Seven default built-in voice styles
- Stable long-form narration with minimal identity drift
- Native audio controls & DSP effects
- Optimized operation lean on your CPU
- Fit for many use cases
- Configurable character voices and identities with voice packs
- Characters can converse with you
- Can voice change into character voices
- Characters can both talk and sing
- Automatic memory saving
- Mood controls for expressive conversation
- Agentic capabilities with tools

## Operation modes

Celune can operate in three major modes, each meant for different uses.

Set the following value in Celune's configuration to change the operation mode:

```yaml
mode: converse  # performance|converse|agent
```

- `performance` only uses the TTS. It is best suited for performance use.
  It does not enable additional features.
- `converse` uses Persona, allowing you to talk with any characters you've set up with Celune.
- `agent` turns Celune into a conversational local agent, allowing her to perform actions
  on your computer while speaking as needed.

## Note on development

Celune is against the stance of "vibe coding" used in development.

None of the 100,000+ lines of code in Celune were created solely using AI. AI tools (such as Claude, Codex, etc.)
were only used to assist in faster development, iteration and solving issues.

All decisions and implementations were reviewed, validated, and approved by human developers.

Celune never was, and will never become an "AI slop" project.

<sub><sup><i>\*laugh\* Maybe I should call it "human slop".</i></sup></sub>

## Documentation

See the [Celune documentation](https://celune.readthedocs.io/en/latest) for setup instructions and user guides.

## License note

Celune is licensed under the Apache 2.0 License, but the software may download certain models
from [Hugging Face](https://huggingface.co) that are of varying licenses,
such as [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) (Qwen, etc.), 
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/deed.en) (Pocket TTS),
[GPL-3.0](https://opensource.org/license/gpl-3.0) (SeedVC),
and [MIT](https://opensource.org/license/mit) (Whisper, Silero VAD, etc.).

Users of Celune will be presented with a license acknowledgement prompt for non-permissive licenses of certain models.

Only Celune 5.0.0 and later are Apache 2.0 licensed. Older releases remain MIT licensed.

## Voices and samples

Celune's voice samples are available in the
[voice sample gallery](https://celune.readthedocs.io/en/latest/development/samples/), including both new 
and old recordings of her voice, and details on how it was created.

The current examples are recorded with Celune's Classic Remastered voice pack, with old recordings still 
available for reference.

## Voice packs

Celune comes with four default packs, allowing you to use all features from the get-go.

No packs need to be installed separately. More packs can be installed or created as needed.
Ask your coding agent for details.

The default packs are:

| Pack               | File                  | Released      | Description                                                                         |
|--------------------|-----------------------|---------------|-------------------------------------------------------------------------------------|
| Classic            | classic.cevoice       | Mar 19, 2026* | Classic Celune voices from her initial inception.                                   |
| Natural            | default_sep19.cevoice | Sep 19, 2026  | Celune's 6-month anniversary voices.                                                |
| Classic Remastered | default.cevoice       | Sep 30, 2026  | New and improved voices that sound natural, and maintain Celune's classic identity. |
| Nol                | nol.cevoice           | Oct 26, 2026  | (NEW IN 5.1.0) Celune's companion character and his unique NYC-style voices.        |

<sub><sup>*Based on Celune's voice debut date. Pack support was released May 16, 2026.</sup></sub>

Browse the `demos` directory for demonstration content from the current version of Celune, as well as 
any past releases.

> [!CAUTION]
> Do not use markup or unknown tags (e.g. `<...>`).
> They may be interpreted as control sequences and cause unexpected vocalizations.
> Refer to the model's known tags before including them.
>
> Do not mix multiple languages in one sentence.
> Keep language boundaries clear and explicit.
>
> Some models may tolerate badly formatted inputs differently, but it is advised to correctly 
> format your text inputs.
>
> **Good:**
>
> ```text
> This is a sentence. This is another sentence.  // correctly bounded text
> [laughter]  // valid tag (in some backends)
> ```
>
> **Bad:**
>
> ```text
> <think>Thinking text.</think>  // invalid tag
> This is a sentence, 中文, 日本語, 한국어.  // ambiguous language boundary
> ```

Samples were captured raw from Celune's output directory, and were not modified.

For details on voice production, check [VOICES.md](./docs/VOICES.md).

> [!NOTE]
> AI-generated voices may occasionally mispronounce words.
> Minor pronunciation errors are expected, and do not necessarily indicate a software defect.
>
> If a word is mispronounced, try spelling it phonetically to guide the voice.

## System Requirements

Celune requires Python 3.12, 3.13, or 3.14. Most features require an NVIDIA RTX 30-series GPU or newer. 
CPU-only operation is supported by compatible CPU backends. Lack of a GPU will otherwise result in a red screen.

Celune also uses system dependencies for specific features:

| Dependency                 | Needed for                                                                        |
| -------------------------- | --------------------------------------------------------------------------------- |
| SoX                        | Audio processing                                                                  |
| Rubber Band                | Speed control: without it, Celune speaks at normal speed                          |
| CUDA Toolkit 12.8 or newer | Building PyTorch from source, not needed with compatible pre-built PyTorch wheels |
| C/C++ compiler             | Building dependencies for the VoxCPM2 backend                                     |
| eSpeak-NG                  | Captioning and caption display                                                    |
| OpenRGB                    | Lighting compatible devices                                                       |
| Symbolic link support      | Recommended on Windows for best performance and compatibility                     |

The UI requires an ANSI-capable terminal. Terminals without ANSI support can use headless mode instead. 
True Color is recommended: without it, themes may look different.

On Linux, if your terminal supports True Color but Celune does not detect it, run Celune with your path:

```bash
# Replace the placeholder with the actual path
COLORTERM=truecolor /path/to/celune
```

If Rubber Band is not installed or fails to run, Celune will speak at normal speed, and speed controls
will be unavailable.

## VRAM presets & requirements

The engine reserves a 2 GiB margin from total VRAM to protect the system and
other applications from memory exhaustion. The table shows the budget available
for model loading and inference.

| Preset   | Total VRAM     | Allocated budget |
| -------- | -------------- | ---------------- |
| `low`    | 6 GiB          | 4 GiB            |
| `medium` | 8 GiB          | 6 GiB            |
| `high`   | 12 GiB         | 10 GiB           |
| `xhigh`  | 16 GiB or more | 14 GiB or more   |

Use of Persona and related features (such as the agent mode) requires the `high` preset, or above.

Large Persona models (~8B parameters) will only run on `xhigh`.
Persona context is variable depending on Celune's operation mode: 2,048 tokens in basic conversation
mode (`mode: converse`), and 8,192 tokens in agent mode (`mode: agent`).

The context size is configurable and can be increased or reduced as needed, and it will be automatically compacted
to prevent context memory exhaustion.

Combinations of presets and model selections were validated where possible, unchecked configurations
display clear warnings and may fail with OOM errors, use them with caution.

The desired preset may be set in Celune's configuration file. Refer to `default_config.yaml` for details.

---
Performance may be reduced when running GPU intensive applications along with Celune.
Automatic resource management may reduce the impact on your GPU, models will be unloaded when Celune is not in use.

Tested on: RTX 5070 (12 GB VRAM)

## Installation

### Run a released build

Download and extract the [latest release](https://github.com/celunah/celune/releases/latest). 
Keep the launcher and `celune-bin` together in the extracted application directory, as it depends on
files in the Celune repository.

Set up the environment:

```bash
python configure.py
```

Then run Celune from the application directory:

```bat
REM Command Prompt
bin\celune

REM PowerShell
.\bin\celune.exe
```
```bash
# Linux
./bin/celune.AppImage
```

### Build from source

Clone the repository:

```bash
git clone https://github.com/celunah/celune
cd celune
```

Install `uv`, then set up the development environment:

```bash
uv sync --dev --all-extras
```

On Windows, OpenZL may fail to compile. Use the following command instead:

```powershell
uv sync --dev --extra api --extra captions
```

Build the launcher:

```bash
# Unsigned build, without tamper protection
cmake --preset unsigned
cmake --build --preset unsigned

# Signed build, with tamper protection
cmake --preset signed
cmake --build --preset signed
```

Backends will have their environments automatically provisioned by Celune the first time they are used.
Run `celune doctor` to check the core environment. Backend environments are not inspected by this checker.

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

#### Linux

```bash
# Ubuntu
sudo apt install sox rubberband-cli

# Arch Linux
sudo pacman -S sox rubberband
```

#### Validate SoX & Rubber Band are installed

```bash
sox --version

# Expected output:
# sox:      SoX v14.4.2 (or newer)

rubberband --version

# Expected output:
# 4.0.0 (or newer)
```

### OpenRGB installation

To install OpenRGB, go to <https://openrgb.org/>, download and install a package appropriate for your platform.

Arch Linux users can obtain OpenRGB from Pacman:

```bash
sudo pacman -S openrgb
```

This will allow Celune to glow up your PC as she speaks.

### eSpeak-NG installation

Install a [Windows version](https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi) on 
your machine, or use your Linux package manager, for example:

```bash
# Ubuntu
sudo apt install espeak-ng

# Arch Linux
sudo pacman -S espeak-ng
```

This will enable captioning and caption display.

### C/C++ compiler setup

Celune's VoxCPM2 backend may require a C/C++ compiler to compile dependencies. To install a suitable compiler,
run one of the following commands:

```bash
# Windows
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# Linux (Ubuntu)
sudo apt install build-essential

# Linux (Arch Linux)
sudo pacman -S base-devel
```

This is not required to use other backends. GGUF inference may or may not need some dependencies.
Otherwise, you may need to install some dependencies manually.

### CUDA Toolkit installation

This step can be skipped if you are using pre-built PyTorch wheels.

Download and install CUDA Toolkit 12.8 or newer: <https://developer.nvidia.com/cuda-toolkit-archive>

Older toolkits may not support every CUDA capability Celune supports, and are not recommended.

Make sure to:

- Select the correct OS and version
- Install both **CUDA Toolkit** and **NVIDIA drivers** (if not already installed)

Note: CUDA Toolkit is not required to run Celune. Not having it will not cause any issues.

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

Without this, Celune may require elevated permissions to run, or use fallback workarounds.

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

For an example, see the [Lua extension](./extensions/test.lua) or the
[Python extension](./extensions/test.py).

The aforementioned extension defines a basic usage case for Celune extensions.

Celune extensions can be made in either Python or Lua. 

Lua extensions use Celune's singleton objects and do not require additional imports to run. 
They inherit the same feature set from Python extensions.

Python extensions can use other third-party modules for deeper integrations with Celune.

## Web UI

The application exposes a web interface for remote access. The WebUI reuses the
application's API commands for control and is available at `/ui/` on the
application's API URL: `http://localhost:2060/` by default.

On public Celune APIs, using the web interface requires you to enter the API
token before you can use it.

## Shortlink

Need a quick and easy shortlink to spread Celune to the public? [Copy link](https://go.lunah.site/celune)

---

<p align="right">
  <i>"Your voice, your way."&#x2000;</i>
  <img src="./resources/branding/celune_88x31_206.png" alt="Celune 88x31 badge" title="enlightened by Celune" width="88" height="31">
</p>

<img src="https://lunah.site/common/signature.png" alt="- l u n a h - signature footer" title="signature project of - l u n a h -" width="2048" height="128">
