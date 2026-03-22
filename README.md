# Celune
Celune is a real-time AI TTS engine focused on natural voice delivery, low-latency playback, and distinct voice styles.

It has been designed for real-time performance on consumer GPUs.

## Features

- Real-time speech generation pipeline
- Distinct voice styles (Calm, Neutral, Energetic)
- Stable long-form narration without drift
- Source-level audio control (no post-processing)
- GPU-accelerated inference

## Voices & samples
Each voice is demonstrated using a short introduction and a longer narration sample to showcase consistency, pacing, and expressiveness.

| Voice     | Intro | Narration |
|-----------|-------|-----------|
| Neutral   | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/neutral_sc.wav) | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/neutral_lc.wav) |
| Calm      | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/calm_sc.wav)    | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/calm_lc.wav)    |
| Energetic | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/energetic_sc.wav) | [▶️ Play](https://gabalpha.github.io/read-audio/?p=https://raw.githubusercontent.com/celunah/celune/main/demos/energetic_lc.wav) |

Samples were captured directly from the live TTS pipeline with no post-processing applied (only silence trimming).

For details on voice production, check the VOICES.md file.

## Requirements

**GPU (CUDA):**
- Minimum: 6 GB VRAM (e.g. GTX 1660 / RTX 2060)
- Recommended: 8 GB+ VRAM (e.g. RTX 3060 or better)

Celune’s core model fits within ~4 GB VRAM, but additional memory is required for runtime overhead, buffering, and stable real-time playback.

Tested on: RTX 5070 (12 GB VRAM)

## Installation

```bash
# Create environment
python -m venv .venv
source ./venv/bin/activate

# Or on Windows:
.venv\Scripts\activate

# Install packages
pip install -U -r requirements.txt

# Run
python main.py

# Or on Unix systems:
chmod +x main.py
./main.py
```

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
