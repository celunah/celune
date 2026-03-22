# Celune
Celune is a real-time AI TTS engine focused on natural voice delivery, low-latency playback, and distinct voice styles.

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
