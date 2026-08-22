# Backends and environments

This page describes how Celune installs, isolates, launches, and extends its
text-to-speech and voice-conversion backends.

## Manifest-driven installation

The canonical backend registry is `celune.backends.environment.BACKEND_MANIFESTS`.
Each manifest includes the backend ID, kind (`tts` or `vc`), requirements,
worker module/class, optional Python version, PyTorch indexes, and a revision
used for its fingerprint.

With `isolated_backends: true`, environments live below the Celune application
data directory in an `environments/` tree. The manager uses `uv`, an exclusive
lock, and a manifest fingerprint. A ready environment contains a usable
interpreter and `manifest.json`; changing requirements or platform details
creates a new fingerprint rather than silently reusing an incompatible one.

## Registered manifests

| ID | Kind | Worker | Extra requirements |
| --- | --- | --- | --- |
| `mini` | TTS | `celune.backends.tts.mini:Mini` | `pocket-tts>=2.1.0` |
| `qwen3` | TTS | `celune.backends.tts.qwen3:Qwen3` | `faster-qwen3-tts>=0.2.4` |
| `dotstts` | TTS | `celune.backends.tts.dotstts:DotsTtsMF` | Celune's `dots.tts` fork; Python 3.12. |
| `voxcpm2` | TTS | `celune.backends.tts.voxcpm2:VoxCPM2` | `voxcpm>=2.0.0`; Python 3.12. |
| `gpt-sovits` | TTS | `celune.backends.tts.gpt_sovits:GPTSoVITS` | GPT-SoVITS family dependencies. |
| `seed-vc` | VC | `celune.backends.vc.seedvc:CeluneSeedVCBackend` | Celune's Seed-VC fork. |

Workers share a compatibility baseline containing Hugging Face Hub and
`hf-xet`, Transformers below 5 in the worker environment, Lingua, librosa,
llvmlite, NumPy/Numba, Pillow, platformdirs, psutil, sounddevice, soundfile,
and Zstandard, plus the CEDTS-compatible PyTorch 2.11 CUDA 12.8 worker stack.
The core project itself currently resolves its own CUDA 13.0 stack; the two
environments are intentionally not the same lockfile.

## Backend behavior

### Qwen3

`qwen3` streams at 12.5 chunks per second and supports `zh-cn`, `en`, `ja`,
`ko`, `de`, `fr`, `ru`, `pt`, `es`, and `it`. Medium and larger VRAM presets use
the 1.7B clone model; low VRAM can select the 0.6B model. It reads a CEVOICE
reference WAV and per-voice transcript. `qwen3_x_vector_only` favors identity
stability over full reference expressiveness.

### Mini

`mini` adapts Pocket TTS, streams at 12.5 chunks per second, and supports
English, French, German, Italian, Portuguese, and Spanish. It is the supported
CPU-friendly path and still uses pack reference data for cloning.

### VoxCPM2

`voxcpm2` streams at 6.25 chunks per second and covers the multilingual list
implemented by the adapter, including Polish, English, Chinese, Japanese,
Korean, and many European and Southeast Asian languages. It uses reference WAV
plus a positive per-voice `cfg_scale`. Native build tooling may be required by
its dependencies.

### dots.tts MF

`dotstts` is a 6.25-chunk-per-second diffusion backend using Celune's fork of
the upstream package. Use the fork declared by the manifest; the upstream
package can carry incompatible build requirements.

### GPT-SoVITS

`gpt-sovits` supports the v2 Pro/ProPlus, v4, and v3 family order implemented by
the current adapter, with language support for Chinese, English, Japanese,
Korean, and Cantonese. Its source/runtime is installed under Celune's runtime
data directory and it can use a custom Text2Semantic checkpoint. It usually
needs a reference of at least three seconds and may show accent drift; v4 has
the least drift in the current implementation. The output is normalized into
Celune's playback format.

## Adding a backend

1. Implement the existing TTS or VC base contract.
2. Add a manifest with all worker-only requirements and the worker entrypoint.
3. Implement the CEDTS operation/capability behavior instead of importing the
   package into the core path.
4. Declare languages, chunk rate, voice/reference requirements, and failure
   behavior in the adapter.
5. Add focused tests for loading, streaming, cancellation, normalization, and
   reload rollback.
6. Update this page and the relevant user-facing capability table immediately.

The core must remain importable without the backend package installed.
