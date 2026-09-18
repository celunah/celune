# Backends

This page describes how Celune resolves, launches, and extends its
text-to-speech and voice-conversion backends.

## Application environment

Normal backend dependencies belong to Celune's configured application
environment. The resolver imports the selected backend lazily, so
backend-specific packages must not be imported by core modules at startup.
Explicit CEDTS worker callers may still use the manifest-backed environment
manager; normal application configuration does not select that path.

## Registered manifests

| ID | Kind | Worker | Extra requirements |
| --- | --- | --- | --- |
| `mini` | TTS | `celune.backends.tts.mini:Mini` | `pocket-tts>=2.1.0` |
| `qwen3` | TTS | `celune.backends.tts.qwen3:Qwen3` | `faster-qwen3-tts>=0.2.4` |
| `fireredtts3` | TTS | `celune.backends.tts.fireredtts3:FireRedTTS3` | FireRedTTS3 source, Transformers 5.6.2, and TorchCodec 0.16.0; BF16 transformer path with PyTorch SDPA and latent streaming. |
| `dotstts` | TTS | `celune.backends.tts.dotstts:DotsTtsMF` | Celune's `dots.tts` fork; Python 3.12. |
| `voxcpm2` | TTS | `celune.backends.tts.voxcpm2:VoxCPM2` | `voxcpm>=2.0.0`; Python 3.12. |
| `luxtts` | TTS | `celune.backends.tts.luxtts:LuxTTS` | CPU-only ONNX LuxTTS worker, its ZipVoice/LinaCodec VCS dependencies, and English prompt transcription. |
| `seed-vc` | VC | `celune.backends.vc.seedvc:CeluneSeedVCBackend` | Celune's Seed-VC fork. |

Most workers share a compatibility baseline containing Hugging Face Hub and
`hf-xet`, Transformers below 5 in the worker environment, Lingua, librosa,
llvmlite, NumPy/Numba, Pillow, platformdirs, psutil, sounddevice, soundfile,
and Zstandard, plus the CEDTS-compatible PyTorch 2.11 CUDA 12.8 worker stack.
LuxTTS uses that same pinned PyTorch stack while its inference runtime remains
configured for `device="cpu"` and uses CPU ONNX Runtime, so selecting it does
not require CUDA hardware. Its model and reference implementation are
[YatharthS/LuxTTS](https://huggingface.co/YatharthS/LuxTTS) and
[LuxTTS](https://github.com/ysharma3501/LuxTTS); those third-party assets retain
their own Apache-2.0 licensing.
FireRedTTS3 is intentionally separate from that Hugging Face portion: its
manifest uses `huggingface-hub>=1.5.0,<2.0.0` and `transformers==5.6.2`, which
cannot be resolved alongside the shared Hub-below-1 and Transformers-below-5
constraints. The core project itself currently resolves its own CUDA 13.0
stack; the core and backend environments are intentionally not the same
lockfile.

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

### FireRedTTS3

`fireredtts3` adapts the FireRedTTS3 base model for zero-shot voice cloning
across 24 languages and 21 Chinese dialect tags. It consumes the active voice's
reference WAV and exact `reference_text`, then streams progressive 24 kHz
audio chunks to the CEDTS worker. FireRed's autoregressive core emits four
RedAE frames per generation step; Celune keeps the RedAE decoder's Qwen3 KV
cache and incrementally overlap-adds its ISTFT frames, forwarding each newly
stable audio segment without waiting for the final waveform. Celune loads the
RedAE and transformer checkpoints directly in BF16 before moving them to CUDA
and runs the FireRed
generation path under BF16 autocast so its FP32 prompt tensors remain
compatible with the loaded model. During construction,
every FireRed Qwen configuration that requests `flash_attention_2` is replaced
with PyTorch SDPA. The backend reports model-construction stages through the
CEDTS progress and log callbacks, so a blocking load remains observable. The
source and Transformers modules are primed before the worker announces
readiness; this keeps their native SciPy imports out of the CEDTS request
thread on Windows without loading model weights during worker startup. The
official source-only repository is downloaded into Celune runtime data on first
use, and the model snapshot is cached through Hugging Face Hub. This backend
therefore does not build or require the unsupported `flash-attn` package.
Its redundant upstream text-front-end completion notice is filtered through
Celune's shared runtime log suppression list; fallback warnings and failures
remain visible.

### VoxCPM2

`voxcpm2` streams at 6.25 chunks per second and covers the multilingual list
implemented by the adapter, including Polish, English, Chinese, Japanese,
Korean, and many European and Southeast Asian languages. It uses reference WAV
plus a positive per-voice `cfg_scale`. Native build tooling may be required by
its dependencies.

### dots.tts MF

`dotstts` is a 6.25-chunk-per-second diffusion backend using Celune's fork of
the upstream package. It receives the complete reference WAV because the
runtime identifies and removes its prompt span before yielding generated audio;
truncating the WAV while keeping its full transcript can make the prompt tail
appear in the response. Use the fork declared by the manifest; the upstream
package can carry incompatible build requirements.

### LuxTTS

`luxtts` adapts the [LuxTTS](https://github.com/ysharma3501/LuxTTS) CPU path
from the [YatharthS/LuxTTS model card](https://huggingface.co/YatharthS/LuxTTS).
It supports English voice cloning, uses the active voice's reference WAV, and
has LuxTTS transcribe the prompt internally. Celune supplies a five-second
prompt window, requests the native 48 kHz waveform path, and emits one complete
waveform per request. The adapter adds the decoder look-ahead frames required by
Vocos, corrects the CPU ONNX duration ratio so short responses retain generated
frames after prompt removal, adds a short reference boundary to prevent prompt
tail leakage, uses that boundary's silent acoustic feature for decoder context
instead of repeating voiced output, and removes only the final two decoder hops
so the last phoneme is not cut off.
Empty or too-short acoustic output becomes a user-facing short-input warning.
The backend runs with `device="cpu"` and two ONNX
threads by default; its isolated manifest uses the shared PyTorch 2.11 CUDA
12.8 stack and adds the upstream Piper wheel page as a `find-links` source so
the environment can resolve its phonemizer dependency. Its installer also
passes uv's `--no-sources` option so LinaCodec cannot redirect PyTorch to its
CUDA 12.6 source override. Preloading also
caches the `openai/whisper-tiny` transcriber required by the upstream CPU
constructor before Celune enables Hugging Face offline mode for model loading.
The worker imports the LuxTTS runtime during its handshake so native runtime
imports do not occur inside a request thread. Celune's startup warm-up uses a
full sentence because the upstream LuxTTS vocoder cannot decode the previous
one-character warm-up input. If a backend raises during an utterance, Celune
reports the generation error and releases the complete speech pipeline lease,
including its typed TTS, speech-queue, and playback ownership, so a later
utterance can still be admitted. The known LuxTTS short-input vocoder shape and
empty-reduction failures return Celune to idle and are presented as a warning
asking the user to enter a longer utterance; other generation failures retain
the normal error treatment.

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

When a backend candidate fails during a hot reload, Celune reports the detailed
failure to the log, restores the previous backend when possible, and then
invokes the app-facing error callback with the localized reload failure. The
UI can therefore show the failure after rollback instead of silently returning
to an idle-looking screen. Backend installer diagnostics from `uv` are decoded
as UTF-8 with replacement for malformed bytes, preserving Unicode resolver
symbols on Windows instead of displaying code-page mojibake.

The core must remain importable without the backend package installed.
