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
| `luxtts` | TTS | `celune.backends.tts.luxtts:LuxTTS` | CUDA-first LuxTTS worker with CPU ONNX fallback, its ZipVoice/LinaCodec VCS dependencies, and English prompt transcription. |
| `seed-vc` | VC | `celune.backends.vc.seedvc:CeluneSeedVCBackend` | Celune's Seed-VC fork. |

Most workers share a compatibility baseline containing Hugging Face Hub and
`hf-xet`, Transformers below 5 in the worker environment, Lingua, librosa,
llvmlite, NumPy/Numba, Pillow, platformdirs, psutil, sounddevice, soundfile,
and Zstandard, plus the CEDTS-compatible PyTorch 2.11 CUDA 12.8 worker stack.
LuxTTS uses that same pinned PyTorch stack. It selects the native PyTorch
checkpoint and CUDA runtime when `torch.cuda.is_available()` is true, and uses
the ONNX CPU path only when no usable CUDA runtime exists. Its model and
reference implementation are
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
complete reference WAV and exact `reference_text`, then streams progressive 24 kHz
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

`luxtts` adapts the [LuxTTS](https://github.com/ysharma3501/LuxTTS) native GPU
path and ONNX CPU fallback from the [YatharthS/LuxTTS model
card](https://huggingface.co/YatharthS/LuxTTS). It supports English voice
cloning, uses the active voice's reference WAV, and has LuxTTS transcribe the
prompt internally. Celune supplies a five-second prompt window, requests the
native 48 kHz waveform path, and emits one complete waveform per request. The
installed upstream `generate_speech` API does not expose incremental acoustic
frames, so this backend cannot provide true model-time streaming yet. The
adapter adds the decoder look-ahead frames required by
Vocos, corrects the CPU ONNX duration ratio so short responses retain generated
frames after prompt removal, adds a short reference boundary to prevent prompt
tail leakage, uses that boundary's silent acoustic feature for decoder context
instead of repeating voiced output, and removes only the final two decoder hops
so the last phoneme is not cut off.
Empty or too-short acoustic output becomes a user-facing short-input warning.
The backend uses the native GPU model when CUDA is available; its CPU fallback
uses two ONNX threads by default. Its isolated manifest uses the shared
PyTorch 2.11 CUDA 12.8 stack and adds the upstream Piper wheel page as a
`find-links` source so the environment can resolve its phonemizer dependency.
Its installer also
passes uv's `--no-sources` option so LinaCodec cannot redirect PyTorch to its
CUDA 12.6 source override. Preloading also
caches the device-appropriate `openai/whisper-tiny` or
`openai/whisper-base` transcriber required by the upstream constructor before
Celune enables Hugging Face offline mode for model loading.
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

## Model weight contracts

`celune.backends.tts.contracts` records the pinned Hugging Face revision and
weight inventory for every supported TTS backend. Each safetensors artifact is
identified by its expected size and SHA-256 digest, then described by its
tensor count, parameter count, dtype distribution, and canonical tensor
inventory digest. The inventory digest covers tensor names, shapes, and dtypes;
it lets a worker reject missing, unexpected, or structurally changed weights
without embedding a several-thousand-name list in the source tree.

The contract includes the currently supported model variants:

- Pocket TTS uses `lunahr/pocket-tts-ungated`, not the gated upstream
  repository. Its language variants are separate contracts because the
  current French `french_24l` artifact has a different tensor inventory from
  the other selected language artifacts.
- Qwen3 includes both the 0.6B and 1.7B Base checkpoints and their shared F32
  speech tokenizer.
- FireRedTTS3 records the upstream F32 source inventories separately from its
  runtime dtype rules: the Qwen backbone, stop head, and RedAE encoder are
  BF16, while the flow and decoder paths remain F32.
- LuxTTS is represented by its Torch/ONNX/BIN artifacts because its model
  repository does not use safetensors for the selected runtime path.

`validate_safetensors_artifact` validates a cached artifact without loading
its tensors into VRAM. `validate_model_state` validates a loaded component's
exact structural inventory, runtime dtypes, parameter count, and finite values.
Both validators raise `celune.exceptions.InvalidCheckpoint` on failure. The
exception exposes the backend, checkpoint filename and path, and—when a
specific tensor is responsible—the tensor name, owning layer, shape, dtype,
expected dtype, and actual dtype. Contract lookup failures remain
`ModelContractError` because they indicate a missing Celune contract rather
than a corrupt checkpoint.
Each quantizable component also carries a `QuantizationRule`. It lists the
linear-module suffixes that are safe for weight-only conversion and explicit
module names that must remain at their contract dtype. The runtime validates
the BF16 state before conversion and calls `validate_model_state` again with
`allow_quantized=True` afterward; that mode permits only approved INT8/FP8
weights and continues checking counts, parameter totals, runtime dtypes for
other tensors, and finite values. `celune.backends.tts.quantization` selects
INT8 for Ampere (`sm80`/`sm86`) and FP8 for `sm89` or newer, using TorchAO's
weight-only configs. It does not quantize embeddings, norms, output heads,
speaker-conditioning paths, or vocoders. Conversion releases the temporary
pre-quantization state references before TorchAO replaces weights. Quantization
is performed in place on the model's current device; Celune never stages a
live CUDA component on CPU and back to CUDA, avoiding a second full device
allocation during backend loading. After conversion it clears unreferenced
CUDA cache blocks, so the runtime retains only the quantized model storage in
VRAM.
Component resolution checks the contract component name on the backend wrapper
and its nested `model` before falling back to a native root module. This keeps
TorchAO scoped to the declared component—for example, Pocket TTS's `flow_lm`
or dots.tts's `core`—instead of quantizing a wrapper that also owns a vocoder
or speaker encoder.

When importing or using integrations that may load TorchAO, Celune skips only
the redundant pytree registration for Enum classes that the active PyTorch
version already supports as opaque compile values. Other TorchAO constant
registrations remain unchanged, so this compatibility path removes the known
`register_constant()` deprecation warning without disabling quantization. The
Transformers imports are also deferred into this boundary because its
quantizer registry can import TorchAO eagerly. The same boundary covers the
lazy Qwen3 voice-embedding model used by post-speech voice analysis.
The isolated worker log bridge also suppresses the known TorchAO invalid-escape
source warning and PyTorch's Windows/macOS redirect-support note; backend
tracebacks and actionable errors remain visible.

When quantized loading, the startup speech probe, or a later speech generation
fails, the backend disables quantization, unloads the model, and reloads the
same model in BF16. A failed BF16 recovery invokes the fatal engine transition
immediately.

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
