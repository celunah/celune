# Backends

This page describes how Celune resolves, launches, and extends its
text-to-speech and voice-conversion backends.

## Application environment

Normal backend dependencies belong to Celune's configured application
environment. The resolver imports the selected backend lazily, so
backend-specific packages must not be imported by core modules at startup.
Explicit CEDTS worker callers may still use the manifest-backed environment
manager; normal application configuration does not select that path.
`BackendEnvironmentManager` in `celune/backends/environment.py` stores each
worker environment under its backend and manifest fingerprint. After the
current environment is ready, it removes older fingerprinted environments for
that backend. If installation fails, the prior environment remains available.
`tests/backends/test_environment.py` covers this cleanup behavior.

If a CEDTS worker exits before its first handshake frame, Celune reports the
unexpected end of the protocol stream. When worker stderr contains an
exception, the early-startup error also includes its final exception line;
traceback frames remain in the worker diagnostics rather than the error
screen.

## Registered manifests

| ID | Kind | Worker | Extra requirements |
| --- | --- | --- | --- |
| `mini` | TTS | `celune.backends.tts.mini:Mini` | `pocket-tts>=3.2.0` |
| `qwen3` | TTS | `celune.backends.tts.qwen3:Qwen3` | `faster-qwen3-tts>=0.2.4` |
| `fireredtts3` | TTS | `celune.backends.tts.fireredtts3:FireRedTTS3` | FireRedTTS3 source, Transformers 5.6.2, and TorchCodec 0.16.0; BF16 transformer path with PyTorch SDPA and latent streaming. |
| `dotstts` | TTS | `celune.backends.tts.dotstts:DotsTtsMF` | Celune's `dots.tts` fork; Python 3.12. |
| `voxcpm2` | TTS | `celune.backends.tts.voxcpm2:VoxCPM2` | `voxcpm>=2.0.0`; Python 3.12. |
| `ming3` | TTS | `celune.backends.tts.ming3:Ming3` | Multilingual Ming 3.0 voice cloning through audio.cpp v0.9.3's native streaming CUDA server. |
| `soulxsinger` | TTS | `celune.backends.tts.soulxsinger:SoulXSinger` | Score-conditioned singing; Python 3.12, pinned upstream source and Transformers 4.41.2, `iso639-lang` for CESN, explicit English IPA, CUDA. |
| `seed-vc` | VC | `celune.backends.vc.seedvc:CeluneSeedVCBackend` | Celune's Seed-VC fork. |

Most workers share a compatibility baseline that includes GitPython for
Celune's shared repository helpers, Hugging Face Hub and `hf-xet`, Transformers
below 5, Lingua, librosa, llvmlite, NumPy/Numba, Pillow, platformdirs, psutil,
sounddevice, soundfile, and Zstandard. They also use the CEDTS-compatible
PyTorch 2.11 CUDA 12.8 worker stack.
Ming3 uses the shared worker environment for Celune integration and starts a
pinned audio.cpp v0.9.3 CUDA server as a child process. On Windows it downloads
the CUDA 13.3 binary and runtime archives; on Linux it uses the CUDA 12.8
release. The server runs only on loopback and is stopped when the backend
runtime is released. The Ming weights are converted GGUF packages from
[audio-cpp](https://huggingface.co/audio-cpp/Ming-3.0-TTS-0.5B-GGUF), pinned
separately from the official
[Ming 3.0 model card](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B).
The model's MIT license applies to the weights; Celune's source license does not
replace third-party model terms.
FireRedTTS3 is intentionally separate from that Hugging Face portion: its
manifest uses `huggingface-hub>=1.5.0,<2.0.0` and `transformers==5.6.2`, which
cannot be resolved alongside the shared Hub-below-1 and Transformers-below-5
constraints. The core project itself currently resolves its own CUDA 13.0
stack; the core and backend environments are intentionally not the same
lockfile. SoulX-Singer pins Transformers 4.41.2 to match its upstream
requirements; its custom Llama decoder omits the `position_embeddings`
argument required by newer `LlamaAttention` implementations. Its worker uses
Python 3.12 because Transformers 4.41.2 selects `tokenizers==0.19.1`, whose
Windows wheel is available for Python 3.12; Python 3.13 falls back to a source
distribution that fails to parse its incomplete project metadata.

## Backend behavior

### Qwen3

`qwen3` streams at 12.5 chunks per second and supports `zh-cn`, `en`, `ja`,
`ko`, `de`, `fr`, `ru`, `pt`, `es`, and `it`. Medium and larger VRAM presets use
the 1.7B clone model; low VRAM can select the 0.6B model. It reads a CEVOICE
reference WAV and per-voice transcript. `qwen3_x_vector_only` favors identity
stability over full reference expressiveness.

### Mini

`mini` adapts Pocket TTS 3.2.0 or newer, streams at 12.5 chunks per second, and
supports English, French, German, Italian, Portuguese, and Spanish. Version
3.2.0 adds the 24-layer French model used by Celune's pinned snapshot
([upstream release](https://github.com/kyutai-labs/pocket-tts/releases/tag/v3.2.0)).
Celune matches each snapshot language variant to its corresponding package
config, which keeps the model architecture aligned with its weights. It uses
`lsd_decode_steps=2` with the pre-3.0.0 API and `sampler_decode_steps=2` with the
3.0.0-or-newer API. Generated model configs select the matching tokenizer
loader for either the JSON tokenizer or SentencePiece model stored in the
snapshot. Mini remains the supported CPU-friendly path and uses pack reference
data for cloning. If a language-specific reload fails during generation,
Celune clears the model reference and enters its fatal state instead of
continuing with a stale model.

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

### Ming3

`ming3` adapts the official [Ming 3.0 TTS 0.5B model](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) and the pinned [audio.cpp v0.9.3 runtime](https://github.com/0xShug0/audio.cpp/tree/v0.9.3). The official Python wrapper collects the complete waveform before returning; Celune instead uses audio.cpp's `streaming` session and server-sent PCM deltas, which arrive while acoustic generation is active. This meets the backend's model-time streaming contract without patching the Python package.

The backend supports Chinese, English, Japanese, Korean, German, French, and
Spanish. It clones from the active CEVOICE/CECHAR voice's reference WAV and exact
`reference_text`. The server runs one request at a time on the active CUDA
device, binds only to loopback, eagerly loads one selected GGUF, and is owned by
the CEDTS backend runtime. Celune downloads and verifies the pinned Windows CUDA
13.3 binary and CUDA runtime packages or the Linux CUDA 12.8 package, then stores
them under application data. It selects Q8_0 weights when `quantize: true` and
BF16 weights when quantization is disabled. A load failure does not silently
switch the selected precision. The model card is MIT-licensed; its usage
disclaimer does not mark it research-only or non-commercial.

When Celune's language is `Auto`, the submission pipeline uses Lingua to detect
the text language. Ming receives that detected language when supported and falls
back to English for unsupported or unrecognized values; it does not use Qwen3's
`Auto` language token.

| Celune quantization | Package | Download size |
| --- | --- | ---: |
| Enabled (`true`) | Q8_0 GGUF | 1.74 GB |
| Disabled (`false`) | BF16 GGUF | 3.00 GB |

Startup downloads the selected model package and native runtime during
`preload_models`; `load_model` waits for the server to report that the model is
resident. Generation decodes SSE PCM deltas into Celune's normalized float32
44.1 kHz audio chunks and rejects an empty, incomplete, too-short, or
implausibly long response. The runtime log is retained in Celune's temporary
data directory when server startup or inference fails.

### SoulX-Singer

`soulxsinger` implements the score-controlled path from the
[SoulX-Singer model](https://huggingface.co/Soul-AILab/SoulX-Singer) and its
[pinned upstream source](https://github.com/Soul-AILab/SoulX-Singer/tree/81aeb3ae772c70093c3de74dc23c92d983801ae4).
The model repository is licensed under Apache-2.0; its terms are separate from
Celune's source license. The registry pins both model weights and source code.
The worker downloads only `config.yaml` and `model.pt` from the reviewed model
revision, then strictly loads the checkpoint state dictionary. Preloading also
prepares the pinned source checkout. High-precision mode additionally downloads
the pinned `rmvpe/rmvpe.pt` checkpoint from the
[SoulX-Singer preprocessing model](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess),
which is also listed under its Apache-2.0 license. That checkpoint is not
downloaded when score mode is selected. The score author supplies IPA directly,
so the worker does not install or download G2P dictionaries. On Windows, the
isolated SoulX worker defaults `OPENBLAS_NUM_THREADS` to `1` unless the host
already sets it, limiting BLAS thread oversubscription.
During worker startup, SoulX imports its pinned model and processor modules
before CEDTS starts handling requests. Model loading reuses those cached modules
and reports numeric progress through config and checkpoint loading, weight
application, CUDA transfer, and score-processor setup without logging each
internal step. Direct callers still load the pinned modules on demand.
The score processor decodes its reference WAV through `soundfile` and calls
`torchaudio.functional.resample` only when resampling is needed, so this path
does not require TorchCodec or FFmpeg.
Concurrent worker startups serialize checkout preparation with a cross-process
lock. Each worker rechecks the pinned checkout after acquiring the lock, so a
worker can reuse a checkout another process just installed. The checkout is
cloned directly into its revision directory while holding the lock, avoiding a
directory rename that Windows can reject while another process scans the tree.
An incomplete checkout is removed before the next installation attempt.

The backend accepts English CESN and requires an active CECHAR v5 pack with a
hidden `sing` WAV and `cesn` reference. It uses that reference as the prompt,
converts each explicitly supplied IPA syllable into the model's English
ARPAbet inventory, maps pitches to MIDI note numbers, and converts CESN beat
lengths to seconds using each section's BPM. It does not infer pronunciation or
syllable boundaries from the lyric spelling. Unsupported IPA sounds raise a
backend error before model inference. See the [CESN format](../formats/cesn.md)
for the notation and conversion behavior.
Changes in singer, tempo, key, or scale remain section boundaries. By default,
the model synthesizes each section in one call and Celune concatenates the
resulting audio. With `backend_options.soulxsinger.streaming: true`, Celune
splits sections at event boundaries around 1.92 seconds (three default TTS
chunk lengths), keeping ties and slurs intact, then yields each completed chunk
for playback. Every chunk remains a complete SoulX inference call; Celune does
not stream from inside the model's reverse-diffusion pass. This mode can reset
expression at chunk boundaries and adds a model call plus onset context for
each chunk, so it is opt-in. See
[backend-specific options](../user-guide/configuration.md#backend-specific-options)
for the configuration contract. By default, each section uses one
score-conditioned inference pass. With
`backend_options.soulxsinger.high_precision: true`, Celune first makes a
score-conditioned teacher pass, extracts its framewise F0 with RMVPE, recenters
each note's median pitch on the written CESN pitch, then performs a second
melody-conditioned pass. This keeps the score's note centers while retaining
within-note pitch movement from the teacher performance. The singer reference
uses its measured F0 track, cached with its prompt tensors. High-precision mode
therefore takes longer and uses more memory than the default score path.
Each section is conditioned with a two-second leading rest so the first sung
event is not placed at the model's generation boundary. Celune removes those
rest samples and their F0 frames before returning audio, preserving the written
score timing.

The startup probe is CESN notation with explicit `[bpm:120,key:c,scale:major]`
metadata and the synthetic note `[a4:4][ɑ] Ah`. A singing backend rejects input
without any note or rest event with `No CESN notation found.` It requires a CUDA
GPU. If the source checkout, model weights, or singing reference cannot be
prepared, the backend raises an English technical error and does not synthesize
the request.

## Generation timing and reference reuse

The speech pipeline owns the canonical `TTFC` measurement. Its clock starts
when Celune enqueues a `SpeechRequest` and records the first nonempty,
non-silent audio chunk after sample-rate conversion and configured DSP,
immediately before buffering it. This includes speech-queue wait, inference
scheduler wait, backend chunk aggregation, conversion, and DSP. Backend
`first_chunk_time` metadata does not override this value. Generation speed and
RTF use a separate timer that begins when the worker starts processing, so
request-queue delay does not make model throughput appear slower. `TTFC`
measures audio ready for the playback buffer; it does not measure when playback
becomes audible.

Streaming adapters retain their configured chunk batches and final-chunk
metadata. Once the first batch is ready, they release it when the next
nonempty backend chunk arrives instead of waiting for a complete second batch.
This preserves final-chunk reporting while reducing the initial buffering delay.

Reference audio truncated to Celune's ten-second limit is reused while its
resolved path, modification time, and size remain unchanged. FireRedTTS3 also
caches each loaded voice's resampled prompt latents and speaker embedding while
the model is loaded. Its feature cache retains up to four references and clears
on model unload.

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
- Ming3 uses separate pinned BF16 and Q8_0 GGUF artifacts from the audio.cpp
  package repository; the source model card remains the consent and license
  identity.

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
version-2 weight-only configs. It does not quantize embeddings, norms, output
heads, speaker-conditioning paths, or vocoders. Conversion releases the
temporary pre-quantization state references before TorchAO replaces weights.
The heavy VoxCPM2, dots.tts, and FireRedTTS3 loaders construct the model on
CPU, quantize the contract-approved components there, and transfer the single
resulting model to CUDA once. This prevents the full BF16 model and its
quantized replacement from overlapping in VRAM during conversion. The lighter
backends retain their existing load order. After conversion Celune clears
unreferenced CUDA cache blocks, so the runtime retains only the quantized model
storage and backend-required unquantized components in VRAM.
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

Runtime KV-cache quantization is separate from TorchAO's weight-only policy.
With `quantize_kv_cache: true`, Persona and the Qwen3 and FireRedTTS3 decoder
caches store older full-attention keys and values as INT8 on Ampere or FP8 on
`sm89+`, while keeping a short recent-token tail in the model's compute dtype.
Attention receives dequantized values in that dtype. Qwen3 uses its dynamic
quantized cache for prompt prefill and transfers it into fixed-capacity
quantized storage for CUDA-graph decoding. The graph cache keeps the recent
tail in the compute dtype and uses a shared dense workspace for attention;
older keys and values remain INT8 or FP8 in resident cache storage. When cache
construction is unsupported by the device or cache layout, Qwen3 keeps its
native graph cache. If an installed Qwen API lacks Celune's graph-cache adapter
surface, Qwen3 retains the dynamic quantized-cache fallback.

VoxCPM2 uses two custom static MiniCPM caches, so its adapter supplies compact
prefix storage and dequantizes each prefix for attention while preserving the
existing cache capacity bound. Mini/Pocket TTS, dots.tts MeanFlow, and Ming3 do
not use autoregressive KV caches. Whisper and Needle are outside this policy.

VoxCPM2 does not expose a Transformers cache boundary. Its two private static
MiniCPM caches are therefore bounded at the adapter boundary to 2048 positions
before the model is used, and are rebuilt at that capacity when a CPU-quantized
runtime moves to CUDA. This removes the unused 8192-position resident
allocation without modifying the isolated upstream package. The cap is sized
for Celune's ten-second reference limit and 512-step speech chunks; extending
those limits requires revisiting the adapter cap and checking the model's
position budget.

When quantized loading, the startup speech probe, or a later speech generation
fails, the backend disables quantization, unloads the model, and reloads the
same model in BF16. A failed BF16 recovery invokes the fatal engine transition
immediately.

## CESN singing hook

`CeluneBackend` accepts `singing: bool = False`. A singing-capable TTS backend
opts in by passing `singing=True` to the base initializer. For a request with
CESN tags, Celune parses the original notation, validates the explicit language
against the backend's `supported_languages`, and records warnings in the runtime
log before calling `cesn_to_model(text)`. It passes the user's original source
so backend validation can report exact line and character locations. The hook
is called once before generation chunking.

`celune/cesn.py` owns X-SAMPA conversion, parsing, singing validation,
syllabification, synthesis preparation, and canonical rendering. It calls the
backend hook before the pipeline chunks the request. A backend resolves its
active singer when the notation does not name one.

Each `CESNSection` has one singer, language, BPM, key, and scale. A change to
any of those values begins a new section. A singing backend must reset its
inference state at each section boundary and process the section metadata.
SoulX-Singer emits each generated section as a separate audio chunk and includes
`section_caption` and `section_final` in the chunk timing metadata. The core
uses the queued frame span to switch the caption when playback reaches that
section. A backend that combines multiple sections into one chunk cannot provide
section-specific captions. Parsed blocks retain note/IPA pairs and positioned
rests; out-of-scale notes remain available after a `CESNNotationWarning` is
displayed.

`[tuplet:a,b]` scales the next `a` pitched-note or rest events by `b/a` and
expires automatically. The parser preserves this ratio on each event and
exposes its effective duration in quarter-note beats so a backend can convert
it to the model's timing units. Tuplets may span adjacent blocks but cannot
overlap or cross a section boundary.

The core parser requires an explicit `[lang:...]` tag and rejects languages
unsupported by the selected model. Each note group uses the order
`[extra tags][notes][raw IPA] lyrics`; the author supplies one dot-separated
IPA syllable per pitched note. `[singer:...]` resolves a character name to a
CECHAR v5 pack with a hidden `sing` voice. See
[`parse_cesn()` and the CESN format](../formats/cesn.md) for tag syntax, data
types, and errors.

The hook signature remains `cesn_to_model(self, text: str) -> str`. Its input is
the complete source notation, which may contain X-SAMPA or omit a singer tag.
Override it to parse the sections and translate them into the selected model's
notation while preserving source locations for validation errors. The base
implementation returns the input unchanged. Ordinary TTS backends leave
`singing` false and do not invoke CESN parsing.

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

## See also

- [CEVOICE and CECHAR formats](../CEVOICE.md)
- [Speech guide](../user-guide/speech.md)
