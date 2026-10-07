# Configuration

Celune loads YAML configuration from the user application-data directory. The
`configure.py` setup helper creates `config.yaml` from the repository's
`default_config.yaml`; if setup was skipped, the first launch creates it
instead. On startup, Celune synchronizes an existing file with the current
default schema: known user values are preserved, new defaults are added, and
obsolete options are removed recursively. The synchronized file is written
before the interface opens. `celune config view` prints the active file and
`celune config edit` opens it in the system editor.

## Core settings

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `null` | TTS or VC backend name. `null` lets Celune choose its normal backend. |
| `quantize` | `true` | Try contract-approved TTS weight-only quantization: INT8 on Ampere and FP8 on sm89 or newer, with BF16 recovery. LuxTTS is excluded. |
| `quantize_kv_cache` | `true` | Store Persona's attention-cache prefix in INT8 on Ampere or FP8 on `sm89+`; use `false` for the normal dynamic cache. |
| `voice_bundle` | `default` | CEVOICE/CECHAR name or path. |
| `log_level` | `info` | `info`, `verbose`, or `debug`. |
| `locale` | `null` | Locale override; `null` uses system detection. |
| `mode` | `converse` | `speak`, `converse`, or `agent`. |
| `vram` | `medium` | Model-size and memory preset: `low`, `medium`, `high`, or `xhigh`. Persona and standard agent models need `high`; smart 8B-tier Persona models need `xhigh`. |
| `headless` | `false` | Suppress the Textual interface. |
| `headless_nocolor` | `false` | Suppress color in headless output. |
| `theme` | `dark` | `dark` or `light`; a pack can supply its own accent colors. |
| `use_normalizer` | `false` | Enable the optional text normalizer. |
| `captions` | `false` | Enable IPA forced alignment for speech caption timing. |
| `ipa` | `false` | Enable IPA-oriented text handling where supported. |
| `audio_api` | `null` | Explicit sounddevice host API. |
| `input_device` | `null` | Input device index/name. |
| `output_device` | `null` | Output device index/name. |

`backend`, `voice_bundle`, and `mode` are the three settings that most directly
change runtime behavior. Backend-specific settings should stay in their
documented namespace instead of being duplicated at the top level.

### VRAM budgets and profiles

Each preset reserves 2 GiB for Windows/Linux and other GPU users. Celune's
budget is therefore 4 GiB at `low`, 6 GiB at `medium`, 10 GiB at `high`, and
14 GiB at `xhigh`. Preset names still describe minimum total GPU capacity.

Persona context is capped at 2,048 tokens outside agent mode and 8,192 tokens
in agent mode. Standard Persona models are available at `high`; smart models,
including the registered 8B-tier variants, require `xhigh`.

Celune checks exact hardware and model profiles when one is available. A
confirmed configuration that exceeds its preset budget or current headroom is
rejected. An unprofiled configuration is allowed to continue and logs:
`This configuration is unprofiled and may not work on your hardware configuration.`
It can still fail at load or inference time if the GPU runs out of memory.

### TTS quantization

Set `quantize: true` (the default) to reduce supported TTS model VRAM use.
LuxTTS is excluded because its runtime is not eligible for TorchAO weight-only
conversion. Celune selects TorchAO
weight-only INT8 for Ampere devices before native FP8 support and FP8 for
Ada-class (`sm89`) and newer devices. The model contracts identify the linear
attention and feed-forward layers eligible for conversion; embeddings, norms,
output heads, speaker conditioning, and vocoders remain in BF16 or their
backend-required dtype.

For VoxCPM2, dots.tts, and FireRedTTS3, Celune quantizes the model on CPU
before transferring it to CUDA. This prevents the full BF16 model and its
quantized replacement from overlapping in VRAM during conversion. The lighter
backends retain their existing loading path. For conversion, startup speech
probing, or generation failures other than CUDA out-of-memory, Celune unloads
the quantized model and retries in BF16. If that recovery also fails, startup
enters the normal fatal state. A CUDA out-of-memory error does not retry with
larger BF16 weights; Celune releases partial allocations, reports the failure,
and returns speech generation to an idle state. Set `quantize: false` to opt
out.

`log_level` controls exception detail as well as ordinary diagnostics. `info`
keeps handled failures concise, `verbose` appends the exception message, and
`debug` appends the full traceback and writes it to Celune's runtime traceback
file. Public API responses continue to use safe, localized summaries; the
additional detail is written to the configured runtime logs instead.

In the Textual interface, `/settings` opens the configuration manager. Nested
YAML values are shown with human-readable labels that preserve names such as
API, T2S, LuxTTS, and Persona. ENTER writes the edited values to the active
`config.yaml`, fades the interface through the normal shutdown transition, and
then requests a silent launcher-managed restart; the terminal title changes to
`Restarting`. ESC leaves the file unchanged.

## YouTube downloads

Celune uses `yt-dlp` for YouTube URLs supplied to audio playback and upgrades
it with the default EJS support group. YouTube extraction may also need a
supported JavaScript runtime; configure its executable when it is not on
`PATH`. These values are passed to the downloader only for YouTube playback.

```yaml
youtube:
  cookies_file: null
  cookies_from_browser: null
  po_token: null
  player_client: null
  js_runtimes: null
  remote_components: null
  extractor_args: []
```

| Key | Values | Purpose |
| --- | --- | --- |
| `youtube.cookies_file` | Path or `null` | Netscape-format cookies file. |
| `youtube.cookies_from_browser` | Browser selector or `null` | Read cookies from a local browser, for example `chrome` or `edge`. |
| `youtube.po_token` | Token spec or list | Passes one or more `yt-dlp` PO-token specs, such as `web.gvs+TOKEN`. |
| `youtube.player_client` | Client name or list | Selects YouTube clients such as `web_embedded`. |
| `youtube.js_runtimes` | Runtime spec or list | Enables runtimes such as `deno` or `node:C:/path/node.exe`. |
| `youtube.remote_components` | Component or list | Allows components such as `ejs:npm` when supported by the runtime. |
| `youtube.extractor_args` | List of `IE_KEY:ARGS` values | Supplies advanced `yt-dlp` extractor arguments. |

Use either `cookies_file` or `cookies_from_browser`; when both are present,
Celune uses `cookies_file`. Keep cookie files and PO tokens private. Browser
cookies can expire or rotate, and using an account with `yt-dlp` may trigger
provider security checks. PO tokens are client- and session-specific, so
refresh them when YouTube rejects an otherwise valid media URL.

For current runtime and token requirements, see the upstream [`yt-dlp` EJS
guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS) and [PO-token
guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).

## Speech buffering and playback

Smart buffering starts playback after a minimum amount of audio, adapts the
playback speed while generation catches up, and protects already-buffered audio
from aggressive changes. Playback also uses a persistent output writer and an
adaptive reserve. The reserve starts at 2 seconds and can grow to 30 seconds
when Celune detects high system/process CPU usage, delayed scheduling, slow
output writes, or a PortAudio underflow. The output stream requests high
latency device buffering to add a second hardware-side reserve; this may add
latency but prevents short CPU spikes from becoming audible gaps.

Playback contention is an engine-owned policy and is not user-configurable. A
persistent queue reader and output writer keep queue waits and stream writes
off the mixer polling path; the mixer yields briefly between blocks so the
writer can continue draining during CPU spikes. Debug timing traces are
sampled at a low rate instead of being written for every block.

The `Celune.playback_buffer_seconds`, `Celune.playback_contention_level`, and
`Celune.playback_underflows` properties expose live diagnostics for integrations
and troubleshooting. Stage timing properties identify where contention is being
felt: `playback_queue_wait_seconds` measures producer backpressure,
`playback_generation_gap_seconds` measures per-source chunk gaps,
`playback_writer_wait_seconds` measures application-side writer delay,
`playback_writer_gap_seconds` measures gaps between output writes,
`playback_writer_write_seconds` measures the stream call, and
`playback_rebuffer_wait_seconds` measures cumulative reserve-gate waiting.
Debug traces include these values as `queue_wait`, `generation_gap`,
`rebuffer_wait`, `writer_wait`, `writer_gap`, and `writer_write` fields. A large
queue or generation gap points upstream; a large writer wait or writer gap
points to thread scheduling; a large writer-write value points to the audio
device or driver.

Runtime speech controls are also exposed as `/speed`, `/reverb`, `/seed`, and
Python properties on `Celune`. The command values are deliberately narrower
than the Python values; see [Text UI](../interfaces/tui.md).

## Speech captions

```yaml
captions: true
```

When enabled, Celune uses Sadda's `sadda-speech/wav2vec2-espeak-ctc` model to
align generated speech words through eSpeak IPA phonemes. The acoustic model and
ONNX Runtime extra are optional. On Windows, install the Python dependencies
with:

```powershell
uv sync --dev --extra api --extra captions
```

Also install the `espeak-ng` executable and make it available on `PATH`. When
captions are enabled, Celune prepares the Apache-2.0-licensed acoustic model in
the background during startup without blocking startup; the first alignment can
still wait if preparation is ongoing. See [speech](speech.md) for alignment and
troubleshooting details. With `captions: false`, Celune does not load or download
the alignment model.

## Sleep and model lifecycle

```yaml
sleep:
  enabled: true
  timeout: 10
  unload_persona: true
  normalizer: true
  tts: false
```

When enabled, idle time can unload selected model components. `unload_persona`
and `normalizer` release their respective memory; `tts` controls whether the
active TTS model is unloaded. `Celune.wake_from_sleep()` restores the runtime.
Sleep is a lifecycle feature, not a process shutdown.

## REST API

```yaml
api:
  enabled: true
  host: 127.0.0.1
  port: 2060
  token: null
  rate_limit_per_minute: 60
```

Keep the API on loopback when no authentication token is configured. A token
may be supplied in YAML or through `CELUNE_API_TOKEN`; authenticated network
binding is documented in [REST API](../API.md). The API extra is required for
FastAPI/Gradio deployment.

## Persona and memory

```yaml
persona:
  context_size: 2048
  compact_at: 75
  max_turns: null
  debug_overrides: false
  model_id: Qwen/Qwen3-VL-4B-Instruct
  speech_model_id: openai/whisper-large-v3-turbo
  speech_language: auto
  speech_end_delay_seconds: 1.5
  memory:
    max_short_term_messages: 20
    auto_classifier: true
    auto_classifier_min_confidence: 0.82
    auto_classifier_max_candidates: 3
    automatic_max_age_days: 60
    context_compaction_enabled: true
    context_compaction_keep_recent_messages: 8
    context_summary_max_characters: 1200
    storage_dir: null
    semantic_similarity_threshold: 0.62
    fallback_token_overlap_threshold: 1
    semantic_embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

Persona is available in `converse` and `agent` modes; it is not enabled or
disabled through a Persona-local switch. `context_size` bounds each ordinary
Persona request, `compact_at` documents the context percentage at which
history should be compacted, and `max_turns: null` leaves the turn count
unbounded unless the memory settings impose a shorter history. Context is
capped at 2,048 tokens outside agent mode and 8,192 tokens in agent mode.

Persona generation uses the configured `quantize_kv_cache` policy on CUDA. The
quantized cache keeps a short BF16 tail for recent tokens and stores older
keys and values with one scale per token. INT8 is selected for Ampere GPUs;
FP8 is selected for `sm89` and newer GPUs. The cache is dequantized only for
the attention operation, so the model still computes attention in its normal
dtype. Unsupported cache layouts, unavailable FP8 support, or a cache
runtime failure fall back to the regular dynamic cache for that request.
`context_size` is an upper bound: input encoding is limited to that bound while
leaving room for at least one response token, and generation is capped by the
remaining space after the actual prompt tokens are counted. The dynamic cache
grows with the encoded prompt and generated tokens; it does not reserve the
configured maximum. Celune releases the request cache after each response.

The default Whisper model is pinned to a specific Hugging Face commit so its
memory profile stays tied to the measured weights. A custom `speech_model_id`
continues to use that repository's current revision and has no confirmed VRAM
profile unless it is measured separately.

## Agent settings

```yaml
agent:
  fs_tools: true
  max_loops: 20
  max_tokens: null
  context_size: 8192
  compact_at: 75
```

`fs_tools` enables the local filesystem and process tool catalog. Agent task
limits are applied when Celune creates a task; `null` for `max_tokens` means
that generation is bounded only by the model and context limits. Agent context
is capped at 8,192 tokens. Routing and classification requests use the same
ceiling to limit transient KV-cache allocation. Routing prompts contain only
the current input and active task
metadata; they do not retain conversational history. Persona generation
requests use the configured KV-cache policy and release unused CUDA allocator
blocks after each response. In agent mode, Celune records the actual prompt and
completion token counts. `compact_at` triggers compaction of older Persona
history according to the memory settings. Celune updates the task's history
snapshot and uses a new request-scoped cache for the next generation, releasing
the previous generation's cache and pruned history references. Compaction is a
soft threshold; `context_size` remains the hard maximum for the prompt and its
response.

Persona is independent of TTS backend selection. The model registry in
`celune.constants` pins allowed remote-code revisions; changing a model ID does
not grant arbitrary remote code. Memory records are character-scoped and are
stored under the Persona data directory. Explicit memory requests are favored;
the classifier is optional and confidence-gated.

Persona and standard agent models require at least the `high` preset. Smart
8B-tier Persona models require `xhigh`. Selecting an incompatible preset
disables the corresponding feature; Celune does not raise the configured VRAM
target automatically.

## Voice conversion

Voice-conversion pitch shift and F0 conditioning are runtime controls rather
than YAML settings. Live capture always performs voice-activity detection. The
optional `live-vc-ai` extra supplies Silero VAD; when it is unavailable, Celune
uses its built-in energy detector instead.

## Configuration precedence

The effective value is selected in this order for keys with explicit support:

1. A direct constructor or API argument.
2. The process environment override, such as `CELUNE_BACKEND`.
3. The user's YAML configuration.
4. The bundled `default_config.yaml` value.

Do not edit the bundled default to configure one machine. It is the migration
source for new installs and future default keys.
