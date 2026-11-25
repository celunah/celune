# Configuration

Celune loads YAML configuration from the user application-data directory. The
`configure.py` setup helper creates `config.yaml` from the repository's
`default_config.yaml`. If setup was skipped, the first launch creates it
instead. On startup, Celune synchronizes an existing file with the current
default schema: known user values are preserved, new defaults are added, and
obsolete options are removed recursively. The synchronized file is written
before the interface opens. `celune config view` prints the active file and
`celune config edit` opens it in the system editor.

## Core settings

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `null` | TTS or VC backend name. `null` lets the application choose its normal backend. |
| `inference_mode` | `native` | Use native model libraries or GGUF runtimes where supported. Takes effect after restarting Celune. |
| `backend_options` | `qwen3.x_vector_only: false`, `soulxsinger.high_precision: true` | Constructor options passed only to the named backend. |
| `commercial` | `false` | Declare commercial use. Celune then allows only models whose reviewed terms confirm commercial use. |
| `quantize` | `true` | Try contract-approved native TTS quantization: INT8 on Ampere and FP8 on sm89 or newer, with BF16 recovery. Ming3 and supported audio.cpp TTS adapters select Q8_0 when enabled and BF16 otherwise. |
| `quantize_kv_cache` | `true` | Store supported Persona, Qwen3, FireRedTTS3, and VoxCPM2 cache prefixes in INT8 on Ampere or FP8 on `sm89+`. Use `false` to keep their native cache dtype. |
| `voice_bundle` | `default` | CEVOICE/CECHAR name or path. |
| `log_level` | `info` | `info`, `verbose`, or `debug`. |
| `locale` | `null` | Locale override. `null` uses system detection. |
| `rpc.enabled` | `true` | Show Celune's activity through Discord Rich Presence. |
| `rpc.show_utterances` | `true` | Show speech caption chunks in Rich Presence. |
| `mode` | `performance` | `performance`, `converse`, or `agent`. The former `speak` name remains compatible. |
| `vram` | `medium` | Model-size and memory preset: `low`, `medium`, `high`, or `xhigh`. Persona and standard agent models need `high`. Smart 8B-tier Persona models need `xhigh`. |
| `headless` | `false` | Suppress the Textual interface. |
| `headless_nocolor` | `false` | Suppress color in headless output. |
| `theme` | `dark` | `dark` or `light`. A pack can supply its own accent colors. |
| `use_normalizer` | `false` | Enable the optional text normalizer. |
| `captions` | `false` | Enable IPA forced alignment for speech caption timing. |
| `ipa` | `false` | Enable IPA-oriented text handling where supported. |
| `audio_api` | `null` | Explicit sounddevice host API. |
| `input_device` | `null` | Input device index/name. |
| `output_device` | `null` | Output device index/name. |

`backend`, `voice_bundle`, `mode`, and `inference_mode` are settings that most
directly change runtime behavior. Backend-specific settings should stay in
their documented namespace instead of being duplicated at the top level.

`inference_mode` accepts `native` or `gguf`. In GGUF mode, Persona uses
llama.cpp for the reviewed Qwen3-VL 4B package when available. Other Persona
models fall back to Transformers. TTS backends select audio.cpp only when they
declare support. Native-only backends stay on their existing runtime. Ming3
requires audio.cpp in either mode. Celune logs the requested and effective
runtime, and backend workers report their loaded model through CEDTS metadata.
See [inference runtimes](../development/inference-runtimes.md) for the exact
compatibility and artifact matrix.

### Discord Rich Presence

When `rpc.enabled` is true, Celune publishes her current activity over
Discord's local IPC connection. Discord labels her activity as
`Listening to Celune`, with a state such as idle, thinking, generating,
speaking, sleeping, or reloading as its status detail. If `rpc.show_utterances`
is true, the detail changes to `Celune says:` for one second when speech starts,
then displays available caption text in chunks no more often than every six
seconds. Each detail is limited to Discord's 128-character activity field.
This uses the existing caption pipeline. Enable `captions` to produce aligned
captions for TTS speech. When speech ends, the detail returns to Celune's
current state. Caption text is sent to Discord only when
`rpc.show_utterances` is enabled.
Discord limits clients to five activity updates per 20 seconds, so Celune also
rate-limits state changes and shutdown clears across the same rolling window
([Discord Gateway documentation](https://github.com/discord/discord-api-docs/blob/main/developers/events/gateway-events.mdx)).
Celune does not use a bot, OAuth authorization, or message, friend, guild, or
voice APIs. Discord must be running and configured to display current activity.
If the IPC connection is unavailable, Celune keeps running and retries. She
clears the activity during shutdown when the rate limit allows. Set
`rpc.enabled: false` to disable Rich Presence. Set `rpc.show_utterances: false`
to keep caption text private while leaving state details enabled.

### Backend-specific options

Put backend-only constructor options under `backend_options` and the backend's
registered name. The application passes only the selected backend's options to
that backend's isolated worker. SoulX-Singer defaults to F0 melody-conditioning
mode and full-section inference. Set `high_precision` to `false` for score conditioning
or `streaming` to `true` for shorter inference chunks:

```yaml
backend_options:
  qwen3:
    x_vector_only: false
  soulxsinger:
    high_precision: true
    streaming: false
```

`backend_options.qwen3.x_vector_only` defaults to `false`, preserving full
reference prosody and expression. Set it to `true` when speaker identity
stability matters more than expressive conditioning.

With `streaming: true`, Celune divides long singing sections into chunks of
about 1.92 seconds, equal to three default TTS chunk lengths, at event
boundaries. The backend keeps tied and slurred note chains together. Each chunk
is a separate inference call, so playback can begin before the complete section
is generated. The default `false` preserves full-section generation and phrasing.

This mode downloads SoulX-Singer's pinned RMVPE F0 extractor checkpoint if it is
not already cached. Celune measures the singer reference's F0, generates a
score-conditioned teacher performance, extracts its natural pitch movement,
re-centers each note on the written pitch, and then synthesizes a final
melody-conditioned performance. This adds a second inference pass, so it takes
longer than the default score mode. CESN still specifies note pitches. This
option does not add explicit pitch bends or cents offsets to the notation.

Set `commercial: true` when Celune is used commercially. Models marked
non-commercial or research-only, models with unknown license restrictions, and
models whose terms still need review are blocked before download or load. With
`commercial: false`, Celune uses the normal model license consent flow.

### VRAM budgets and profiles

Each preset reserves 2 GiB for Windows/Linux and other GPU users. Celune's
budget is therefore 4 GiB at `low`, 6 GiB at `medium`, 10 GiB at `high`, and
14 GiB at `xhigh`. Preset names still describe minimum total GPU capacity.

Persona context is capped at 2,048 tokens outside agent mode and 8,192 tokens
in agent mode. Standard Persona models are available at `high`. Smart models,
including the registered 8B-tier variants, require `xhigh`.

Celune checks exact hardware and model profiles when one is available. A
confirmed configuration that exceeds its preset budget or current headroom is
rejected. An unprofiled configuration is allowed to continue and logs:
`This configuration is unprofiled and may not work on your hardware configuration.`
It can still fail at load or inference time if the GPU runs out of memory.
The `quantize_kv_cache` setting is part of this profile identity. Existing
Qwen3 measurements cover native TTS caches, so quantized-cache configurations
remain unprofiled until measured separately.

### TTS quantization

Set `quantize: true` (the default) to reduce VRAM use for supported TTS
models. Ming3 and supported audio.cpp backends use Q8_0 when enabled and BF16
when disabled. Celune selects INT8 on Ampere GPUs and FP8 on Ada-class (`sm89`)
and newer GPUs. If quantized loading or generation fails, Celune may retry in
BF16. It does not retry CUDA out-of-memory failures with the larger BF16 model.
Set `quantize: false` to opt out.

`log_level` controls exception detail as well as ordinary diagnostics. `info`
keeps handled failures concise, `verbose` appends the exception message, and
`debug` appends the full traceback and writes it to Celune's runtime traceback
file. Public API responses continue to use safe, localized summaries. The
additional detail is written to the configured runtime logs instead.

In the Textual interface, `/settings` opens the configuration manager. Nested
YAML values are shown with human-readable labels that preserve names such as
API, T2S, Ming3, and Persona. ENTER writes the edited values to the active
`config.yaml`, fades the interface through the normal shutdown transition, and
then requests a silent launcher-managed restart. The terminal title changes to
`Restarting`. ESC leaves the file unchanged.

String settings that allow null can be edited by typing a value directly. Enter
`null` or `None` without quotes to save YAML null. Enter `"null"` or `"None"` to
save the word as a string. Autocomplete suggestions remain available for
settings that provide them.

## YouTube downloads

Celune uses `yt-dlp` for YouTube URLs supplied to audio playback and upgrades
it with the default EJS support group. YouTube extraction may also need a
supported JavaScript runtime. Configure its executable when it is not on
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

Use either `cookies_file` or `cookies_from_browser`. When both are present,
Celune uses `cookies_file`. Keep cookie files and PO tokens private. Browser
cookies can expire or rotate, and using an account with `yt-dlp` may trigger
provider security checks. PO tokens are client- and session-specific, so
refresh them when YouTube rejects an otherwise valid media URL.

For current runtime and token requirements, see the upstream [`yt-dlp` EJS
guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS) and [PO-token
guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide).

## Speech buffering and playback

Smart buffering keeps playback smooth while speech is generated. An automatic
reserve adapts to system load and audio-device underflows. Live microphone voice
conversion bypasses this reserve to avoid extra latency. These behaviors are
automatic and are not user-configurable. Integrations can read playback
statistics through the [Python API](../reference/public-api.md).

Use `/speed`, `/reverb`, and `/seed` to adjust speech. See the
[Textual UI guide](../interfaces/tui.md) for command details.

## Speech captions

```yaml
captions: true
```

See [Speech](speech.md#long-form-speech) for caption requirements, setup, and
fallback behavior. With `captions: false` (the default), Celune does not load or
download the alignment model.

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
and `normalizer` release their respective memory. `tts` controls whether the
active TTS model is unloaded. `Celune.wake_from_sleep()` restores the runtime.
An explicit speech request automatically wakes Celune first and proceeds after
model readiness returns. If waking fails, the speech request is not queued.
Sleep is a lifecycle feature, not a process shutdown.

## REST API

```yaml
api:
  enabled: true
  access: same_machine
  port: 2060
  token: null
  rate_limit_per_minute: 60
```

Celune creates and persists an API key automatically when `token` is `null` and
`CELUNE_API_TOKEN` is unset.
When `token` is `null`, a nonempty `CELUNE_API_TOKEN` is used instead. Set
`token: ""` to disable API authentication. This explicit empty value also
overrides `CELUNE_API_TOKEN`. Any other token value is used as the API key.
With authentication disabled, keep `access: same_machine` unless the API is
intended to be available without a key to other devices.
Use `access: same_machine` for localhost, `same_network` for LAN devices, or
`everyone` for a temporary public Cloudflare tunnel. The API extra is required
for FastAPI/Gradio deployment. See [REST API](../API.md) for key storage,
network binding, tunnel shutdown, and browser authentication details.

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

Persona is available in `converse` and `agent` modes. `max_turns: null` leaves
the turn count unbounded unless memory settings impose a shorter history.
Context is capped at 2,048 tokens outside agent mode and 8,192 tokens in agent
mode. The shared `compact_at` setting controls when older conversation history
is summarized. Memory options are character-scoped. See [Persona](persona.md)
for memory behavior and [Agent settings](#agent-settings) for context limits.

The default Whisper model is pinned to a measured revision. A custom
`speech_model_id` uses the repository's current revision and has no confirmed
VRAM profile unless measured separately.

## Agent settings

```yaml
agent:
  fs_tools: true
  max_loops: 20
  max_tokens: null
  context_size: 8192
```

`fs_tools` enables local filesystem and process tools. `null` for `max_tokens`
uses the model and context limits. Agent context is capped at 8,192 tokens, and
`persona.compact_at` controls when older conversation history is summarized.
Persona model selection is independent of TTS backend selection. Model IDs use
reviewed remote-code revisions. See [Agent mode](agent.md) for tool permissions
and [Persona](persona.md) for memory behavior.

Persona and standard agent models require at least the `high` preset. Smart
8B-tier Persona models require `xhigh`. Selecting an incompatible preset
disables the corresponding feature. Celune does not raise the configured VRAM
target automatically.

## Voice conversion

Voice-conversion F0 conditioning is a runtime control rather than a YAML
setting. Live capture always performs voice-activity detection. The selected
CUDA extra supplies Silero VAD. When it is unavailable, the application uses
its built-in energy detector instead. With Silero enabled, the energy detector
also catches speech onset while the background model catches up. Live capture
preserves microphone audio through a 300 ms VAD hangover so delayed or briefly
missed decisions do not cut off speech boundaries.

## Configuration precedence

The effective value is selected in this order for keys with explicit support:

1. A direct constructor or API argument.
2. The process environment override, such as `CELUNE_BACKEND`.
3. The user's YAML configuration.
4. The bundled `default_config.yaml` value.

Do not edit the bundled default to configure one machine. It is the migration
source for new installs and future default keys.
