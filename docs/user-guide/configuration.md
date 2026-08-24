# Configuration

Celune loads YAML configuration from the user application-data directory. The
`configure.py` setup helper creates `config.yaml` from the repository's
`default_config.yaml`; if setup was skipped, the first launch creates it
instead. Celune then merges newly introduced defaults into an existing file
without discarding the user's values. `celune config view` prints the active
file and `celune config edit` opens it in the system editor.

## Core settings

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `null` | TTS or VC backend name. `null` lets Celune choose its normal backend. |
| `voice_bundle` | `default` | CEVOICE/CECHAR name or path. |
| `log_level` | `info` | `info`, `verbose`, or `debug`. |
| `locale` | `null` | Locale override; `null` uses system detection. |
| `mode` | `converse` | `speak`, `converse`, or `agent`. |
| `vram` | `medium` | Model-size and memory preset: `low`, `medium`, `high`, or `xhigh`. |
| `headless` | `false` | Suppress the Textual interface. |
| `headless_nocolor` | `false` | Suppress color in headless output. |
| `theme` | `dark` | `dark` or `light`; a pack can supply its own accent colors. |
| `use_normalizer` | `false` | Enable the optional text normalizer. |
| `ipa` | `false` | Enable IPA-oriented text handling where supported. |
| `audio_api` | `null` | Explicit sounddevice host API. |
| `input_device` | `null` | Input device index/name. |
| `output_device` | `null` | Output device index/name. |

`backend`, `voice_bundle`, and `mode` are the three settings that most directly
change runtime behavior. Backend-specific settings should stay in their
documented namespace instead of being duplicated at the top level.

In the Textual interface, `/settings` opens the configuration manager. Nested
YAML values are shown with human-readable labels that preserve names such as
API, T2S, GPT-SoVITS, and Persona. ENTER writes the edited values to the active
`config.yaml` before requesting a silent launcher-managed restart; the terminal
title changes to `Restarting`. ESC leaves the file unchanged.

## Speech buffering and playback

Smart buffering starts playback after a minimum amount of audio, adapts the
playback speed while generation catches up, and protects already-buffered audio
from aggressive changes. Playback CPU protection is an engine-owned policy and
is not user-configurable.

Runtime speech controls are also exposed as `/speed`, `/reverb`, `/seed`, and
Python properties on `Celune`. The command values are deliberately narrower
than the Python values; see [Text UI](../interfaces/tui.md).

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
  context_size: 8192
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
unbounded unless the memory settings impose a shorter history.

## Agent settings

```yaml
agent:
  fs_tools: true
  max_loops: 20
  max_tokens: null
  context_size: 32768
  compact_at: 75
```

`fs_tools` enables the local filesystem and process tool catalog. Agent task
limits are applied when Celune creates a task; `null` for `max_tokens` means
that generation is bounded only by the model and context limits.

Persona is independent of TTS backend selection. The model registry in
`celune.constants` pins allowed remote-code revisions; changing a model ID does
not grant arbitrary remote code. Memory records are character-scoped and are
stored under the Persona data directory. Explicit memory requests are favored;
the classifier is optional and confidence-gated.

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
