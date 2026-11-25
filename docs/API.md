# REST API

Celune can expose a local REST API for speech, singing, sound effects, and voice control.
The API is enabled by default in `default_config.yaml`:

```yaml
api:
  enabled: true
  access: same_machine
  port: 2060
  token: null
  rate_limit_per_minute: 60
```

## Access and authentication

Celune generates a key automatically when `api.token` is `null` and
`CELUNE_API_TOKEN` is unset. A nonempty `CELUNE_API_TOKEN` is used when the
configuration token is `null`. The generated key uses the `C-` prefix followed
by 25 uppercase base-36 characters, and persists in Celune's app-data
`api.key` file. Set `api.token: ""` to disable API authentication. This explicit
empty value also overrides `CELUNE_API_TOKEN`. Any other token value is used as
the API key. With authentication disabled, requests to the API and WebUI are
not gated by a key, so keep the listener on localhost unless unauthenticated
network access is intended. The helper `celune.api.get_or_create_api_key()`
returns the generated or configured key to Python integrations.
`celune.api.generate_api_key()` creates a new key without storing it.

Choose `api.access` to control where Celune listens:

| Value | Listener | Reach |
| --- | --- | --- |
| `same_machine` | `127.0.0.1` | Programs and browsers on this computer. |
| `same_network` | `0.0.0.0` | Other devices that can reach this computer on the local network. |
| `everyone` | `127.0.0.1` plus `cloudflared` | A temporary public Cloudflare Quick Tunnel. |

The `same_network` mode may need a host-firewall rule for local devices. Router
port forwarding is needed only if you also want to reach that listener from
outside the local network. The `everyone` mode requires `cloudflared` on `PATH`.
The application stops the quick-tunnel process when it closes. If `cloudflared`
is unavailable, the application keeps the API on loopback and reports that the
public tunnel did not start.

When the API starts in `everyone` mode, the application uses the Cloudflare URL
in its startup message when `cloudflared` publishes it within ten seconds. If
the tunnel fails or does not publish a URL by then, the startup message uses the
loopback API URL. A public URL that appears later is logged separately.

The API is an optional core extra. Install it with
`uv sync --extra api --extra cu132` (or select another CUDA extra). Keep a
network-facing listener behind authentication and an appropriate firewall. The
API can start speech, load voices, and play uploaded audio.

Authenticated requests may send either header:

```http
Authorization: Bearer YOUR_API_KEY
X-Celune-Token: YOUR_API_KEY
```

## API key helpers

`celune.api.generate_api_key() -> str` creates a new key using Python's
cryptographic `secrets` generator. It takes no arguments and does not write to
disk:

```python
from celune.api import generate_api_key

new_key = generate_api_key()
```

`celune.api.get_or_create_api_key(configured_key: Optional[str] = None, path: Optional[Path] = None) -> str`
uses a configured key first, then `CELUNE_API_TOKEN`, and otherwise loads or
creates the persisted generated key. Its optional `path` argument selects a
different key file:

```python
from celune.api import get_or_create_api_key

api_key = get_or_create_api_key()
```

`celune.paths.api_key_path(create_parent: bool = False) -> Path` returns the
default key-file path. Set `create_parent=True` when the caller will create or
write the file:

```python
from celune.paths import api_key_path

key_file = api_key_path()
api_key = key_file.read_text(encoding="utf-8").strip()
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1` | Return Celune's current status. |
| `GET` | `/v1/version` | Return the running Celune version. |
| `GET` | `/` | Redirect to the mounted WebUI at `/ui/`. |
| `GET` | `/ui/` | Load the mounted WebUI. |
| `GET` | `/ui/key` | Show the API-key prompt when authentication is enabled. |
| `POST` | `/ui/key` | Exchange the API key for a browser WebUI session. |
| `POST` | `/v1/think` | Ask Celune to think through Persona and reply asynchronously through the application's normal speech pipeline. |
| `POST` | `/v1/speak` | Queue speech and return one complete FLAC or WAV response. |
| `POST` | `/v1/audio/speech` | Accept the OpenAI audio speech request shape and return MP3, FLAC, or WAV. |
| `POST` | `/v1/audio/transcriptions` | Transcribe an uploaded audio file with the current Persona Whisper model. |
| `POST` | `/v1/chat/completions` | Generate a text completion with the current Persona LLM. |
| `POST` | `/v1/speak/stream` | Return generated speech progressively as multipart FLAC or WAV chunks. |
| `POST` | `/v1/sing` | Validate CESN, queue singing, and stream FLAC or WAV. |
| `POST` | `/v1/speak/async` | Add speech to the bounded FIFO and return `202 Accepted` with a job ID. |
| `GET` | `/v1/speak/jobs/{job_id}` | Poll an async speech job. Returns `202` while pending and its selected audio format when complete. |
| `POST` | `/v1/speak/jobs/{job_id}/cancel` | Cancel a queued job or stop an active async speech job. |
| `WebSocket` | `/v1/ws/tasks/{job_id}` | Replay and stream typed lifecycle events for an async speech job. |
| `POST` | `/v1/voice` | Change the active voice. |
| `POST` | `/v1/sfx` | Upload and play a sound effect. |
| `POST` | `/v1/convert` | Upload audio for voice conversion in VC mode. |
| `GET` | `/v1/models` | List OpenAI-compatible available models and Celune's consent catalog. |
| `POST` | `/v1/models/consent` | Record consent for one exact model revision. |

`/v1/sing` uses the same `content`, optional `save`, and `response_format`
fields as `/v1/speak`. It requires a singing-capable backend and valid CESN
notation. It returns
`409 Conflict` when the backend is not ready for singing and `422 Unprocessable
Content` with a line and column for malformed notation. The optional MCP server
is available at `/mcp`. See [MCP server](interfaces/mcp.md).

When a singing backend is active, `/v1/speak` and `/v1/speak/async` return
`409 Conflict` with instructions to submit CESN to `/v1/sing`. The async route
does not create a job. `/v1/think` also returns `409 Conflict` because Persona
replies are spoken prose, not CESN scores.

`/v1` and `/v1/version` are authenticated JSON endpoints. The root redirect is
useful when a browser opens the API port. Programmatic clients should call the
`/v1` paths directly. `/ui` is the Gradio WebUI mounted by the optional API
runtime.

## Model terms and consent

Before loading a restricted, unregistered, unknown-license, or unreviewed model,
Celune asks for consent to that exact Hugging Face commit. The Textual UI shows
the localized license prompt and keeps its bottom-left status at
`registry.waiting_status` while it waits. The WebUI displays an obstructing
consent modal with license details, restriction flags, and the source link.

`GET /v1/models` lists the built-in catalog, including model ID, display name,
category, pinned revision, license and license URL, non-commercial and
research-only flags, optional `supports_vision` capability, consent state,
whether commercial mode is enabled, and whether each model's reviewed terms
confirm commercial use. Custom model
metadata is included in a `428 Precondition Required` response when an
operation first needs consent. When commercial mode blocks a model operation,
Celune returns `403 Forbidden` with the model metadata and a localized reason.

`POST /v1/models/consent` accepts a model ID, a full commit SHA, and an
`accepted` boolean. A successful acknowledgement is stored in Celune app data
for that exact model and revision. If the model changes revision, it requires a
new acknowledgement. After a `428` response, submit consent and retry the
operation. A declined decision returns `403 Forbidden` and the affected load
does not continue.

See [Model registry and license consent](development/model-registry.md) for
catalog ownership, revision pinning, and consent workflow details.

## Request and response shapes

The JSON models are intentionally small:

```json
{"content":"Hello from Celune.","save":true}
```

`SpeakRequest.content` is a non-empty string, `save` defaults to `true`, and
`response_format` defaults to `flac`. The supported formats are `flac` and
`wav`. `ThinkRequest` has `content`. `VoiceRequest` has `voice_name`.

Synchronous speech returns `audio/flac` or `audio/wav` according to
`response_format`. Audio is encoded as signed 24-bit PCM at 48 kHz. Async
speech returns a task object with `status`, `job_id`, and `location`. Polling
returns JSON while queued or running and the selected format when complete.
Task event objects contain `task_id`, `event`, `status`, optional `message`,
`severity`, `current`, `total`, `location`, and `error`. WebSocket commands
currently accept:

```json
{"command":"cancel"}
```

The server validates the command and returns a typed command result. A socket
disconnect removes only that subscriber. It does not cancel the job.

## OpenAI-compatible API

Celune exposes OpenAI-compatible text-to-speech, transcription, and chat
completion routes. The endpoints use Celune's currently configured local models
and do not contact OpenAI. Persona-backed endpoints appear only when the active
operation mode enables Persona and its model is ready.

### Text-to-speech

`POST /v1/audio/speech` accepts the OpenAI audio speech request fields `model`,
`input`, `voice`, `response_format`, `speed`, `instructions`, and
`stream_format`. Text inputs to `/v1/audio/speech`, `/v1/speak`, `/v1/sing`, and
`/v1/think` are limited to 65,536 characters. Chat completions accept at most 64
messages with up to 16,384 characters per message. Oversized fields or message
lists return FastAPI's standard `422` validation response. Celune uses its
currently selected TTS backend. The `model`
field is required for client compatibility and does not switch the backend.
`voice` must match one of Celune's configured voice names. Selecting a
different voice makes it the active Celune voice, as with `POST /v1/voice`.

Celune returns MP3 by default, using the standard `audio/mpeg` media type. Set
`response_format` to `flac` for PCM24 FLAC or `wav` for PCM24 WAV.
`speed` accepts values from `0.25` to `4.0` and uses Rubber Band when it is
available. If Rubber Band is unavailable, Celune uses normal speed.
`instructions` are added to the active voice prompt when the backend supports
voice prompts. `stream_format` supports `audio`; SSE is not supported.
`response_format` supports `mp3`, `flac`, and `wav`. Requests use the usual
bearer API key when authentication is enabled. Speech responses are returned to
the requesting client without being sent to the host playback mixer. Speech
submitted directly through the main TUI continues to play on the host.

```json
{
  "model": "tts-1",
  "input": "Hello from Celune.",
  "voice": "Balanced",
  "response_format": "wav",
  "speed": 1.0,
  "instructions": "Speak clearly and warmly."
}
```

For example, the OpenAI Python client can use Celune as its base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:2060/v1",
    api_key="YOUR_CELUNE_API_KEY",
)
response = client.audio.speech.create(
    model="tts-1",
    voice="Balanced",
    input="Hello from Celune.",
    response_format="wav",
)
response.write_to_file("celune.wav")
```

The route supports Celune TTS with `mp3`, `flac`, and `wav` responses. Other
OpenAI audio encodings are not supported. Celune-specific operations such as CESN
singing, voice conversion, Persona, and sound effects remain available through
their `/v1` endpoints.

### Audio transcriptions

`POST /v1/audio/transcriptions` accepts a multipart `file`, `model`, optional
`language`, and `response_format`. `whisper-1` selects the configured Persona
speech model, which defaults to `openai/whisper-large-v3-turbo`. The
`whisper-tiny` alias selects `openai/whisper-tiny`. The exact configured model
ID is also accepted. Files must be readable by Celune's SoundFile decoder and
must not exceed 25 MB.

`response_format` supports `json` (the default), `text`, `verbose_json`, `srt`,
and `vtt`. When `language` is omitted, Celune uses the configured Persona speech
language or automatic detection. Transcription is available when Persona is
enabled and ready, except while voice-conversion input mode is active. The
Whisper model is loaded for the request and unloaded afterward. A busy ASR
component returns `409 Conflict`; pending model consent returns `428`.

```python
with open("input.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json",
    )
print(transcript.text)
```

### Chat completions

`POST /v1/chat/completions` accepts text-only `system`, `developer`, `user`,
and `assistant` messages. Use `celune-current` or the active Persona model ID
for `model`. Celune applies the active character's Persona prompt and the
messages supplied by the client. The request does not speak the response or
change Celune's stored conversation history. Requests are limited to 64 text
messages, each at most 16,384 characters.

The route supports one choice and the `max_tokens` or `max_completion_tokens`,
`temperature`, and `top_p` parameters. Streaming, tool calls, and multimodal
message content are not supported. Chat completions require Persona to be
enabled and ready in the active operation mode. Requests return `409 Conflict`
while Persona is unavailable or its model is busy.

```python
completion = client.chat.completions.create(
    model="celune-current",
    messages=[{"role": "user", "content": "Hello, Celune."}],
)
print(completion.choices[0].message.content)
```

### Model discovery

`GET /v1/models` includes the OpenAI list fields `object` and `data` while
retaining Celune's existing `models` consent catalog. It lists `tts-1` only
when text-to-speech is available. Persona model IDs, `celune-current`,
`whisper-1`, and `whisper-tiny` appear only when their mode and readiness
requirements are met.

## Think

Use `/v1/think` when the client wants Celune to respond through Persona instead of speaking the provided text literally.
The endpoint returns as soon as Celune accepts the request. The spoken reply is produced later through Celune's normal playback pipeline.

Linux, macOS, or Git Bash:

```bash
curl -i -X POST http://127.0.0.1:2060/v1/think \
  -H "Content-Type: application/json" \
  -d '{"content":"What changed since the last build?"}'
```

PowerShell:

```powershell
curl.exe -i -X POST http://127.0.0.1:2060/v1/think -H "Content-Type: application/json" -d '{"content":"What changed since the last build?"}'
```

Command Prompt:

```bat
curl.exe -i -X POST http://127.0.0.1:2060/v1/think ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"What changed since the last build?\"}"
```

Response:

- `202 application/json` with `{"status":"accepted"}` when Celune accepts the Persona request.
- `409 application/json` when Celune is busy or the Persona input queue is full.
- `503 application/json` when Celune is unavailable.

Submitting another request does not cancel speech that is already playing. An
explicit `/stop` command or speech-job cancellation is required to interrupt it.

## Speak (Synchronous)

Use `/v1/speak` when the client wants the generated audio on the same request.
The request stays open until generation finishes.

Linux, macOS, or Git Bash:

```bash
curl -X POST http://127.0.0.1:2060/v1/speak \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello from Celune.","save":true}' \
  --output celune.flac
```

PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:2060/v1/speak -H "Content-Type: application/json" -d '{"content":"Hello from Celune.","save":true}' --output celune.flac
```

Command Prompt:

```bat
curl.exe -X POST http://127.0.0.1:2060/v1/speak ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"Hello from Celune.\",\"save\":true}" ^
  --output celune.flac
```

Response:

- `200 audio/flac` by default, or `audio/wav` when requested.
- `409 application/json` when Celune is unavailable or cannot accept speech.

Submitting another synchronous speech request does not cancel speech that is
already playing. Speech requests are queued in FIFO order and begin when the
shared pipeline is available. Use `/v1/speak/async` when the client needs a
job ID and polling instead of an open audio response.

## Speak (Streaming)

Use `/v1/speak/stream` with the same request fields as `/v1/speak` when the
client can consume multipart responses and wants
audio before the full utterance has finished. Each part is an independent,
complete PCM24 FLAC or WAV file at 48 kHz stereo. Set `response_format` to
`wav` to receive WAV parts. The response uses
`multipart/mixed; boundary=celune-audio`. Clients decode each part and append its
frames in order. A disconnected client stops the active request. A slow client
applies backpressure to audio delivery.

```bash
curl -N -X POST http://127.0.0.1:2060/v1/speak/stream \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello from Celune.","save":false}' \
  --output celune.multipart
```

The synchronous `/v1/speak` endpoint returns one complete FLAC response by
default. Set `response_format` to `wav` for one complete WAV response.
`/v1/speak/stream` queues requests behind active speech and begins streaming
when their turn reaches the shared pipeline. Use the asynchronous endpoint when
requests should return a job ID immediately.

## Speak (Asynchronous)

Use `/v1/speak/async` when the client needs a quick acknowledgement and can poll
for the result later. Celune runs one API speech job at a time and keeps up to
four waiting jobs in FIFO order. If that waiting queue is full, the endpoint
returns `429 Too Many Requests` with `queue_full`.

Linux, macOS, or Git Bash:

```bash
curl -i -X POST http://127.0.0.1:2060/v1/speak/async \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello from Celune.","save":true}'
```

PowerShell:

```powershell
curl.exe -i -X POST http://127.0.0.1:2060/v1/speak/async -H "Content-Type: application/json" -d '{"content":"Hello from Celune.","save":true}'
```

Command Prompt:

```bat
curl.exe -i -X POST http://127.0.0.1:2060/v1/speak/async ^
  -H "Content-Type: application/json" ^
  -d "{\"content\":\"Hello from Celune.\",\"save\":true}"
```

Accepted response:

```json
{
  "status": "accepted",
  "job_id": "6f98c8f3a4a94a049d84dd9fce8a63c5",
  "location": "/v1/speak/jobs/6f98c8f3a4a94a049d84dd9fce8a63c5"
}
```

Poll the `location` until the job completes:

```bash
curl -i http://127.0.0.1:2060/v1/speak/jobs/6f98c8f3a4a94a049d84dd9fce8a63c5
```

When the job completes, it returns the format selected in the submission's
`response_format` field, which defaults to `flac`:

```bash
curl http://127.0.0.1:2060/v1/speak/jobs/6f98c8f3a4a94a049d84dd9fce8a63c5 \
  --output celune.flac
```

Job responses:

- `202 application/json` with `{"status":"queued"}` or `{"status":"running"}` while Celune is still working.
- `200 audio/flac` by default, or `audio/wav` when WAV was selected.
- `404 application/json` if the job ID is unknown.
- `500 application/json` if the job failed.
- `429 application/json` from submission when the waiting queue is full.

Queued and running jobs remain available for their full duration. Terminal jobs
are kept in memory for 15 minutes after their last status update and are not
persisted across Celune restarts. Queued jobs wait until the shared speech
pipeline is available, including while local speech is playing. Cancelling a
queued job removes it from the FIFO. Cancelling a running job targets that
request without stopping a replacement request.

## Request scheduling

TTS and Persona inference share a priority scheduler for the active accelerator.
When both are waiting, speech inference runs first. An inference already in
progress is allowed to finish. This keeps waiting Persona work from starting
ahead of speech that is ready to use the accelerator.

### Task WebSocket Events

After receiving a job ID from `/v1/speak/async`, connect to
`ws://127.0.0.1:2060/v1/ws/tasks/{job_id}`. The connection replays retained events
so clients may connect after speech has already started, then streams live events:

```json
{"task_id":"...","event":"started","status":"running"}
{"task_id":"...","event":"progress","status":"running","current":1.0}
{"task_id":"...","event":"log","status":"running","message":"Generating"}
{"task_id":"...","event":"completed","status":"completed","location":"/v1/speak/jobs/..."}
```

Every event includes its task ID. Terminal events are `completed`, `failed`, or
`cancelled`. A disconnected WebSocket only removes that client subscription. It
does not stop or alter the Celune task. Cancellation is explicit through
`POST /v1/speak/jobs/{job_id}/cancel` or by sending this API-layer command through
the WebSocket:

```json
{"command":"cancel"}
```

The WebSocket layer never runs speech generation itself and does not expose
internal exception details. Polling and synchronous speech endpoints remain
available alongside the event stream.

## Voice

Linux, macOS, or Git Bash:

```bash
curl -X POST http://127.0.0.1:2060/v1/voice \
  -H "Content-Type: application/json" \
  -d '{"voice_name":"Balanced"}'
```

PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:2060/v1/voice -H "Content-Type: application/json" -d '{"voice_name":"Balanced"}'
```

Command Prompt:

```bat
curl.exe -X POST http://127.0.0.1:2060/v1/voice ^
  -H "Content-Type: application/json" ^
  -d "{\"voice_name\":\"Balanced\"}"
```

Response:

- `200 application/json` with `{"status":"ok"}` when the voice was changed.
- `400 application/json` when the voice name is unknown.
- `500 application/json` when the voice could not be changed.

## Sound Effects

```bash
curl -X POST http://127.0.0.1:2060/v1/sfx \
  -F "file=@sound.wav" \
  -F "keep=true" \
  --output sfx.flac
```

Response:

- `200 audio/flac` with the uploaded sound effect encoded as FLAC.
- `400 application/json` when the upload is not valid audio.
- `409 application/json` when Celune cannot play the sound right now.
- `413 application/json` when the upload is too large.

## Voice conversion

The conversion endpoint accepts multipart form data:

```bash
curl -X POST http://127.0.0.1:2060/v1/convert \
  -F "file=@input.wav" \
  -F "f0_condition=false" \
  --output converted.flac
```

`file` is required. `f0_condition` is an optional boolean override. The endpoint returns `audio/flac`
when conversion succeeds, `409` when VC mode is not active or the engine is
busy, `400` for malformed audio/fields, `413` when the upload limit is
exceeded, and `500` when the conversion backend raises an internal request
failure. See [Voice conversion](user-guide/voice-conversion.md) for the
interactive capture path.

## Authentication, limits, and errors

When an API key is configured, send either:

```http
Authorization: Bearer YOUR_API_KEY
```

or:

```http
X-Celune-Token: YOUR_API_KEY
```

The configured per-minute rate limit applies independently to API calls and
WebUI generation actions. WebSocket connection and reconnection events do not
count toward either request limit. Authenticated WebSocket connections still
require a valid session or API key and a same-origin request. Only WebUI speech
and conversion submissions consume the WebUI generation limit. Polling,
page-load refreshes, voice and recording controls, and other Gradio transport
requests do not consume it. Failed validation, busy-state, missing-job, and
backend errors are returned as JSON with an HTTP status instead of exposing raw
internal tracebacks. Queued and running jobs remain available until terminal;
terminal jobs are retained for 15 minutes after their last status update. Jobs
are not persisted across a restart.

## WebUI behavior

The Gradio WebUI is mounted at `/ui` and uses the same Celune engine, voice
state, and Persona and agent submission path as the other interfaces. It
supports text input, voice selection, and `ALT+R` capture when the active
mode supports recording. Touch and coarse-pointer layouts show a visible
recording button as an alternative to the shortcut. The WebUI does not include
the TUI settings editor.

When API authentication is enabled, `/ui` prompts for an API key. A valid key
creates an opaque, signed `HttpOnly`, `SameSite=Strict` cookie scoped to `/ui`
for 30 days with a sliding expiry. The cookie contains a random client ID and
expiry, not the API key, and is invalidated when the configured key changes.
The browser does not store the key in its URL or browser storage. The key
exchange is rate-limited, and browser writes reject cross-origin requests.
WebUI actions and WebSocket connections require a valid session and same-origin
request. Setting `api.token: ""` disables the prompt and session requirement,
as described under
[Access and authentication](#access-and-authentication).

A reverse proxy or Cloudflare Tunnel must forward `/ui/assets/` to Celune
without rewriting that path. A remembered browser may need to authenticate
again after its network or browser request signature changes.

## Embedding the server

Python integrations can bind the engine explicitly:

```python
from celune.api import run_api, start_api

run_api(celune, host="127.0.0.1", port=2060, token=None)
# Or keep the main thread free:
thread = start_api(celune, host="127.0.0.1", port=2060)
```

`run_api()` blocks in Uvicorn. `start_api()` starts a daemon thread and waits
for startup confirmation. Its optional `on_started(host, port)` callback runs
when the socket starts listening. Startup logging reports the listener URL by
default. Set `defer_startup_log=True` with an `on_started` callback when that
callback reports a different public URL. Without a callback, Celune still logs
the listener URL. The supporting calls `bind_celune()`,
`configure_api_security()`, `resolve_api_host()`, `audio_bytes()`, and
`stream_headers()` are the existing integration hooks for custom hosts. Use
them instead of mounting a second FastAPI app around the engine. Core callback
properties are single-registration slots: assigning the same callable to the
same callback property again raises `ValueError`, so integrations should
replace a handler only when they intentionally change the callback.

## See also

- [Configuration](user-guide/configuration.md) for the API access setting.
- [Validation](development/validation.md) for the repository verification commands.
