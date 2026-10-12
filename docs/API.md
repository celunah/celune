# REST API

Celune can expose a local REST API for speech, sound effects, and voice control.
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

Celune generates a key automatically when `api.token` and
`CELUNE_API_TOKEN` are unset. The generated key uses the `C-` prefix followed
by 25 uppercase base-36 characters, and persists in Celune's app-data
`api.key` file. The helper `celune.api.get_or_create_api_key()` returns
that key to Python integrations; `celune.api.generate_api_key()` creates
a new key without storing it.

Choose `api.access` to control where Celune listens:

| Value | Listener | Reach |
| --- | --- | --- |
| `same_machine` | `127.0.0.1` | Programs and browsers on this computer. |
| `same_network` | `0.0.0.0` | Other devices that can reach this computer on the local network. |
| `everyone` | `127.0.0.1` plus `cloudflared` | A temporary public Cloudflare Quick Tunnel. |

The `same_network` mode may need a host-firewall rule for local devices. Router
port forwarding is needed only if you also want to reach that listener from
outside the local network. The `everyone` mode requires `cloudflared` on `PATH`;
Celune stops the quick-tunnel process when it closes. If `cloudflared` is
unavailable, Celune keeps the API on loopback and reports that the public
tunnel did not start.

When the API starts in `everyone` mode, Celune uses the Cloudflare URL in its
startup message when `cloudflared` publishes it within ten seconds. If the
tunnel fails or does not publish a URL by then, the startup message uses the
loopback API URL; a public URL that appears later is logged separately.

The API is an optional core extra. Install it with `uv sync --extra api` (or
include `--all-extras` during Linux development). Keep a network-facing listener
behind authentication and an appropriate firewall; the API can start speech,
load voices, and play uploaded audio.

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
| `GET` | `/` | Redirect to the mounted WebUI at `/ui`. |
| `GET` | `/ui/key` | Show the WebUI API-key prompt. |
| `POST` | `/ui/key` | Exchange the API key for a browser WebUI session. |
| `POST` | `/v1/think` | Ask Celune to think through Persona and reply asynchronously through her normal speech pipeline. |
| `POST` | `/v1/speak` | Queue speech and keep the HTTP request open until `audio/flac` is ready. |
| `POST` | `/v1/speak/async` | Queue speech and return `202 Accepted` immediately with a job ID. |
| `GET` | `/v1/speak/jobs/{job_id}` | Poll an async speech job; returns `202` while pending and `audio/flac` when complete. |
| `POST` | `/v1/speak/jobs/{job_id}/cancel` | Explicitly request cancellation of an active async speech job. |
| `WebSocket` | `/v1/ws/tasks/{job_id}` | Replay and stream typed lifecycle events for an async speech job. |
| `POST` | `/v1/voice` | Change the active voice. |
| `POST` | `/v1/sfx` | Upload and play a sound effect. |
| `POST` | `/v1/convert` | Upload audio for voice conversion in VC mode. |
| `GET` | `/v1/models` | List built-in models, pinned revisions, license details, and consent state. |
| `POST` | `/v1/models/consent` | Record consent for one exact model revision. |

`/v1` and `/v1/version` are authenticated JSON endpoints. The root redirect is
useful when a browser opens the API port; programmatic clients should call the
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
research-only flags, consent state, whether commercial mode is enabled, and
whether each model's reviewed terms confirm commercial use. Custom model
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

`SpeakRequest.content` is a non-empty string and `save` defaults to `true`.
`ThinkRequest` has `content`. `VoiceRequest` has `voice_name`.

Synchronous speech returns `audio/flac`. Async speech returns a task object
with `status`, `job_id`, and `location`; polling returns JSON while queued or
running and FLAC when complete. Task event objects contain `task_id`, `event`,
`status`, optional `message`, `severity`, `current`, `total`, `location`, and
`error`. WebSocket commands currently accept:

```json
{"command":"cancel"}
```

The server validates the command and returns a typed command result. A socket
disconnect removes only that subscriber; it does not cancel the job.

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

- `202 application/json` with `{"status":"accepted"}` when Celune starts processing the Persona request.
- `409 application/json` when Celune is busy.
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

- `200 audio/flac` when speech is generated.
- `409 application/json` when Celune is busy or unavailable.

Submitting another speech request does not cancel speech that is already
playing; the request receives the busy response until the shared pipeline is
available.

## Speak (Asynchronous)

Use `/v1/speak/async` when the client needs a quick acknowledgement and can poll for the result later.

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

When the job returns `200 audio/flac`, save the audio:

```bash
curl http://127.0.0.1:2060/v1/speak/jobs/6f98c8f3a4a94a049d84dd9fce8a63c5 \
  --output celune.flac
```

Job responses:

- `202 application/json` with `{"status":"queued"}` or `{"status":"running"}` while Celune is still working.
- `200 audio/flac` when the generated speech is ready.
- `404 application/json` if the job ID is unknown.
- `500 application/json` if the job failed.

Async jobs are kept in memory for 15 minutes and are not persisted across Celune restarts.

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
`cancelled`. A disconnected WebSocket only removes that client subscription; it
does not stop or alter the Celune task. Cancellation is explicit through
`POST /v1/speak/jobs/{job_id}/cancel` or by sending this API-layer command through
the WebSocket:

```json
{"command":"cancel"}
```

The WebSocket layer never runs speech generation itself and does not expose
internal exception details. Existing polling and synchronous speech endpoints
remain available unchanged.

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

The configured per-minute rate limit applies independently to API calls, WebUI
actions, and WebUI WebSocket connections. WebSocket reconnects do not consume
the WebUI action budget. Read-only Gradio background refresh callbacks do not
count against the action limit. Failed validation, busy-state, missing-job, and
backend errors are returned as JSON with an HTTP status instead of exposing raw
internal tracebacks. Async jobs are retained in memory for 15 minutes and are
not persisted across a restart.

## WebUI behavior

The WebUI is a Gradio application mounted at `/ui`. It uses the same engine,
Persona/agent submission path, slash-command handler, voice state, VC state,
speech lifecycle callbacks, and one-shot upload/microphone VC conversion. TUI
timed state updates are delivered through the CEDTS frontend update channel;
the browser keeps a polling fallback for reconnects and standalone API hosts.
Browser playback is returned as 16-bit PCM; Celune's engine continues to use
normalized float32 audio internally.

Celune links a standalone stylesheet that applies the active palette as CSS.
The three read-only background
refresh callbacks use direct, unqueued requests and are excluded from the
per-minute action limit; speech, conversion, and other WebUI actions remain
limited.

Celune serves its page scripts, styles, and bundled fonts under `/ui/assets/`.
Gradio serves its own hashed JavaScript and CSS bundles from that same path
through the mounted application. A reverse proxy or Cloudflare Tunnel must
forward `/ui/assets/` requests to Celune without rewriting their paths.

Unauthenticated access to `/ui` redirects to `/ui/key`, which displays a
blocking API-key prompt. Submitting the key sends it to `POST /ui/key` in a
JSON `api_key` field. A valid key creates an opaque, `HttpOnly`,
`SameSite=Strict` cookie scoped to `/ui` for 30 days, then the browser returns
to `/ui`. The signed cookie carries a random browser client ID and expiry, not
the API key, and remains valid across Celune restarts. Active sessions are
extended as the browser uses the WebUI. Changing the API key invalidates
existing sessions; another browser without its own cookie still sees the key
prompt. The API key is not stored in the URL or browser storage. The API
security middleware checks the session on WebUI HTTP actions and WebSocket
connections; requests without a valid API key or session receive `401`, even
if the prompt is removed from the page with browser developer tools. API
routes, including read-only status and version routes, also require the API
key. The authentication request is rate limited, and browser writes and
WebSocket connections must come from the same origin.

Celune also remembers a successful browser authentication for 30 days in its
app-data directory. The record is a keyed fingerprint of the client's
resolved address (Cloudflare's forwarded address when tunneled) and browser
request headers; it does not contain the API key, address, or raw headers. This
lets the same client pass the gate after a Quick Tunnel hostname changes and
receive a fresh host-scoped cookie. A different address or browser signature
still receives the key prompt. Changing networks or browser settings can change
that signature and require the key again. Two clients behind one public address
with identical browser headers cannot be distinguished by the server and share
this remembered grant.

The browser submits this body to `POST /ui/key`:

```json
{"api_key":"YOUR_API_KEY"}
```

`api_key` must contain 1 to 4096 characters. A valid key returns
`{"authenticated":true}` with the session cookie. An invalid key returns
`401`, a cross-origin request returns `403`, and an attempt over the configured
request limit returns `429`. The prompt shows “Invalid API key” for `401`
responses; other HTTP or network errors show “An error occurred while
authenticating.”

The WebUI exposes the active voice as a compact button beside the text input;
clicking it cycles through the available voices using the same Core switching
path as the TUI. The voice and VC conversion buttons follow the active Celune
palette, including their disabled, hover, and keyboard-focus states. The WebUI
does not expose the TUI's settings editor. When live Persona or VC capture is
available, press `ALT+R` to toggle recording; the same shortcut is shown in the
bottom-right resource area. If the API extra is absent, the REST/WebUI surface
cannot be started, but the core Python/TUI runtime remains separate.

## Embedding the server

Python integrations can bind the engine explicitly:

```python
from celune.api import run_api, start_api

run_api(celune, host="127.0.0.1", port=2060, token=None)
# Or keep the main thread free:
thread = start_api(celune, host="127.0.0.1", port=2060)
```

`run_api()` blocks in Uvicorn. `start_api()` starts a daemon thread and waits
for startup confirmation; its optional `on_started(host, port)` callback runs
when the socket starts listening. Startup logging reports the listener URL by
default. Set `defer_startup_log=True` with an `on_started` callback when that
callback reports a different public URL; without a callback, Celune still logs
the listener URL. The supporting calls `bind_celune()`,
`configure_api_security()`, `resolve_api_host()`, `audio_bytes()`, and
`stream_headers()` are the existing integration hooks for custom hosts; use
them instead of mounting a second FastAPI app around the engine. Core callback
properties are single-registration slots: assigning the same callable to the
same callback property again raises `ValueError`, so integrations should
replace a handler only when they intentionally change the callback.

## See also

- [Configuration](user-guide/configuration.md) for the API access setting.
- [Validation](development/validation.md) for the repository verification commands.
