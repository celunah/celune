# REST API

Celune can expose a local REST API for speech, sound effects, and voice control.
The API is enabled by default in `default_config.yaml`:

```yaml
api:
  enabled: true
  host: 127.0.0.1
  port: 2060
  token: null
  rate_limit_per_minute: 60
```

If no token is configured, Celune binds the API to `127.0.0.1`.
If a token is configured through `api.token` or `CELUNE_API_TOKEN`, Celune can bind to `0.0.0.0`.

The API is an optional core extra. Install it with `uv sync --extra api` (or
include `--all-extras` during Linux development). Keep a network-facing listener
behind authentication and an appropriate firewall; the API can start speech,
load voices, and play uploaded audio.

Authenticated requests may send either header:

```http
Authorization: Bearer YOUR_TOKEN
X-Celune-Token: YOUR_TOKEN
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1` | Return Celune's current status. |
| `GET` | `/v1/version` | Return the running Celune version. |
| `GET` | `/` | Redirect to the mounted WebUI at `/ui`. |
| `POST` | `/v1/think` | Ask Celune to think through Persona and reply asynchronously through her normal speech pipeline. |
| `POST` | `/v1/speak` | Queue speech and keep the HTTP request open until `audio/flac` is ready. |
| `POST` | `/v1/speak/async` | Queue speech and return `202 Accepted` immediately with a job ID. |
| `GET` | `/v1/speak/jobs/{job_id}` | Poll an async speech job; returns `202` while pending and `audio/flac` when complete. |
| `POST` | `/v1/speak/jobs/{job_id}/cancel` | Explicitly request cancellation of an active async speech job. |
| `WebSocket` | `/v1/ws/tasks/{job_id}` | Replay and stream typed lifecycle events for an async speech job. |
| `POST` | `/v1/voice` | Change the active voice. |
| `POST` | `/v1/sfx` | Upload and play a sound effect. |
| `POST` | `/v1/convert` | Upload audio for voice conversion in VC mode. |

`/v1` and `/v1/version` are JSON endpoints. The root redirect is useful when a
browser opens the API port; programmatic clients should call the `/v1` paths
directly. `/ui` is the Gradio WebUI mounted by the optional API runtime.

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
  -F "pitch_shift=-2" \
  -F "f0_condition=false" \
  --output converted.flac
```

`file` is required. `pitch_shift` is an optional integer override and
`f0_condition` is an optional boolean override. The endpoint returns `audio/flac`
when conversion succeeds, `409` when VC mode is not active or the engine is
busy, `400` for malformed audio/fields, `413` when the upload limit is
exceeded, and `500` when the conversion backend raises an internal request
failure. See [Voice conversion](user-guide/voice-conversion.md) for the
interactive capture path.

## Authentication, limits, and errors

When a token is configured, send either:

```http
Authorization: Bearer YOUR_TOKEN
```

or:

```http
X-Celune-Token: YOUR_TOKEN
```

The configured per-minute rate limit applies to API calls. Failed validation,
busy-state, missing-job, and backend errors are returned as JSON with an HTTP
status instead of exposing raw internal tracebacks. Async jobs are retained in
memory for 15 minutes and are not persisted across a restart.

## WebUI behavior

The WebUI is a Gradio application mounted at `/ui`. It uses the same engine,
Persona/agent submission path, slash-command handler, voice state, VC state,
speech lifecycle callbacks, logs, and stop control as the Textual UI. TUI
timed state updates are delivered through the CEDTS frontend update channel;
the browser keeps a polling fallback for reconnects and standalone API hosts.
The browser also provides a voice selector and one-shot upload/microphone VC
conversion. Its Record button delegates Persona or live VC capture to the
active Textual runtime, and its Settings editor persists the same YAML
configuration. If the API extra is absent, the REST/WebUI surface cannot be
started, but the core Python/TUI runtime remains separate.

## Embedding the server

Python integrations can bind the engine explicitly:

```python
from celune.api import run_api, start_api

run_api(celune, host="127.0.0.1", port=2060, token=None)
# Or keep the main thread free:
thread = start_api(celune, host="127.0.0.1", port=2060)
```

`run_api()` blocks in Uvicorn. `start_api()` starts a daemon thread and waits
for startup confirmation. The supporting calls `bind_celune()`,
`configure_api_security()`, `resolve_api_host()`, `audio_bytes()`, and
`stream_headers()` are the existing integration hooks for custom hosts; use
them instead of mounting a second FastAPI app around the engine. Core callback
properties are single-registration slots: assigning the same callable to the
same callback property again raises `ValueError`, so integrations should
replace a handler only when they intentionally change the callback.
