# Celune REST API

Celune can expose a local REST API for speech, sound effects, and voice control.
The API is enabled by default in `default_config.yaml`:

```yaml
api:
  enabled: true
  host: 0.0.0.0
  port: 2060
  token: null
  rate_limit_per_minute: 60
```

If no token is configured, Celune binds the API to `127.0.0.1`.
If a token is configured through `api.token` or `CELUNE_API_TOKEN`, Celune can bind to `0.0.0.0`.

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
| `POST` | `/v1/speak` | Queue speech and keep the HTTP request open until `audio/flac` is ready. |
| `POST` | `/v1/speak/async` | Queue speech and return `202 Accepted` immediately with a job ID. |
| `GET` | `/v1/speak/jobs/{job_id}` | Poll an async speech job; returns `202` while pending and `audio/flac` when complete. |
| `POST` | `/v1/voice` | Change the active voice. |
| `POST` | `/v1/sfx` | Upload and play a sound effect. |

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
