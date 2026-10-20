# MCP server

The MCP server lets MCP-compatible clients call Celune's live runtime. It shares the REST API process, authentication, and request limits, so tools act on the same active backend and voice pack as the app.

## Connect

The server uses Streamable HTTP at `/mcp`. With the default API port, connect to:

```text
http://127.0.0.1:2060/mcp
```

Send the Celune API key in the `Authorization` header:

```http
Authorization: Bearer YOUR_API_KEY
```

The API key is the configured `api.token`, `CELUNE_API_TOKEN`, or generated key stored in the app-data `api.key` file. The endpoint uses the same authentication and per-client request limit as REST routes. For network access, use the machine address accepted by the API listener; MCP validates the Host against loopback and the machine's active network interfaces.

Install the optional API dependencies with `uv sync --extra api`. The MCP endpoint starts with the API and shuts down with it. It does not start a second Celune process.

## Tools

| Tool | Arguments | Result |
| --- | --- | --- |
| `speak` | `content: str`, `save: bool = true` | Generated FLAC audio. Requires a speaking backend. |
| `sing` | `content: str`, `save: bool = true` | Generated FLAC audio. Requires a singing backend and valid CESN. |
| `run_persona` | `content: str` | Text confirming that Persona accepted the request. |
| `change_voice` | Exactly one of `voice_name: str` or `pack_name: str` | Text confirming the selected voice or pack. |
| `change_backend` | `backend_name: str` | Text confirming the selected installed TTS or VC backend. |

Speech and singing tools return an MCP audio content block with MIME type `audio/flac`. They queue audio through Celune's normal generation and playback path. `save` defaults to `true`, matching `/v1/speak` and `/v1/sing`.

`change_voice` accepts either voice name from the active pack, or an installed voice pack's filename or stem. A voice name must exist in the active pack and is subject to Celune's normal reload rules. Voice selection remains locked while a singing backend is active. A pack change on a singing backend succeeds only when the selected pack contains a singing entry.

`change_backend` accepts the canonical installed backend names, including TTS and voice-conversion backends. A failed reload is returned as a tool error from the normal Celune control path.

## Errors and compatibility

`sing` checks the active backend and parses CESN before queueing. Missing notation or parse failures return a tool error; CESN parse diagnostics include the source line and column. REST clients receive `409 Conflict` when singing is unavailable or the request cannot be queued, and `422 Unprocessable Content` for invalid notation.

The MCP endpoint is available only when the optional API dependencies are installed. Local and LAN MCP clients must send the API key. See [REST API](../API.md) for API access settings and [CESN](../formats/cesn.md) for score syntax.

## See also

- [REST API](../API.md)
- [CESN format](../formats/cesn.md)
