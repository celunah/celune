# CEDTS transport

CEDTS is Celune's private **C**elune **E**xtensible **D**ata **T**ransport
**S**tandard for isolated backend workers and in-process frontend timed-update
notifications. It is an implementation contract, not a public network API.
The current protocol version is `1`.

The design keeps control frames and binary payloads separate, uses explicit
lengths, and never relies on backend stdout for protocol data. Worker logs are
forwarded through a separate diagnostic path so a third-party library cannot
corrupt the transport by printing to stdout.

## Streams and framing

### Control frames

Each control frame is:

```text
uint32 network-order frame length
UTF-8 compact JSON object
```

JSON is encoded with `allow_nan=False`. A zero-length frame is reserved as a
boundary/termination marker where the stream reader needs one.

### Binary frames

Each binary frame is:

```text
uint32 network-order frame length
uint16 identifier length
uint64 payload length
identifier bytes
payload bytes
```

The frame length covers the binary header, identifier, and payload. Binary
payloads are referenced by name from the JSON envelope instead of being copied
into JSON. A zero-size frame marks the end of a binary collection.

## Packet envelope

Control packets use these fields:

| Field | Meaning |
| --- | --- |
| `cedts_version` | Protocol version, currently `1`. |
| `kind` | Packet kind. |
| `message_id` | Unique ID for a request/event. |
| `reply_to` | Message ID being answered, when applicable. |
| `operation` | Operation or lifecycle verb. |
| `data` | JSON-compatible typed value. |
| `payloads` | Named binary descriptors. |

The packet kinds are `hello`, `hello_ack`, `ready`, `request`, `response`,
`event`, `callback`, `progress`, `cancel`, `cancel_ack`, `error`, `ping`,
`pong`, `shutdown`, and `shutdown_ack`. The `event` kind supports the
`ui_timed_update` operation for TUI-to-WebUI state synchronization.

### Frontend timed updates

The TUI publishes one CEDTS-framed `event` with operation
`ui_timed_update` whenever shared timed state changes. The WebUI accepts only
newer sequence numbers for the bound runtime and uses the transmitted resource
page, theme, status text, severity, and marquee offset. Browser polling remains
as a stale-channel fallback for standalone or reconnecting clients.

The update data has this layout:

| Field | Meaning |
| --- | --- |
| `runtime_id` | Bound Celune runtime identity. |
| `sequence` | Monotonic update number for that runtime. |
| `emitted_at` | TUI monotonic emission timestamp. |
| `resource_page` | Current shared resource-footer page. |
| `theme_name` | Active Celune theme. |
| `status_text` | Current status text. |
| `status_severity` | Current status severity. |
| `status_marquee_offset` | Current TUI marquee offset. |

The channel is intentionally separate from backend worker operations and does
not expose a second public socket or permit arbitrary frontend commands.

## Handshake

The worker sends `hello` with version 1, its supported operations, supported
media types, a `describe` result, and capability flags. The peer replies with
`hello_ack` containing the negotiated intersection. Limits are negotiated by
taking the smaller peer limit. Current capability flags cover streaming,
cancellation, and callbacks; the core callback capability is currently false.

The supported operation names are:

`describe`, `model_is_available_locally`, `preload_models`, `load_model`,
`unload_model`, `convert`, `call`, and `generate_stream`.

Protocol-level operations include `cancel`, `fatal`, `handshake`, `protocol`,
`ready`, and `shutdown`.

## Media and typed values

Supported media types include:

- `audio/pcm_f32le`
- `audio/pcm_s16le`
- `application/octet-stream`
- `application/x-tensor`
- `image/jpeg`
- `image/png`

Supported scalar dtypes include `bool`, `float32`, `float64`, `int8`, `int16`,
and `uint8`. Typed values include `numpy_array`,
`voice_conversion_request`, `audio_output`, `backend_generation`, and `tuple`.

Audio descriptors declare dtype, shape, sample rate, and channel count. Float
audio must already be normalized to -1 through 1 and is peak-limited to 0.95
when encoded. Signed 16-bit audio is converted to float32 using `/32768` when
decoded. The application-level canonical format remains 48 kHz stereo float32;
the descriptor's sample rate is authoritative at the worker boundary.

## Limits

The protocol rejects oversized or deeply nested input before handing it to a
backend. Current limits include:

| Limit | Value |
| --- | ---: |
| Control frame | 1 MiB |
| Binary frame | 8 MiB |
| Aggregate payload collection | 64 MiB |
| Payload descriptors/collections | 1,024 |
| JSON depth | 64 |
| String length | 1 MiB |
| Array shape dimensions | 8 |
| Shape dimension | 16 Mi elements |
| Identifier length | 256 bytes |

These limits are part of the protocol safety boundary. A worker should return a
typed protocol/payload error instead of attempting to allocate an unbounded
buffer.

## Requests, streaming, and cancellation

The worker allows one active request at a time. `generate_stream`, `convert`,
and `call` can be cancelled through the cancellation path where the backend
supports it. Progress packets are throttled to approximately 100 ms. Streaming
outputs carry typed audio or backend-generation values and finish with a
terminal response/event.

Worker stderr is retained for failure diagnostics, but CEDTS applies the same
known-benign runtime-message suppression list used by Celune's local runtime
log redirect. This keeps isolated workers from reintroducing filtered model,
Transformers, tqdm, and Triton notices into the UI log; actionable worker
messages remain visible and retained for error reporting.

State events use `loading`, `ready`, `processing`, `streaming`, `paused`,
`cancelling`, `cancelled`, `completed`, `failed`, and `shutdown_requested`.
Event operations include the matching state names plus `fatal`.

Cancellation is cooperative: the core requests it, the worker acknowledges it,
and the backend decides how quickly to release the current operation. A fatal
protocol error tears down the worker rather than trying to reuse a stream with
unknown framing state.

## Errors and implementation API

The transport defines these exception categories:

`CEDTSError`, `CEDTSStreamError`, `CEDTSEOFError`, `CEDTSTimeoutError`,
`CEDTSProtocolError`, and `CEDTSPayloadError`.

The implementation lives in `celune.cedts.protocol` and is used by
`celune.cedts.worker` and `celune.cedts.remote`. Backend authors should
use the existing encoders/decoders and typed descriptors rather than creating
a parallel socket or serialization layer.
