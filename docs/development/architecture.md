# Architecture

Celune is organized as one core engine with several frontends and isolated
model workers. The important rule is that frontends share the engine; they do
not implement separate speech pipelines.

## Runtime layers

```text
CLI / Textual UI / Headless UI / REST + WebUI / Extensions
                         │
                    Celune engine
       ┌─────────────────┼─────────────────┐
   Persona             Agent             Pipeline
       │                 │                 │
 CEVOICE/CECHAR     typed tools       buffers/DSP/audio
                         │                 │
              core-side backend proxy
                         │ CEDTS
                isolated backend worker
```

The engine owns configuration, singleton lifecycle, queues, component locks,
events, model readiness, and shutdown. Persona and agent requests enter the
same speech pipeline used by direct TTS. Voice conversion enters the same
playback path through an audio-input request.

## Startup and lazy imports

`celune.__init__` configures cache defaults and exposes lazy root symbols. The
entrypoint mounts the loading UI before importing heavy model/UI dependencies.
This keeps a blank/pre-loading phase responsive and prevents a model-library
import from taking over the first visible frame. Runtime dependencies are
loaded only when the selected path needs them.

## Speech pipeline

1. A frontend submits text, audio, or an SFX request.
2. The engine validates the current mode and component locks.
3. Text is normalized/segmented and placed in the bounded text queue.
4. The backend generates chunks, locally or through a CEDTS worker.
5. The smart buffer decides when to start and how to protect playback.
6. Audio is resampled/normalized, optionally pitch-shifted or reverberated.
7. The audio worker writes to the sounddevice stream and emits lifecycle events.
8. The final 48 kHz stereo FLAC is saved when requested.

Cancellation travels through the same queue and worker boundary. It does not
create a second “stop” path that could leave a backend generation alive.

## Component locks

Reloads and concurrent operations use named component locks for TTS, VC, audio,
Persona, and agent work. A voice/backend/character switch captures a rollback
snapshot before replacing state. If a reload fails, the runtime attempts to
restore the previous working configuration rather than leaving half-loaded
objects exposed.

## Backend isolation

Core application dependencies and backend dependencies are separate. A
`BackendManifest` declares the worker requirements, module/class entrypoint,
Python version, indexes, and fingerprint. The environment manager creates a
per-backend `venv`, records `manifest.json`, and launches the worker with the
matching interpreter. See [Backends and environments](backends.md).

The worker's stdout/stderr is not the control channel. CEDTS control frames,
binary payloads, progress, cancellation, and fatal state are typed and length
framed. See [CEDTS](../formats/cedts.md).

## Persona and agent

Persona keeps bounded short-term history and character-scoped long-term memory.
Agent mode adds planning, schema validation, approvals, and task lifecycle on
top of Persona. The agent can choose only registered tools; local management is
an explicit opt-in catalog, not a hidden fallback.

## Shutdown

UI unmount, `CTRL+Q`, API shutdown, process-loss detection, and `Celune.close()`
converge on idempotent teardown. Active live recording is stopped, workers are
asked to shut down, streams are closed, models release their state, and event
listeners receive terminal notifications. A fatal state can stop generation
without pretending that the engine is healthy.
