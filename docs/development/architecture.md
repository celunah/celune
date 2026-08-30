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

`celune.__init__` configures cache defaults and exposes lazy root symbols. A
Celune binary registers the default Celune palette and mounts the loading screen
before importing or initializing the engine, Persona, agent runtime, backend
environments, audio backends, or model libraries. The first-frame path is
limited to launcher state and lightweight Textual/loading/theme primitives; the
existing post-frame worker loads the selected runtime and later applies
pack-derived colors. This keeps startup responsive and lets `CTRL+C` use the UI
shutdown path even while runtime loading fails.

`Celune`, `CeluneUI`, and `CeluneHeadlessUI` are marked `@final` and reject
runtime subclass creation. Integrations should use their callbacks, protocols,
backend interfaces, and composition points rather than inheriting from these
single-instance runtime classes.

When verbose or debug logging is enabled, the loading overlay shows only three
startup checkpoints: checking dependencies immediately on process startup,
loading the core at `Celune` construction, and initializing the core during
`Celune.load()`. The checkpoints are emitted at those boundaries, so the
loading screen does not present later UI/runtime/model work as separate startup
stages. The current log line reports the latest information-level runtime
message while initialization is in progress.

The terminal title follows the same startup and runtime transitions using the
format `Celune ・ state ・ action`. The state and action portion is limited to
20 characters, so titles remain visible in narrow terminal tabs; detailed
status and diagnostic text stays in the loading screen or main log.

## Speech pipeline

1. A frontend submits text, audio, or an SFX request.
2. The engine validates the current mode and component locks.
3. Text is normalized/segmented and placed in the bounded text queue.
4. The backend generates chunks, locally or through a CEDTS worker.
5. The smart buffer and contention monitor decide when to start and how much
   output reserve to maintain.
6. Audio is resampled/normalized, optionally pitch-shifted or reverberated.
7. A persistent playback input reader feeds the mixer, and a persistent audio
   writer drains mixed blocks to the sounddevice stream. The mixer yields
   briefly after each submitted block so the writer can keep its reserve moving
   during CPU contention. Playback traces are sampled rather than written for
   every block, separating bounded-queue backpressure, per-source generation
   gaps, writer scheduling gaps, stream-write duration, and adaptive rebuffer
   waiting without adding file I/O to the audio clock.
8. The final 48 kHz stereo FLAC is saved when requested.

Cancellation travels through the same queue and worker boundary. It does not
create a second “stop” path that could leave a backend generation alive.
Playback completion is source-aware: speech completion closes speech captions
without waiting for unrelated SFX overlays, while the global idle transition
still waits for every source. Output failures clear the source maps, mark
playback complete, and release any held pipeline lease.

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
matching interpreter. See [Backends](backends.md).

The worker's stdout/stderr is not the control channel. CEDTS control frames,
binary payloads, progress, cancellation, fatal state, and frontend timed
updates are typed and length framed. The TUI publishes timed UI state through
the in-process frontend channel so the WebUI does not maintain a second clock.
See [CEDTS](../formats/cedts.md).

## Persona and agent

Persona keeps bounded short-term history and character-scoped long-term memory.
Agent mode adds planning, schema validation, approvals, and task lifecycle on
top of Persona. The agent can choose only registered tools; local management is
an explicit opt-in catalog, not a hidden fallback.

## Shutdown

UI unmount, `CTRL+Q`, API shutdown, process-loss detection, and `Celune.close()`
converge on idempotent teardown. Active live recording is stopped, workers are
asked to shut down, streams are closed, models release their state, and event
listeners receive terminal notifications. Pipeline blocking work uses daemon
threads rather than the event loop's default executor, so a cancelled backend
operation cannot make `asyncio.run()` wait for the executor's 300-second join
window during restart. A fatal state can stop generation without pretending
that the engine is healthy.
