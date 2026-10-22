# Source map

This page maps the repository's implementation packages to the behavior they
own. It is the index to use when a feature needs a precise code entrypoint.

## Runtime and public surfaces

| Path | Responsibility |
| --- | --- |
| `celune/__init__.py` | Lazy package exports and build metadata. |
| `celune/exceptions.py` | Canonical Celune-owned exception hierarchy and structured failure metadata. |
| `celune/celune.py` | Concrete singleton façade, state properties, constructor, and declared public calls. |
| `celune/core/lifecycle.py` | Engine construction, shutdown, cleanup, worker startup, Persona lifecycle, and callbacks. |
| `celune/core/loading.py` | Runtime loading, warmup, normalizer loading, and normalization. |
| `celune/core/reload.py` | Backend/character reload transactions, rollback, and language reload coordination. |
| `celune/core/voice.py` | Voice, backend, and CEVOICE changes plus extension setup. |
| `celune/core/sleep.py` | Sleep/wake orchestration and runtime restoration. |
| `celune/core/speech.py` | Engine-level speech, audio submission, and SFX entry points coordinated with the pipeline. |
| `celune/core/conversation.py` | Persona request orchestration, response delivery, and the conversation worker. |
| `celune/core/agent.py` | Engine-specific agent coordination and result delivery. |
| `celune/core/callbacks.py` | State, callback, status, and event publication helpers. |
| `celune/core/concurrency.py` | Shared component ownership locks and daemon-thread async bridging. |
| `celune/entrypoint.py` | CLI dispatch, CEVOICE pack commands, startup, doctor, config commands, updates, and headless/interactive handoff. |
| `celune/updater.py` | Signed compiled updates, Git updates, and launcher handoff. |
| `celune/signature.py` | Ed25519 manifest verification and appended keyring handling. |
| `main.py` | Lightweight interpreter-safe launcher. |
| `celune/config.py` | Config coercion, merge/default behavior, logging, device, and environment lookup. |
| `celune/constants.py` | Names, sample-rate/VRAM/context constants, release-keyring trust anchor, and exit codes. |
| `celune/terminal.py` | TTY capability, terminal title, and runtime log handling. |
| `celune/modes.py` | Operation-mode resolution and mode capability predicates. |
| `celune/paths.py` | User data, cache, runtime, pack, output, config, and migration paths. |
| `celune/runtime.py` | Lazy runtime imports and shared test/headless runtime assembly. |

## Audio and pipeline

| Path | Responsibility |
| --- | --- |
| `celune/pipeline/submission.py` | Speech/audio request submission, prepared audio, singing setup, and SFX requests. |
| `celune/pipeline/generation.py` | Generation worker, segmentation, synthesis, and generated chunk handling. |
| `celune/pipeline/buffering.py` | Bounded buffering, contention policy, and release coordination. |
| `celune/pipeline/playback.py` | Playback workers, stream lifecycle, cancellation, and signal playback. |
| `celune/pipeline/captions.py` | Caption alignment, playback-source tracking, and completion state. |
| `celune/pipeline/output.py` | Saved speech/SFX output and output metadata helpers. |
| `celune/pipeline/timing.py` | Shared speech-pipeline timing constants. |
| `celune/audio/server.py` | Host audio-server recovery. |
| `celune/audio/resampling.py` | Sample-rate conversion at boundaries. |
| `celune/audio/dsp.py` | Pitch/speed support, readiness signals, silence detection, and streaming reverb. |
| `celune/vc.py` | Stable VC helper entrypoints for normalization, RMS/VAD, live conversion, and diagnostic capture. |
| `celune/modeling.py` | CeluneNorm loading and quantized autoregressive cache helpers. |
| `celune/chroma.py` | OpenRGB connection, audio reactivity, sleep/wake, and fatal glow state. |
| `celune/theme/colors.py` | Built-in themes and pack color integration. |
| `celune/analysis.py` | Voice metrics, embedding similarity, traits, assessments, plots, and reports. |

## Backend boundary

| Path | Responsibility |
| --- | --- |
| `celune/backends/tts/base.py` | TTS backend contract and common model/voice behavior. |
| `celune/backends/tts/contracts.py` | Pinned TTS model revisions, weight inventories, and validation helpers. |
| `celune/backends/tts/quantization.py` | GPU capability selection and contract-approved TorchAO weight quantization. |
| `celune/backends/tts/mini.py` | Pocket TTS adapter. |
| `celune/backends/tts/qwen3.py` | Qwen3 streaming voice-cloning adapter. |
| `celune/backends/tts/fireredtts3.py` | FireRedTTS3 CEDTS adapter, source bootstrap, and BF16 loading. |
| `celune/backends/tts/voxcpm2.py` | VoxCPM2 streaming adapter and CFG metadata. |
| `celune/backends/tts/dotstts.py` | Celune's forked dots.tts adapter. |
| `celune/backends/tts/luxtts.py` | CUDA-first LuxTTS voice-cloning adapter with CPU fallback. |
| `celune/backends/vc/base.py` | Voice-conversion backend contract. |
| `celune/backends/vc/seedvc.py` | Seed-VC file and live conversion adapter. |
| `celune/backends/environment.py` | Per-backend manifests, uv environments, fingerprints, and readiness. |
| `celune/cedts/remote.py` | Core-side proxy around a worker process. |
| `celune/cedts/worker.py` | Worker-side operation dispatch and lifecycle. |
| `celune/cedts/protocol.py` | CEDTS serialization, framing, negotiation, and limits. |
| `celune/cedts/ui.py` | CEDTS-framed TUI-to-WebUI timed-update channel. |
| `celune/cedts/bootstrap.py` | CEDTS worker process bootstrap and isolated import setup. |

## Character and Persona

| Path | Responsibility |
| --- | --- |
| `celune/cevoice.py` | CEVOICE/CECHAR readers, writers, validation, lazy asset materialization, and pack selection. |
| `celune/cesn.py` | CESN parsing, X-SAMPA conversion, singing validation/preparation, and canonical rendering. |
| `celune/persona/runtime.py` | Model loading, quantization, generation, and context capacity. |
| `celune/persona/cache.py` | Device-aware INT8/FP8 Persona KV-cache storage and dynamic-cache fallback boundary. |
| `celune/persona/impl.py` | Configured Persona client and engine integration. |
| `celune/persona/conversation.py` | Persona prompt construction, context, history, memory extraction, and request helpers. |
| `celune/persona/asr.py` | Whisper transcription, timestamps, and speech-input lifecycle. |
| `celune/persona/capabilities.py` | Text/vision/upload/emotion capability detection. |
| `celune/persona/memory.py` | Character-scoped JSON memory, semantic retrieval, fallback matching, and filtering. |
| `celune/persona/emotion.py` | Emotion model outputs and response-style cues. |
| `celune/persona/prompts.py` | System/task prompts and routing contracts. |
| `celune/persona/paths.py` | Character slug and Persona override paths. |

## Agent and integrations

| Path | Responsibility |
| --- | --- |
| `celune/agent/runtime.py` | Task state, iteration accounting, approvals, choices, cancellation, and execution. |
| `celune/agent/routing.py` | Input/approval/choice/cancellation routing. |
| `celune/core/agent.py` | Core-owned coordination between routed requests, engine state, tools, and speech delivery. |
| `celune/agent/tools.py` | Typed built-in and optional local-management tool catalog. |
| `celune/agent/needle/__init__.py` | Public Needle model, checkpoint, and tool-selection package surface. |
| `celune/agent/needle/impl.py` | Registered-tool selection and schema validation. |
| `celune/agent/needle/checkpoints.py` | Needle checkpoint validation/preparation. |
| `celune/agent/needle/models.py` | Inference-only PyTorch Needle model implementation. |
| `celune/agent/persona.py` | Persona-to-agent bridge. |
| `celune/agent/contracts.py` | Public agent data contracts. |
| `celune/extensions/base.py` | Extension context and base class. |
| `celune/extensions/manager.py` | Discovery, registration, invocation, and teardown. |
| `celune/extensions/events.py` | Event names, subscription, and dispatch. |
| `celune/dataclasses/events.py` | Event payload dataclasses. |
| `celune/api.py` | Documented API entrypoints, FastAPI application wiring, endpoints, WebUI mount, and public-tunnel lifecycle. |
| `celune/server/session.py` | Shared typed API/WebUI session state. |
| `celune/server/auth.py` | API authentication and rate limiting. |
| `celune/server/jobs.py` | Speech jobs, task subscriptions, and transport audio helpers. |
| `celune/webui/` | Gradio presentation, actions, and command handlers. |
| `celune/ui/commands.py` | Slash-command parser and command behavior. |
| `celune/ui/app.py` | Textual application composition and presentation hooks. |
| `celune/ui/capture.py`, `runtime.py`, `interaction.py`, `theme.py` | Textual recording, startup/runtime, interaction, and theme workflows. |
| `celune/ui/headless.py` | Non-interactive log/status surface. |

## Scripts and bundled data

| Path | Responsibility |
| --- | --- |
| `scripts/cac.py` | Interactive/simple CEVOICE pack creator. |
| `scripts/run_ci.py` | Canonical Windows-safe CI process wrapper. |
| `scripts/update_docstrings.py` | Docstring coverage helper; inspect its diff for mechanical collateral. |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/` | Pinned libsodium build, native launcher verification, Nuitka core build, and release packaging. |
| `scripts/make_signing_key.py` | Protected release-key creation and catalog registration. |
| `scripts/append_release_keyring.py` | Root-sign and append the release-key catalog to the launcher. |
| `scripts/celune-bin.cmd` | Windows compiled-binary handoff. |
| `scripts/ci_warnings.py` | CI warning annotations. |
| `scripts/write_update_manifest.py` | Release update metadata. |
| `celune/release_keyring.json` | Public manifest-signing key catalog embedded in signed launcher trailers. |
| `launcher/release_keys.h` | Pinned keyring root trust anchor for the C launcher. |
| `celune/lang/` | Localized runtime strings. |
| `voices/` | Bundled CEVOICE packs. |

The tests mirror these boundaries: `tests/core/`, `tests/pipeline/`,
`tests/persona/`, `tests/agent/`, `tests/agent/needle/`, `tests/backends/`,
`tests/server/`, and `tests/ui/`. Standalone CEVOICE and CESN tests remain under
`tests/` because those packages are cohesive modules with their own contracts.
