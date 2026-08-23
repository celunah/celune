# Source map

This page maps the repository's implementation packages to the behavior they
own. It is the index to use when a feature needs a precise code entrypoint.

## Runtime and public surfaces

| Path | Responsibility |
| --- | --- |
| `celune/__init__.py` | Lazy package exports and build metadata. |
| `celune/celune.py` | Singleton engine, lifecycle, backend/voice switching, Persona, agent, and public calls. |
| `celune/entrypoint.py` | CLI dispatch, startup, doctor, config commands, updates, and headless/interactive handoff. |
| `main.py` | Lightweight interpreter-safe launcher. |
| `celune/config.py` | Config coercion, merge/default behavior, logging, device, and environment lookup. |
| `celune/constants.py` | Names, sample-rate/VRAM/context constants, model registry, and exit codes. |
| `celune/modes.py` | Operation-mode resolution and mode capability predicates. |
| `celune/paths.py` | User data, cache, runtime, pack, output, config, and migration paths. |
| `celune/runtime.py` | Lazy runtime imports and shared test/headless runtime assembly. |

## Audio and pipeline

| Path | Responsibility |
| --- | --- |
| `celune/pipeline.py` | Text segmentation, generation queue, smart buffer, playback worker, SFX, cancellation, and audio input routing. |
| `celune/audio/server.py` | Host audio-server recovery. |
| `celune/audio/resampling.py` | Sample-rate conversion at boundaries. |
| `celune/audio/dsp.py` | Pitch/speed support, readiness signals, silence detection, and streaming reverb. |
| `celune/vc.py` | VC normalization, RMS/VAD, preroll/hangover, and live-capture helpers. |
| `celune/chroma.py` | OpenRGB connection, audio reactivity, sleep/wake, and fatal glow state. |
| `celune/theme/colors.py` | Built-in themes and pack color integration. |
| `celune/analysis.py` | Voice metrics, embedding similarity, traits, assessments, plots, and reports. |

## Backend boundary

| Path | Responsibility |
| --- | --- |
| `celune/backends/tts/base.py` | TTS backend contract and common model/voice behavior. |
| `celune/backends/tts/mini.py` | Pocket TTS adapter. |
| `celune/backends/tts/qwen3.py` | Qwen3 streaming voice-cloning adapter. |
| `celune/backends/tts/voxcpm2.py` | VoxCPM2 streaming adapter and CFG metadata. |
| `celune/backends/tts/dotstts.py` | Celune's forked dots.tts adapter. |
| `celune/backends/tts/gpt_sovits.py` | GPT-SoVITS family adapter. |
| `celune/backends/vc/base.py` | Voice-conversion backend contract. |
| `celune/backends/vc/seedvc.py` | Seed-VC file and live conversion adapter. |
| `celune/backends/environment.py` | Per-backend manifests, uv environments, fingerprints, and readiness. |
| `celune/cedts/remote.py` | Core-side proxy around a worker process. |
| `celune/cedts/worker.py` | Worker-side operation dispatch and lifecycle. |
| `celune/cedts/protocol.py` | CEDTS serialization, framing, negotiation, and limits. |
| `celune/cedts/bootstrap.py` | CEDTS worker process bootstrap and isolated import setup. |

## Character and Persona

| Path | Responsibility |
| --- | --- |
| `celune/cevoice.py` | CEVOICE/CECHAR readers, writers, validation, lazy asset materialization, and pack selection. |
| `celune/persona/runtime.py` | Model loading, quantization, generation, and context capacity. |
| `celune/persona/impl.py` | Configured Persona client and engine integration. |
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
| `celune/agent/tools.py` | Typed built-in and optional local-management tool catalog. |
| `celune/agent/needle.py` | Registered-tool selection and schema validation. |
| `celune/agent/needle_checkpoint.py` | Needle checkpoint validation/preparation. |
| `celune/agent/persona.py` | Persona-to-agent bridge. |
| `celune/agent/contracts.py` | Public agent data contracts. |
| `celune/extensions/base.py` | Extension context and base class. |
| `celune/extensions/manager.py` | Discovery, registration, invocation, and teardown. |
| `celune/extensions/events.py` | Event names, subscription, and dispatch. |
| `celune/dataclasses/events.py` | Event payload dataclasses. |
| `celune/api.py` | FastAPI models, endpoints, WebSocket tasks, and WebUI mount. |
| `celune/ui/commands.py` | Slash-command parser and command behavior. |
| `celune/ui/app.py` | Textual UI, loading, captions, recording, themes, and shutdown. |
| `celune/ui/headless.py` | Non-interactive log/status surface. |

## Scripts and assets

| Path | Responsibility |
| --- | --- |
| `scripts/cac.py` | Interactive/simple CEVOICE pack creator. |
| `scripts/run_ci.py` | Canonical Windows-safe CI process wrapper. |
| `scripts/update_docstrings.py` | Docstring coverage helper; inspect its diff for mechanical collateral. |
| `scripts/build_nuitka.ps1` / `.sh` | Windows/Linux launcher builds. |
| `scripts/celune-bin.cmd` | Windows compiled-binary handoff. |
| `scripts/ci_warnings.py` | CI warning annotations. |
| `scripts/write_update_manifest.py` | Release update metadata. |
| `celune/assets/` | Tutorial audio and packaged runtime assets. |
| `voices/` | Bundled CEVOICE packs. |

The tests mirror these boundaries. `tests/test_cedts_*`, `tests/test_cevoice_*`,
`tests/test_agent_*`, `tests/test_persona_*`, API tests, UI-command tests, and
pipeline tests are useful starting points when changing one subsystem.
