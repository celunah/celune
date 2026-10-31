---
name: implement-celune-backend
description: Implement or repair a Celune TTS or voice-conversion backend from upstream inspection through live CEDTS verification. Use when changing a backend adapter, manifest, or worker integration.
---

# Implement a Celune backend

Use this workflow when adding or materially changing a Celune TTS or voice-conversion backend. The goal is a backend that matches its upstream contract, fits Celune's existing architecture, and completes real CEDTS operations without hanging. Follow repository `AGENTS.md` instructions and the user's explicit constraints throughout.

## 1. Inspect the upstream implementation

- Identify the exact upstream package version, source revision, or commit Celune will use. Read the implementation at that version, not only its README or a current online example.
- Trace the model construction, load, inference, streaming, and unload paths. Record required inputs, output types and sample rates, device and dtype behavior, blocking work, error behavior, and which operations are genuinely supported upstream.
- Separate upstream capabilities from Celune's planned adapter behavior. Do not claim streaming, cancellation, or a capability based only on a similarly named API.

## 2. Compare Celune's backend specification with upstream

- Read the backend contract and nearby implementations, plus `docs/development/backends.md`. For worker-backed behavior, trace the relevant paths through `celune/cedts/worker.py`, `celune/cedts/remote.py`, and `celune/cedts/protocol.py`.
- Compare upstream requirements and behavior with Celune's base class, typing contract, manifest, worker entrypoint, configuration, model metadata/consent, audio format, and documented capabilities. Identify each required adapter or compatibility layer before coding.
- Verify model-license facts against the upstream source and keep legal license terms distinct from usage disclaimers and research-only metadata. Preserve Celune's consent flow; do not bypass a pending confirmation.
- Check recent related backend implementations and fixes for reusable patterns. Reuse their demonstrated lessons, but verify each pattern against this backend rather than copying backend-specific assumptions.
- Preserve known intentional upstream opt-outs, fallbacks, and platform constraints. Do not re-enable a disabled upstream optimization unless the user requests it or its compatibility has been established for Celune's full supported pipeline.

## 3. Implement against Celune's existing architecture

- Make the smallest change that fulfills the verified upstream and Celune contracts. Keep backend-only dependencies in the backend environment and avoid importing them from core startup paths.
- Preserve CEDTS framing: worker stdout is protocol transport. Route diagnostics through Celune's logging/progress mechanisms or the established stderr path; never let upstream output corrupt protocol frames.
- Make long startup and model-loading stages observable. Move expensive imports out of request handling only when source/runtime evidence shows they block CEDTS responsiveness and the worker lifecycle provides a safe initialization point.
- Return audio in Celune's normalized `np.float32` format and use the existing backend, buffering, and streaming abstractions. Keep streaming claims consistent with when the first chunk is actually yielded.
- Add the focused tests, manifest/configuration updates, localization, and documentation required by the changed behavior. Do not silently weaken an upstream feature or mask a failure to make integration pass.

## 4. Run the backend through Celune and resolve hangs

- Prefer launching Celune with `python main.py` from the repository root while iterating. This exercises Celune without requiring a compiled rebuild for each backend edit.
- Before a substantial GPU inference check, verify that no other workload is using the required GPU and avoid launching duplicate Celune instances or overlapping generations. If the current request has not already authorized GPU use, ask before starting it.
- Exercise the actual CEDTS lifecycle for the target backend: worker readiness/description, `preload_models`, `load_model`, and inference through `generate_stream` for TTS or the corresponding conversion operation for VC. Exercise unload/reload when the implementation changes that lifecycle. Confirm that output arrives, the operation completes, and streaming begins when the backend claims it does.
- Validate the returned audio: it must be non-empty and have a plausible duration for the utterance (or, for VC, relative to the source audio). Treat empty audio or audio that is implausibly long as an inference failure. Investigate and fix the cause, then repeat the runtime check; do not advance to post-runtime checks while this remains unresolved.
- Observe each operation's start/end, Celune status and logs, CEDTS progress, worker liveness and stderr, and any traceback. Distinguish environment creation, dependency installation, model download, and expected warmup from a stalled operation; elapsed time by itself is not proof of a hang.
- If an operation appears stuck, do not stack another reload, backend switch, or generation on top of it. First collect enough evidence to identify the exact operation and stage: timestamps, last progress/log message, Celune and worker process state, worker stderr/traceback, and whether the operation or its documented timeout resolves. Diagnose the underlying cause instead of treating a longer timeout or a restart alone as a fix.
- After fixing the cause, stop and restart Celune, preferably again with `python main.py`, then repeat the same operation on the same backend. Continue until load and inference complete without hanging. If that cannot be demonstrated, report the backend as not yet runtime-verified and keep investigating before moving on.

## 5. Complete post-runtime checks

Only after Celune has loaded the backend and completed inference without hanging should it be considered implemented. Then finish the repository's required post-task work: update behavior documentation and localization, update docstrings, format and order imports, remove `__pycache__` directories, and run the canonical CI workflow from `AGENTS.md`. Report live-runtime and CI results separately; CI does not substitute for the CEDTS runtime check.
