# Python API

The package root is intentionally small:

```python
from celune import Celune, CeluneContext, CeluneExtension, subscribe
```

Only create one `Celune` engine and one UI owner per process. Multiple engines
can load duplicate GPU models, compete for audio devices, and violate the
runtime's singleton lifecycle.

## Package metadata

| Name | Meaning |
| --- | --- |
| `celune.__version__` | Project version. |
| `celune.REVISION` | Build/revision marker. |
| `celune.__tagline__` | User-facing tagline. |
| `celune.__codename__` | Release codename. |
| `celune.__comment__` | Release comment. |

The root lazily resolves `Celune`, `CeluneContext`, `CeluneExtension`, and
`subscribe` so importing `celune` does not eagerly load the heavy engine/UI.

## `Celune` construction

The constructor accepts a configuration mapping plus optional backend, VC,
device, callback, log-level, agent-selector, and test-mode arguments. The
common integration shape is:

```python
from celune import Celune

engine = Celune(config={"mode": "speak", "voice_bundle": "default"})
try:
    engine.load()
    engine.say("Hello from Python.")
finally:
    engine.close()
```

The constructor does not replace the need for `load()`: loading initializes the
selected model and backend resources. The engine owns queues, model state,
Persona state, events, and audio lifecycle until `close()`.

## Lifecycle and state

| Call/property | Contract |
| --- | --- |
| `load() -> bool` | Load the configured character, backend, model, and audio runtime. |
| `close() -> None` | Stop work, release backend/model/audio state, and make shutdown idempotent. |
| `fatal() -> None` | Enter the fatal runtime path and notify listeners. |
| `wait_until_idle_async(timeout=None) -> bool` | Await the shared playback/generation idle condition. |
| `sleep_enabled` | Whether configured sleep is active. |
| `sleep_timeout_seconds` | Resolved idle timeout. |
| `enter_sleep_mode() -> bool` | Unload configured components and enter sleep. |
| `wake_from_sleep() -> bool` | Restore the runtime synchronously. |
| `enter_sleep_mode_async() -> bool` | Async sleep operation. |
| `wake_from_sleep_async() -> bool` | Async wake operation. |
| `loaded`, `sleeping`, `cur_state`, `locked` | Read lifecycle and queue state. |
| `mode`, `input_mode` | Read or set the resolved operation/input route where the caller owns mode control. |
| `backend`, `vc_backend` | Active backend objects. Use their stable names/capabilities rather than backend-specific imports. |
| `tts_backend`, `model_name` | Active backend/model identifiers. |
| `current_voice`, `voices` | Selected voice and available voice names. |
| `current_character`, `voice_bundle_is_default` | Selected character identity and pack provenance. |

`load()` may download models and create isolated environments. Callers that run
inside an event loop should prefer the async reload methods below.

## Backends, voices, and character packs

| Call | Contract |
| --- | --- |
| `set_voices(voices: tuple[str, ...])` | Replace the exposed voice-name sequence for an integration or test backend. |
| `load_voice_bundle(bundle=None) -> bool` | Load/validate a pack and update its voice metadata. |
| `load_available_voices() -> bool` | Refresh voice availability from the active bundle/backend. |
| `set_voice(name) -> bool` | Request a voice change without waiting for the reload to finish. |
| `set_voice_and_wait(name, timeout=30.0) -> bool` | Change voice and wait for completion. |
| `set_voice_async(name, timeout=30.0) -> bool` | Async voice change. |
| `set_backend(spec) -> bool` | Request a backend change. |
| `set_backend_and_wait(spec, timeout=30.0) -> bool` | Change backend and wait for completion. |
| `set_backend_async(spec, timeout=30.0) -> bool` | Async backend change. |
| `set_cevoice(bundle) -> bool` | Request a CEVOICE/CECHAR change. |
| `set_cevoice_and_wait(bundle, timeout=30.0) -> bool` | Change character pack and wait for completion. |
| `set_cevoice_async(bundle, timeout=30.0) -> bool` | Async character-pack change. |
| `with_backend(name)` | Context manager that restores the previous backend. |
| `with_cevoice(bundle)` | Context manager that restores the previous character pack. |
| `change_voice(voice)` | Callback-oriented voice-change path used by the pipeline. |
| `voice_prompt_supported()` | Report whether the active backend supports voice prompting. |
| `effective_voice_prompt` | Read the prompt after backend capability filtering. |
| `voice_prompt` | Get/set the active voice prompt. |
| `set_voice_prompt` through agent/API surfaces | Use the validated runtime property rather than mutating backend internals. |

Reload calls coordinate with component locks and the CEDTS worker. A caller that
needs to speak immediately after a change should await the `*_and_wait()` or
async form.

## Speech and Persona

| Call | Contract |
| --- | --- |
| `say(text, save=True, display_text=None) -> bool` | Queue literal text for TTS. `display_text` changes only logs/captions. |
| `say_async(text, save=True, display_text=None) -> bool` | Non-blocking async form of `say`. |
| `say_stream(text, save=True) -> SpeechStreamQueue` | Queue speech and mirror generated audio chunks to a bounded queue. |
| `say_stream_async(text, save=True) -> SpeechStreamQueue` | Async form of streaming speech. |
| `think(text) -> bool` | Route text through Persona and the shared speech pipeline. |
| `think_async(text) -> bool` | Async Persona turn. |
| `normalize(text) -> Optional[str]` | Apply the optional text normalizer; returns `None` when unavailable/rejected. |
| `force_stop_speech() -> bool` | Interrupt active speech and queued generation. |
| `force_stop_speech_async() -> bool` | Async interruption path. |
| `try_play_signal(name) -> bool` | Play a readiness/sleep/working/error signal. |
| `is_voice_conversion_mode` | Report whether text-to-speech calls are currently unavailable. |
| `vision.capabilities()` | Read text/vision/upload/emotion support when Persona is loaded. |
| `persona_history`, `persona_session_summary` | Read bounded session context for integrations. |
| `persona_attachments` | Pending attachment descriptors for the next vision turn. |
| `agent_needle_ready`, `agent_needle_error` | Read agent selector availability/diagnostics. |

`say()` and `think()` use the same playback queue used by the UI/API. Do not
call a backend's generator directly from application code.

## Audio and voice conversion

| Call | Contract |
| --- | --- |
| `submit_audio(audio, sample_rate, label="audio input", pitch_shift=None, f0_condition=None, ...) -> bool` | Submit audio to the active non-TTS mode for playback/conversion. |
| `convert_audio(audio, sample_rate, label="audio input", pitch_shift=None, f0_condition=None) -> Optional[AudioOutput]` | Convert one block and return the output metadata/audio. |
| `convert_live_audio(audio, sample_rate, label="audio input", pitch_shift=None, f0_condition=None) -> Optional[AudioOutput]` | Convert one low-latency live block while preserving VC overlap state. |
| `stop_live_audio() -> None` | Reset live VC backend state. |
| `play(sound_path, keep=False, volume=1.0) -> bool` | Queue an SFX file, optionally prepending it to the next saved utterance. |
| `play_audio(audio, sample_rate, label="uploaded SFX", keep=False) -> bool` | Queue decoded SFX data. |
| `speed` | Get/set playback speed. |
| `reverb.strength` | Get/set DSP reverb strength from 0 through 1. |
| `playback_buffer_seconds` | Read the application-side seconds currently reserved for output. |
| `playback_contention_level` | Read smoothed output contention from 0 through 1. |
| `playback_underflows` | Read the cumulative PortAudio output-underflow count for this runtime. |
| `playback_queue_wait_seconds` | Read the latest time spent waiting to enqueue a chunk into the bounded playback queue. |
| `playback_generation_gap_seconds` | Read the latest time since the previous chunk from the same source was enqueued. |
| `playback_writer_wait_seconds` | Read the latest time a mixed block waited for the output writer thread. |
| `playback_writer_gap_seconds` | Read the latest gap between the previous write finishing and this write starting. |
| `playback_writer_write_seconds` | Read the latest time spent inside the output stream write. |
| `playback_rebuffer_wait_seconds` | Read cumulative time spent waiting for the adaptive reserve target. |
| `vc_pitch_shift`, `vc_f0_condition` | Read/write VC controls. |

Arrays passed to audio calls should be numeric mono/stereo data; Celune converts
them to normalized float32. The returned `AudioOutput` includes the actual
sample rate and channel data.

## Events and extensions

```python
from celune import CeluneExtension, subscribe


@subscribe("audio_end")
def record_audio_end(event) -> None:
    print(event.saved_path)
```

`setup_extensions()` initializes the extension manager. `CeluneExtension`
provides `log`, `say`, `think`, `play`, `status`, `set_voice`,
`with_backend`, and `with_cevoice`; the `CeluneContext` object contains the
same operations in a narrower form. Event names and payloads are documented in
[Extensions](../interfaces/extensions.md).

## CEVOICE calls

The stable pack API is:

```python
from celune.cevoice import CEVoice, write_cevoice, write_cechar_v4

bundle = CEVoice.open("voices/default.cevoice")
wav_bytes = bundle.read_asset("balanced", "wav")
write_cevoice("legacy-compatible.cevoice", voices, metadata, voice_metadata)
write_cechar_v4("character.cechar", files, metadata, compression=0)
```

`CEVoice.open()` validates the header, metadata, safe names, ranges, and
checksums. `read_asset()` validates the selected asset digest before returning
bytes. `write_cevoice()` writes the legacy-compatible layout;
`write_cechar_v4()` is the preferred writer for new packs. See
[CEVOICE and CECHAR](../CEVOICE.md) for the binary schemas and constraints.

## Supporting module calls

| Module | Public call families |
| --- | --- |
| `celune.audio.resampling` | `resample_audio()` for declared-rate conversion. |
| `celune.audio.server` | `restart_audio_server()` for device/stream recovery. |
| `celune.audio.dsp` | `pitch_shift_audio()`, readiness/sleep/working/error signals, `is_silent_utterance()`, and `StreamingPedalboardReverb`. |
| `celune.vc` | Pitch clamping, RMS/VAD helpers, frame calculations, and `LiveVoiceActivityDetector`. |
| `celune.analysis` | Audio metrics, reference similarity, trait computation, radar plots, reports, and `analyze_voice_audio()`. |
| `celune.paths` | App-data, runtime, Hugging Face cache, voice, backend-environment, config, output, and migration paths. |
| `celune.extensions` | Extension base, context, manager, event dispatcher, and event decorator. |
| `celune.api` | `configure_api_security()`, `resolve_api_host()`, `bind_celune()`, `run_api()`, `start_api()`, `audio_bytes()`, and `stream_headers()`. |
| `celune.backends.environment` | `BackendManifest`, `BackendEnvironment`, and `BackendEnvironmentManager`. |
| `celune.cedts.protocol` | CEDTS framing, typed values, handshake, and stream exceptions. |

Internal helpers, private backend implementations, UI widget methods, and
dataclass fields not listed above are implementation details. Extend the
existing abstractions when a new integration needs one; do not fork the
pipeline or invent a second event/worker protocol.
