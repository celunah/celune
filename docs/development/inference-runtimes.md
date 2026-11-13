# Inference runtimes

This page defines Celune's `native` and `gguf` inference selection, the runtime
artifacts currently registered for those paths, and how to identify the
effective runtime. It is for Celune users and contributors validating Persona,
TTS, or the CEDTS worker boundary.

## Configuration and selection

Set the top-level `inference_mode` key to `native` or `gguf`. The default is
`native`, including for configurations created before the setting existed.
Changes take effect after Celune restarts. Invalid values fail through the
normal configuration error path.

Persona and TTS resolve independently. In `gguf` mode Persona selects a
reviewed llama.cpp package only for the exact model and quantization pair in
the matrix below. Other Persona models use Transformers. A supported TTS
backend selects audio.cpp; native-only backends such as SoulX-Singer continue
to use their native adapter. Ming3 requires audio.cpp in both modes.

When a GGUF runtime or package cannot initialize, Celune closes the candidate
and attempts native inference once. It logs the reason and keeps the effective
runtime for that loaded component. Generation errors are returned through the
current backend error path and do not replay a request through another
runtime. Consent, policy, cancellation, and invalid-request errors remain
direct. TTS model identity and runtime metadata arrive from the worker in a
CEDTS `metadata` event; Persona logs the same model, effective runtime, and
quantization details in core.

## Persona compatibility

The built-in Persona registry currently contains 14 model IDs. Only the exact
`Qwen/Qwen3-VL-4B-Instruct` entry has a matching reviewed GGUF package. The
other 13 entries, including 4B derivatives, 8B Qwen3-VL variants, Qwen3.5, and
Gemma 4, use Transformers in `gguf` mode. An unsupported model logs one reason
and falls back to native inference.

| Persona model and requested precision | GGUF model file | Vision projector | GGUF quantization | Eligible platform |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-VL-4B-Instruct`, Celune 4-bit preset | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | `Q4_K_M` | Windows x64 and Linux x86_64 with a supported NVIDIA CUDA device and driver |
| `Qwen/Qwen3-VL-4B-Instruct`, Celune 8-bit preset | `Qwen3VL-4B-Instruct-Q8_0.gguf` | `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | `Q8_0` | Windows x64 and Linux x86_64 with a supported NVIDIA CUDA device and driver |

Both model/projector pairs come from `Qwen/Qwen3-VL-4B-Instruct-GGUF` at
revision `1cd86afb9a95c410a6038ab3b40d8b578c892266`. Celune checks the exact
file size and SHA-256 before use:

| File | Size in bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | 2,497,281,664 | `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a` |
| `Qwen3VL-4B-Instruct-Q8_0.gguf` | 4,280,406,144 | `054721f478bc5fa6beffb7f38eae575d45298f88cbb8d2f83ef675a727863eb1` |
| `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | 453,974,304 | `30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d` |

The llama.cpp server is pinned to release `b12110`, commit
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`. Each supported platform uses the
server archive plus its matching CUDA-library archive. Celune verifies both
archives before extraction and caches the combined runtime under application
data:

| Platform | Runtime archive | Size in bytes | SHA-256 |
| --- | --- | ---: | --- |
| Windows x64, CUDA 12.4 | `llama-b12110-bin-win-cuda-12.4-x64.zip` | 287,014,657 | `1ce755bc3a061889234048f88e33328e17971eaad73c4666cfeb26026b317fee` |
| Windows x64, CUDA 12.4 libraries | `cudart-llama-bin-win-cuda-12.4-x64.zip` | 391,443,627 | `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` |
| Ubuntu Linux x64, CUDA 12.8 | `llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz` | 187,314,621 | `4af1e1da23661db4a5b84b29aae16ebc053df689c5b8b1427c4670f2c6406d8c` |
| Ubuntu Linux x64, CUDA 12.8 libraries | `cudart-llama-b12110-bin-ubuntu-cuda-12.8-x64.tar.gz` | 594,371,477 | `6f1d562086d84ae85df880d3711079ec225602b51d530608e5001600625067dd` |

The Qwen GGUF entry supports text and JPEG/PNG image attachments. Video is
rejected with a clear error. llama.cpp does not expose the native hidden-state
probe used for emotion analysis, so Celune uses the existing fallback. The
server listens only on loopback and is stopped when Persona unloads.

## TTS compatibility

`supports_audio_cpp` and `requires_audio_cpp` are TTS backend declarations.
Both default to `false`; `requires_audio_cpp=true` is invalid unless support is
also true. Runtime selection happens before the worker environment is installed
or model weights are downloaded.

| Backend | audio.cpp flags | `native` mode | `gguf` mode |
| --- | --- | --- | --- |
| Mini / Pocket TTS | `true / false` | Native adapter | audio.cpp for verified English, French, German, Italian, Portuguese, and Spanish packages; otherwise native |
| Qwen3 TTS | `true / false` | Native adapter | audio.cpp for verified 0.6B and 1.7B Base packages; other variants fall back to native |
| FireRedTTS3 | `false / false` | Native adapter | Native fallback; audio.cpp v0.9.3 FireRedTTS3 is offline-only |
| Dots TTS MeanFlow | `true / false` | Native adapter | audio.cpp for the verified MeanFlow package; unsupported model IDs fall back to native |
| VoxCPM2 | `true / false` | Native adapter | audio.cpp for the verified VoxCPM2 package; unsupported load options fall back to native |
| Ming3 | `true / true` | audio.cpp | audio.cpp |
| SoulX-Singer | `false / false` | Native adapter | Native fallback |

The enabled streaming adapters use the exact GGUF filenames, sizes, SHA-256
digests, model IDs, quantization labels, and package revision recorded in
`celune/backends/tts/audio_cpp_models.py`. All package files come from
`audio-cpp/audio.cpp-gguf` revision
`73ce4162f6cce370dd533c6782b772135c1b2a8e`. Each downloaded file is checked
against its pinned size and SHA-256 before the runtime can load it. The runtime
is audio.cpp `v0.9.3` and currently advertises Windows x64 with CUDA 13.3 and
Linux x86_64 with CUDA 12.8.

| Backend and package | Verified streaming inputs and options |
| --- | --- |
| Mini / Pocket TTS | English, French, German, Italian, Portuguese, and Spanish; reference WAV, temperature, and seed. |
| Qwen3 TTS Base | 0.6B and 1.7B; reference WAV and transcript, language, x-vector-only mode, sampling controls, maximum tokens, and seed. |
| Dots TTS MeanFlow | Reference WAV and transcript, language, maximum tokens, seed, 4 Euler steps, and speaker scale 1.5. |
| VoxCPM2 | Reference WAV and transcript, style instruction prefix, voice CFG scale, 4 inference steps, maximum tokens, seed, and streaming with bad-case retry disabled. |

FireRedTTS3 remains native because the pinned audio.cpp guide marks its TTS
implementation offline-only, which cannot satisfy Celune's streaming contract.
Seed-VC is a voice-conversion backend and does not participate in TTS runtime
selection. The feature audit uses the pinned [audio.cpp v0.9.3 TTS guide](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/docs/tts.md),
the [Qwen3 TTS guide](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/docs/models/qwen3_tts.md),
and the [DotTTS guide](https://github.com/0xShug0/audio.cpp/blob/v0.9.3/docs/models/dots_tts.md).

Ming3 uses audio.cpp release `v0.9.3`. Its GGUF files are pinned to repository
revision `2eb8d37c7f10672e69c8aa33091baa04c461b66c`:

| File | Size in bytes | SHA-256 | `quant` metadata |
| --- | ---: | --- | --- |
| `ming-3.0-tts-0.5b-bf16.gguf` | 2,998,284,288 | `43f108a4e6bb54af13071d0cb345a094e0b011b3b8318d8fb2c61753733f5312` | `full` |
| `ming-3.0-tts-0.5b-q8_0.gguf` | 1,742,280,896 | `fcf2aa25e8e9cf2065a936e506581735c64d6255fb7869f09fbb61a2dde09aa1` | `Q8_0` |

The audio.cpp runtime supports Windows x64 with CUDA 13.3 and Ubuntu Linux x64
with CUDA 12.8. Windows downloads both `audio-v0.9.3-bin-windows-x64-cuda13.3.zip`
(`0bd3988be10c31ec078c40df2bd0d4266e660804f6d5083aeb05d1309d091e43`) and
`audio-v0.9.3-cudart-windows-x64-cuda13.3.zip`
(`839eb20f684a8aab64895b1e466b32ce9b29dfc19dc1ce15b2ef35218512ac0a`). Linux
downloads `audio-v0.9.3-bin-ubuntu-x64-cuda12.8-colab.tar.gz`
(`5be3e8d515fb244387de87958668ebb729d9442de6a53df12e4c718ea918419d`). Celune
checks the archive SHA-256 and model size/checksum before launching the
loopback streaming server.

## Failure behavior and verification

If a required TTS runtime cannot initialize at startup, Celune enters its
recoverable startup error state, names the backend and failure, skips the ready
event and readiness cue, and allows configuration recovery. Backend reloads
use the existing rollback path to restore the last working backend. If
Persona cannot initialize either runtime while TTS is healthy, Celune marks
Persona unavailable and continues in speech-only mode.

Check the logs for `inference requested=... effective=...` and the subsequent
`Loaded ... with ... inference at ... precision` line. For TTS, inspect the
worker-to-core CEDTS `metadata` event or the remote backend's `model_metadata`
value. The metadata packet uses exactly `model`, `type`, and `quant`; `type` is
`native` or `gguf`.

Unit coverage checks mode selection, unsupported-model fallback, failed GGUF
initialization, consent errors, generation failure behavior, protocol
validation, metadata delivery, package identity, and streamed PCM conversion.
The adapter compatibility list records guide-verified package features. The
Persona model and TTS entries still require real-weight load, generation,
cancellation, and unload checks on every advertised platform before a release
can claim live verification.

## See also

- [Configuration](../user-guide/configuration.md)
- [CEDTS transport](../formats/cedts.md)
- [Backend development](backends.md)
