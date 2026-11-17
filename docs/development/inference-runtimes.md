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

When a GGUF runtime or package cannot initialize, the application closes the
candidate and attempts native inference once. It logs the reason and keeps
the effective runtime for that loaded component. Generation errors are returned
through the current backend error path and do not replay a request through another
runtime. Consent, policy, cancellation, and invalid-request errors remain
direct. TTS model identity and runtime metadata arrive from the worker in a
CEDTS `metadata` event; Persona logs the same model, effective runtime, and
quantization details in core.

## Persona compatibility

The built-in Persona registry currently contains 14 model IDs. All 14 now have
a pinned GGUF package for the `4bit` and `8bit` Persona precision settings.
`Qwen/Qwen3-VL-4B-Instruct` uses its reviewed Qwen package. The other 13 use
matching mradermacher GGUF packages listed below and in
`celune/registry/gguf_catalog.py`. An unsupported model or precision logs one
reason and falls back to native inference.

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

The remaining built-in models use these exact mradermacher artifacts. For each
repo, `4bit` selects its `Q4_K_M` file and `8bit` selects its `Q8_0` file; both
use the listed `mmproj-Q8_0` vision projector. Sizes are bytes. Each GGUF repo
is pinned to the immutable revision shown, and Celune verifies every file's
size and SHA-256 before starting llama-server.

| Persona model ID | GGUF repo and revision | Q4_K_M file (bytes; SHA-256) | Q8_0 file (bytes; SHA-256) | Projector file (bytes; SHA-256) |
| --- | --- | --- | --- | --- |
| `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` | [mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF/tree/abf99adf548eb2aa6f9a3d56f976968996959e7f) @ `abf99adf548eb2aa6f9a3d56f976968996959e7f` | `Huihui-Qwen3-VL-4B-Instruct-abliterated.Q4_K_M.gguf` (2,497,282,496; `20d58872549c13f5476622d2385027e1b346298558822eca8aa33cb8846d5765`) | `Huihui-Qwen3-VL-4B-Instruct-abliterated.Q8_0.gguf` (4,280,406,976; `aba89057758004b0349b3b8ccc7f453386f9389ed6dea71a716be7ab5ede9a92`) | `Huihui-Qwen3-VL-4B-Instruct-abliterated.mmproj-Q8_0.gguf` (453,974,624; `cd4503cef5b48c55da5f04edd06e88f3c10344129f19bd9ebffdab32d0590a02`) |
| `Qwen/Qwen3-VL-8B-Instruct` | [mradermacher/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-GGUF/tree/499df3a68793b65eaec9035247e32f5cffa346b1) @ `499df3a68793b65eaec9035247e32f5cffa346b1` | `Qwen3-VL-8B-Instruct.Q4_K_M.gguf` (5,027,785,344; `659ea2c20fe881bba78e6c7a41dc206ab4de993b3df80f554c18b098b673792e`) | `Qwen3-VL-8B-Instruct.Q8_0.gguf` (8,709,520,000; `2370e7ec38c2ff0ee8b5aefaccbb86018a7bf44381a4970a0039832a9016ca2d`) | `Qwen3-VL-8B-Instruct.mmproj-Q8_0.gguf` (752,289,824; `7e848608a42de7e514803662b922f3835ce4191cde2709af1a2882d682c67718`) |
| `huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated` | [mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF/tree/03f1e8f28be1177d430866b31e0bc9a939c669d6) @ `03f1e8f28be1177d430866b31e0bc9a939c669d6` | `Huihui-Qwen3-VL-8B-Instruct-abliterated.Q4_K_M.gguf` (5,027,785,728; `c9592053aab8874021ecd3f80e45cc12dd167454bebfb215a36bdf690f22dc52`) | `Huihui-Qwen3-VL-8B-Instruct-abliterated.Q8_0.gguf` (8,709,520,384; `2a80cab9a07998164351e47c67380cbd10df95afbe8f3726688b82723689c64c`) | `Huihui-Qwen3-VL-8B-Instruct-abliterated.mmproj-Q8_0.gguf` (752,290,176; `8083b2ccf0351ce673fca1dd5667ad1fd49861c72dccebb16f3dd32aa6c42fb6`) |
| `Qwen/Qwen3-VL-8B-Thinking` | [mradermacher/Qwen3-VL-8B-Thinking-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-8B-Thinking-GGUF/tree/e92647a6203cbcb918c7efe7b08b245db7291681) @ `e92647a6203cbcb918c7efe7b08b245db7291681` | `Qwen3-VL-8B-Thinking.Q4_K_M.gguf` (5,027,785,248; `e495720e0e823fa8fb9f082a4012ccb412deebc9f03fa83dac7e6e15b477c693`) | `Qwen3-VL-8B-Thinking.Q8_0.gguf` (8,709,519,904; `00673fa25542eda8b95d4c01f8f73cf864fe64640fd83bfddd78573eee2d2537`) | `Qwen3-VL-8B-Thinking.mmproj-Q8_0.gguf` (752,289,824; `09add99dd97f82aa04bd39ec2460e099b9bfb29b7f20e7605346fc85b6dfa87e`) |
| `huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated` | [mradermacher/Huihui-Qwen3-VL-8B-Thinking-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-VL-8B-Thinking-abliterated-GGUF/tree/7b20225df54c856f085e3d0d120ca6cab3aa10b6) @ `7b20225df54c856f085e3d0d120ca6cab3aa10b6` | `Huihui-Qwen3-VL-8B-Thinking-abliterated.Q4_K_M.gguf` (5,027,785,632; `478f1c18f5cd3bcbd1a1e6ec36e30451d2ad39566161176f019c678c1a30090b`) | `Huihui-Qwen3-VL-8B-Thinking-abliterated.Q8_0.gguf` (8,709,520,288; `2af6547ea5630a4e9dd7267815ce7a0358ef2ed426c0d4fd02c4493814bb37a4`) | `Huihui-Qwen3-VL-8B-Thinking-abliterated.mmproj-Q8_0.gguf` (752,290,176; `1cadefa6bfa43cf64b34d4e87a3a1bb6ca3f9ed6261b04010a5e7ecbfc6f0023`) |
| `Qwen/Qwen3.5-4B` | [mradermacher/Qwen3.5-4B-GGUF](https://huggingface.co/mradermacher/Qwen3.5-4B-GGUF/tree/85aed52869ec5ef32259f80686311048bc260494) @ `85aed52869ec5ef32259f80686311048bc260494` | `Qwen3.5-4B.Q4_K_M.gguf` (2,708,804,800; `51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39`) | `Qwen3.5-4B.Q8_0.gguf` (4,482,403,520; `722af6f11b1f8052ae4854a82947e8f37102275f6b0edc402146144aefa840ab`) | `Qwen3.5-4B.mmproj-Q8_0.gguf` (366,894,656; `40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad`) |
| `huihui-ai/Huihui-Qwen3.5-4B-abliterated` | [mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF/tree/9d3aebb998fcf084086a3d1c60c31035fcfbf8e1) @ `9d3aebb998fcf084086a3d1c60c31035fcfbf8e1` | `Huihui-Qwen3.5-4B-abliterated.Q4_K_M.gguf` (2,707,514,688; `423f10b6ec2d99c3378143d7cd3b80eb4887b3ed92103103ac59173b404f4f7c`) | `Huihui-Qwen3.5-4B-abliterated.Q8_0.gguf` (4,482,403,648; `1e78f9b48c8b69242fa45bbbbeb439ed818a71381a6362130d62a38561c3756d`) | `Huihui-Qwen3.5-4B-abliterated.mmproj-Q8_0.gguf` (366,894,720; `650e2af8754b8b10e21debdbaf352cab43cd2435ec8ed56b893b9b581ae4a95c`) |
| `Qwen/Qwen3.5-9B` | [mradermacher/Qwen3.5-9B-GGUF](https://huggingface.co/mradermacher/Qwen3.5-9B-GGUF/tree/664ad08c870c0a6e80798eb4a98e2335ec6e93b4) @ `664ad08c870c0a6e80798eb4a98e2335ec6e93b4` | `Qwen3.5-9B.Q4_K_M.gguf` (5,627,045,120; `9fa52e37c8293ee02cf277634f8d3a392dc3239b8ccef2315b4c8c9ef223e7fb`) | `Qwen3.5-9B.Q8_0.gguf` (9,527,502,080; `e7057b92e524bb7bdf824f84b268b4ed64ea0336b51cf442afec22b2a996e132`) | `Qwen3.5-9B.mmproj-Q8_0.gguf` (624,229,696; `6ae4061987abccb86185f3b4e2ae0e4c6375bfd3dc98c4e473eda7545877155c`) |
| `huihui-ai/Huihui-Qwen3.5-9B-abliterated` | [mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF/tree/419ca34f424adbd9276c09e0271cb6b718e8da1d) @ `419ca34f424adbd9276c09e0271cb6b718e8da1d` | `Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf` (5,627,045,248; `ea1858ef4dc4b648b8dbb44612962a0333e945060dd0545ac0f28d7c4416e4b3`) | `Huihui-Qwen3.5-9B-abliterated.Q8_0.gguf` (9,527,502,208; `9df7ce00f4df74344dd788ade97c412ef9850e58b44ff337e56fbf07be24a97f`) | `Huihui-Qwen3.5-9B-abliterated.mmproj-Q8_0.gguf` (624,229,760; `cf54897b6bbfa89c809fdc2b8236e0977bed70760b39ee8f11d87296e1d0e046`) |
| `google/gemma-4-E2B-it` | [mradermacher/gemma-4-E2B-it-GGUF](https://huggingface.co/mradermacher/gemma-4-E2B-it-GGUF/tree/cb7ceaccb7b0973a17ab6d39fe6d2b11987ea8b5) @ `cb7ceaccb7b0973a17ab6d39fe6d2b11987ea8b5` | `gemma-4-E2B-it.Q4_K_M.gguf` (3,427,878,336; `7e6e3d87757ca0560a4b0be80dc0bb480bb793b749b317f7aeda81b5ea8260a0`) | `gemma-4-E2B-it.Q8_0.gguf` (4,967,495,104; `803d2049854ebff08a199f9dfce0c1f25688127c63da5a0c4a8376ccba822c91`) | `gemma-4-E2B-it.mmproj-Q8_0.gguf` (557,367,808; `4e1b21a6ddf52e07387be624523c899591d463cd3051819eb6b8acad3f2cc6a6`) |
| `huihui-ai/Huihui-gemma-4-E2B-it-abliterated` | [mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF/tree/0538cae708ae060ba4fff720297b79c672c339ff) @ `0538cae708ae060ba4fff720297b79c672c339ff` | `Huihui-gemma-4-E2B-it-abliterated.Q4_K_M.gguf` (3,427,874,272; `55044454df35c2e3b9f4ca7a52b27cc7b057b3b3547cd114183196c31f5062a8`) | `Huihui-gemma-4-E2B-it-abliterated.Q8_0.gguf` (4,967,491,040; `c127fb9c6f3867969ceb79078e30eae3c62603aea92598400ac5a93bb2936431`) | `Huihui-gemma-4-E2B-it-abliterated.mmproj-Q8_0.gguf` (557,368,128; `030b1a496787c3ca9240552a3d5488db0ba50c6d6cdc878793513340d736765c`) |
| `google/gemma-4-E4B-it` | [mradermacher/gemma-4-E4B-it-GGUF](https://huggingface.co/mradermacher/gemma-4-E4B-it-GGUF/tree/bccd4df108bbde6cd495f2594c17a5182bc59304) @ `bccd4df108bbde6cd495f2594c17a5182bc59304` | `gemma-4-E4B-it.Q4_K_M.gguf` (5,335,290,304; `e5555564868766605415f0e1401feac79c9327333c051521934c90aea67df897`) | `gemma-4-E4B-it.Q8_0.gguf` (8,031,240,640; `0802b70ad2e46c1db03b45504cd3feedb5d329b24ad98078c5418d14674c4aaa`) | `gemma-4-E4B-it.mmproj-Q8_0.gguf` (559,874,560; `2d5aa169b4b61a9281e9106acbe73d8f2437e316ba9657f65eb3e3332f5c598a`) |
| `huihui-ai/Huihui-gemma-4-E4B-it-abliterated` | [mradermacher/Huihui-gemma-4-E4B-it-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-gemma-4-E4B-it-abliterated-GGUF/tree/580a4a13e9633a0f123befcfab222cfcd9a36b77) @ `580a4a13e9633a0f123befcfab222cfcd9a36b77` | `Huihui-gemma-4-E4B-it-abliterated.Q4_K_M.gguf` (5,335,286,272; `d793116ec11cacac0228e7c907a40e9765969d76dd45708f0f638df008718f87`) | `Huihui-gemma-4-E4B-it-abliterated.Q8_0.gguf` (8,031,236,608; `8bb670d57c5dc042c6ef9571f8b86674b4d1835e55708ae5ac5b40aa16687036`) | `Huihui-gemma-4-E4B-it-abliterated.mmproj-Q8_0.gguf` (559,874,880; `46f248feeb9288819cdd6a239dfa42335780dcf085165247cbe2c56cae403839`) |

The Persona GGUF adapter accepts text and JPEG/PNG image attachments and
rejects video with a clear error. llama.cpp does not expose the native
hidden-state probe used for emotion analysis, so Celune uses the existing
fallback. The server listens only on loopback and is stopped when Persona
unloads.

## TTS compatibility

`supports_audio_cpp` and `requires_audio_cpp` are TTS backend declarations.
Both default to `false`; `requires_audio_cpp=true` is invalid unless support is
also true. Runtime selection happens before the worker environment is installed
or model weights are downloaded.

All audio.cpp TTS adapters use one loopback server process managed by the core.
CEDTS workers stay isolated and register their verified model package with that
server when loading it. The server keeps at most one model resident, so changing
to another audio.cpp backend replaces the loaded GGUF while reusing the same
server process and cached executable. The core stops the server after its last
audio.cpp worker closes. If the shared server cannot start, optional backends
fall back to native inference; Ming3 reports a startup failure because it
requires audio.cpp.

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
`celune/registry/audio_cpp_models.py`. All package files come from
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

If a required TTS runtime cannot initialize at startup, the application enters
a recoverable startup error state, names the backend and failure, skips the
ready event and readiness cue, and allows configuration recovery. Backend reloads
use the existing rollback path to restore the last working backend. If
Persona cannot initialize either runtime while TTS is healthy, Celune marks
Persona unavailable and continues in speech-only mode.

Check the logs for `inference requested=... effective=...` and the subsequent
`Loaded ... with ... inference at ... precision` line. For TTS, inspect the
worker-to-core CEDTS `metadata` event or the remote backend's `model_metadata`
value. The metadata packet uses exactly `model`, `type`, and `quant`; `type` is
`native` or `gguf`. A native fallback warning now includes the precise reason
that the selected package, load options, or shared server could not be used.

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
