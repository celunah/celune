# SPDX-License-Identifier: Apache-2.0
"""Pinned audio.cpp GGUF packages verified against the TTS adapters."""

from dataclasses import dataclass
from typing import Optional

from ...registry.tts import (
    DOTS_TTS_MODEL_ID,
    POCKET_TTS_MODEL_ID,
    QWEN3_TTS_0_6B_MODEL_ID,
    QWEN3_TTS_1_7B_MODEL_ID,
    VOXCPM2_MODEL_ID,
)

AUDIOCPP_GGUF_REPO = "audio-cpp/audio.cpp-gguf"
AUDIOCPP_GGUF_REVISION = "73ce4162f6cce370dd533c6782b772135c1b2a8e"


@dataclass(frozen=True)
class AudioCppModelArtifact:
    """Integrity and feature metadata for one audio.cpp GGUF artifact."""

    filename: str
    size: int
    sha256: str
    quantization: str
    family: str
    model_id: str
    language: Optional[str] = None


def _artifact(
    filename: str,
    size: int,
    sha256: str,
    quantization: str,
    family: str,
    model_id: str,
    language: Optional[str] = None,
) -> AudioCppModelArtifact:
    """Create one pinned package entry without repeating shared fields."""
    return AudioCppModelArtifact(
        filename=filename,
        size=size,
        sha256=sha256,
        quantization=quantization,
        family=family,
        model_id=model_id,
        language=language,
    )


_POCKET = {
    "en": (
        _artifact(
            "PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
            127867968,
            "3f71cf6a51f93c2f2e554e98970a3f49fb9a98d0c64bed5a815e65be3d2ff588",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "en",
        ),
        _artifact(
            "PocketTTS-GGUF/english/pocket-tts-english-bf16.gguf",
            219107328,
            "24ab011c231ea33e7e8b88453c8faec02cdeb37c9b62dbb946ec5c806ec35902",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "en",
        ),
    ),
    "fr": (
        _artifact(
            "PocketTTS-GGUF/french/pocket-tts-french-q8_0.gguf",
            128117408,
            "ae7e6e536c20245d216a7ec9404124cf67f913380bf41a6380c55d6ca9dbaef5",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "fr",
        ),
        _artifact(
            "PocketTTS-GGUF/french/pocket-tts-french-bf16.gguf",
            219356768,
            "f7b197184e36d9f744fbc3e224e685de8fc977ab748e3612732a8706e9096541",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "fr",
        ),
    ),
    "de": (
        _artifact(
            "PocketTTS-GGUF/german/pocket-tts-german-q8_0.gguf",
            127868448,
            "d99a966c42026dbe119863ad8a8c96ddd3c8d2e800dfe2ab75a0caa579293268",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "de",
        ),
        _artifact(
            "PocketTTS-GGUF/german/pocket-tts-german-bf16.gguf",
            219107808,
            "e8b47c42f9c377eed09ad6d3e4e35031ca35dbc69ef55992c2cabafbc24885c6",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "de",
        ),
    ),
    "it": (
        _artifact(
            "PocketTTS-GGUF/italian/pocket-tts-italian-q8_0.gguf",
            127868704,
            "aa25eb6c0d73fa90a26c07ed5453f44c321ff483b33e96e5c3693b6f76ef9149",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "it",
        ),
        _artifact(
            "PocketTTS-GGUF/italian/pocket-tts-italian-bf16.gguf",
            219108064,
            "853805d9b7fe25d33ecdc609dc62686d2df88f2ac1041a8009c4ca1bdfb4d51d",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "it",
        ),
    ),
    "pt": (
        _artifact(
            "PocketTTS-GGUF/portuguese/pocket-tts-portuguese-q8_0.gguf",
            127869632,
            "e6e76fde1770586b11fab5a7d4fa07006d911098718991e242ca6006bd0513cd",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "pt",
        ),
        _artifact(
            "PocketTTS-GGUF/portuguese/pocket-tts-portuguese-bf16.gguf",
            219108992,
            "0ba1e195f8225299218ee8a731457e3121fa171b0f6135bcc8afc43ff96faa46",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "pt",
        ),
    ),
    "es": (
        _artifact(
            "PocketTTS-GGUF/spanish/pocket-tts-spanish-q8_0.gguf",
            127869504,
            "d2645064c191d0b9b8daf6d24d2b8a310d89f694c9b3523c1258bc9455adcaff",
            "Q8_0",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "es",
        ),
        _artifact(
            "PocketTTS-GGUF/spanish/pocket-tts-spanish-bf16.gguf",
            219108864,
            "aefafb0186b16847aa184c28d2bf0310ccb5999f1fde98c416786b268fa8d232",
            "BF16",
            "pocket_tts",
            POCKET_TTS_MODEL_ID,
            "es",
        ),
    ),
}

_QWEN = {
    (QWEN3_TTS_0_6B_MODEL_ID, "q8_0"): _artifact(
        "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-q8_0.gguf",
        1991217728,
        "c7d0e6c9485a25ae1f6a4c0796f0023230a0051c44cbd2bd05d73770f2cce637",
        "Q8_0",
        "qwen3_tts",
        QWEN3_TTS_0_6B_MODEL_ID,
    ),
    (QWEN3_TTS_0_6B_MODEL_ID, "bf16"): _artifact(
        "Qwen3-TTS-12Hz-0.6B-Base-GGUF/qwen3-tts-12hz-0.6b-base-bf16.gguf",
        2516161088,
        "b730cd6dfe549d7afd2b4735f57558744d5118db11ba0f7b8d2d42278cc431c7",
        "BF16",
        "qwen3_tts",
        QWEN3_TTS_0_6B_MODEL_ID,
    ),
    (QWEN3_TTS_1_7B_MODEL_ID, "q8_0"): _artifact(
        "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf",
        2695186240,
        "92641a8022d521efd79a5db45e952589b3f2b781680f9abfcf26f075d37b86c6",
        "Q8_0",
        "qwen3_tts",
        QWEN3_TTS_1_7B_MODEL_ID,
    ),
    (QWEN3_TTS_1_7B_MODEL_ID, "bf16"): _artifact(
        "Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-bf16.gguf",
        4203169600,
        "db237375e0f04d10fd0ac0afc2855e6844fdcb04a7d471e8a4b6c36bdb6030d7",
        "BF16",
        "qwen3_tts",
        QWEN3_TTS_1_7B_MODEL_ID,
    ),
}

_DOTS = {
    "q8_0": _artifact(
        "DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf",
        2965377472,
        "abc4c27db2a6750119d70f5e66e9c268c80c47ff4659867bbba124717f05ae7a",
        "Q8_0",
        "dots_tts",
        DOTS_TTS_MODEL_ID,
    ),
    "bf16": _artifact(
        "DotTTS-MF-GGUF/dots-tts-mf-bf16.gguf",
        4791358304,
        "4738dd818d4a5be1ad0162c2828711da99182ab044b7d72a7833facd6cc0b7f8",
        "BF16",
        "dots_tts",
        DOTS_TTS_MODEL_ID,
    ),
}

_VOXCPM2 = {
    "q8_0": _artifact(
        "VoxCPM2-GGUF/voxcpm2-q8_0.gguf",
        2955007488,
        "8df1faf274e1bdeedef1fd9ddd6aba6632536f66f2e080991d2352af35ceef72",
        "Q8_0",
        "voxcpm2",
        VOXCPM2_MODEL_ID,
    ),
    "bf16": _artifact(
        "VoxCPM2-GGUF/voxcpm2-bf16.gguf",
        4772295296,
        "dc48028ebf08b9fd2456cbf5fa1ab52568d16660e34d0fa418c35e251ca1ecab",
        "BF16",
        "voxcpm2",
        VOXCPM2_MODEL_ID,
    ),
}


def audio_cpp_model_artifact(
    backend_id: str,
    model_id: str,
    *,
    quantize: bool,
    language: Optional[str] = None,
) -> Optional[AudioCppModelArtifact]:
    """Resolve an exact, pinned audio.cpp package for one backend load."""
    quantization = "q8_0" if quantize else "bf16"
    if backend_id == "mini" and model_id == POCKET_TTS_MODEL_ID and language:
        language_artifacts = _POCKET.get(language)
        if language_artifacts is not None:
            return language_artifacts[0 if quantize else 1]
        return None
    if backend_id == "qwen3":
        return _QWEN.get((model_id, quantization))
    if backend_id == "dotstts" and model_id == DOTS_TTS_MODEL_ID:
        return _DOTS[quantization]
    if backend_id == "voxcpm2" and model_id == VOXCPM2_MODEL_ID:
        return _VOXCPM2[quantization]
    return None
