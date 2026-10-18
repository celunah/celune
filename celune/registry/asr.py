# SPDX-License-Identifier: Apache-2.0
"""Reviewed metadata for automatic speech recognition models."""

from .types import ModelMetadata

DEFAULT_PERSONA_SPEECH_MODEL_ID = "openai/whisper-large-v3-turbo"
DEFAULT_PERSONA_SPEECH_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
WHISPER_TINY_MODEL_ID = "openai/whisper-tiny"
WHISPER_TINY_MODEL_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
WHISPER_BASE_MODEL_ID = "openai/whisper-base"
WHISPER_BASE_MODEL_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"

_APACHE_2_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_MIT_LICENSE_URL = "https://opensource.org/license/mit"
_HUGGING_FACE_ROOT = "https://huggingface.co/"

ASR_MODELS: tuple[ModelMetadata, ...] = (
    ModelMetadata(
        model_id=DEFAULT_PERSONA_SPEECH_MODEL_ID,
        display_name="Whisper large-v3 Turbo",
        category="asr",
        revision=DEFAULT_PERSONA_SPEECH_MODEL_REVISION,
        license_id="mit",
        license_url=_MIT_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{DEFAULT_PERSONA_SPEECH_MODEL_ID}",
        roles=("persona_speech_input",),
    ),
    ModelMetadata(
        model_id=WHISPER_TINY_MODEL_ID,
        display_name="Whisper Tiny",
        category="asr",
        revision=WHISPER_TINY_MODEL_REVISION,
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{WHISPER_TINY_MODEL_ID}",
        roles=("tts_transcription",),
    ),
    ModelMetadata(
        model_id=WHISPER_BASE_MODEL_ID,
        display_name="Whisper Base",
        category="asr",
        revision=WHISPER_BASE_MODEL_REVISION,
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{WHISPER_BASE_MODEL_ID}",
        roles=("tts_transcription",),
    ),
)
