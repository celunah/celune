# SPDX-License-Identifier: Apache-2.0
"""Reviewed metadata for Celune's embedding and supporting models."""

from .types import ModelMetadata

VOICE_EMBEDDING_MODEL = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
VOICE_EMBEDDING_MODEL_REVISION = "7577f61c42737fc8064bba773e2a18602df92803"
PERSONA_MEMORY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSONA_EMOTION_MODEL = "lunahr/emotispace-128"
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v2.0-ctx2048"
CAPTION_ALIGNMENT_MODEL_ID = "sadda-speech/wav2vec2-espeak-ctc"
NEEDLE_MODEL_ID = "Cactus-Compute/needle"
NEEDLE_MODEL_REVISION = "5f89b4307696d669c3df1d38ae057e6e1728b107"

_APACHE_2_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_MIT_LICENSE_URL = "https://opensource.org/license/mit"
_GPL_3_LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.html"
_HUGGING_FACE_ROOT = "https://huggingface.co/"

OTHER_MODELS: tuple[ModelMetadata, ...] = (
    ModelMetadata(
        model_id=VOICE_EMBEDDING_MODEL,
        display_name="Qwen3 Voice Embedding 12Hz 1.7B",
        category="other",
        revision=VOICE_EMBEDDING_MODEL_REVISION,
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{VOICE_EMBEDDING_MODEL}",
        roles=("voice_embedding",),
    ),
    ModelMetadata(
        model_id=PERSONA_MEMORY_EMBEDDING_MODEL,
        display_name="all-MiniLM-L6-v2",
        category="other",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(f"{_HUGGING_FACE_ROOT}{PERSONA_MEMORY_EMBEDDING_MODEL}"),
        roles=("persona_memory_embedding",),
    ),
    ModelMetadata(
        model_id=PERSONA_EMOTION_MODEL,
        display_name="Emotispace 128",
        category="other",
        revision="34c72d6d2304e90b4bf27b5a61f6771458bf9bea",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{PERSONA_EMOTION_MODEL}",
        roles=("persona_emotion",),
    ),
    ModelMetadata(
        model_id=NORMALIZER_MODEL_ID,
        display_name="CeluneNorm 0.6B v2.0 Context 2048",
        category="other",
        revision="f387f219bc6cfb87d6b142f96e5ac6cb8b1774d0",
        license_id="mit",
        license_url=_MIT_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{NORMALIZER_MODEL_ID}",
        roles=("text_normalizer",),
    ),
    ModelMetadata(
        model_id=CAPTION_ALIGNMENT_MODEL_ID,
        display_name="wav2vec2 espeak CTC",
        category="other",
        revision="fce273e5ab08dcc6090fdf51b5893cf76b038e70",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{CAPTION_ALIGNMENT_MODEL_ID}",
        roles=("caption_alignment",),
    ),
    ModelMetadata(
        model_id=NEEDLE_MODEL_ID,
        display_name="Needle",
        category="other",
        revision=NEEDLE_MODEL_REVISION,
        license_id="mit",
        license_url=_MIT_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}{NEEDLE_MODEL_ID}",
        roles=("agent",),
    ),
    ModelMetadata(
        model_id="Plachta/Seed-VC",
        display_name="Seed-VC",
        category="other",
        revision="257283f9f41585055e8f858fba4fd044e5caed6e",
        license_id="gpl-3.0",
        license_url=_GPL_3_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Plachta/Seed-VC",
        roles=("seed_vc", "voice_conversion"),
    ),
    ModelMetadata(
        model_id="Plachta/ASTRAL-quantization",
        display_name="ASTRAL Quantization",
        category="other",
        revision="4a2e9679f76eb03753adc8c503e3c23bb9c22f26",
        license_id=None,
        license_url=None,
        non_commercial=None,
        research_only=None,
        repository_url=f"{_HUGGING_FACE_ROOT}Plachta/ASTRAL-quantization",
        roles=("seed_vc", "voice_conversion_quantizer"),
        terms_review_required=True,
    ),
    ModelMetadata(
        model_id="funasr/campplus",
        display_name="CAM++ Speaker Encoder",
        category="other",
        revision="e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}funasr/campplus",
        roles=("seed_vc", "voice_conversion_speaker_encoder"),
    ),
    ModelMetadata(
        model_id="FunAudioLLM/CosyVoice-300M",
        display_name="CosyVoice 300M",
        category="other",
        revision="24c40509c3c5ea6fe06b5f8790ff99e3714a6bee",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}FunAudioLLM/CosyVoice-300M",
        roles=("seed_vc", "voice_conversion_flow"),
    ),
    ModelMetadata(
        model_id="lj1995/VoiceConversionWebUI",
        display_name="VoiceConversionWebUI",
        category="other",
        revision="e6d0c1a17da07c33557852f9dfa2bd44cc75737d",
        license_id="mit",
        license_url=_MIT_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}lj1995/VoiceConversionWebUI",
        roles=("seed_vc", "voice_conversion_vocoder"),
    ),
)
