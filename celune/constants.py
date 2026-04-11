"""Shared Celune model constants."""

VOICE_MODELS = {
    "balanced": "lunahr/Celune-1.7B-Neutral",
    "calm": "lunahr/Celune-1.7B-Calm",
    "bold": "lunahr/Celune-1.7B-Energetic",
    "upbeat": "lunahr/Celune-1.7B-Upbeat",
}

DEFAULT_VOICE = "balanced"
DEFAULT_MODEL_ID = VOICE_MODELS[DEFAULT_VOICE]
ALL_VOICE_MODEL_IDS = list(VOICE_MODELS.values())
NORMALIZER_MODEL_ID = "lunahr/CeluneNorm-0.6B-v1.1"
