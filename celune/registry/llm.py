# SPDX-License-Identifier: Apache-2.0
"""Reviewed metadata for Persona and agent language models."""

from typing import Literal, Optional

from .types import ModelMetadata

PERSONA_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

_APACHE_2_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
_HUGGING_FACE_ROOT = "https://huggingface.co/"
_PERSONA_ROLES = ("persona", "agent")

LLM_MODELS: tuple[ModelMetadata, ...] = (
    ModelMetadata(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        display_name="Qwen3 VL 4B Instruct",
        category="llm",
        revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Qwen/Qwen3-VL-4B-Instruct",
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        display_name="Qwen3 VL 4B Instruct Abliterated",
        category="llm",
        revision="ce72a7c22aacb493fb94478de3bfbe834c61844a",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(
            f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
        ),
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        display_name="Qwen3 VL 8B Instruct",
        category="llm",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Qwen/Qwen3-VL-8B-Instruct",
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        display_name="Qwen3 VL 8B Instruct Abliterated",
        category="llm",
        revision="b47a0690b22eaf1d9a63874d967a03781c90f9cf",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(
            f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"
        ),
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="Qwen/Qwen3-VL-8B-Thinking",
        display_name="Qwen3 VL 8B Thinking",
        category="llm",
        revision="92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Qwen/Qwen3-VL-8B-Thinking",
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated",
        display_name="Qwen3 VL 8B Thinking Abliterated",
        category="llm",
        revision="34bbf0d131d799ef233b2c20b074fbc9a0179ead",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(
            f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"
        ),
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="Qwen/Qwen3.5-4B",
        display_name="Qwen3.5 4B",
        category="llm",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Qwen/Qwen3.5-4B",
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        display_name="Qwen3.5 4B Abliterated",
        category="llm",
        revision="5581467dfd52bf338c782006a6cdce05c42594be",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="Qwen/Qwen3.5-9B",
        display_name="Qwen3.5 9B",
        category="llm",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}Qwen/Qwen3.5-9B",
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        display_name="Qwen3.5 9B Abliterated",
        category="llm",
        revision="05b9e7c9b978ba29bdb8f50a49c30e4b91183339",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="google/gemma-4-E2B-it",
        display_name="Gemma 4 E2B Instruct",
        category="llm",
        revision="3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}google/gemma-4-E2B-it",
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
        display_name="Gemma 4 E2B Instruct Abliterated",
        category="llm",
        revision="3d1e3d50d7a04585ce4ded197b2fd7a90c04647c",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(
            f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-gemma-4-E2B-it-abliterated"
        ),
        roles=_PERSONA_ROLES,
        tier="standard",
    ),
    ModelMetadata(
        model_id="google/gemma-4-E4B-it",
        display_name="Gemma 4 E4B Instruct",
        category="llm",
        revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=f"{_HUGGING_FACE_ROOT}google/gemma-4-E4B-it",
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
    ModelMetadata(
        model_id="huihui-ai/Huihui-gemma-4-E4B-it-abliterated",
        display_name="Gemma 4 E4B Instruct Abliterated",
        category="llm",
        revision="03ce1f3a982b544afb03878ce80e7f042bcdc172",
        license_id="apache-2.0",
        license_url=_APACHE_2_LICENSE_URL,
        non_commercial=False,
        research_only=False,
        repository_url=(
            f"{_HUGGING_FACE_ROOT}huihui-ai/Huihui-gemma-4-E4B-it-abliterated"
        ),
        roles=_PERSONA_ROLES,
        tier="smart",
    ),
)


def persona_model_tier(
    model_id: str,
) -> Optional[Literal["standard", "smart"]]:
    """Return the hardware tier configured for a Persona model ID.

    Args:
        model_id: Full Hugging Face model repository ID.

    Returns:
        Optional[Literal["standard", "smart"]]: The tier, or ``None`` for an
            unregistered model.
    """
    for model in LLM_MODELS:
        if model.model_id == model_id:
            return model.tier
    return None
