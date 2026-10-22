# SPDX-License-Identifier: Apache-2.0
"""Persona request construction and conversation context."""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, cast

from .paths import persona_override_files
from .emotion import PersonaEmotionAnalyzer
from .capabilities import PersonaCapabilities
from .memory import (
    MEMORY_CLASSIFIER_SYSTEM_PROMPT,
    PersonaMemoryStore,
    classifier_memory_candidates,
)
from .prompts import (
    PersonaCard,
    PersonaContext,
    CharacterProfile,
    PersonaPromptBuilder,
    PersonaSourceMaterial,
    RetrievedMemoryBundle,
)
from .impl import (
    persona_config,
    persona_model_id,
    pack_persona_text,
    pack_identity_text,
    pack_persona_lines,
    default_persona_age,
    persona_context_size,
    persona_quantization,
    persona_style_traits,
    default_persona_gender,
    default_persona_context,
    default_persona_persona,
    persona_session_summary,
    persona_history_messages,
    persona_pending_attachments,
    persona_active_character_name,
    persona_debug_overrides_enabled,
)

from ..utils import format_error_message
from ..typing.common import JSON, JSONSerializable
from ..registry.other import PERSONA_MEMORY_EMBEDDING_MODEL
from ..typing.persona import PersonaModel, PersonaTokenizer
from ..typing.agent import (
    ToolCall,
    AgentContext,
    AgentToolSchema,
)
from ..constants import (
    AGENT_CONTEXT_SPACE,
    AGENT_ROUTING_CONTEXT_SPACE,
)
from ..cevoice import (
    default_loader,
    bundle_character_name,
    persona_files_from_bundle,
    persona_metadata_from_manifest,
)

if TYPE_CHECKING:
    from ..celune import Celune
    from ..typing.persona import PersonaClientResponse

_MemoryClassifier = Callable[[JSON], "PersonaClientResponse"]


def _config_text(engine: Celune, key: str, default: str) -> str:
    """Read a configured string with a fallback."""
    value = engine.config.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


def _config_lines(engine: Celune, key: str) -> tuple[str, ...]:
    """Normalize a scalar or list Persona setting to non-empty lines."""
    value = engine.config.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        return tuple(
            item.strip() for item in value if isinstance(item, str) and item.strip()
        )
    return ()


def build_persona_character_card(engine: Celune) -> str:
    """Build the compact character and persona summary sent with requests.

    Args:
        engine: The instance of Celune to use.

    Returns:
        str: The formatted Persona character card and summary.
    """
    context = build_persona_context(engine, "")
    return f"{context.character_profile.render()}\n\n{context.persona_card.render()}"


def _persona_emotion_analyzer(engine: Celune) -> Optional[PersonaEmotionAnalyzer]:
    """Return the configured Persona emotion analyzer for this engine."""
    existing = getattr(engine, "persona_emotion_analyzer", None)

    emotion_config = persona_config(engine.config).get("emotion")
    if isinstance(emotion_config, dict):
        enabled = emotion_config.get("enabled", True)
        if isinstance(enabled, bool) and not enabled:
            return None
        user_weight = emotion_config.get("user_weight", 0.75)
        assistant_weight = emotion_config.get("assistant_weight", 0.25)
        decay_power = emotion_config.get("history_decay_power", 3.0)
        analyzer = PersonaEmotionAnalyzer(
            user_weight=float(user_weight)
            if isinstance(user_weight, (int, float))
            and not isinstance(user_weight, bool)
            else 0.75,
            assistant_weight=float(assistant_weight)
            if isinstance(assistant_weight, (int, float))
            and not isinstance(assistant_weight, bool)
            else 0.25,
            history_decay_power=float(decay_power)
            if isinstance(decay_power, (int, float))
            and not isinstance(decay_power, bool)
            else 3.0,
        )
    else:
        analyzer = (
            existing
            if isinstance(existing, PersonaEmotionAnalyzer)
            else PersonaEmotionAnalyzer()
        )

    if isinstance(existing, PersonaEmotionAnalyzer) and existing is not analyzer:
        existing.user_weight = analyzer.user_weight
        existing.assistant_weight = analyzer.assistant_weight
        existing.history_decay_power = analyzer.history_decay_power

    vision = getattr(engine, "vision", None)
    get_capabilities = getattr(vision, "capabilities", None)
    capabilities = get_capabilities() if callable(get_capabilities) else None
    get_emotion_backend = getattr(vision, "emotion_backend", None)
    emotion_backend = cast(
        Optional[tuple[PersonaTokenizer, PersonaModel]],
        get_emotion_backend() if callable(get_emotion_backend) else None,
    )
    if (
        isinstance(capabilities, PersonaCapabilities)
        and not capabilities.emotion_probes
    ) or emotion_backend is None:
        analyzer.clear_vlm()
    else:
        analyzer.bind_vlm(*emotion_backend)

    engine.persona_emotion_analyzer = analyzer
    return analyzer


def _persona_mood_or_state(
    engine: Celune,
    request: str,
) -> str:
    """Return the Persona state string for the current request."""

    configured_state = _config_text(engine, "persona_state", "")
    if configured_state:
        return configured_state

    analyzer = _persona_emotion_analyzer(engine)
    if analyzer is None:
        return "Neutral."

    summary = analyzer.summarize_history(persona_history_messages(engine), request)
    if summary is None or not summary.target_state.strip():
        emotion_warning = (
            f"Persona emotion analysis fell back to Neutral: {analyzer.last_error}"
            if analyzer.last_error.strip()
            else "Persona emotion analysis fell back to Neutral."
        )
        log = getattr(engine, "log", None)
        if callable(log):
            log(emotion_warning, "warning", loglevel="verbose")
        return "Neutral."
    return summary.target_state


def persona_memory_store(engine: Celune) -> Optional[PersonaMemoryStore]:
    """Return the configured Persona memory store for this engine."""
    existing = getattr(engine, "persona_memory_store", None)
    if isinstance(existing, PersonaMemoryStore):
        return existing

    memory_config = persona_config(engine.config).get("memory")
    normalized_memory = memory_config if isinstance(memory_config, dict) else {}
    enabled = normalized_memory.get("enabled", True)
    if isinstance(enabled, bool) and not enabled:
        return None

    similarity_threshold = normalized_memory.get("semantic_similarity_threshold", 0.62)
    overlap_threshold = normalized_memory.get("fallback_token_overlap_threshold", 1)
    automatic_max_age_days = normalized_memory.get("automatic_max_age_days", 60)
    embedding_model = normalized_memory.get("semantic_embedding_model")
    embedding_model_name = (
        embedding_model.strip()
        if isinstance(embedding_model, str) and embedding_model.strip()
        else None
    )
    configured_storage = normalized_memory.get("storage_dir")
    storage_dir = (
        configured_storage.strip()
        if isinstance(configured_storage, str) and configured_storage.strip()
        else None
    )
    store = PersonaMemoryStore(
        storage_dir=storage_dir,
        semantic_similarity_threshold=float(similarity_threshold)
        if isinstance(similarity_threshold, (int, float))
        and not isinstance(similarity_threshold, bool)
        else 0.62,
        fallback_token_overlap_threshold=int(overlap_threshold)
        if isinstance(overlap_threshold, (int, float))
        and not isinstance(overlap_threshold, bool)
        else 1,
        automatic_max_age_days=automatic_max_age_days
        if automatic_max_age_days is None
        or (
            isinstance(automatic_max_age_days, int)
            and not isinstance(automatic_max_age_days, bool)
        )
        else 60,
        embedding_model=embedding_model_name or PERSONA_MEMORY_EMBEDDING_MODEL,
    )

    engine.persona_memory_store = store
    return store


def store_persona_memories(engine: Celune, request: str) -> None:
    """Persist long-term memory candidates extracted from the user request."""
    store = persona_memory_store(engine)
    if store is None:
        return

    character_name = persona_active_character_name(engine)
    if not character_name.strip():
        return

    store.remember_from_user_message(character_name, request)


def _persona_memory_classifier_context(
    engine: Celune,
    current_request: Optional[str] = None,
) -> str:
    """Build bounded user-authored context for the memory classifier.

    Args:
        engine: Celune engine holding the current Persona history.
        current_request: Current user message, which is added separately.

    Returns:
        str: Prior user-authored messages without assistant-generated content.
    """
    sections: list[str] = []
    normalized_request = current_request.strip() if current_request else ""
    messages = persona_history_messages(engine)
    messages = [
        message
        for message in messages
        if message.get("role") == "user"
        and str(message.get("content", "")).strip() != normalized_request
    ]
    if messages:
        sections.append(
            "Recent user messages:\n"
            + "\n".join(str(message["content"]) for message in messages)
        )
    return "\n\n".join(sections)


def classify_persona_memories(engine: Celune, request: str) -> None:
    """Classify and persist unmatched durable user facts without blocking reply logic."""

    store = persona_memory_store(engine)
    if store is None or store.collect_candidates(request):
        return

    memory_config = persona_config(engine.config).get("memory")
    normalized_memory = memory_config if isinstance(memory_config, dict) else {}
    enabled = normalized_memory.get("auto_classifier", True)
    if isinstance(enabled, bool) and not enabled:
        return

    classifier = cast(
        "Optional[_MemoryClassifier]",
        getattr(getattr(engine, "vision", None), "classify_memory", None),
    )
    if not callable(classifier):
        return

    minimum_confidence = normalized_memory.get("auto_classifier_min_confidence", 0.82)
    if isinstance(minimum_confidence, bool) or not isinstance(
        minimum_confidence, (int, float)
    ):
        minimum_confidence = 0.82
    maximum_candidates = normalized_memory.get("auto_classifier_max_candidates", 3)
    if isinstance(maximum_candidates, bool) or not isinstance(
        maximum_candidates, (int, float)
    ):
        maximum_candidates = 3

    context = _persona_memory_classifier_context(engine, request)
    if request.strip():
        context = f"{context}\n\nCurrent user message:\n{request.strip()}".strip()

    payload: JSON = {
        "format": "celune_memory_classifier",
        "format_version": 1,
        "model": persona_model_id(engine.config),
        "quantization": persona_quantization(engine.config),
        "quantized": True,
        "system": MEMORY_CLASSIFIER_SYSTEM_PROMPT,
        "user": context,
        "request": context,
        "messages": [
            {"role": "system", "content": MEMORY_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        "max_new_tokens": 180,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }

    try:
        response = classifier(payload)
        response.raise_for_status()
        candidates = classifier_memory_candidates(
            extract_persona_text(response.json()),
            minimum_confidence=float(minimum_confidence),
            maximum_candidates=max(1, int(maximum_candidates)),
        )
        character_name = persona_active_character_name(engine)
        if not character_name.strip():
            return
        for candidate in candidates:
            store.remember(
                character_name,
                candidate.content,
                importance=candidate.importance,
                explicit=False,
            )
    except Exception as error:
        log = getattr(engine, "log", None)
        if callable(log):
            log(
                format_error_message(
                    "Persona memory classification failed",
                    error,
                    engine.log_level,
                ),
                loglevel="verbose",
            )


def _build_retrieved_memory_bundle(
    engine: Celune, request: str
) -> RetrievedMemoryBundle:
    """Return retrieved long-term memory for the current request."""

    direct_memories = getattr(engine, "retrieved_long_term_memory", None)
    if isinstance(direct_memories, list):
        memories = [
            memory.strip()
            for memory in direct_memories
            if isinstance(memory, str) and memory.strip()
        ]
        return RetrievedMemoryBundle(memories=tuple(memories))

    store = persona_memory_store(engine)
    if store is not None:
        character_name = persona_active_character_name(engine)
        memories = tuple(
            record.content
            for record in store.retrieve(character_name, request.strip())
            if record.content.strip()
        )
        if memories:
            return RetrievedMemoryBundle(memories=memories)

    return RetrievedMemoryBundle(
        memories=_config_lines(engine, "persona_long_term_memory")
    )


def persona_manifest_files(engine: Celune) -> dict[str, str]:
    """Return whitelisted persona Markdown files for the active engine persona."""
    loader = default_loader()
    if loader is None:
        return {}
    pack_persona = persona_metadata_from_manifest(loader.bundle.metadata)
    current_persona = getattr(engine, "current_character_persona", None)
    if pack_persona is not None:
        if current_persona != pack_persona:
            return {}
    else:
        current_character = getattr(engine, "current_character", None)
        bundle_name = bundle_character_name(loader.bundle)
        if not (
            isinstance(current_character, str)
            and isinstance(bundle_name, str)
            and current_character.strip()
            and current_character.strip() == bundle_name.strip()
        ):
            return {}
    files = persona_files_from_bundle(loader.bundle)
    if persona_debug_overrides_enabled(engine.config):
        files.update(persona_override_files(persona_active_character_name(engine)))
    return files


def _legacy_identity_source(profile: CharacterProfile) -> str:
    """Render legacy identity metadata into CECHAR v3-style source material."""
    lines: list[str] = []
    if profile.name.strip():
        lines.append(f"Name: {profile.name.strip()}")
    if profile.age.strip():
        lines.append(f"Age: {profile.age.strip()}")
    if profile.gender.strip():
        lines.append(f"Gender: {profile.gender.strip()}")
    if profile.profile.strip():
        if lines:
            lines.append("")
        lines.append(profile.profile.strip())
    return "\n".join(lines).strip()


def _legacy_personality_source(engine: Celune) -> str:
    """Render legacy persona settings into the v3 personality source slot."""

    blocks: list[str] = []
    persona_text = _config_text(
        engine,
        "persona_persona",
        default_persona_persona(),
    )
    if persona_text:
        blocks.append(persona_text)

    prompt_rules = pack_persona_lines(engine, "prompt_rules")
    if prompt_rules:
        blocks.append("\n".join(f"- {line}" for line in prompt_rules))

    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _legacy_speech_style_source(engine: Celune) -> str:
    """Render legacy speech-style metadata into the v3 speech-style slot."""
    blocks: list[str] = []
    speaking_style = pack_persona_text(engine, "speaking_style")
    if speaking_style:
        blocks.append(speaking_style)

    traits = persona_style_traits(engine)
    trait_lines = [
        f"- Warmth: {traits['warmth']}",
        f"- Directness: {traits['directness']}",
        f"- Humor: {traits['humor']}",
        f"- Detail: {traits['detail']}",
        f"- Formality: {traits['formality']}",
        f"- Enthusiasm: {traits['enthusiasm']}",
    ]
    blocks.append("\n".join(trait_lines))
    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _legacy_boundaries_source(engine: Celune) -> str:
    """Render legacy boundary lines into the v3 boundaries source slot."""
    lines = pack_persona_lines(engine, "boundaries")
    return "\n".join(f"- {line}" for line in lines)


def _legacy_examples_source(engine: Celune) -> str:
    """Render legacy example dialogue into the v3 examples source slot."""
    return "\n".join(pack_persona_lines(engine, "example_dialogue")).strip()


def _configured_persona_instructions(engine: Celune) -> str:
    """Return explicit user Persona settings after pack-provided behavior."""
    blocks: list[str] = []
    configured_persona = engine.config.get("persona_persona")
    if isinstance(configured_persona, str) and configured_persona.strip():
        blocks.append(f"Persona guidance:\n{configured_persona.strip()}")
    configured_context = engine.config.get("persona_context")
    if isinstance(configured_context, str) and configured_context.strip():
        blocks.append(f"User-provided context:\n{configured_context.strip()}")
    return "\n\n".join(blocks)


def _build_persona_source_material(
    engine: Celune,
    character_profile: CharacterProfile,
) -> PersonaSourceMaterial:
    """Build v3 prompt source material from package files with legacy fallback."""
    persona_files = persona_manifest_files(engine)
    return PersonaSourceMaterial(
        identity=persona_files.get("identity.md", "")
        or _legacy_identity_source(character_profile),
        soul=persona_files.get("soul.md", ""),
        personality=persona_files.get("personality.md", "")
        or _legacy_personality_source(engine),
        speech_style=persona_files.get("speech_style.md", "")
        or _legacy_speech_style_source(engine),
        boundaries=persona_files.get("boundaries.md", "")
        or _legacy_boundaries_source(engine),
        examples=persona_files.get("examples.md", "")
        or _legacy_examples_source(engine),
    )


def build_persona_context(
    engine: Celune,
    request: str,
    *,
    agent_context: Optional[AgentContext] = None,
    include_memory_context: bool = True,
    tool_schemas: tuple[AgentToolSchema, ...] = (),
    pending_tool_call: Optional[ToolCall] = None,
) -> PersonaContext:
    """Build structured Persona context for one user request.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.
        agent_context: Optional existing agent callback context for task-aware prompts.
        include_memory_context: Whether to include stored memories and summaries.
        tool_schemas: Existing tool schemas available to the agent runtime.
        pending_tool_call: Optional validated-boundary tool call awaiting runtime handling.

    Returns:
        PersonaContext: The built context for Persona.
    """

    name = persona_active_character_name(engine)
    voice = getattr(engine, "current_voice", None) or "balanced"
    voice_prompt = effective_voice_prompt(engine)
    traits = persona_style_traits(engine)

    voice_notes = f"Selected voice: {voice}."
    if isinstance(voice_prompt, str) and voice_prompt.strip():
        voice_notes = f"{voice_notes}\nVoice prompt: {voice_prompt.strip()}"

    character_profile = CharacterProfile(
        name=name,
        age=pack_identity_text(engine, "age")
        or _config_text(engine, "persona_character_age", default_persona_age(engine)),
        gender=pack_identity_text(engine, "gender")
        or _config_text(
            engine, "persona_character_gender", default_persona_gender(engine)
        ),
        profile=pack_identity_text(engine, "profile")
        or _config_text(engine, "persona_character_profile", ""),
    )
    persona_card = PersonaCard(
        persona=_config_text(
            engine,
            "persona_persona",
            default_persona_persona(),
        ),
        warmth=traits["warmth"],
        directness=traits["directness"],
        humor=traits["humor"],
        detail=traits["detail"],
        formality=traits["formality"],
        enthusiasm=traits["enthusiasm"],
        context=_config_text(
            engine,
            "persona_context",
            default_persona_context(),
        ),
        voice=voice_notes,
        speaking_style=pack_persona_text(engine, "speaking_style"),
        boundaries=pack_persona_lines(engine, "boundaries"),
        prompt_rules=pack_persona_lines(engine, "prompt_rules"),
        example_dialogue=pack_persona_lines(engine, "example_dialogue"),
    )
    persona_source_material = _build_persona_source_material(engine, character_profile)
    mood_or_state = _persona_mood_or_state(engine, request)

    return PersonaContext(
        character_profile=character_profile,
        persona_card=persona_card,
        persona_source_material=persona_source_material,
        mood_or_state=mood_or_state,
        conversation_summary=(
            persona_session_summary(engine)
            if include_memory_context and agent_context is None
            else ""
        ),
        retrieved_long_term_memory=(
            _build_retrieved_memory_bundle(engine, request)
            if include_memory_context and agent_context is None
            else RetrievedMemoryBundle()
        ),
        user_instructions=_configured_persona_instructions(engine),
        agent_context=agent_context,
        tool_schemas=tool_schemas,
        pending_tool_call=pending_tool_call,
    )


def effective_voice_prompt(engine: Celune) -> Optional[str]:
    """Return the active voice prompt only when the engine supports it."""
    supported = getattr(engine, "voice_prompt_supported", None)
    if callable(supported) and not supported():
        return None
    if supported is False:
        return None

    voice_prompt = getattr(engine, "voice_prompt", None)
    return voice_prompt if isinstance(voice_prompt, str) else None


def build_persona_messages(
    engine: Celune,
    request: str,
    *,
    context: Optional[PersonaContext] = None,
) -> list[JSON]:
    """Build OpenAI-style messages for the Persona model.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.
        context: Optional prebuilt context to keep the system prompt consistent.

    Returns:
        list[JSON]: A list of JSON objects containing current message history.
    """
    resolved_context = context or build_persona_context(engine, request)
    attachments = persona_pending_attachments(engine)
    user_content: JSONSerializable = request.strip()
    if attachments:
        user_content = [
            *attachments,
            {"type": "text", "text": request.strip()},
        ]

    messages: list[JSON] = [
        cast(
            JSON,
            {"role": "system", "content": PersonaPromptBuilder.build(resolved_context)},
        )
    ]
    if resolved_context.agent_context is None:
        messages.extend(persona_history_messages(engine))
    messages.append(cast(JSON, {"role": "user", "content": user_content}))
    return messages


def build_persona_request(
    engine: Celune,
    request: str,
    *,
    agent_context: Optional[AgentContext] = None,
    tool_schemas: tuple[AgentToolSchema, ...] = (),
    pending_tool_call: Optional[ToolCall] = None,
) -> JSON:
    """Build the JSON payload sent to the Persona model.

    Args:
        engine: The instance of Celune to use.
        request: The user's request.
        agent_context: Optional existing agent callback context for task-aware prompts.
        tool_schemas: Existing tool schemas available to the agent runtime.
        pending_tool_call: Optional tool call awaiting runtime handling.

    Returns:
        JSON: The JSON payload to be sent to the Persona model.
    """
    context = build_persona_context(
        engine,
        request,
        agent_context=agent_context,
        tool_schemas=tool_schemas,
        pending_tool_call=pending_tool_call,
    )
    character_card = (
        f"{context.character_profile.render()}\n\n{context.persona_card.render()}"
    )
    system_prompt = PersonaPromptBuilder.build(context)
    clean_request = request.strip()
    context_size = persona_context_size(engine.config)
    if agent_context is not None:
        context_size = (
            min(agent_context.task.config.context_size, AGENT_CONTEXT_SPACE)
            if agent_context.task is not None
            else AGENT_ROUTING_CONTEXT_SPACE
        )
    return {
        "format": "celune_persona_request",
        "format_version": 1,
        "model": persona_model_id(engine.config),
        "quantization": persona_quantization(engine.config),
        "quantized": True,
        "character": getattr(engine, "current_character", None) or "Unknown",
        "voice": getattr(engine, "current_voice", None) or "balanced",
        "character_card": character_card,
        "system": system_prompt,
        "user": clean_request,
        "request": clean_request,
        "context_space": context_size,
        "messages": cast(
            JSONSerializable,
            build_persona_messages(engine, clean_request, context=context),
        ),
    }


def extract_persona_text(payload: JSONSerializable) -> str:
    """Extract spoken text from common Persona response payload shapes."""
    if isinstance(payload, str):
        return payload.strip()

    if not isinstance(payload, dict):
        return ""

    for key in ("text", "response", "reply", "output", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return extract_persona_text(first)

    return ""
