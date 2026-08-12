# SPDX-License-Identifier: MIT
"""Celune-managed Persona runtime helpers."""

import contextlib
import io
import os
import re
from collections.abc import Generator, Mapping
from typing import Optional

from ..cevoice import (
    CEVoicePersona,
    default_loader,
    merge_persona_metadata,
    persona_metadata_from_voice,
)
from ..config import Config
from ..constants import (
    DEFAULT_PERSONA_CONTEXT,
    DEFAULT_PERSONA_DESCRIPTION,
    PERSONA_DEFAULT_MODEL_ID,
    PERSONA_HISTORY_MESSAGES,
)
from ..i18n import string
from ..modes import (
    has_explicit_operation_mode,
    mode_allows_persona,
    resolve_operation_mode,
)
from ..typing.aliases import LogCallback
from ..typing.common import JSON, JSONSerializable
from ..typing.persona import (
    PersonaClientResponse,
    PersonaEngineView,
    PersonaModel,
    PersonaTokenizer,
)
from ..vram import resolve_vram_preset
from .capabilities import PersonaCapabilities
from .runtime import PersonaRuntime, request_from_json, response_to_json

PERSONA_QUANTIZATION = "4bit"
_CONVERSATION_SUMMARY_SYSTEM_PROMPT = (
    "You are a neutral conversation summarizer. Do not roleplay, imitate a "
    "character, answer the user, or use a character's speaking style. Summarize "
    "the supplied conversation in concise factual prose. Preserve important "
    "facts, decisions, preferences, unresolved issues, and relevant emotional "
    "context. Ignore greetings and filler. Output only the summary prose, with "
    "no XML tags, role labels, or heading."
)
_SUMMARY_PREFIX = "Conversation context:"
_SUMMARY_SPLIT_RE = re.compile(
    r"(?:\r?\n+|(?<=[.!?\u3002\uff01\uff1f\u061f])\s*|[;\uff1b]\s*)"
)
_SUMMARY_WRAPPER_RE = re.compile(r"</?(?:conversation_summary|summary)>", re.IGNORECASE)
_SUMMARY_LABEL_RE = re.compile(
    r"^\s*(?:summary|earlier summary|conversation summary|conversation context|"
    r"user|assistant|celune)\s*:\s*",
    re.IGNORECASE,
)


class PersonaClient:
    """In-process Persona client adapter used by Celune."""

    def __init__(
        self,
        config: Optional[Mapping[str, JSONSerializable]] = None,
        log: Optional[LogCallback] = None,
        log_dev: Optional[LogCallback] = None,
    ) -> None:
        self.runtime = PersonaRuntime(config=config)
        self.config = config
        self.log = log or log_dev

    @contextlib.contextmanager
    def _capture_backend_output(self) -> Generator[None, None, None]:
        """Route Persona backend stderr into Celune developer logs."""
        if self.log is None:
            yield
            return

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            yield

        for line in stderr_buffer.getvalue().splitlines():
            text = line.strip()
            if text:
                self.log(
                    string("persona.backend_diagnostic", message=text),
                    "warning",
                    loglevel="verbose",
                )

    def load(
        self,
        model_id: str,
        quantization: str = PERSONA_QUANTIZATION,
    ) -> None:
        """Explicitly load the Persona runtime.

        Args:
            model_id: The Persona model ID.
            quantization: The requested quantization mode.
        """
        with self._capture_backend_output():
            self.runtime.load(model_id, quantization)

    def post(self, json: dict[str, JSONSerializable]) -> PersonaClientResponse:
        """Handle a Persona generation request without leaving the process.

        Args:
            json: A JSON serializable payload to be sent.

        Returns:
            PersonaClientResponse: The client response received from Persona.
        """
        request = request_from_json(json)
        with self._capture_backend_output():
            response = self.runtime.generate(request)
        return PersonaClientResponse(response_to_json(response))

    def classify_memory(
        self, json: dict[str, JSONSerializable]
    ) -> PersonaClientResponse:
        """Classify durable user-memory candidates through the same Persona runtime.

        Args:
            json: A structured memory-classification request.

        Returns:
            PersonaClientResponse: The classifier response using the local Persona model.
        """
        return self.post(json)

    def summarize_history(
        self,
        messages: list[JSON],
        previous_summary: str = "",
        maximum_characters: int = 1200,
    ) -> str:
        """Summarize conversation context without the CEVOICE persona prompt.

        Args:
            messages: Older conversation turns that are being compacted.
            previous_summary: The summary produced by the previous compaction pass.
            maximum_characters: Maximum desired length of the returned summary.

        Returns:
            str: Neutral summary prose returned by the active Persona model.
        """
        context_sections: list[str] = []
        if previous_summary.strip():
            context_sections.append(f"Existing summary:\n{previous_summary.strip()}")
        if messages:
            turns = "\n".join(
                f"{message.get('role', 'unknown')}: {' '.join(str(message.get('content', '')).split())}"
                for message in messages
            )
            context_sections.append(f"Conversation turns:\n{turns}")
        context = "\n\n".join(context_sections).strip()
        if not context:
            return ""

        response = self.post(
            {
                "format": "celune_conversation_summary",
                "format_version": 1,
                "model": persona_model_id(self.config),
                "quantization": persona_quantization(self.config or {}),
                "quantized": True,
                "system": _CONVERSATION_SUMMARY_SYSTEM_PROMPT,
                "user": context,
                "request": context,
                "messages": [
                    {
                        "role": "system",
                        "content": _CONVERSATION_SUMMARY_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": context},
                ],
                "max_new_tokens": max(64, min(240, maximum_characters // 3)),
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            }
        )
        payload = response.json()
        summary = payload.get("response", payload.get("text", ""))
        return summary.strip() if isinstance(summary, str) else ""

    def close(self) -> None:
        """Release Persona runtime state."""
        self.runtime.close()

    def emotion_backend(self) -> Optional[tuple[PersonaTokenizer, PersonaModel]]:
        """Return the active VLM components for local emotion analysis."""
        return self.runtime.emotion_backend()

    def capabilities(self) -> PersonaCapabilities:
        """Return the capabilities of the active Persona architecture."""
        return self.runtime.capabilities()


def persona_config(config: Mapping[str, JSONSerializable]) -> Config:
    """Return the normalized configuration block for the persona system.

    Args:
        config: The configuration data for the persona system.

    Returns:
        Config: The normalized configuration data for the persona system.
    """
    raw = config.get("persona", config.get("pyop", {}))
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    elif raw is None or not isinstance(raw, dict):
        raw = {}

    return dict(raw)


def persona_debug_overrides_enabled(
    config: Mapping[str, JSONSerializable],
) -> bool:
    """Return whether app-data Persona Markdown overrides are enabled.

    Args:
        config: Celune's current configuration.

    Returns:
        bool: Whether Persona should load character files from app data.
    """
    return persona_config(config).get("debug_overrides") is True


def _config_text(
    engine: PersonaEngineView,
    key: str,
    default: str = "",
) -> str:
    """Return one text configuration value from an engine-like object."""
    config = getattr(engine, "config", {})
    if not isinstance(config, Mapping):
        return default

    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def pack_persona(engine: PersonaEngineView) -> Optional[CEVoicePersona]:
    """Return typed CEVOICE persona metadata attached to the current engine.

    Args:
        engine: Celune-like runtime object that may expose persona metadata.

    Returns:
        Optional[CEVoicePersona]: The active persona metadata when present and typed.
    """
    base_persona = getattr(engine, "current_character_persona", None)
    base_persona = base_persona if isinstance(base_persona, CEVoicePersona) else None
    backend = getattr(engine, "backend", None)
    if getattr(backend, "uses_voice_bundles", False) is not True:
        return base_persona

    loader = default_loader()
    current_voice = getattr(engine, "current_voice", None)
    if loader is None:
        return base_persona

    return merge_persona_metadata(
        base_persona,
        persona_metadata_from_voice(loader.bundle, current_voice),
    )


def pack_identity_text(engine: PersonaEngineView, field_name: str) -> str:
    """Read one CEVOICE persona identity field when present.

    Args:
        engine: Celune-like runtime object that may expose persona metadata.
        field_name: Identity-field attribute name to read from the persona.

    Returns:
        str: Trimmed field value, or an empty string when the field is unavailable.
    """
    persona = pack_persona(engine)
    if persona is None:
        return ""
    identity = persona.identity
    value = getattr(identity, field_name, "")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def pack_persona_text(engine: PersonaEngineView, field_name: str) -> str:
    """Read one top-level CEVOICE persona text field when present.

    Args:
        engine: Celune-like runtime object that may expose persona metadata.
        field_name: Top-level persona attribute name to read.

    Returns:
        str: Trimmed field value, or an empty string when the field is unavailable.
    """
    persona = pack_persona(engine)
    if persona is None:
        return ""
    value = getattr(persona, field_name, "")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def pack_persona_lines(engine: PersonaEngineView, field_name: str) -> tuple[str, ...]:
    """Read one CEVOICE persona text-list field when present.

    Args:
        engine: Celune-like runtime object that may expose persona metadata.
        field_name: Persona tuple field to read and normalize into non-empty lines.

    Returns:
        tuple[str, ...]: Trimmed non-empty lines from the requested persona field.
    """
    persona = pack_persona(engine)
    if persona is None:
        return ()
    raw = getattr(persona, field_name, ())
    if not isinstance(raw, tuple):
        return ()
    lines = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    return tuple(lines)


def persona_active_character_name(engine: PersonaEngineView) -> str:
    """Return the active character name used for Persona memory isolation.

    Args:
        engine: Celune-like runtime object holding the current character selection.

    Returns:
        str: Active character name, falling back to config defaults when needed.
    """
    current_character = getattr(engine, "current_character", None)
    if isinstance(current_character, str) and current_character.strip():
        return current_character.strip()

    pack_identity_name = pack_identity_text(engine, "name")
    if pack_identity_name:
        return pack_identity_name

    return _config_text(engine, "persona_character_name", "Unknown")


def uses_default_celune_identity(engine: PersonaEngineView) -> bool:
    """Return whether Persona defaults should use Celune's canonical identity.

    Args:
        engine: Celune-like runtime object holding the active voice bundle state.

    Returns:
        bool: ``True`` when the default Celune voice bundle is active for Celune.
    """
    if not bool(getattr(engine, "voice_bundle_is_default", False)):
        return False
    return persona_active_character_name(engine).strip().casefold() == "celune"


def default_persona_persona() -> str:
    """Return the default persona instructions for the active character.

    Returns:
        str: Built-in fallback system prompt used for Persona conversations.
    """
    return DEFAULT_PERSONA_DESCRIPTION


def default_persona_age(engine: PersonaEngineView) -> str:
    """Return the default age for the active character source.

    Args:
        engine: Celune-like runtime object used to choose default identity values.

    Returns:
        str: Default age string for the active persona source.
    """
    if uses_default_celune_identity(engine):
        return "28"
    return "unknown"


def default_persona_gender(engine: PersonaEngineView) -> str:
    """Return a conservative gender default for the active character source.

    Args:
        engine: Celune-like runtime object used to choose default identity values.

    Returns:
        str: Default gender string for the active persona source.
    """
    if uses_default_celune_identity(engine):
        return "female"
    return "unknown"


def default_persona_context() -> str:
    """Return the default interaction context for the active character source.

    Returns:
        str: Built-in fallback environment and relationship context.
    """
    return DEFAULT_PERSONA_CONTEXT


def persona_style_traits(engine: PersonaEngineView) -> dict[str, str]:
    """Return the configured speaking-style traits for a Persona request.

    Args:
        engine: Celune-like runtime object that may expose persona style metadata.

    Returns:
        dict[str, str]: Style-trait values merged with Celune's default trait set.
    """
    traits = {
        "warmth": "mid",
        "directness": "mid",
        "humor": "low",
        "detail": "mid",
        "formality": "mid",
        "enthusiasm": "mid",
    }
    persona = pack_persona(engine)
    if persona is None:
        return traits

    style = persona.style
    configured = {
        "warmth": style.warmth,
        "directness": style.directness,
        "humor": style.humor,
        "detail": style.detail,
        "formality": style.formality,
        "enthusiasm": style.enthusiasm,
    }
    for key, value in configured.items():
        if value.strip():
            traits[key] = value.strip()
    return traits


def persona_short_term_history_limit(engine: PersonaEngineView) -> int:
    """Return the configured short-term memory length for Persona.

    Args:
        engine: Celune-like runtime object whose config may override the default.

    Returns:
        int: Maximum number of recent chat messages to keep in short-term memory.
    """
    config = getattr(engine, "config", {})
    memory = (
        persona_config(config).get("memory") if isinstance(config, Mapping) else None
    )
    if isinstance(memory, dict):
        configured = memory.get("max_short_term_messages")
        if isinstance(configured, bool):
            return PERSONA_HISTORY_MESSAGES
        if isinstance(configured, (int, float)):
            return max(0, int(configured))
        if isinstance(configured, str):
            stripped = configured.strip()
            if stripped:
                try:
                    return max(0, int(stripped))
                except ValueError:
                    return PERSONA_HISTORY_MESSAGES
    return PERSONA_HISTORY_MESSAGES


def persona_history_messages(engine: PersonaEngineView) -> list[JSON]:
    """Return prior Persona chat messages in OpenAI chat format.

    Args:
        engine: Celune-like runtime object that stores prior Persona messages.

    Returns:
        list[JSON]: Sanitized chat-history entries ready for the Persona API.
    """
    history = getattr(engine, "persona_history", [])
    if not isinstance(history, list):
        return []

    messages: list[JSON] = []
    limit = persona_short_term_history_limit(engine)
    window = history if limit <= 0 else history[-limit:]
    for item in window:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")
        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({"role": role, "content": content.strip()})

    return messages


def persona_session_summary(engine: PersonaEngineView) -> str:
    """Return the bounded summary of older Persona conversation turns."""
    summary = getattr(engine, "persona_session_summary", "")
    return summary.strip() if isinstance(summary, str) else ""


def _summary_fragments(text: str) -> list[str]:
    """Extract clean, sentence-sized fragments from one summary source."""
    cleaned = _SUMMARY_WRAPPER_RE.sub(" ", text)
    fragments: list[str] = []
    for raw_fragment in _SUMMARY_SPLIT_RE.split(cleaned):
        fragment = raw_fragment.strip()
        while True:
            unlabeled = _SUMMARY_LABEL_RE.sub("", fragment)
            if unlabeled == fragment:
                break
            fragment = unlabeled.strip()
        fragment = " ".join(fragment.split()).strip(" -:;")
        if len(fragment) >= 12:
            fragments.append(fragment)
    return fragments


def _summary_fragment_key(fragment: str) -> str:
    """Normalize one summary fragment for duplicate detection."""
    return " ".join(fragment.casefold().strip(".!?").split())


def _build_persona_summary(
    previous_summary: str,
    old_messages: list[dict[str, str]],
    maximum_characters: int,
) -> str:
    """Build a bounded deterministic fallback digest without recursive nesting."""
    sources: list[str] = [
        content
        for message in old_messages
        for content in [message.get("content", "")]
        if isinstance(content, str)
    ]

    if previous_summary:
        sources.append(previous_summary)

    fragments: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for fragment in _summary_fragments(source):
            key = _summary_fragment_key(fragment)
            if not key or key in seen:
                continue
            seen.add(key)
            fragments.append(fragment)

    if not fragments:
        return ""

    prefix_length = len(_SUMMARY_PREFIX) + 1
    available = max(0, maximum_characters - prefix_length)
    selected: list[str] = []
    used = 0
    for fragment in fragments:
        separator_length = 1 if selected else 0
        remaining = available - used - separator_length
        if remaining < 12:
            break
        if len(fragment) > remaining:
            fragment = f"{fragment[: max(0, remaining - 3)].rstrip()}..."
        selected.append(fragment)
        used += separator_length + len(fragment)

    return f"{_SUMMARY_PREFIX} {' '.join(selected)}".strip()


def _vlm_persona_summary(
    engine: PersonaEngineView,
    previous_summary: str,
    old_messages: list[dict[str, str]],
    maximum_characters: int,
) -> str:
    """Ask the active Persona VLM for a neutral summary when supported."""
    vision = getattr(engine, "vision", None)
    summarize_history = getattr(vision, "summarize_history", None)
    if not callable(summarize_history):
        return ""

    try:
        generated = summarize_history(
            old_messages,
            previous_summary,
            maximum_characters,
        )
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(generated, str) or not generated.strip():
        return ""

    return _build_persona_summary(
        "",
        [{"role": "assistant", "content": generated}],
        maximum_characters,
    )


def compact_persona_history(engine: PersonaEngineView) -> None:
    """Compact older Persona turns through a neutral VLM summary request.

    Args:
        engine: Celune-like runtime whose Persona history should be summarized.
    """
    history = getattr(engine, "persona_history", None)
    if not isinstance(history, list):
        return

    limit = persona_short_term_history_limit(engine)
    if limit <= 0:
        history.clear()
        return
    if len(history) <= limit:
        return

    config = getattr(engine, "config", {})
    raw_memory = (
        persona_config(config).get("memory") if isinstance(config, Mapping) else None
    )
    memory = raw_memory if isinstance(raw_memory, dict) else {}
    enabled = memory.get("context_compaction_enabled", True)
    if isinstance(enabled, bool) and not enabled:
        del history[:-limit]
        return

    keep_recent = memory.get("context_compaction_keep_recent_messages", min(limit, 8))
    if isinstance(keep_recent, bool) or not isinstance(keep_recent, (int, float)):
        keep_recent = min(limit, 8)
    keep_count = max(1, min(limit, int(keep_recent)))

    previous_summary = persona_session_summary(engine)
    old_messages: list[dict[str, str]] = []
    for message in history[:-keep_count]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        normalized = " ".join(content.split())
        if not normalized:
            continue
        old_messages.append({"role": role, "content": normalized})

    maximum_characters = memory.get("context_summary_max_characters", 1200)
    if isinstance(maximum_characters, bool) or not isinstance(
        maximum_characters, (int, float)
    ):
        maximum_characters = 1200
    maximum_characters = max(240, int(maximum_characters))
    summary = _vlm_persona_summary(
        engine,
        previous_summary,
        old_messages,
        maximum_characters,
    ) or _build_persona_summary(previous_summary, old_messages, maximum_characters)

    setattr(engine, "persona_session_summary", summary)
    del history[:-keep_count]


def persona_attachment_source(path: str) -> str:
    """Return a qwen-vl-utils-safe attachment path or URI.

    Args:
        path: Attachment path or file URI captured by the Persona UI.

    Returns:
        str: Normalized path or URI suitable for Qwen vision attachments.
    """
    source = path.strip()
    if os.name == "nt" and source.startswith("file:///"):
        without_scheme = source.removeprefix("file:///")
        if len(without_scheme) >= 2 and without_scheme[1] == ":":
            return without_scheme
    return source


def persona_pending_attachments(engine: PersonaEngineView) -> list[JSON]:
    """Return pending Persona attachments in Qwen chat content format.

    Args:
        engine: Celune-like runtime object that stores staged Persona attachments.

    Returns:
        list[JSON]: Attachment content blocks formatted for the Persona request.
    """
    attachments = getattr(engine, "persona_attachments", [])
    if not isinstance(attachments, list):
        return []

    vision = getattr(engine, "vision", None)
    get_capabilities = getattr(vision, "capabilities", None)
    if callable(get_capabilities):
        capabilities = get_capabilities()
        if (
            isinstance(capabilities, PersonaCapabilities)
            and not capabilities.image_uploads
        ):
            return []

    content: list[JSON] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue

        kind = attachment.get("type")
        path = attachment.get("path")
        if kind in {"image", "video"} and isinstance(path, str) and path.strip():
            content.append({"type": kind, kind: persona_attachment_source(path)})

    return content


def persona_enabled(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether Celune should try to use Persona.

    Args:
        config: Celune's current configuration.

    Returns:
        bool: Whether Persona is enabled.
    """
    mode_allowed = mode_allows_persona(resolve_operation_mode(config))
    vram_allowed = resolve_vram_preset(config).persona_enabled
    configured_persona = bool(persona_config(config).get("enabled", True))
    return (
        mode_allowed
        and vram_allowed
        and (has_explicit_operation_mode(config) or configured_persona)
    )


def persona_talkback_enabled(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether regular UI input should go through persona talkback.

    Args:
        config: Celune's current configuration.

    Returns:
        bool: Whether Celune should use Persona if enabled, or not.
    """
    return persona_enabled(config) and bool(
        persona_config(config).get("talkback", True)
    )


def persona_quantization(config: Mapping[str, JSONSerializable]) -> str:
    """Return the Persona quantization mode permitted by the VRAM tier.

    Args:
        config: Celune's current configuration.

    Returns:
        str: The resolved permitted quantization type for Persona from current VRAM tier.
    """
    return resolve_vram_preset(config).persona_quantization


def persona_model_id(config: Optional[Mapping[str, JSONSerializable]] = None) -> str:
    """Return the Persona model ID Celune should load.

    Args:
        config: Celune's current configuration.

    Returns:
        str: The currently selected Persona model ID: Qwen/Qwen2.5-VL-3B-Instruct or a derivative.
    """
    if config is not None:
        configured = persona_config(config).get("model_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()

    return PERSONA_DEFAULT_MODEL_ID


def persona_is_available() -> bool:
    """Check whether the in-process Persona runtime can be used.

    Returns:
        bool: Whether the in-process Persona is usable.
    """
    try:
        PersonaRuntime()
        return True
    except Exception:  # noqa
        return False


def create_persona_client(
    config: Optional[Mapping[str, JSONSerializable]] = None,
    log: Optional[LogCallback] = None,
    log_dev: Optional[LogCallback] = None,
) -> Optional[PersonaClient]:
    """Create a Celune-managed in-process Persona client when enabled.

    Args:
        config: Celune's current configuration.
        log: The logging callback to Celune's UI.

    Returns:
        Optional[PersonaClient]: ``PersonaClient`` if Persona is enabled, else ``None``.
    """
    if config is not None and not persona_enabled(config):
        return None

    if not persona_is_available():
        return None

    return PersonaClient(config=config, log=log, log_dev=log_dev)
