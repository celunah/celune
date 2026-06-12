# SPDX-License-Identifier: MIT
"""Celune-managed Persona runtime helpers."""

import os
import io
import contextlib
from collections.abc import Mapping
from typing import Callable, Optional, Generator, Any

from ..config import Config
from ..cevoice import CEVoicePersona
from ..vram import resolve_vram_preset
from .runtime import PersonaRuntime, request_from_json, response_to_json
from ..constants import (
    DEFAULT_PERSONA_CONTEXT,
    DEFAULT_PERSONA_DESCRIPTION,
    JSON,
    JSONSerializable,
    PERSONA_HISTORY_MESSAGES,
    PERSONA_MODEL_ID,
)

PERSONA_QUANTIZATION = "4bit"
DevLogCallback = Callable[[str, str], None]


class PersonaClientResponse:
    """Small response shim matching the local HTTP client contract."""

    def __init__(self, payload: dict[str, JSONSerializable]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Mirror the ``httpx`` response API for local in-process calls."""

    def json(self) -> dict[str, JSONSerializable]:
        """Return the stored response payload.

        Returns:
            str: The stored JSON response payload.
        """
        return dict(self._payload)


class PersonaClient:
    """In-process Persona client adapter used by Celune."""

    def __init__(
        self,
        config: Optional[Mapping[str, JSONSerializable]] = None,
        log_dev: Optional[DevLogCallback] = None,
    ) -> None:
        self.runtime = PersonaRuntime(config=config)
        self.config = config
        self.log_dev = log_dev

    @contextlib.contextmanager
    def _capture_backend_output(self) -> Generator[None, None, None]:
        """Route Persona backend stdout/stderr into Celune developer logs."""
        if self.log_dev is None:
            yield
            return

        stderr_buffer = io.StringIO()
        with contextlib.redirect_stderr(stderr_buffer):
            yield

        for line in stderr_buffer.getvalue().splitlines():
            text = line.strip()
            if text:
                self.log_dev(f"[PERSONA] {text}", "warning")

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

    def close(self) -> None:
        """Release Persona runtime state."""
        self.runtime.close()


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
    elif raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raw = {}

    return dict(raw)


def _config_text(
    engine: Any,
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


def pack_persona(engine: Any) -> Optional[CEVoicePersona]:
    """Return typed CEVOICE persona metadata attached to the current engine.

    Args:
        engine: Celune-like runtime object that may expose persona metadata.

    Returns:
        Optional[CEVoicePersona]: The active persona metadata when present and typed.
    """
    persona = getattr(engine, "current_character_persona", None)
    return persona if isinstance(persona, CEVoicePersona) else None


def pack_identity_text(engine: Any, field_name: str) -> str:
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


def pack_persona_text(engine: Any, field_name: str) -> str:
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


def pack_persona_lines(engine: Any, field_name: str) -> tuple[str, ...]:
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


def persona_active_character_name(engine: Any) -> str:
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


def uses_default_celune_identity(engine: Any) -> bool:
    """Return whether Persona defaults should use Celune's canonical identity.

    Args:
        engine: Celune-like runtime object holding the active voice bundle state.

    Returns:
        bool: ``True`` when the default Celune voice bundle is active for Celune.
    """
    if not bool(getattr(engine, "voice_bundle_is_default", False)):
        return False
    return persona_active_character_name(engine).strip().lower() == "celune"


def default_persona_persona() -> str:
    """Return the default persona instructions for the active character.

    Returns:
        str: Built-in fallback system prompt used for Persona conversations.
    """
    return DEFAULT_PERSONA_DESCRIPTION


def default_persona_age(engine: Any) -> str:
    """Return the default age for the active character source.

    Args:
        engine: Celune-like runtime object used to choose default identity values.

    Returns:
        str: Default age string for the active persona source.
    """
    if uses_default_celune_identity(engine):
        return "28"
    return "unknown"


def default_persona_gender(engine: Any) -> str:
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


def persona_style_traits(engine: Any) -> dict[str, str]:
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


def persona_short_term_history_limit(engine: Any) -> int:
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


def persona_history_limit() -> int:
    """Return the default short-term memory length for Persona.

    Returns:
        int: Built-in fallback message-window length for Persona history.
    """
    return PERSONA_HISTORY_MESSAGES


def persona_history_messages(engine: Any) -> list[JSON]:
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


def persona_pending_attachments(engine: Any) -> list[JSON]:
    """Return pending Persona attachments in Qwen chat content format.

    Args:
        engine: Celune-like runtime object that stores staged Persona attachments.

    Returns:
        list[JSON]: Attachment content blocks formatted for the Persona request.
    """
    attachments = getattr(engine, "persona_attachments", [])
    if not isinstance(attachments, list):
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
    return resolve_vram_preset(config).persona_enabled and bool(
        persona_config(config).get("enabled", True)
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

    return PERSONA_MODEL_ID


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
    log_dev: Optional[DevLogCallback] = None,
) -> Optional[PersonaClient]:
    """Create a Celune-managed in-process Persona client when enabled.

    Args:
        config: Celune's current configuration.
        log_dev: The logging callback to Celune's UI.

    Returns:
        Optional[PersonaClient]: ``PersonaClient`` if Persona is enabled, else ``None``.
    """
    if config is not None and not persona_enabled(config):
        return None

    if not persona_is_available():
        return None

    return PersonaClient(config=config, log_dev=log_dev)
