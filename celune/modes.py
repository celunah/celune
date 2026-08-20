# SPDX-License-Identifier: Apache-2.0
"""Global Celune operation modes and their feature gates."""

from __future__ import annotations

from typing import Optional, cast
from collections.abc import Mapping

from .config import config_value
from .typing.modes import OperationMode
from .typing.common import JSONSerializable

OPERATION_MODES: tuple[OperationMode, ...] = ("speak", "converse", "agent")
_LEGACY_INPUT_MODES = {"text_to_speech", "tts", "voice_conversion", "revoice"}

# Agent mode is now owned by the production AgentRuntime integration.
AGENT_MODE_REDIRECT_TARGET: Optional[OperationMode] = None


def resolve_operation_mode(
    config: Optional[Mapping[str, JSONSerializable]],
    requested_mode: Optional[str] = None,
) -> OperationMode:
    """Resolve the global operation mode from config or an explicit override.

    Args:
        config: Celune's loaded configuration.
        requested_mode: Optional mode supplied by a caller with higher priority.

    Returns:
        OperationMode: The resolved global mode.

    Raises:
        ValueError: If an explicitly configured operation mode is unsupported.
    """
    candidate = requested_mode
    if candidate is None and config is not None:
        configured = config_value(config, "mode")
        if isinstance(configured, str) and configured.strip():
            candidate = configured

    if candidate is not None:
        normalized = candidate.strip().casefold()
        if normalized in _LEGACY_INPUT_MODES:
            candidate = None
        elif normalized in OPERATION_MODES:
            resolved = normalized
            if resolved == "agent" and AGENT_MODE_REDIRECT_TARGET is not None:
                return AGENT_MODE_REDIRECT_TARGET
            # HACK: Pyrefly's type inference is flaky here, and may randomly turn OperationMode into str
            return cast(OperationMode, resolved)  # type: ignore[redundant-cast]
        else:
            raise ValueError(f"unknown Celune operation mode: '{candidate}'")

    if config is not None:
        persona = config.get("persona", config.get("pyop", {}))
        if isinstance(persona, dict) and persona.get("enabled") is False:
            return "speak"
    return "converse"


def mode_allows_persona(mode: OperationMode) -> bool:
    """Return whether a global mode may load Persona."""
    return mode in {"converse", "agent"}


def has_explicit_operation_mode(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether config explicitly selects a global operation mode."""
    configured = config.get("mode")
    return (
        isinstance(configured, str) and configured.strip().casefold() in OPERATION_MODES
    )


def mode_allows_agents(mode: OperationMode) -> bool:
    """Return whether a global mode enables the agent runtime contract."""
    return mode == "agent"
