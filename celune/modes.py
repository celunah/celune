# SPDX-License-Identifier: Apache-2.0
"""Global Celune operation modes and their feature gates."""

from __future__ import annotations

# noinspection PyProtectedMember
from typing import Optional, cast
from collections.abc import Mapping

from .utils import discard
from .config import config_value
from .typing.common import JSONSerializable
from .typing.modes import OPERATION_MODES, OperationMode

_LEGACY_OPERATION_MODE_ALIASES: dict[str, OperationMode] = {"speak": "performance"}
_LEGACY_INPUT_MODES = {"text_to_speech", "tts", "voice_conversion", "revoice"}

# legacy, please remove
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
            discard(candidate)
        else:
            resolved = _LEGACY_OPERATION_MODE_ALIASES.get(normalized, normalized)
            if resolved not in OPERATION_MODES:
                raise ValueError(f"unknown Celune operation mode: '{candidate}'")
            if resolved == "agent" and AGENT_MODE_REDIRECT_TARGET is not None:
                return AGENT_MODE_REDIRECT_TARGET
            # HACK: Pyrefly's type inference is flaky here, and may randomly turn OperationMode into str
            return cast(OperationMode, resolved)  # type: ignore[redundant-cast]

    return "converse"


def mode_allows_persona(mode: OperationMode) -> bool:
    """Return whether a global mode may load Persona."""
    return mode in {"converse", "agent"}


def has_explicit_operation_mode(config: Mapping[str, JSONSerializable]) -> bool:
    """Return whether config explicitly selects a global operation mode."""
    configured = config.get("mode")
    return isinstance(configured, str) and configured.strip().casefold() in (
        *OPERATION_MODES,
        *_LEGACY_OPERATION_MODE_ALIASES,
    )


def mode_allows_agents(mode: OperationMode) -> bool:
    """Return whether a global mode enables the agent runtime contract."""
    return mode == "agent"
