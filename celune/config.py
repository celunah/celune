# SPDX-License-Identifier: MIT
"""Configuration helpers for Celune."""

import os
from copy import deepcopy
from typing import Optional
from collections.abc import Mapping

from .constants import JSONSerializable

ENABLED_ENV_VALUES = {"1", "true", "on", "yes", "enabled"}
type Config = dict[str, JSONSerializable]


def env_bool(name: str, fallback: bool = False) -> bool:
    """Read a boolean environment variable with strict enabled values.

    Args:
        name: Environment variable name.
        fallback: Value to use when the variable is unset.

    Returns:
        bool: ``True`` only for known enabled strings. Other set values are treated as disabled. When the variable is
        unset, ``fallback`` is returned.
    """
    value = os.getenv(name)
    if value is None:
        return fallback
    return value.strip().lower() in ENABLED_ENV_VALUES


def config_value(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
    default: JSONSerializable = None,
) -> JSONSerializable:
    """Safely read a value from the loaded YAML configuration.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        key: Configuration key to read.
        default: Value returned when config or key is missing.

    Returns:
        JSONSerializable: The configured value or ``default``.
    """
    if not config:
        return default
    return config.get(key, default)


def config_bool(
    config: Optional[Mapping[str, JSONSerializable]],
    env_name: str,
    config_key: str,
    default: bool = False,
) -> bool:
    """Read a boolean setting where env vars take precedence over config.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        env_name: Environment variable name to check first.
        config_key: Configuration key used when the environment variable is unset.
        default: Fallback value when no setting is present.

    Returns:
        bool: The resolved boolean setting.
    """
    return env_bool(env_name, bool(config_value(config, config_key, default)))


def merge_missing_defaults(
    config: Optional[Mapping[str, JSONSerializable]],
    defaults: Mapping[str, JSONSerializable],
) -> tuple[Config, bool]:
    """Fill missing configuration fields from defaults without overriding users.

    Args:
        config: Loaded user configuration, or ``None`` for an empty config.
        defaults: Default configuration fields to merge into ``config``.

    Returns:
        tuple[Config, bool]: The merged configuration and whether any fields were added.
    """
    merged: Config = dict(deepcopy(config)) if config is not None else {}
    changed = False

    for key, default_value in defaults.items():
        if key not in merged:
            merged[key] = deepcopy(default_value)
            changed = True
            continue

        current_value = merged[key]
        if isinstance(current_value, dict) and isinstance(default_value, dict):
            nested, nested_changed = merge_missing_defaults(
                current_value,
                default_value,
            )
            if nested_changed:
                merged[key] = nested
                changed = True

    return merged, changed
