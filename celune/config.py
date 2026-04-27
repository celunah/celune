"""Configuration helpers for Celune."""

import os
from typing import Any, Optional

ENABLED_ENV_VALUES = {"1", "true", "on", "yes", "enabled"}


def env_bool(name: str, fallback: bool = False) -> bool:
    """Read a boolean environment variable with strict enabled values.

    Args:
        name: Environment variable name.
        fallback: Value to use when the variable is unset.

    Returns:
        bool: ``True`` only for known enabled strings. Any other set value is
            treated as disabled. When the variable is unset, ``fallback`` is
            returned.
    """
    value = os.getenv(name)
    if value is None:
        return fallback
    return value.strip().lower() in ENABLED_ENV_VALUES


def config_value(
    config: Optional[dict[str, Any]], key: str, default: Any = None
) -> Any:
    """Safely read a value from the loaded YAML configuration.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        key: Configuration key to read.
        default: Value returned when config or key is missing.

    Returns:
        Any: The configured value or ``default``.
    """
    if not config:
        return default
    return config.get(key, default)


def config_bool(
    config: Optional[dict[str, Any]],
    env_name: str,
    config_key: str,
    default: bool = False,
) -> bool:
    """Read a boolean setting where env vars take precedence over config.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        env_name: Environment variable name to check first.
        config_key: Configuration key used when the environment variable is
            unset.
        default: Fallback value when no setting is present.

    Returns:
        bool: The resolved boolean setting.
    """
    return env_bool(env_name, bool(config_value(config, config_key, default)))
