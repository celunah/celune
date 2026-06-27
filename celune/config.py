# SPDX-License-Identifier: MIT
"""Configuration helpers for Celune."""

import os
from copy import deepcopy
from typing import Optional, Union, Literal
from collections.abc import Mapping

import sounddevice as sd

from .constants import APP_NAME
from .i18n import string
from .typing.common import Config, JSONSerializable

ENABLED_ENV_VALUES = {"1", "true", "on", "yes", "enabled"}
AudioDeviceConfig = Optional[Union[int, str]]
AudioDeviceDirection = Literal["input", "output"]


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


def config_audio_device(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
) -> AudioDeviceConfig:
    """Resolve one nullable audio-device selection from configuration.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        key: Configuration key containing the device name or index.

    Returns:
        AudioDeviceConfig: ``None`` for the system default device, or the configured device name/index.
    """
    value = config_value(config, key)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def resolve_audio_device(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
    direction: AudioDeviceDirection,
) -> AudioDeviceConfig:
    """Resolve one configured audio device into an exact PortAudio selector."""
    resolved, _device_info = resolve_audio_device_with_info(config, key, direction)
    return resolved


def resolve_audio_device_with_info(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
    direction: AudioDeviceDirection,
) -> tuple[AudioDeviceConfig, Optional[Mapping[str, object]]]:
    """Resolve one configured audio device into an exact PortAudio selector.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        key: Configuration key containing the device name or index.
        direction: Whether the device must support ``"input"`` or ``"output"``.

    Returns:
        tuple[AudioDeviceConfig, Optional[Mapping[str, object]]]: The resolved selector and optional direct device info.

    Raises:
        ValueError: The configured device name matches multiple devices.
    """
    configured = config_audio_device(config, key)
    if configured is None or isinstance(configured, int):
        return configured, None

    channel_key = (
        "max_input_channels" if direction == "input" else "max_output_channels"
    )
    query_kind = "input" if direction == "input" else "output"
    try:
        direct_info = sd.query_devices(device=configured, kind=query_kind)
    except ValueError:
        direct_info = None
    else:
        if (
            isinstance(direct_info, Mapping)
            and int(direct_info.get(channel_key, 0)) > 0
        ):
            # PortAudio already resolved this selector successfully, so reuse the
            # returned device info and avoid a second global device scan.
            return configured, direct_info

    hostapis = sd.query_hostapis()
    matches: list[tuple[int, str]] = []
    all_devices = sd.query_devices()

    if isinstance(all_devices, Mapping):
        name = str(all_devices.get("name", ""))
        if str(configured).casefold() in name.casefold():
            if int(all_devices.get(channel_key, 0)) > 0:
                return configured, all_devices
        return configured, None

    for index, info in enumerate(all_devices):
        if str(configured).casefold() not in str(info["name"]).casefold():
            continue
        if int(info.get(channel_key, 0)) <= 0:
            continue
        hostapi_name = str(hostapis[int(info["hostapi"])]["name"])
        matches.append((index, f"[{index}] {info['name']}, {hostapi_name}"))

    if len(matches) == 1:
        return matches[0][0], None

    if len(matches) > 1:
        matches_text = "\n".join(f"- {label}" for _, label in matches)
        raise ValueError(
            string(
                "config.audio_device_multiple_matches",
                device_kind=string(f"config.audio_device_kind_{query_kind}"),
                device_name=configured,
                matches=matches_text,
                app_name=APP_NAME,
            )
        )

    return configured, None


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
