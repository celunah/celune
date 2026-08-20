# SPDX-License-Identifier: MIT
"""Configuration helpers for Celune."""

import os
from copy import deepcopy
from typing import Optional, cast
from collections.abc import Mapping, Sequence

import sounddevice as sd

from .i18n import string
from .constants import APP_NAME
from .typing.aliases import LogLevel
from .typing.common import Config, JSONSerializable
from .typing.config import (
    AudioHostApi,
    AudioDeviceConfig,
    AudioDeviceDirection,
    AudioDeviceInfoValue,
    AudioDeviceQueryResult,
)

ENABLED_ENV_VALUES = {"1", "true", "on", "yes", "enabled"}
LOG_LEVELS: tuple[LogLevel, ...] = ("info", "verbose", "debug")


def normalize_log_level(value: object, default: LogLevel = "info") -> LogLevel:
    """Return one supported log level, falling back when the value is invalid."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in LOG_LEVELS:
            return normalized  # type: ignore[return-value]
    return default


WINDOWS_AUDIO_HOSTAPIS: dict[str, str] = {
    "wasapi": "Windows WASAPI",
    "directsound": "Windows DirectSound",
}


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


def config_log_level(
    config: Optional[Mapping[str, JSONSerializable]],
    env_name: str = "CELUNE_LOG_LEVEL",
    config_key: str = "log_level",
    default: LogLevel = "info",
) -> LogLevel:
    """Resolve one configured Celune log level."""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        configured = config_value(config, config_key, default)
        raw_value = configured if isinstance(configured, str) else default

    return normalize_log_level(raw_value, default)


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
        if not stripped:
            return None
        if os.name != "nt":
            return stripped

        configured_hostapi = config_audio_api(config)
        if configured_hostapi is None:
            return stripped

        normalized_name, parsed_hostapi = _parse_audio_device_selector(
            stripped,
            configured_hostapi,
        )

        hostapi_key = (
            parsed_hostapi if parsed_hostapi is not None else configured_hostapi
        ) or "wasapi"

        return f"{normalized_name}, {WINDOWS_AUDIO_HOSTAPIS[hostapi_key]}"
    return None


def config_audio_api(
    config: Optional[Mapping[str, JSONSerializable]],
) -> AudioHostApi:
    """Resolve the optional Windows audio host API selector from config.

    Args:
        config: Loaded configuration dictionary, or ``None``.

    Returns:
        AudioHostApi: One supported Windows host API key, or ``None`` when unset or invalid.
    """
    value = config_value(config, "audio_api")
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    if normalized in WINDOWS_AUDIO_HOSTAPIS:
        return cast(AudioHostApi, normalized)
    return None


def format_audio_device_name(
    info: Mapping[str, AudioDeviceInfoValue],
    hostapis: Optional[list[Mapping[str, AudioDeviceInfoValue]]] = None,
) -> str:
    """Format one runtime-facing audio device label.

    Args:
        info: Sounddevice information for the resolved device.
        hostapis: Optional host API list used to append the host API label on Windows.

    Returns:
        str: The formatted device label.
    """
    name = str(info.get("name", ""))
    hostapi_name = _hostapi_name(info, hostapis)
    if os.name == "nt" and hostapi_name:
        return f"{name}, {hostapi_name}"
    return name


def _hostapi_name(
    info: Mapping[str, AudioDeviceInfoValue],
    hostapis: Optional[list[Mapping[str, AudioDeviceInfoValue]]] = None,
) -> Optional[str]:
    """Resolve one host API display name from sounddevice info."""
    if hostapis is None:
        # noinspection PyBroadException
        try:
            queried = sd.query_hostapis()
        except Exception:
            return None
        if not isinstance(queried, Sequence) or isinstance(queried, (str, bytes)):
            return None
        hostapis = list(queried)

    hostapi_index = info.get("hostapi")
    if not isinstance(hostapi_index, int):
        return None
    if hostapi_index < 0 or hostapi_index >= len(hostapis):
        return None
    return str(hostapis[hostapi_index].get("name", ""))


def _parse_audio_device_selector(
    configured: str,
    configured_hostapi: AudioHostApi,
) -> tuple[str, AudioHostApi]:
    """Split a configured device selector into name and optional host API."""
    if "," not in configured:
        return configured, configured_hostapi

    device_name, suffix = configured.rsplit(",", 1)
    normalized_suffix = suffix.strip().casefold()
    for hostapi_key, hostapi_name in WINDOWS_AUDIO_HOSTAPIS.items():
        if normalized_suffix in {hostapi_key, hostapi_name.casefold()}:
            return device_name.strip(), cast(AudioHostApi, hostapi_key)
    return configured, configured_hostapi


def _find_matching_device_index(
    all_devices: AudioDeviceQueryResult,
    configured_name: str,
    channel_key: str,
    configured_hostapi: AudioHostApi,
    hostapis: Optional[list[Mapping[str, AudioDeviceInfoValue]]] = None,
) -> Optional[int]:
    """Return one exact device index when the configured selector identifies a single device."""
    if isinstance(all_devices, Mapping):
        if (
            configured_name.casefold()
            not in str(all_devices.get("name", "")).casefold()
        ):
            return None
        if int(all_devices.get(channel_key, 0)) <= 0:
            return None
        hostapi_name = _hostapi_name(all_devices, hostapis)
        if (
            configured_hostapi is not None
            and hostapi_name != WINDOWS_AUDIO_HOSTAPIS[configured_hostapi]
        ):
            return None
        return 0

    if not isinstance(all_devices, Sequence) or isinstance(all_devices, (str, bytes)):
        return None

    matches: list[int] = []
    for index, info in enumerate(all_devices):
        if configured_name.casefold() not in str(info.get("name", "")).casefold():
            continue
        if int(info.get(channel_key, 0)) <= 0:
            continue
        hostapi_name = _hostapi_name(info, hostapis)
        if (
            configured_hostapi is not None
            and hostapi_name != WINDOWS_AUDIO_HOSTAPIS[configured_hostapi]
        ):
            continue
        matches.append(index)

    if len(matches) == 1:
        return matches[0]
    return None


def resolve_audio_device(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
    direction: AudioDeviceDirection,
) -> AudioDeviceConfig:
    """Resolve one configured audio device into an exact PortAudio selector.

    Args:
        config: Configuration mapping that may declare one device selector.
        key: Config key to resolve from the mapping.
        direction: Whether the selector is for input or output audio.

    Returns:
        AudioDeviceConfig: Exact selector data suitable for PortAudio lookup.
    """
    resolved, _device_info = resolve_audio_device_with_info(config, key, direction)
    return resolved


def resolve_audio_device_with_info(
    config: Optional[Mapping[str, JSONSerializable]],
    key: str,
    direction: AudioDeviceDirection,
) -> tuple[AudioDeviceConfig, Optional[Mapping[str, AudioDeviceInfoValue]]]:
    """Resolve one configured audio device into an exact PortAudio selector.

    Args:
        config: Loaded configuration dictionary, or ``None``.
        key: Configuration key containing the device name or index.
        direction: Whether the device must support ``"input"`` or ``"output"``.

    Returns:
        tuple[AudioDeviceConfig, Optional[Mapping[str, AudioDeviceInfoValue]]]: The resolved selector and optional
        direct device info.

    Raises:
        ValueError: The configured device name matches multiple devices.
    """
    configured = config_audio_device(config, key)
    if configured is None or isinstance(configured, int):
        return configured, None

    configured_hostapi = config_audio_api(config) if os.name == "nt" else None
    configured_name, configured_hostapi = _parse_audio_device_selector(
        configured,
        configured_hostapi,
    )

    channel_key = (
        "max_input_channels" if direction == "input" else "max_output_channels"
    )
    query_kind = "input" if direction == "input" else "output"
    try:
        direct_info = sd.query_devices(device=configured_name, kind=query_kind)
    except ValueError:
        # noinspection PyUnusedLocal
        direct_info = None
    else:
        if (
            isinstance(direct_info, Mapping)
            and int(direct_info.get(channel_key, 0)) > 0
            and (
                configured_hostapi is None
                or _hostapi_name(direct_info)
                == WINDOWS_AUDIO_HOSTAPIS[configured_hostapi]
            )
        ):
            all_devices = sd.query_devices()
            resolved_index = _find_matching_device_index(
                all_devices,
                configured_name,
                channel_key,
                configured_hostapi,
                hostapis=(
                    sd.query_hostapis() if configured_hostapi is not None else None
                ),
            )
            if resolved_index is not None:
                return resolved_index, direct_info

            # PortAudio already resolved this selector successfully, so reuse the
            # returned device info and avoid a second global device scan.
            return configured_name, direct_info

    hostapis = sd.query_hostapis()
    matches: list[tuple[int, str]] = []
    all_devices = sd.query_devices()

    if isinstance(all_devices, Mapping):
        name = str(all_devices.get("name", ""))
        if (
            configured_name.casefold() in name.casefold()
            and int(all_devices.get(channel_key, 0)) > 0
        ):
            hostapi_name = _hostapi_name(
                all_devices,
                list(hostapis)
                if isinstance(hostapis, Sequence)
                and not isinstance(hostapis, (str, bytes))
                else None,
            )
            if (
                configured_hostapi is None
                or hostapi_name == WINDOWS_AUDIO_HOSTAPIS[configured_hostapi]
            ):
                return configured_name, all_devices
        return configured_name, None

    for index, info in enumerate(all_devices):
        if configured_name.casefold() not in str(info["name"]).casefold():
            continue
        if int(info.get(channel_key, 0)) <= 0:
            continue
        hostapi_name = _hostapi_name(
            info,
            list(hostapis)
            if isinstance(hostapis, Sequence) and not isinstance(hostapis, (str, bytes))
            else None,
        )
        if hostapi_name is None:
            continue
        if (
            configured_hostapi is not None
            and hostapi_name != WINDOWS_AUDIO_HOSTAPIS[configured_hostapi]
        ):
            continue
        matches.append((index, f"[{index}] {info['name']}, {hostapi_name}"))

    if len(matches) == 1:
        return matches[0][0], None

    if len(matches) > 1:
        matches_text = "\n".join(f"- {label}" for _, label in matches)
        raise ValueError(
            string(
                "config.audio_device_multiple_matches",
                device_kind=string(f"config.audio_device_kind_{query_kind}"),
                device_name=configured_name,
                matches=matches_text,
                app_name=APP_NAME,
            )
        )

    return configured_name, None


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
