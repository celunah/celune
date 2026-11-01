# SPDX-License-Identifier: Apache-2.0
"""Detect the speaker positions exposed by Celune's configured output."""

import os
import re
import sys
import json
import uuid
import ctypes
import shutil
import subprocess
from collections.abc import Mapping
from collections.abc import Callable
from typing import NamedTuple, Optional, Union, cast

import sounddevice as sd

from ..i18n import string
from ..typing.common import JSONSerializable
from ..config import resolve_audio_device_with_info
from ..typing.config import AudioDeviceInfo, AudioDeviceInfoValue

_POSITION_KEYS = {
    "mono": "commands.audio_test_mono",
    "front_left": "commands.audio_test_left",
    "front_right": "commands.audio_test_right",
    "front_center": "commands.audio_test_center",
    "lfe": "commands.audio_test_subwoofer",
    "back_left": "commands.audio_test_back_left",
    "back_right": "commands.audio_test_back_right",
    "front_left_of_center": "commands.audio_test_front_left_of_center",
    "front_right_of_center": "commands.audio_test_front_right_of_center",
    "back_center": "commands.audio_test_back_center",
    "side_left": "commands.audio_test_side_left",
    "side_right": "commands.audio_test_side_right",
    "top_center": "commands.audio_test_top_center",
    "top_front_left": "commands.audio_test_top_front_left",
    "top_front_center": "commands.audio_test_top_front_center",
    "top_front_right": "commands.audio_test_top_front_right",
    "top_back_left": "commands.audio_test_top_back_left",
    "top_back_center": "commands.audio_test_top_back_center",
    "top_back_right": "commands.audio_test_top_back_right",
}

_WINDOWS_CHANNEL_MASKS = (
    (0x00001, "front_left"),
    (0x00002, "front_right"),
    (0x00004, "front_center"),
    (0x00008, "lfe"),
    (0x00010, "back_left"),
    (0x00020, "back_right"),
    (0x00040, "front_left_of_center"),
    (0x00080, "front_right_of_center"),
    (0x00100, "back_center"),
    (0x00200, "side_left"),
    (0x00400, "side_right"),
    (0x00800, "top_center"),
    (0x01000, "top_front_left"),
    (0x02000, "top_front_center"),
    (0x04000, "top_front_right"),
    (0x08000, "top_back_left"),
    (0x10000, "top_back_center"),
    (0x20000, "top_back_right"),
)

_PULSE_CHANNEL_NAMES = {
    "mono": "mono",
    "front-left": "front_left",
    "front-right": "front_right",
    "front-center": "front_center",
    "lfe": "lfe",
    "rear-left": "back_left",
    "rear-right": "back_right",
    "rear-center": "back_center",
    "side-left": "side_left",
    "side-right": "side_right",
    "top-center": "top_center",
    "top-front-left": "top_front_left",
    "top-front-center": "top_front_center",
    "top-front-right": "top_front_right",
    "top-rear-left": "top_back_left",
    "top-rear-center": "top_back_center",
    "top-rear-right": "top_back_right",
    "fl": "front_left",
    "fr": "front_right",
    "fc": "front_center",
    "rl": "back_left",
    "rr": "back_right",
    "rc": "back_center",
    "sl": "side_left",
    "sr": "side_right",
    "tfl": "top_front_left",
    "tfc": "top_front_center",
    "tfr": "top_front_right",
    "trl": "top_back_left",
    "trc": "top_back_center",
    "trr": "top_back_right",
}

_FALLBACK_LAYOUTS = {
    1: ("mono",),
    2: ("front_left", "front_right"),
    3: ("front_left", "front_right", "front_center"),
    4: ("front_left", "front_right", "back_left", "back_right"),
    5: (
        "front_left",
        "front_right",
        "front_center",
        "side_left",
        "side_right",
    ),
    6: (
        "front_left",
        "front_right",
        "front_center",
        "lfe",
        "side_left",
        "side_right",
    ),
    7: (
        "front_left",
        "front_right",
        "front_center",
        "lfe",
        "back_center",
        "side_left",
        "side_right",
    ),
    8: (
        "front_left",
        "front_right",
        "front_center",
        "lfe",
        "back_left",
        "back_right",
        "side_left",
        "side_right",
    ),
}


class SpeakerChannel(NamedTuple):
    """One detected output channel and its localized spoken name."""

    position: str
    name: str


def detect_speaker_channels(
    config: Mapping[str, JSONSerializable],
) -> tuple[SpeakerChannel, ...]:
    """Return localized labels for the configured output's available channels.

    Args:
        config: Celune's loaded audio and host API settings.

    Returns:
        tuple[SpeakerChannel, ...]: Channel positions and their spoken labels, ordered for playback.

    Raises:
        ValueError: A configured output device name matches multiple devices.
    """
    device_key = (
        "output_recording_device"
        if "output_recording_device" in config
        else "output_device"
    )
    selector, device_info = resolve_audio_device_with_info(config, device_key, "output")
    if device_info is None:
        queried_info = sd.query_devices(device=selector, kind="output")
        device_info = cast(AudioDeviceInfo, queried_info)

    channel_value = device_info.get("max_output_channels", 0)
    if not isinstance(channel_value, int) or channel_value <= 0:
        return ()
    channel_count = channel_value

    positions: Optional[tuple[str, ...]] = None
    if os.name == "nt" and _uses_default_output(selector, device_info):
        positions = _windows_default_positions()
    elif sys.platform.startswith("linux"):
        positions = _pulse_output_positions(
            str(device_info.get("name", "")),
            use_default=_uses_default_output(selector, device_info),
        )

    if positions is None or len(positions) > channel_count:
        positions = _FALLBACK_LAYOUTS.get(channel_count)
    elif positions:
        channel_count = len(positions)
    if positions is None:
        positions = ("front_left", "front_right") + tuple(
            f"channel_{index}" for index in range(3, channel_count + 1)
        )

    return tuple(
        SpeakerChannel(
            position,
            string(_POSITION_KEYS[position])
            if position in _POSITION_KEYS
            else string("commands.audio_test_channel", number=index),
        )
        for index, position in enumerate(positions[:channel_count], start=1)
    )


def _uses_default_output(
    selector: Optional[Union[int, str]],
    device_info: Mapping[str, AudioDeviceInfoValue],
) -> bool:
    """Return whether PortAudio resolved the configured output to its default."""
    if selector is None:
        return True
    try:
        default_info = sd.query_devices(kind="output")
    except (sd.PortAudioError, ValueError):
        return False
    selected_index = device_info.get("index")
    return isinstance(selected_index, int) and selected_index == default_info.get(
        "index"
    )


def _pulse_output_positions(
    device_name: str,
    *,
    use_default: bool,
) -> Optional[tuple[str, ...]]:
    """Read the selected PulseAudio or PipeWire sink's declared channel map."""
    pactl = shutil.which("pactl")
    if pactl is None:
        return None
    try:
        result = subprocess.run(
            (pactl, "--format=json", "list", "sinks"),
            capture_output=True,
            check=False,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None

    try:
        sinks = cast(JSONSerializable, json.loads(result.stdout))
    except json.JSONDecodeError:
        return None
    if not isinstance(sinks, list):
        return None

    default_name = _pulse_default_sink(pactl) if use_default else None
    normalized_device = _normalize_device_name(device_name)
    for sink_value in sinks:
        if not isinstance(sink_value, Mapping):
            continue
        sink = cast(Mapping[str, JSONSerializable], sink_value)
        name = str(sink.get("name", ""))
        description = str(sink.get("description", ""))
        if default_name is not None:
            matches = name == default_name
        else:
            normalized_names = (
                _normalize_device_name(name),
                _normalize_device_name(description),
            )
            matches = any(
                normalized_device
                and (
                    normalized_device == candidate
                    or normalized_device in candidate
                    or candidate in normalized_device
                )
                for candidate in normalized_names
            )
        if not matches:
            continue

        channel_map = sink.get("channel_map")
        if isinstance(channel_map, str):
            channel_names = tuple(
                item.strip() for item in channel_map.split(",") if item.strip()
            )
        elif isinstance(channel_map, list):
            channel_names = tuple(item for item in channel_map if isinstance(item, str))
        else:
            return None

        positions = tuple(
            _PULSE_CHANNEL_NAMES.get(name.casefold(), f"channel_{index}")
            for index, name in enumerate(channel_names, start=1)
        )
        return positions or None
    return None


def _pulse_default_sink(pactl: str) -> Optional[str]:
    """Return the audio server's default sink name when it can be queried."""
    try:
        result = subprocess.run(
            (pactl, "get-default-sink"),
            capture_output=True,
            check=False,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _normalize_device_name(value: str) -> str:
    """Remove host API decorations before comparing output device names."""
    normalized = re.sub(
        r"\s+\((?:MME|WASAPI|Windows DirectSound|PulseAudio|ALSA)\)$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[^a-z0-9]+", "", normalized.casefold())


def _windows_channel_positions(
    channel_mask: int,
    channel_count: int,
) -> Optional[tuple[str, ...]]:
    """Translate a WAVEFORMATEXTENSIBLE speaker mask into channel order."""
    positions = tuple(
        position for mask, position in _WINDOWS_CHANNEL_MASKS if channel_mask & mask
    )
    return positions if len(positions) == channel_count else None


def _windows_default_positions() -> Optional[tuple[str, ...]]:
    """Read Windows' active default render endpoint channel mask."""
    if not hasattr(ctypes, "WinDLL"):
        return None

    try:
        return _read_windows_default_positions()
    except (OSError, ValueError, ctypes.ArgumentError):
        return None


def _read_windows_default_positions() -> Optional[tuple[str, ...]]:
    """Read the default render endpoint's WAVEFORMATEXTENSIBLE mask."""
    from ctypes import wintypes

    class Guid(ctypes.Structure):
        """Windows GUID layout used by Core Audio COM interfaces."""

        _fields_ = [
            ("data1", wintypes.DWORD),
            ("data2", wintypes.WORD),
            ("data3", wintypes.WORD),
            ("data4", ctypes.c_ubyte * 8),
        ]

    class WaveFormatEx(ctypes.Structure):
        """Windows WAVEFORMATEX structure with SDK-compatible packing."""

        _pack_ = 2
        _fields_ = [
            ("format_tag", wintypes.WORD),
            ("channels", wintypes.WORD),
            ("samples_per_second", wintypes.DWORD),
            ("average_bytes_per_second", wintypes.DWORD),
            ("block_align", wintypes.WORD),
            ("bits_per_sample", wintypes.WORD),
            ("extra_size", wintypes.WORD),
        ]

    class WaveFormatExtensible(ctypes.Structure):
        """WAVEFORMATEXTENSIBLE fields that follow WAVEFORMATEX."""

        _pack_ = 2
        _fields_ = [
            ("format", WaveFormatEx),
            ("valid_bits_per_sample", wintypes.WORD),
            ("channel_mask", wintypes.DWORD),
            ("subformat", Guid),
        ]

    def make_guid(value: str) -> Guid:
        """Create the in-memory GUID value expected by COM."""
        raw = uuid.UUID(value).bytes_le
        return Guid(
            int.from_bytes(raw[0:4], "little"),
            int.from_bytes(raw[4:6], "little"),
            int.from_bytes(raw[6:8], "little"),
            (ctypes.c_ubyte * 8).from_buffer_copy(raw[8:16]),
        )

    def method(
        instance: ctypes.c_void_p,
        slot: int,
        result: type,
        *args: type,
    ) -> Callable[..., int]:
        """Resolve one COM vtable method using its explicit calling convention."""
        vtable = ctypes.cast(
            instance,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ).contents
        function_name = "WINFUNCTYPE"
        function_type = cast(
            Callable[..., Callable[..., int]],
            getattr(ctypes, function_name),
        )
        callback_type = function_type(result, ctypes.c_void_p, *args)
        return cast(Callable[..., int], callback_type(vtable[slot]))

    def release(instance: Optional[ctypes.c_void_p]) -> None:
        """Release one COM interface pointer."""
        if instance:
            method(instance, 2, wintypes.ULONG)(instance)

    loader_name = "WinDLL"
    dll_loader = cast(Callable[..., ctypes.CDLL], getattr(ctypes, loader_name))
    ole32 = dll_loader("ole32", use_last_error=True)
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = ctypes.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(Guid),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(Guid),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = ctypes.c_long

    initialized = ole32.CoInitializeEx(None, 0)
    if initialized < 0:
        return None

    enumerator = ctypes.c_void_p()
    device = ctypes.c_void_p()
    audio_client = ctypes.c_void_p()
    wave_format = ctypes.c_void_p()
    try:
        class_id = make_guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
        interface_id = make_guid("A95664D2-9614-4F35-A746-DE8DB63617E6")
        audio_client_id = make_guid("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
        if (
            ole32.CoCreateInstance(
                ctypes.byref(class_id),
                None,
                0x17,
                ctypes.byref(interface_id),
                ctypes.byref(enumerator),
            )
            < 0
        ):
            return None

        get_default_endpoint = method(
            enumerator,
            4,
            ctypes.c_long,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p),
        )
        if get_default_endpoint(enumerator, 0, 0, ctypes.byref(device)) < 0:
            return None

        activate = method(
            device,
            3,
            ctypes.c_long,
            ctypes.POINTER(Guid),
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        if (
            activate(
                device,
                ctypes.byref(audio_client_id),
                0x17,
                None,
                ctypes.byref(audio_client),
            )
            < 0
        ):
            return None

        get_mix_format = method(
            audio_client,
            8,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_void_p),
        )
        if get_mix_format(audio_client, ctypes.byref(wave_format)) < 0:
            return None

        base_format = ctypes.cast(
            wave_format,
            ctypes.POINTER(WaveFormatEx),
        ).contents
        if base_format.format_tag != 0xFFFE or base_format.extra_size < 22:
            return None
        extensible_format = ctypes.cast(
            wave_format,
            ctypes.POINTER(WaveFormatExtensible),
        ).contents
        return _windows_channel_positions(
            extensible_format.channel_mask,
            base_format.channels,
        )
    finally:
        if wave_format:
            ole32.CoTaskMemFree(wave_format)
        release(audio_client)
        release(device)
        release(enumerator)
        ole32.CoUninitialize()
