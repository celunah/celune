# SPDX-License-Identifier: MIT
"""Configuration-specific type aliases."""

from typing import Union, Literal, Optional
from collections.abc import Mapping, Sequence

type AudioDeviceConfig = Optional[Union[int, str]]
type AudioDeviceDirection = Literal["input", "output"]
type AudioDeviceInfoValue = Union[bool, int, float, str]
type AudioDeviceInfo = Mapping[str, AudioDeviceInfoValue]
type AudioDeviceQueryResult = Union[AudioDeviceInfo, Sequence[AudioDeviceInfo]]
type AudioHostApi = Optional[Literal["wasapi", "directsound"]]
