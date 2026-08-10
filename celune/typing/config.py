# SPDX-License-Identifier: Apache-2.0
"""Configuration-specific type aliases."""

from collections.abc import Mapping, Sequence
from typing import Literal, Optional, Union

type AudioDeviceConfig = Optional[Union[int, str]]
type AudioDeviceDirection = Literal["input", "output"]
type AudioDeviceInfoValue = Union[bool, int, float, str]
type AudioDeviceInfo = Mapping[str, AudioDeviceInfoValue]
type AudioDeviceQueryResult = Union[AudioDeviceInfo, Sequence[AudioDeviceInfo]]
type AudioHostApi = Optional[Literal["wasapi", "directsound"]]
