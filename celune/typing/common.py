# SPDX-License-Identifier: Apache-2.0
"""Shared Celune type aliases."""

from collections.abc import Mapping
from typing import Union, Literal, Optional

type JSONSerializable = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONSerializable"],
    dict[str, "JSONSerializable"],
]
type JSON = dict[str, JSONSerializable]
type RGB = tuple[int, int, int]
type Config = dict[str, JSONSerializable]
type TerminalConfig = Mapping[str, JSONSerializable]
type VramTier = Literal["low", "medium", "high", "xhigh"]
type VideoMetadataScalar = Optional[Union[bool, int, float, str]]
