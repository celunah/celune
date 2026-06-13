"""Shared Celune type aliases."""

from collections.abc import Mapping
from typing import Literal, Optional, Union

JSONSerializable = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JSONSerializable"],
    dict[str, "JSONSerializable"],
]
JSON = dict[str, JSONSerializable]
RGB = tuple[int, int, int]
Config = dict[str, JSONSerializable]
TerminalConfig = Mapping[str, JSONSerializable]
ColorMode = Literal["auto", "truecolor", "terminal-default", "ansi", "none"]
ResolvedColorMode = Literal["truecolor", "terminal-default", "ansi", "none"]
VramTier = Literal["low", "medium", "high", "xhigh"]
VideoMetadataScalar = Optional[Union[bool, int, float, str]]
