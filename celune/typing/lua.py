"""Type aliases exchanged with the embedded Lua runtime."""

from pathlib import Path
from collections.abc import Callable
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..celune import Celune

type LuaValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["LuaValue"],
    dict[str, "LuaValue"],
]
type LuaGlobalValue = Union[LuaValue, "Celune", Callable[..., None]]
type LuaScalar = Union[None, bool, int, float, str, Exception, Path]
