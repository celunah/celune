from pathlib import Path
from typing import Union, Callable, TYPE_CHECKING

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
type _LuaGlobalValue = Union[LuaValue, "Celune", Callable[..., None]]
type _LuaScalar = Union[None, bool, int, float, str, Exception, Path]
