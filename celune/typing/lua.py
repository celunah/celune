"""Type aliases exchanged with the embedded Lua runtime."""

from typing import TYPE_CHECKING, Union, Protocol
from pathlib import Path
from collections.abc import Callable

if TYPE_CHECKING:
    # unused my ass
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


class LuaTable(Protocol):
    """Minimal table interface needed by the embedded Lua host."""

    def __getitem__(self, key: str) -> LuaValue:
        """Read one value from the Lua table."""
        raise NotImplementedError

    def __setitem__(self, key: str, value: LuaGlobalValue) -> None:
        """Write one value into the Lua table."""
        raise NotImplementedError


class LuaRuntimeProtocol(Protocol):
    """Minimal Lupa runtime interface used by Celune."""

    def execute(self, source: str, *, name: str) -> LuaValue:
        """Execute one Lua source chunk.

        Args:
            source: Lua source code to execute.
            name: Source name used in Lua diagnostics.
        """
        raise NotImplementedError

    def globals(self) -> LuaTable:
        """Return the runtime's global table."""
        raise NotImplementedError

    def table_from(
        self,
        values: dict[str, LuaValue],
        *,
        recursive: bool,
    ) -> LuaTable:
        """Convert Python values into a Lua table.

        Args:
            values: Mapping of field names to Lua-compatible values.
            recursive: Whether nested mappings should also become Lua tables.
        """
        raise NotImplementedError
