from typing import Protocol

from ..typing.lua import LuaValue, _LuaGlobalValue


class _LuaTable(Protocol):
    """Minimal table interface needed by the embedded Lua host."""

    def __getitem__(self, key: str) -> "LuaValue":
        """Read one value from the Lua table."""
        raise NotImplementedError

    def __setitem__(self, key: str, value: "_LuaGlobalValue") -> None:
        """Write one value into the Lua table."""
        raise NotImplementedError


class _LuaRuntime(Protocol):
    """Minimal Lupa runtime interface used by Celune."""

    def execute(self, source: str, *, name: str) -> "LuaValue":
        """Execute one Lua source chunk.

        Args:
            source: Lua source code to execute.
            name: Source name used in Lua diagnostics.
        """
        raise NotImplementedError

    def globals(self) -> _LuaTable:
        """Return the runtime's global table."""
        raise NotImplementedError

    def table_from(
        self,
        values: dict[str, "LuaValue"],
        *,
        recursive: bool,
    ) -> _LuaTable:
        """Convert Python values into a Lua table.

        Args:
            values: Mapping of field names to values.
            recursive: Whether nested mappings should also become Lua tables.
        """
        raise NotImplementedError
