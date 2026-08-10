# SPDX-License-Identifier: Apache-2.0
"""Extension-facing dataclasses."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .. import __version__
from ..typing.common import JSONSerializable
from ..typing.aliases import LogLevel
from ..typing.extensions import (
    GetStateCallable,
    LogCallable,
    PlayCallable,
    SayCallable,
    SetVoiceCallable,
    StatusCallable,
    ThinkCallable,
    WaitUntilReadyCallable,
)

CELUNE_VERSION = __version__

if TYPE_CHECKING:
    from ..celune import Celune


@dataclass(slots=True)
class CeluneContext:
    """Celune's extension context."""

    log: LogCallable
    say: SayCallable
    think: ThinkCallable
    play: PlayCallable
    status: StatusCallable
    set_voice: SetVoiceCallable
    get_state: GetStateCallable
    wait_until_ready: WaitUntilReadyCallable
    backend_override: Optional[Callable[[str], AbstractContextManager["Celune"]]] = None
    cevoice_override: Optional[
        Callable[[Optional[Union[str, Path]]], AbstractContextManager["Celune"]]
    ] = None
    name: str = "Celune"
    version: str = CELUNE_VERSION
    shared: dict[str, JSONSerializable] = field(default_factory=dict)
    log_level: LogLevel = "info"

    def expose(self, key: str, value: JSONSerializable) -> None:
        """Expose a shared object.

        Args:
            key: Shared-object name to publish.
            value: JSON-serializable value exposed to extensions.
        """
        self.shared[key] = value

    def get(
        self,
        key: str,
        default: JSONSerializable = None,
    ) -> JSONSerializable:
        """Get a shared object.

        Args:
            key: Shared-object name to read.
            default: Fallback returned when the key is missing.

        Returns:
            JSONSerializable: The stored value, or ``default`` when absent.
        """
        return self.shared.get(key, default)

    def with_backend(
        self,
        backend_name: str,
    ) -> AbstractContextManager["Celune"]:
        """Temporarily switch Celune to another backend for one context block.

        Args:
            backend_name: The backend name to activate for the lifetime of the context.

        Returns:
            AbstractContextManager[Celune]: A context manager that restores the previous backend on exit.

        Raises:
            RuntimeError: Backend overrides are unavailable in this context.
        """
        if self.backend_override is None:
            raise RuntimeError("current context can't switch backends")
        return self.backend_override(backend_name)

    def with_cevoice(
        self,
        bundle: Optional[Union[str, Path]],
    ) -> AbstractContextManager["Celune"]:
        """Temporarily switch Celune to another CEVOICE bundle for one context block.

        Args:
            bundle: The CEVOICE bundle name or path to activate temporarily.

        Returns:
            AbstractContextManager[Celune]: A context manager that restores the previous CEVOICE pack on exit.

        Raises:
            RuntimeError: CEVOICE overrides are unavailable in this context.
        """
        if self.cevoice_override is None:
            raise RuntimeError("current context can't switch characters")
        return self.cevoice_override(bundle)
