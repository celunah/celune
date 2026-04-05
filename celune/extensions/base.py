# pylint: disable=R0903, R0902
"""Celune's extension annotations and classes."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from celune import __version__
from celune.exceptions import IncompleteExtensionError


@runtime_checkable
class LogCallable(Protocol):
    """Extension callable logging annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None: ...


@runtime_checkable
class SayCallable(Protocol):
    """Extension callable speech request annotation."""

    def __call__(self, text: str) -> bool: ...


@runtime_checkable
class StatusCallable(Protocol):
    """Extension callable status update annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None: ...


@runtime_checkable
class SetVoiceCallable(Protocol):
    """Extension callable voice setting request annotation."""

    def __call__(self, name: str) -> bool: ...


@runtime_checkable
class GetStateCallable(Protocol):
    """Extension callable state read annotation."""

    def __call__(self) -> str: ...


@runtime_checkable
class WaitUntilReadyCallable(Protocol):
    """Extension callable wait until ready annotation."""

    def __call__(self) -> None: ...


@dataclass(slots=True)
class CeluneContext:
    """Celune's extension context."""

    log: LogCallable
    say: SayCallable
    status: StatusCallable
    set_voice: SetVoiceCallable
    get_state: GetStateCallable
    wait_until_ready: WaitUntilReadyCallable

    name: str = "Celune"
    version: str = __version__
    shared: dict[str, Any] = field(default_factory=dict)

    def expose(self, key: str, value: Any) -> None:
        """Expose a shared object."""
        self.shared[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a shared object."""
        return self.shared.get(key, default)


class CeluneExtension(ABC):
    """Celune extension abstract base class."""

    EXTENSION_NAME = "UnnamedExtension"
    AUTOSTART = False

    def __init__(self, context: CeluneContext) -> None:
        self.ctx = context

    @property
    def name(self) -> str:
        """Current Celune extension name."""
        return self.EXTENSION_NAME

    @property
    def state(self) -> str:
        """Read Celune's current state."""
        return self.ctx.get_state()

    def autostart(self) -> None:
        """Overridable autostart logic function."""
        self.log(f"{self.name} has no autostart, skipping", "warning")

    def invoke(self, *args, **kwargs) -> None:
        """Overridable invocation logic function."""
        raise IncompleteExtensionError(
            f"{self.__class__.__name__}.invoke() is not implemented"
        )

    def log(self, msg: str, severity: str = "info") -> None:
        """Log to Celune's logs."""
        self.ctx.log(f"[{self.name}] {msg}", severity)

    def say(self, text: str) -> None:
        """Make Celune say something."""
        self.ctx.wait_until_ready()
        self.ctx.say(text)

    def status(self, msg: str, severity: str = "info") -> None:
        """Update status display."""
        self.ctx.status(msg, severity)

    def set_voice(self, voice: str) -> None:
        """Change Celune's voice."""
        self.ctx.set_voice(voice)
