"""Extension-facing protocols."""

from typing import Optional, Protocol, runtime_checkable

from ..exceptions import IncompleteExtensionError


@runtime_checkable
class LogCallable(Protocol):
    """Extension callable logging annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Emit a log message."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class DevLogCallable(Protocol):
    """Extension callable developer logging annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Emit a developer log message."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class SayCallable(Protocol):
    """Extension callable speech request annotation."""

    def __call__(
        self,
        text: str,
        save: bool = True,
        display_text: Optional[str] = None,
    ) -> bool:
        """Queue text for speech."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class ThinkCallable(Protocol):
    """Extension callable think request annotation."""

    def __call__(self, text: str) -> bool:
        """Start a think request."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class PlayCallable(Protocol):
    """Extension callable play request annotation."""

    def __call__(
        self,
        sound_path: str,
        keep: bool = False,
        volume: float = 1.0,
    ) -> bool:
        """Queue an audio file for playback."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class StatusCallable(Protocol):
    """Extension callable status update annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Emit a status update."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class SetVoiceCallable(Protocol):
    """Extension callable voice setting request annotation."""

    def __call__(self, name: str) -> bool:
        """Request a voice change."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class GetStateCallable(Protocol):
    """Extension callable state read annotation."""

    def __call__(self) -> str:
        """Read the current runtime state."""
        raise IncompleteExtensionError("protocol not defined")


@runtime_checkable
class WaitUntilReadyCallable(Protocol):
    """Extension callable wait-until-ready annotation."""

    def __call__(self, timeout: float = 30.0) -> bool:
        """Wait for Celune to become ready."""
        raise IncompleteExtensionError("protocol not defined")
