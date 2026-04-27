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

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Emit a log message.

        Args:
            msg: Message text to emit.
            severity: Message severity level.

        Returns:
            None: Implementations forward the message to a logger.
        """
        ...


@runtime_checkable
class SayCallable(Protocol):
    """Extension callable speech request annotation."""

    def __call__(self, text: str, save: bool = True) -> bool:
        """Queue text for speech.

        Args:
            text: Text to synthesize.
            save: Whether to save generated output artifacts.

        Returns:
            bool: ``True`` when the request was accepted.
        """
        ...


@runtime_checkable
class PlayCallable(Protocol):
    """Extension callable play request annotation."""

    def __call__(self, sound_path: str) -> bool:
        """Queue an audio file for playback.

        Args:
            sound_path: Path to the sound file.

        Returns:
            bool: ``True`` when playback was queued.
        """
        ...


@runtime_checkable
class StatusCallable(Protocol):
    """Extension callable status update annotation."""

    def __call__(self, msg: str, severity: str = "info") -> None:
        """Emit a status update.

        Args:
            msg: Status message text.
            severity: Status severity level.

        Returns:
            None: Implementations forward the status update.
        """
        ...


@runtime_checkable
class SetVoiceCallable(Protocol):
    """Extension callable voice setting request annotation."""

    def __call__(self, name: str) -> bool:
        """Request a voice change.

        Args:
            name: Voice name to select.

        Returns:
            bool: ``True`` when the voice change was accepted.
        """
        ...


@runtime_checkable
class GetStateCallable(Protocol):
    """Extension callable state read annotation."""

    def __call__(self) -> str:
        """Read the current runtime state.

        Returns:
            str: Current state name.
        """
        ...


@runtime_checkable
class WaitUntilReadyCallable(Protocol):
    """Extension callable wait until ready annotation."""

    def __call__(self, timeout: float = 30.0) -> bool:
        """Wait for Celune to become ready.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            bool: ``True`` when Celune is ready.
        """
        ...


@dataclass(slots=True)
class CeluneContext:
    """Celune's extension context."""

    log: LogCallable
    say: SayCallable
    play: PlayCallable
    status: StatusCallable
    set_voice: SetVoiceCallable
    get_state: GetStateCallable
    wait_until_ready: WaitUntilReadyCallable

    name: str = "Celune"
    version: str = __version__
    shared: dict[str, Any] = field(default_factory=dict)
    dev: bool = False

    def expose(self, key: str, value: Any) -> None:
        """Expose a shared object.

        Args:
            key: The name used to store the shared value.
            value: The object to expose to other extensions.

        Returns:
            None: This method updates the shared extension context.
        """
        self.shared[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a shared object.

        Args:
            key: The name of the shared value to fetch.
            default: The fallback value returned when the key is missing.

        Returns:
            Any: The stored shared value, or ``default`` when absent.
        """
        return self.shared.get(key, default)


class CeluneExtension(ABC):
    """Celune extension abstract base class."""

    EXTENSION_NAME = "UnnamedExtension"
    AUTOSTART = False

    def __init__(self, context: CeluneContext) -> None:
        """Initialize an extension instance.

        Args:
            context: Shared Celune extension context.

        Returns:
            None: This constructor stores the context for later use.
        """
        self.ctx = context

    @property
    def name(self) -> str:
        """Return the extension's display name.

        Returns:
            str: The extension name exposed to Celune.
        """
        return self.EXTENSION_NAME

    @property
    def state(self) -> str:
        """Read Celune's current state.

        Returns:
            str: The current Celune runtime state string.
        """
        return self.ctx.get_state()

    def autostart(self) -> None:
        """Run extension startup logic.

        Returns:
            None: The default implementation only logs that autostart is skipped.
        """
        self.log(f"{self.name} has no autostart, skipping", "warning")

    def invoke(self, *args, **kwargs) -> None:
        """Run extension invocation logic.

        Args:
            *args: Positional arguments forwarded by the extension manager.
            **kwargs: Keyword arguments forwarded by the extension manager.

        Returns:
            None: Subclasses override this to perform extension work.
        """
        raise IncompleteExtensionError(
            f"{self.__class__.__name__}.invoke() is not implemented"
        )

    def log(self, msg: str, severity: str = "info") -> None:
        """Log to Celune's logs.

        Args:
            msg: The message to append to Celune's log output.
            severity: The message severity level.

        Returns:
            None: This method forwards the message to Celune's logger.
        """
        self.ctx.log(f"[{self.name}] {msg}", severity)

    def say(self, text: str, save: bool = True) -> bool:
        """Make Celune say something.

        Args:
            text: The text to queue for speech synthesis.
            save: Whether to save generated output artifacts.

        Returns:
            bool: ``True`` when the speech request was queued, otherwise ``False``.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.say(text, save=save)

    def play(self, sound_path: str) -> bool:
        """Play arbitrary sound through Celune.

        Args:
            sound_path: The path to the audio file to play.

        Returns:
            bool: ``True`` when playback was queued, otherwise ``False``.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.play(sound_path)

    def status(self, msg: str, severity: str = "info") -> None:
        """Update status display.

        Args:
            msg: The status message to show.
            severity: The status severity level.

        Returns:
            None: This method forwards the status update to Celune.
        """
        self.ctx.status(msg, severity)

    def set_voice(self, voice: str) -> bool:
        """Change Celune's voice.

        Args:
            voice: The voice name to request from Celune.

        Returns:
            bool: ``True`` when the voice change request was accepted.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.set_voice(voice)
