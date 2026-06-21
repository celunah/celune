# SPDX-License-Identifier: MIT
"""Celune's extension annotations and classes."""

from abc import ABC
from pathlib import Path
from typing import Optional, Union

from ..exceptions import IncompleteExtensionError
from ..dataclasses.extensions import CeluneContext


class CeluneExtension(ABC):
    """Celune extension abstract base class."""

    EXTENSION_NAME = "Unknown Extension"
    AUTOSTART = False

    def __init__(self, context: CeluneContext) -> None:
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
        """Run deprecated extension startup logic."""
        self.log(f"{self.name} has no autostart, skipping", "warning")

    def invoke(self, *args, **kwargs) -> None:
        """Run extension invocation logic.

        Args:
            args: Positional arguments forwarded by the extension manager.
            kwargs: Keyword arguments forwarded by the extension manager.

        Raises:
            IncompleteExtensionError: The extension does not override ``invoke``.
        """
        raise IncompleteExtensionError(
            f"{self.__class__.__name__}.invoke() is not implemented"
        )

    def log(self, msg: str, severity: str = "info") -> None:
        """Log to Celune's logs.

        Args:
            msg: The message to append to Celune's log output.
            severity: The message severity level.
        """
        self.ctx.log(f"[{self.name}] {msg}", severity)

    def say(
        self,
        text: str,
        save: bool = True,
        display_text: Optional[str] = None,
    ) -> bool:
        """Make Celune say something.

        Args:
            text: The text to queue for speech synthesis.
            save: Whether to save generated output artifacts.
            display_text: Optional text to show in logs instead of the synthesis text.

        Returns:
            bool: ``True`` when the speech request was queued, otherwise ``False``.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.say(text, save=save, display_text=display_text)

    def think(self, text: str) -> bool:
        """Make Celune process a Persona request, saying it later.

        Args:
            text: The text to have the Persona model process.

        Returns:
            bool: ``True`` if Persona returned a response, otherwise ``False``.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.think(text)

    def play(
        self,
        sound_path: str,
        keep: bool = False,
        volume: float = 1.0,
    ) -> bool:
        """Play arbitrary sound through Celune.

        Args:
            sound_path: The path to the audio file to play.
            keep: Whether to prepend this SFX to the next saved utterance.
            volume: How loud should the SFX be played at.

        Returns:
            bool: ``True`` when playback was queued, otherwise ``False``.
        """
        if not self.ctx.wait_until_ready():
            return False

        return self.ctx.play(sound_path, keep=keep, volume=volume)

    def status(self, msg: str, severity: str = "info") -> None:
        """Update status display.

        Args:
            msg: The status message to show.
            severity: The status severity level.
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

    def with_backend(self, backend_name: str):
        """Temporarily switch Celune to another backend inside a ``with`` block.

        Args:
            backend_name: The backend name to activate temporarily.

        Returns:
            contextlib.AbstractContextManager[None]: A context manager that restores the previous backend on exit.
        """
        return self.ctx.with_backend(backend_name)

    def with_cevoice(self, bundle: Optional[Union[str, Path]]):
        """Temporarily switch Celune to another CEVOICE bundle inside a ``with`` block.

        Args:
            bundle: The CEVOICE bundle name or path to activate temporarily.

        Returns:
            contextlib.AbstractContextManager[None]: A context manager that restores the previous CEVOICE pack on exit.
        """
        return self.ctx.with_cevoice(bundle)
