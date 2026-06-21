# SPDX-License-Identifier: MIT
"""Edit this Celune extension to suit your needs."""

import time
from pathlib import Path

import celune
from celune import CeluneExtension
from celune.utils import discard


class TestExtension(CeluneExtension):
    """A sample Celune extension showcasing all features available in Celune's extension context."""

    EXTENSION_NAME = "Test"

    @celune.subscribe("ready", enabled=False)
    def on_ready(self, event) -> None:
        """Demonstrate extension behavior when Celune becomes ready.

        Args:
            event: Ready event emitted after Celune finishes startup.
        """
        discard(event)

        self.log("Log test")
        time.sleep(1)  # due to threading, this does not block
        self.status("Status test")
        time.sleep(5)
        self.status("Status test (warning)", "warning")
        time.sleep(5)
        self.status("Status test (error)", "error")
        time.sleep(5)
        self.status("Status test (unknown)", "invalid")
        time.sleep(5)
        self.say("Speaking with default voice.")
        time.sleep(1)
        self.set_voice("calm")
        self.say(
            "Speaking with non-default voice."
        )  # this will wait for Calm to load before speaking
        time.sleep(1)
        sfx_path = Path(__file__).resolve().with_name("NOT_TTS.wav")
        self.play(
            str(sfx_path)
        )  # Celune can also play sound effects, regardless of sample rate
        time.sleep(1)
        # Celune can ignore saving artifacts from self.say()
        self.say("You will only hear this once.", save=False)

    @celune.subscribe("voice_changed")
    def on_voice_changed(self, event) -> None:
        """Demonstrate access to typed voice-change event payloads.

        Args:
            event: Voice-change event carrying the previous and new voice names.
        """
        self.log(f"Voice changed from {event.old_voice} to {event.new_voice}.")

    def invoke(self) -> None:
        """Demonstrate manual extension invocation behavior."""
        self.log("You invoked the extension.")
        self.say("You invoked the extension.")
