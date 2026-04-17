"""Edit this Celune extension to suit your needs."""

import time
from pathlib import Path

from celune import CeluneExtension


class TestExtension(CeluneExtension):
    """A sample Celune extension showcasing all features available in Celune's extension context."""

    EXTENSION_NAME = "Test"
    AUTOSTART = True  # if you do not want Celune to load this, set it to False

    def autostart(self) -> None:
        """Demonstrate extension behavior during autostart.

        Returns:
            None: This method performs example logging, speech, and playback work.
        """
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

    def invoke(self) -> None:
        """Demonstrate manual extension invocation behavior.

        Returns:
            None: This method logs and speaks a confirmation message.
        """
        self.log("You invoked the extension.")
        self.say("You invoked the extension.")
