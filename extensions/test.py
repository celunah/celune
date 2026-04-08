"""Edit this Celune extension to suit your needs."""

import time
from celune import CeluneExtension


class TestExtension(CeluneExtension):
    """A sample Celune extension showcasing all features available in Celune's extension context."""

    EXTENSION_NAME = "Test"
    AUTOSTART = True  # if you do not want Celune to load this, set it to False

    def autostart(self) -> None:
        """Demonstration on autostart."""
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
        self.play("extensions/NOT_TTS.wav")  # Celune can also play sound effects, regardless of sample rate

    def invoke(self) -> None:
        """Feedback on invoke."""
        self.log("You invoked the extension.")
        self.say("You invoked the extension.")
