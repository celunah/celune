import time
from celune.extensions.base import CeluneExtension


class TestExtension(CeluneExtension):
    """A sample Celune extension showcasing all the features available in Celune's extension context."""

    EXTENSION_NAME = "Test"
    AUTOSTART = False  # if you do not want Celune to load this, set it to False

    def autostart(self) -> None:
        self.log("Log test")
        time.sleep(1)
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
        self.say("Speaking with non-default voice.")

    def invoke(self) -> None:
        self.log("You invoked the extension.")
        self.say("You invoked the extension.")
