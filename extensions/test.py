import time
from celune import CeluneExtension

class TestExtension(CeluneExtension):
    EXTENSION_NAME = "Greeter"
    AUTOSTART = True

    def autostart(self) -> None:
        # time.sleep(5)  # HACK: race conditions occur if you reload the model too early
        self.set_voice("calm")
        self.say("Hey there. What do you want to make me say today?")

    def invoke(self) -> None:
        self.say("You want me to greet you again? Well, hello. Now, what are you going to make me say?")