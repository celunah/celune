#!/usr/bin/env python3
# pylint: disable=R0902, R0913, R0917, W0718
"""
Celune 2.1.1 - A celestial TTS engine.
She has three tones and can change them on the fly.
She can also run extensions.
"""

import os
import sys

DEV = os.getenv("DEV") in {"1", "true", "True"}

try:
    from celune.celune import Celune
    from celune.ui import CeluneUI
except ModuleNotFoundError as package:
    print(f"Missing dependency: {package.name}")
    print("Celune requires this library to function.")
    print("Try running 'pip install -U -r requirements.txt'.")
    if DEV:
        raise
    sys.exit(1)


def main() -> None:
    """Instantiate and start Celune."""
    sys.stdout.write("\x1b]2;Celune\x07")
    sys.stdout.flush()

    try:
        ui = CeluneUI()
        celune = Celune(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            ref_audio="refs/neutral.wav",
            ref_text="My name is Celune, pronounced Celune. It is a pleasure to meet you.",
            log_callback=ui.tts_log,
            status_callback=ui.safe_status,
            error_callback=ui.error,
            idle_callback=ui.tts_idle,
            queue_avail_callback=ui.tts_queue_avail,
            voice_changed_callback=ui.tts_voice_changed,
            dev=DEV,
        )

        celune.setup_extensions()

        ui.celune = celune
        ui.run()
    except Exception as e:
        print("Celune early initialization failed.")
        if DEV:
            raise
        print(e)
        sys.exit(1)
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?2004l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
