#!/usr/bin/env python3
# pylint: disable=R0902, R0913, R0917, W0718
"""
Celune 2.2.0 - ”It's not just TTS. It's a character.”
Refer to https://github.com/celunah/celune for information about Celune.
"""

import os
import sys
import contextlib

DEV = os.getenv("CELUNE_DEV") in {"1", "true", "on"}

try:
    from celune.celune import Celune
    from celune.ui import CeluneUI
except ModuleNotFoundError as package:
    print(f"Missing dependency: {package.name}")
    print("Celune requires this library to function.")
    print("Try running 'pip install -U -r requirements.txt'.")
    if DEV:
        raise
    print("Run Celune with CELUNE_DEV=1 to get full traceback.")
    sys.exit(1)


def main() -> None:
    """Instantiate and start Celune."""
    try:
        print("\x1b]2;Celune\x07", end="", flush=True)

        with contextlib.suppress(ModuleNotFoundError):
            # AKA
            # from setproctitle import setproctitle
            # setproctitle("Celune")
            __import__("setproctitle").setproctitle("Celune")

        ui = CeluneUI()
        celune = Celune(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            ref_audio="refs/balanced.wav",
            ref_text="My name is Celune, pronounced Celune. It is a pleasure to meet you.",
            log_callback=ui.tts_log,
            status_callback=ui.safe_status,
            error_callback=ui.error,
            idle_callback=ui.tts_idle,
            queue_avail_callback=ui.tts_queue_avail,
            voice_changed_callback=ui.tts_voice_changed,
            chunk_size=16,
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
        print("Run Celune with CELUNE_DEV=1 to get full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
