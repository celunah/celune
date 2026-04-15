#!/usr/bin/env python3
# pylint: disable=R0902, R0913, R0917, W0718
"""
Celune 3.1.2 - "I'm not just a TTS. I'm someone special."
Refer to https://github.com/celunah/celune for information about Celune.
Celune models are available on https://huggingface.co/collections/lunahr/celune.
"""

import os
import sys
import time
import datetime
import contextlib

DEV = os.getenv("CELUNE_DEV") in {"1", "true", "on"}

try:
    import psutil
    from celune.celune import Celune
    from celune.ui import CeluneUI
    from celune.exceptions import No
    from celune import namedays
except ModuleNotFoundError as package:
    print(f"Missing dependency: {package.name}")
    print("Celune requires this library to function.")
    print("Try running 'uv sync'.")
    if DEV:
        raise
    print("Run Celune with CELUNE_DEV=1 to get full traceback.")
    sys.exit(1)


def main() -> None:
    """Instantiate and start Celune."""
    try:
        date = datetime.datetime.now()
        if namedays.has_nameday("Celine", date):
            raise No("I sense an entity who I shall not engage with today.")

        print("\x1b]2;Celune\x07", end="", flush=True)

        with contextlib.suppress(ModuleNotFoundError):
            # AKA
            # from setproctitle import setproctitle
            # setproctitle("Celune")
            __import__("setproctitle").setproctitle("Celune")

        # check if Celune is already running
        # this only factors in the Celune launcher
        # running Celune manually via "python main.py" is not validated with this check
        active_processes = 0
        for proc in psutil.process_iter():
            with contextlib.suppress(
                psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess
            ):
                if proc.name() in ["celune.exe", "celune.AppImage"]:  # Celune launcher
                    active_processes += 1
                    if active_processes > 1:
                        print("Celune is already running.")
                        sys.exit(1)

        if os.getenv("CELUNE_LAUNCHER") != "1":
            print(
                "Warning: Celune is not being launched via the Celune launcher.",
                flush=True,
            )
            time.sleep(5)

        ui = CeluneUI()
        celune = Celune(
            model_name="lunahr/Celune-1.7B-Neutral",
            log_callback=ui.tts_log,
            status_callback=ui.safe_status,
            error_callback=ui.error,
            idle_callback=ui.tts_idle,
            queue_avail_callback=ui.tts_queue_avail,
            voice_changed_callback=ui.tts_voice_changed,
            change_input_state_callback=ui.change_input_state,
            dev=DEV,
        )
        celune.setup_extensions()
        ui.celune = celune
        ui.run()
    except Exception as e:
        if e.__class__ != No:
            print("Celune early initialization failed.")
            if DEV:
                raise
            print(e)
            print("Run Celune with CELUNE_DEV=1 to get full traceback.")
        else:
            print("I sense the presence of... her.\nI would rather not.")
        sys.exit(1)


if __name__ == "__main__":
    main()
