#!/usr/bin/env python3
# pylint: disable=R0902, R0912, R0913, R0915, R0917, W0718
"""
Celune 3.2.2 - "I'm not just a TTS. I'm someone special."
Refer to https://github.com/celunah/celune for information about Celune.
Celune models are available on https://huggingface.co/collections/lunahr/celune.
"""

import os
import sys
import time
import shutil
import datetime
import contextlib

# Setting this environment variable will run Celune in dev mode and provide full tracebacks.
INITIAL_DEV = os.getenv("CELUNE_DEV") in {"1", "true", "on", "yes", "enabled"}
# Setting this environment variable will run Celune in headless mode (aka CEF, Celune Embedded Framework).
INITIAL_HEADLESS = os.getenv("CELUNE_HEADLESS") in {"1", "true", "on", "yes", "enabled"}
# Which backend shall Celune use? She can use Qwen3-TTS (qwen3) or VoxCPM2 (voxcpm2).
INITIAL_BACKEND = os.getenv("CELUNE_BACKEND")

try:
    import yaml
    import psutil
    from celune.celune import Celune
    from celune.exceptions import No
    from celune.namedays import has_nameday
    from celune.ui import CeluneUI, CeluneHeadlessUI, SelectMenu
    from celune.config import config_bool, config_value, env_bool
except ModuleNotFoundError as package:
    print(f"Missing dependency: {package.name}")
    print("Celune requires this library to function.")
    print("Install dependencies with:")
    print("    uv sync")
    print("or install manually:")
    print(f"    pip install {package.name}")
    if INITIAL_DEV:
        raise
    print("Run Celune with CELUNE_DEV=1 to get full traceback.")

    # this error is not controllable by config.yaml due to how Celune exceptions are caught
    print("Other errors may be displayed by setting 'dev: true' in config.yaml.")
    sys.exit(1)


def main() -> None:
    """Instantiate and start Celune.

    Returns:
        None: This function runs the application or exits the process on failure.
    """
    try:
        date = datetime.datetime.now()
        if has_nameday("Celine", date):
            raise No("I sense an entity who I shall not engage with today.")

        print("\x1b]2;Celune\x07", end="", flush=True)

        if not os.path.exists("config.yaml"):
            shutil.copy("default_config.yaml", "config.yaml")
            print("Celune configuration has been created.")

        with open("config.yaml", encoding="utf-8") as cfg:
            config = yaml.safe_load(cfg)

        dev = config_bool(config, "CELUNE_DEV", "dev")
        headless = config_bool(config, "CELUNE_HEADLESS", "headless")
        backend = INITIAL_BACKEND or config_value(config, "backend")

        # ask for default backend if not set yet
        # Celune will save this preference
        if not backend:
            backend = SelectMenu(
                ["Qwen3 - Fast", "VoxCPM2 - High quality"],
                ["qwen3", "voxcpm2"],
                "Which backend should Celune use?",
            ).start()

            if backend == "Fast":
                backend = "qwen3"
            elif backend == "High Quality":
                backend = "voxcpm2"

            config["backend"] = backend
            with open("config.yaml", "w", encoding="utf-8") as cfg:
                yaml.dump(config, cfg)

        if not env_bool("CELUNE_LAUNCHER"):
            print(
                "Warning: Celune is not being launched via the Celune launcher.",
                flush=True,
            )
            time.sleep(5)
        else:
            active_processes = 0
            for proc in psutil.process_iter():
                with contextlib.suppress(
                    psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess
                ):
                    if proc.name() in [
                        "celune.exe",
                        "celune.AppImage",
                    ]:  # Celune launcher
                        active_processes += 1
                        if active_processes > 1:
                            # you do not want to run multiple instances of Celune
                            # your memory doesn't want to either
                            print("Celune is already running.")
                            sys.exit(1)

        if not headless:  # normal mode
            ui = CeluneUI()
            celune = Celune(
                tts_backend=backend,
                log_callback=ui.tts_log,
                status_callback=ui.safe_status,
                error_callback=ui.error,
                idle_callback=ui.tts_idle,
                queue_avail_callback=ui.tts_queue_avail,
                voice_changed_callback=ui.tts_voice_changed,
                change_input_state_callback=ui.change_input_state,
                dev=dev,
                config=config,
            )
            celune.setup_extensions()
            ui.celune = celune
            ui.run()
        else:  # CEF/headless mode
            ui = CeluneHeadlessUI()
            celune = Celune(
                tts_backend=backend,
                log_callback=ui.headless_log,
                error_callback=ui.headless_error,
                dev=dev,
                config=config,
            )
            celune.setup_extensions()
            ui.celune = celune

            if not celune.load():
                celune.close()
                sys.exit(1)

            print("Celune is running in headless mode.")
            print("While in this mode, input is only possible via Celune extensions.")
            ui.run()
    except Exception as e:
        if e.__class__ != No:
            print("An internal error occurred running Celune.")
            if INITIAL_DEV:
                raise
            print(e or "no error description")
            print("Run Celune with CELUNE_DEV=1 to get full traceback.")
            print("Alternatively, set 'dev: true' in config.yaml")
            sys.exit(1)
        else:
            print("I sense the presence of... her.\nI would rather not.")
            sys.exit(103)


if __name__ == "__main__":
    main()
