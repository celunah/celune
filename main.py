#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Celune 3.5.0 - "I'm not just a TTS. I'm someone special."
Refer to https://github.com/celunah/celune for information about Celune.
Celune models are available on https://huggingface.co/collections/lunahr/celune.

This software may be redistributed under the terms of the MIT license, including commercial use.
"""

import os
import sys
import time
import random
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
    from celune.exceptions import No, UpdateError
    from celune.namedays import has_name_day
    from celune.updater import check_for_update, update_to_latest
    from celune.ui import (
        CeluneUI,
        CeluneHeadlessUI,
        CeluneHeadlessBaseUI,
        CeluneTextualUI,
        SelectMenu,
    )
    from celune.config import (
        config_bool,
        config_value,
        env_bool,
        merge_missing_defaults,
    )
    from celune.utils import supports_ansi, indent, title_case
    from celune.constants import ExitCodes
except ModuleNotFoundError as package:
    print(f"You do not have '{package.name}' installed.")
    print("Celune requires this library to function.")
    print()
    print("Set up Celune automatically by running:")
    print("    python setup.py")
    print()
    print("or alternatively with uv:")
    print("    uv sync")
    print()
    print("or install the package manually:")
    print(f"    pip install {package.name}")
    print()
    if INITIAL_DEV:
        with contextlib.suppress(ModuleNotFoundError):
            from rich.traceback import install

            install()

        raise
    print("for full traceback:")
    if os.name == "nt":
        print("set CELUNE_DEV=1")
        print(f"    python {os.path.basename(__file__)}")
    else:
        print(f"    CELUNE_DEV=1 python {os.path.basename(__file__)}")
    print()

    # this error is not controllable by config.yaml due to how Celune exceptions are caught
    print("additional debugging:")
    print("    Set 'dev: true' in config.yaml")
    print()
    sys.exit(4)


def main() -> None:
    """Instantiate and start Celune.

    Returns:
        None: This function runs the application or exits the process on failure.

    Raises:
        No: Celune refuses to run on a blocked name day.
        Exception: Re-raised in development mode for unexpected startup errors.
    """
    try:
        date = datetime.datetime.now()
        if has_name_day("Celine", date) and not env_bool("CELUNE_OVERRIDE_CELINE_DAY"):
            raise No

        if supports_ansi():
            print("\x1b]2;Celune\x07", end="", flush=True)

        if not os.path.exists("config.yaml"):
            shutil.copy("default_config.yaml", "config.yaml")
            print("Celune configuration has been created.")

        with open("config.yaml", encoding="utf-8") as cfg:
            config = yaml.safe_load(cfg)
        with open("default_config.yaml", encoding="utf-8") as cfg:
            default_config = yaml.safe_load(cfg)

        if not isinstance(default_config, dict):
            default_config = {}
        if not isinstance(config, dict):
            config = {}

        config, config_updated = merge_missing_defaults(config, default_config)
        if config_updated:
            with open("config.yaml", "w", encoding="utf-8") as cfg:
                yaml.safe_dump(config, cfg, sort_keys=False)
            print("Celune configuration has been updated with new defaults.")

        dev = config_bool(config, "CELUNE_DEV", "dev")
        headless = config_bool(config, "CELUNE_HEADLESS", "headless")
        backend = INITIAL_BACKEND or config_value(config, "backend")

        # try to update if not up to date
        if not headless and supports_ansi():
            update = check_for_update()
            if update:
                latest_label = f"Celune {update.latest_version}"
                if not update.latest_tag:
                    latest_label = "Celune"  # "I'm not sure if I'm up to date."

                choice = SelectMenu(
                    ["Yes, update now", "No, continue as is"],
                    [True, False],
                    "\n".join(
                        [
                            "New update found.",
                            (
                                f"You are running Celune {update.local_version} "
                                f"({update.local_revision}), latest version is "
                                f"{latest_label} ({update.latest_revision})."
                            ),
                            "Do you want to update?",
                        ]
                    ),
                ).start()

                if choice:
                    print("Updating Celune...")
                    try:
                        update_to_latest()
                    except UpdateError as exc:
                        print(exc)
                        print("Continuing with the current version.")
                        time.sleep(5)
                    else:
                        print(
                            "Celune updated successfully. Restart Celune to apply changes."
                        )
                        sys.exit(ExitCodes.EXIT_PENDING_UPDATE.value)
        elif check_for_update() and not supports_ansi():
            print("This terminal does not support ANSI.")
            print("Attempting to apply update non-interactively...")
            try:
                update_to_latest()
            except UpdateError as exc:
                detail = title_case(str(exc))
                print(f"Celune could not update: {detail}")
                print("Continuing with the current version.")
                time.sleep(5)
            else:
                print("Celune updated successfully. Restart Celune to apply changes.")
                sys.exit(ExitCodes.EXIT_PENDING_UPDATE.value)

        # ask for default backend if not set yet
        # Celune will save this preference
        if not backend and supports_ansi():
            backend = SelectMenu(
                ["Qwen3 - Fast", "VoxCPM2 - High quality"],
                ["qwen3", "voxcpm2"],
                "Which backend should Celune use?",
            ).start()

            if backend == "qwen3":
                print("Qwen3 uses CEVOICE-backed voice cloning by default.")
                print(
                    "Native mode remains available as a deprecated compatibility option."
                )
            elif backend == "voxcpm2":
                print(
                    "Note: VoxCPM2 only supports voice cloning, and it is significantly slower."
                )
                print(
                    "By selecting this backend, you accept any tradeoffs that may occur later on."
                )

            config["backend"] = backend
            with open("config.yaml", "w", encoding="utf-8") as cfg:
                yaml.dump(config, cfg)
        elif not backend and not supports_ansi():
            print("This terminal does not support ANSI.")
            print("Please select a backend manually.")
            print("Refer to Celune's configuration for details.")
            sys.exit(ExitCodes.EXIT_NO_ANSI.value)

        if not env_bool("CELUNE_LAUNCHER"):
            launcher_exe = "celune.exe" if os.name == "nt" else "celune.appimage"
            print("Celune is not being launched via the Celune launcher.")
            print()
            print("To suppress this message, run Celune with:")
            print(indent(f"{launcher_exe}", spaces=4))
            print()
            print("or set the following environment variable:")
            print(indent("CELUNE_LAUNCHER=1", spaces=4))
            time.sleep(5)
        else:
            active_processes = 0
            for proc in psutil.process_iter():
                with contextlib.suppress(
                    psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess
                ):
                    if proc.name() in [
                        "celune.exe",
                        "celune.appimage",
                    ]:  # Celune launcher
                        active_processes += 1
                        if active_processes > 1:
                            # you do not want to run multiple instances of Celune
                            # your memory doesn't want to either
                            print("Celune is already running.")
                            sys.exit(ExitCodes.EXIT_ALREADY_RUNNING.value)

        if not headless and supports_ansi():  # normal mode
            ui: CeluneTextualUI
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
                change_voice_lock_state_callback=ui.change_voice_lock_state,
                progress_callback=ui.safe_progress,
                dev=dev,
                config=config,
            )
            ui.celune = celune
            ui.run()
        elif headless:
            ui_headless: CeluneHeadlessBaseUI
            ui_headless = CeluneHeadlessUI(config)
            celune = Celune(
                tts_backend=backend,
                log_callback=ui_headless.headless_log,
                error_callback=ui_headless.headless_error,
                dev=dev,
                config=config,
            )
            ui_headless.celune = celune

            if not celune.load():
                celune.close()
                sys.exit(ExitCodes.EXIT_FAILURE.value)

            print("Celune is running in headless mode.")
            print("While in this mode, input is only possible via Celune extensions.")
            ui_headless.run()
        else:
            print("This terminal does not support ANSI.")
            print("Celune cannot start in normal mode.")
            print("Hint:")
            print(indent("Try using another terminal application.", spaces=4))
            sys.exit(ExitCodes.EXIT_NO_ANSI.value)
    except Exception as e:
        if e.__class__ != No:
            stdout = getattr(sys.stdout, "underlying_stdout", sys.stdout)
            stderr = getattr(sys.stderr, "underlying_stderr", sys.stderr)
            sys.stdout = stdout
            sys.stderr = stderr

            print("An internal error occurred while Celune was running.")
            if INITIAL_DEV:
                with contextlib.suppress(ModuleNotFoundError):
                    from rich.traceback import install

                    install()

                raise
            print(e or "no error description")
            print("For full traceback:")
            if os.name == "nt":
                print("set CELUNE_DEV=1")
                print(indent("python {os.path.basename(__file__)}", spaces=4))
            else:
                print(
                    indent("CELUNE_DEV=1 python {os.path.basename(__file__)}", spaces=4)
                )
            print()
            print("additional debugging:")
            print(indent("Set 'dev: true' in config.yaml", spaces=4))
            sys.exit(ExitCodes.EXIT_FAILURE.value)

        print("I sense the presence of... her.")
        print("I would rather not.")
        print()
        print("Hint:")
        print(indent("Try again tomorrow.", spaces=4))
        print("or set the following environment variable:")
        print(indent("CELUNE_OVERRIDE_CELINE_DAY=1", spaces=4))
        sys.exit(
            ExitCodes.EXIT_CELINE_DAY.value
            if random.uniform(0, 1) < 0.5
            else ExitCodes.EXIT_CELINE_DAY_SIX_SEVEN.value
        )


if __name__ == "__main__":
    main()
