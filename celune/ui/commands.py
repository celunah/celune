# SPDX-License-Identifier: MIT
"""Slash command handling for the Textual UI."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

from ..exceptions import InvalidExtensionError
from ..utils import format_error

if TYPE_CHECKING:
    from .app import CeluneUI


def tutorial(ui: CeluneUI) -> None:
    """Run actions related to Celune's tutorial.

    Args:
        ui: The instance of CeluneUI that the tutorial will interact with.

    Returns:
        None: This function sends Celune tutorial commands automatically.
    """
    assets = Path(__file__).resolve().parents[1] / "assets"
    if not assets.exists():
        ui.safe_log("No tutorial assets found.", "warning")
        return

    clips = (
        (assets / "tutorial1.wav", None),
        (assets / "tutorial2.wav", lambda: ui.pulse_border("#input")),
        (assets / "tutorial3.wav", lambda: ui.pulse_border("#style")),
        (
            assets / "tutorial4.wav",
            lambda: ui.type_and_send("/help", process_commands=True),
        ),
    )

    ui.begin_tutorial()
    tutorial_token = ui._tutorial_token

    def prepare_and_schedule() -> None:
        """Prepare tutorial clip timings without blocking Textual."""

        def wav_duration(pth: Path) -> float:
            """Return the duration of a WAV file in seconds."""
            if not pth.exists():
                raise FileNotFoundError(f"tutorial clip not found: {pth}")

            info = sf.info(str(pth))
            return info.frames / info.samplerate

        def play_tutorial_clip(pth: Path) -> None:
            """Play a tutorial clip without blocking the Textual message loop."""

            def worker() -> None:
                """Queue tutorial audio on a background thread."""
                try:
                    ui.celune.play(str(pth))
                except Exception as e:
                    ui.safe_log(
                        f"Tutorial playback failed: {format_error(e, ui.celune.dev)}",
                        "warning",
                    )
                    ui.call_from_thread(ui.cancel_tutorial, True)

            threading.Thread(target=worker, daemon=True).start()

        try:
            clip_durations = tuple(
                (path, action, wav_duration(path)) for path, action in clips
            )
        except Exception as e:
            ui.safe_log(
                f"Tutorial failed: {format_error(e, ui.celune.dev)}",
                "warning",
            )
            ui.call_from_thread(ui.cancel_tutorial, True)
            return

        def schedule() -> None:
            """Schedule prepared tutorial actions on the UI thread."""
            if tutorial_token != ui._tutorial_token or not ui._tutorial_active:
                return

            elapsed = 0.0
            gap = 0.15

            for path, action, duration in clip_durations:
                ui.tutorial_after(elapsed, lambda pth=path: play_tutorial_clip(pth))

                if action is not None:
                    ui.tutorial_after(elapsed, action)

                elapsed += duration + gap

            ui.tutorial_after(elapsed, ui.finish_tutorial)

        ui.call_from_thread(schedule)

    threading.Thread(target=prepare_and_schedule, daemon=True).start()


def process_command(ui: CeluneUI, command: str, args: list[str]) -> None:
    """Process Celune control commands.

    Args:
        ui: The instance of the CeluneUI to use here.
        command: The slash-command name without the leading slash.
        args: The parsed command arguments.

    Returns:
        None: This method mutates UI or engine state based on the command.
    """

    ui.input_box.load_text("")
    if command == "help":
        ui.safe_log("Celune help topics")
        ui.safe_log("Available commands:")
        ui.safe_log(
            "Arguments marked in <> are required, those marked in [] are optional."
        )
        ui.safe_log(
            "/consumebuffer <true/false> - Make Celune consume text from the live buffer without "
            "pressing CTRL+ENTER."
        )
        ui.safe_log("Caution: This feature may interfere with typing '...'.", "warning")
        ui.safe_log(
            "/invoke <extension> [args] - Invoke a Celune extension by its name."
        )
        ui.safe_log("/extensions - List currently available Celune extensions.")
        ui.safe_log(
            "/voiceprompt <prompt> - Change Celune's voice prompt. This will allow you to steer her voice."
        )
        ui.safe_log(
            "Caution: Some prompts may cause adverse effects. Choose prompts that enhance personality, "
            "rather than replace it.",
            "warning",
        )
        ui.safe_log("/speed <speed> - Change speaking speed.")
        ui.safe_log("/reverb <strength> - Change reverb strength.")
        ui.safe_log(
            "/play <file> - Play a sound effect by path. Only WAV files are supported."
        )
        ui.safe_log(
            "/seed [seed|random] - Set or clear the seed for speech outputs (VoxCPM2 only)."
        )
        ui.safe_log("/tutorial - Run Celune's tutorial.")
        ui.safe_log("/stop - Terminate ongoing speech.")
        ui.safe_log("/exit - Exit Celune.")
        ui.safe_log("/help - Display this help message.")
        return
    if command == "consumebuffer":
        if not args:
            ui.safe_log("Usage: /consumebuffer <true/false>", "warning")
            return

        if args[0].lower() in ["true", "false"]:
            boolean = args[0].lower() == "true"
            ui.consume_on_boundary = boolean

            if boolean:
                ui.safe_log("Now consuming from live input")
            else:
                ui.safe_log("No longer consuming from live input")
            return
        ui.safe_log(f"Invalid argument for '{command}', must be true/false.", "warning")
        return
    if command == "invoke":
        if not args:
            ui.safe_log("Usage: /invoke <extension> [args]")
            return

        if not ui.celune.extension_manager:
            ui.safe_log("Extension system not initialized.", "warning")
            return

        name = args[0]
        invoke_args = args[1:]

        try:
            ui.celune.extension_manager.invoke(name, *invoke_args)
        except InvalidExtensionError:
            ui.safe_log(f"Extension not found: {name}", "warning")
        except Exception as e:
            ui.safe_log(f"[EXT ERROR] {e}", "error")

        return
    if command == "extensions":
        if not ui.celune.extension_manager:
            ui.safe_log("Extension system not initialized.", "warning")
            return

        names = ui.celune.extension_manager.list_extensions()
        if not names:
            ui.safe_log("No extensions loaded.", "warning")
        else:
            ui.safe_log("Extensions: " + ", ".join(names))
        return
    if command == "voiceprompt":
        if not args:
            ui.safe_log("Usage: /voiceprompt <prompt>", "warning")
            return

        new_prompt = " ".join(args).strip()
        ui.celune.voice_prompt = new_prompt

        if not new_prompt or new_prompt.lower() == "clear":
            ui.celune.voice_prompt = None
            ui.safe_log("Voice prompt cleared.")
            return

        ui.safe_log(f"Voice prompt set to '{new_prompt}'.")
        return
    if command == "speed":
        if not ui.celune.can_use_rubberband:
            ui.safe_log("Celune cannot currently use Rubber Band.", "warning")
            return

        if not args:
            ui.safe_log("Usage: /speed <speed>", "warning")
            return

        try:
            if args[0].endswith("%"):
                args[0] = args[0].rstrip("%")

            speed = float(args[0])
            float_speed = speed / 100.0
            if not 0.8 <= float_speed <= 1.2:
                ui.safe_log("Value out of range. Expected 80-120%.", "warning")
                return
            ui.celune.speed = float_speed
        except ValueError:
            ui.safe_log(f"Invalid argument: {args[0]}", "warning")
        else:
            ui.safe_log(f"Speaking speed set to {args[0]}%.")
        return
    if command == "reverb":
        if not args:
            ui.safe_log("Usage: /reverb <strength>", "warning")
            return

        try:
            if args[0].endswith("%"):
                args[0] = args[0].rstrip("%")

            strength = float(args[0])
            float_strength = strength / 100.0
            if not 0.0 <= float_strength <= 1.0:
                ui.safe_log("Value out of range. Expected 0-100%.", "warning")
                return
            ui.celune.reverb.strength = float_strength
        except ValueError:
            ui.safe_log(f"Invalid argument: {args[0]}", "warning")
        else:
            ui.safe_log(f"Reverb strength set to {args[0]}%.")
        return
    if command == "play":
        if not args:
            ui.safe_log("Usage: /play <path>", "warning")
            return

        try:
            if not ui.celune.play(args[0]):
                return
            ui.safe_log(f"Playing {args[0]}")
        except Exception as e:
            ui.safe_log(
                f"Cannot play this file: {format_error(e, ui.celune.dev)}",
                "error",
            )
            return
        return
    if command == "seed":
        if not args:
            ui.celune.backend.current_seed = None
            ui.celune.backend.random_seed = True
            ui.safe_log("Custom seed removed.")
            return

        if args[0].lower() in ["random", "unset", "none", "off"]:
            ui.celune.backend.current_seed = None
            ui.celune.backend.random_seed = True
            ui.safe_log("Custom seed removed.")
            return

        try:
            value = int(args[0])
        except ValueError:
            ui.safe_log(f"Invalid argument: {args[0]}", "warning")
            return

        if not 0 <= value < 2**32:
            ui.safe_log("Seed must be between 0 and 4294967295.", "warning")
            return

        ui.celune.backend.current_seed = value
        ui.celune.backend.random_seed = False
        ui.safe_log(f"Seed set to {value}.")
        return
    if command == "tutorial":
        tutorial(ui)
        return
    if command == "stop":
        if not ui.celune.force_stop_speech():
            ui.safe_log("Nothing to stop.")
            return

        return
    if command == "exit":
        ui.graceful_exit()
        return

    ui.safe_log(
        f"Unknown command: {command}. Run /help for a list of commands.", "warning"
    )
