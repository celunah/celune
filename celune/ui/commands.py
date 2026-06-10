# SPDX-License-Identifier: MIT
"""Slash command handling for the Textual UI."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, TYPE_CHECKING

import soundfile as sf

from ..paths import project_root
from ..constants import APP_NAME
from ..backends.qwen3 import Qwen3
from ..exceptions import InvalidExtensionError
from ..utils import format_error, replace_ipa, format_number

if TYPE_CHECKING:
    from .app import CeluneUI

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}


def _attachment_source(path: Path) -> str:
    """Return a Persona-friendly attachment source string for one local file."""
    resolved = path.resolve()
    if os.name == "nt":
        return resolved.as_posix()
    return resolved.as_uri()


def attachment_source(path: Path) -> str:
    """Public interface for _attachment_source().

    Args:
        path: The path of the attachment.

    Returns:
        str: The return value of _attachment_source(), containing a Persona-friendly attachment source string.
    """

    return _attachment_source(path)


def _remote_attachment_kind(source: str) -> Optional[str]:
    """Return the attachment kind for one supported remote URL."""
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    suffix = Path(parsed.path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def tutorial(ui: CeluneUI) -> None:
    """Run actions related to the tutorial.

    Args:
        ui: The instance of CeluneUI that the tutorial will interact with.
    """
    assets = project_root() / "celune" / "assets"
    if not assets.exists():
        assets = project_root() / "assets"
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
    tutorial_token = ui.tutorial_token

    def prepare_and_schedule() -> None:
        def wav_duration(pth: Path) -> float:
            if not pth.exists():
                raise FileNotFoundError(f"tutorial clip not found: {pth}")

            info = sf.info(str(pth))
            return info.frames / info.samplerate

        def play_tutorial_clip(pth: Path) -> None:
            def worker() -> None:
                try:
                    ui.celune.play(str(pth))
                except Exception as exc:
                    ui.safe_log(
                        f"Tutorial playback failed: {format_error(exc, ui.celune.dev)}",
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
            if tutorial_token != ui.tutorial_token or not ui.tutorial_active:
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
    """Process slash commands.

    Args:
        ui: The instance of the CeluneUI to use here.
        command: The slash-command name without the leading slash.
        args: The parsed command arguments.
    """

    ui.input_box.load_text("")
    if command == "help":
        ui.safe_log(f"--- {APP_NAME} help topics ---")
        ui.safe_log("Available commands:")
        ui.safe_log(
            "Arguments marked in <> are required, those marked in [] are optional."
        )
        ui.safe_log(
            f"/consumebuffer <true/false> - Make {APP_NAME} consume text from the live buffer without "
            "pressing CTRL+ENTER."
        )
        ui.safe_log("Caution: This feature may interfere with typing '...'.", "warning")
        ui.safe_log(
            f"/invoke <extension> [args] - Invoke a {APP_NAME} extension by its name."
        )
        ui.safe_log(f"/extensions - List currently available {APP_NAME} extensions.")

        if ui.celune.backend.name != "mini":
            ui.safe_log(
                f"/voiceprompt <prompt> - Change {APP_NAME}'s voice prompt. This will allow you to steer her voice."
            )
            ui.safe_log(
                "Caution: Some prompts may cause adverse effects. Choose prompts that enhance personality, "
                "rather than replace it.",
                "warning",
            )

        ui.safe_log("/speed <speed> - Change speaking speed.")
        ui.safe_log("/reverb <strength> - Change reverb strength.")

        if ui.celune.backend.name == "qwen3":
            ui.safe_log(
                "/xvectoronly <true/false> - Toggle Qwen3 identity-only cloning."
            )

        ui.safe_log("/play <file> - Play a sound effect by path.")

        if ui.celune.vision is not None:
            ui.safe_log(
                "/attach <file> [file...] - Attach one or more images or videos to the next persona reply."
            )
            ui.safe_log(
                "/say <text> - Speak text directly and bypass Persona for this one message."
            )

        ui.safe_log(
            "/seed [seed|random] - Set or clear the seed for speech outputs, affecting pronunciation and/or prosody."
        )
        ui.safe_log(f"/tutorial - Run {APP_NAME}'s tutorial.")
        ui.safe_log("/stop - Terminate ongoing speech.")
        ui.safe_log(f"/exit - Exit {APP_NAME}.")
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
        voice_prompt_supported = getattr(ui.celune, "voice_prompt_supported", None)
        if callable(voice_prompt_supported) and not voice_prompt_supported():
            ui.celune.voice_prompt = None
            ui.safe_log(
                "Voice prompts are unavailable with the currently loaded model.",
                "warning",
            )
            return

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
            ui.safe_log(f"{APP_NAME} cannot currently use Rubber Band.", "warning")
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
    if command == "xvectoronly":
        backend = ui.celune.backend
        if not isinstance(backend, Qwen3):
            ui.safe_log(
                "This setting is only available on the Qwen3 backend.", "warning"
            )
            return

        if not args:
            ui.safe_log("Usage: /xvectoronly <true/false>", "warning")
            return

        value = args[0].lower()
        if value not in {"true", "false"}:
            ui.safe_log(
                f"Invalid argument for '{command}', must be true/false.",
                "warning",
            )
            return

        backend.x_vector_only = value == "true"
        state = "enabled" if backend.x_vector_only else "disabled"
        ui.safe_log(f"Qwen3 identity-only cloning {state}.")
        return
    if command == "play":
        if not args:
            ui.safe_log("Usage: /play <path> [volume]", "warning")
            return

        try:
            volume = 1.0
            if len(args) >= 2:
                try:
                    volume = float(args[1])
                except ValueError:
                    ui.safe_log(
                        f"Invalid volume for '{command}', must be numeric.",
                        "warning",
                    )
                    return

            def worker() -> None:
                try:
                    if not ui.celune.play(args[0], volume=volume):
                        return
                    if args[0].startswith("https://"):
                        ui.safe_log(
                            f"Playing YouTube audio at {format_number(volume * 100)}% volume"
                        )
                    else:
                        ui.safe_log(
                            f"Playing {args[0]} at {format_number(volume * 100)}% volume"
                        )
                except Exception as exc:
                    ui.safe_log(
                        f"Cannot play this audio: {format_error(exc, ui.celune.dev)}",
                        "error",
                    )

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            ui.safe_log(
                f"Cannot play this file: {format_error(e, ui.celune.dev)}",
                "error",
            )
            return
        return
    if command == "attach":
        if not args:
            ui.safe_log("Usage: /attach <file> [file...]", "warning")
            return

        if len(args) == 1 and args[0].lower() in {"clear", "reset", "none"}:
            ui.celune.persona_attachments.clear()
            ui.safe_log("Attachments cleared.")
            return

        vision = getattr(ui.celune, "vision", "available")
        if vision is None:
            ui.safe_log(
                f"Cannot add attachments while {APP_NAME} is running in speech-only mode.",
                "warning",
            )
            return

        added: list[str] = []
        for raw_path in args:
            remote_kind = _remote_attachment_kind(raw_path)
            if remote_kind is not None:
                parsed = urlparse(raw_path)
                name = Path(parsed.path).name or raw_path
                ui.celune.persona_attachments.append(
                    {"type": remote_kind, "path": raw_path, "name": name}
                )
                added.append(name)
                continue

            path = Path(raw_path).expanduser()
            if not path.exists() or not path.is_file():
                ui.safe_log(f"Attachment not found: {raw_path}", "warning")
                continue

            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                kind = "image"
            elif suffix in VIDEO_EXTENSIONS:
                kind = "video"
            else:
                ui.safe_log(f"Unsupported attachment type: {raw_path}", "warning")
                continue

            resolved = path.resolve()
            ui.celune.persona_attachments.append(
                {
                    "type": kind,
                    "path": _attachment_source(resolved),
                    "name": resolved.name,
                }
            )
            added.append(resolved.name)

        if not added:
            return

        count = len(ui.celune.persona_attachments)
        ui.safe_log(
            f"Attached {', '.join(added)}. {count} attachments will be sent in the next pass."
        )
        return
    if command == "say":
        if not args:
            ui.safe_log("Usage: /say <text>", "warning")
            return

        raw_text = " ".join(args).strip()
        if not raw_text:
            ui.safe_log("Usage: /say <text>", "warning")
            return

        if not ui.celune.vision:
            ui.safe_log("This command is redundant in the current operation mode.")
            ui.safe_log("Submit inputs normally instead.")
            return

        ipa_decoded, unmatched = replace_ipa(raw_text, strict=True)
        if unmatched > 0:
            safe_log_dev = getattr(ui, "safe_log_dev", None)
            if callable(safe_log_dev):
                safe_log_dev(
                    f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                    "warning",
                )
            else:
                ui.safe_log(
                    f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                    "warning",
                )

        ui.celune.say(ipa_decoded, display_text=raw_text)
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
            ui.safe_log(f"Seed must be between 0 and {2**32 - 1}.", "warning")
            return

        ui.celune.backend.current_seed = value
        ui.celune.backend.random_seed = False
        ui.safe_log(f"Seed set to {value}.")
        return
    if command == "tutorial":
        ui.safe_log(
            f"Tutorial activated. Listen to what's said to learn how to use {APP_NAME}."
        )
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
