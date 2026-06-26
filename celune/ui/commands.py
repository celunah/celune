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
from ..backends.tts.qwen3 import Qwen3
from ..exceptions import InvalidExtensionError
from ..utils import format_error, replace_ipa, format_number
from ..cevoice import active_bundle_path, resolve_bundle_path
from ..i18n import string

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
        ui.safe_log(string("commands.no_tutorial_assets"), "warning")
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
                        string(
                            "commands.tutorial_playback_failed",
                            error=format_error(exc, ui.celune.dev),
                        ),
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
                string(
                    "commands.tutorial_failed",
                    error=format_error(e, ui.celune.dev),
                ),
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

    def refresh_vc_controls() -> None:
        refresh = getattr(ui, "refresh_vc_controls", None)
        if callable(refresh):
            refresh()

    def set_vc_f0_condition(enabled: bool) -> None:
        setter = getattr(ui, "set_vc_f0_condition", None)
        if callable(setter):
            setter(enabled)
            return

        ui.celune.vc_f0_condition = enabled
        backend = getattr(ui.celune, "vc_backend", None)
        if backend is not None and hasattr(backend, "f0_condition"):
            setattr(backend, "f0_condition", enabled)
        refresh_vc_controls()
        ui.safe_log(
            string(
                "commands.vcmode_set",
                mode=string("ui.vc_mode_sing" if enabled else "ui.vc_mode_talk"),
            )
        )

    def set_vc_pitch_shift(value: int) -> None:
        setter = getattr(ui, "set_vc_pitch_shift", None)
        if callable(setter):
            setter(value)
            return

        ui.celune.vc_pitch_shift = value
        backend = getattr(ui.celune, "vc_backend", None)
        if backend is not None and hasattr(backend, "pitch_shift"):
            setattr(backend, "pitch_shift", value)
        refresh_vc_controls()
        ui.safe_log(string("commands.vcpitch_set", value=value))

    if command == "help":
        ui.safe_log(string("commands.help_header", app_name=APP_NAME))
        ui.safe_log(string("commands.help_available"))
        ui.safe_log(string("commands.help_arguments"))
        ui.safe_log(string("commands.help_consumebuffer", app_name=APP_NAME))
        ui.safe_log(string("commands.help_consumebuffer_caution"), "warning")
        ui.safe_log(string("commands.help_invoke", app_name=APP_NAME))
        ui.safe_log(string("commands.help_extensions", app_name=APP_NAME))

        if ui.celune.backend.name != "mini":
            ui.safe_log(string("commands.help_voiceprompt", app_name=APP_NAME))
            ui.safe_log(string("commands.help_voiceprompt_caution"), "warning")

        ui.safe_log(string("commands.help_speed"))
        ui.safe_log(string("commands.help_reverb"))
        ui.safe_log(string("commands.help_backend"))
        ui.safe_log(string("commands.help_cevoice"))
        if getattr(ui.celune, "input_mode", "text_to_speech") == "voice_conversion":
            ui.safe_log(string("commands.help_vc"))
            ui.safe_log(string("commands.help_vcmode"))
            ui.safe_log(string("commands.help_vcpitch"))

        if ui.celune.backend.name == "qwen3":
            ui.safe_log(string("commands.help_xvectoronly"))

        ui.safe_log(string("commands.help_play"))

        if ui.celune.vision is not None:
            ui.safe_log(string("commands.help_attach"))
            ui.safe_log(string("commands.help_say"))

        ui.safe_log(string("commands.help_seed"))
        ui.safe_log(string("commands.help_tutorial", app_name=APP_NAME))
        ui.safe_log(string("commands.help_stop"))
        ui.safe_log(string("commands.help_exit", app_name=APP_NAME))
        ui.safe_log(string("commands.help_help"))
        return
    if command == "consumebuffer":
        if not args:
            ui.safe_log(string("commands.usage_consumebuffer"), "warning")
            return

        if args[0].lower() in ["true", "false"]:
            boolean = args[0].lower() == "true"
            ui.consume_on_boundary = boolean

            if boolean:
                ui.safe_log(string("commands.consuming_live_input"))
            else:
                ui.safe_log(string("commands.not_consuming_live_input"))
            return
        ui.safe_log(
            string("commands.invalid_true_false_argument", command=command), "warning"
        )
        return
    if command == "invoke":
        if not args:
            ui.safe_log(string("commands.usage_invoke"))
            return

        if not ui.celune.extension_manager:
            ui.safe_log(string("commands.extension_system_not_initialized"), "warning")
            return

        name = args[0]
        invoke_args = args[1:]

        try:
            ui.celune.extension_manager.invoke(name, *invoke_args)
        except InvalidExtensionError:
            ui.safe_log(string("commands.extension_not_found", name=name), "warning")
        except Exception as e:
            ui.safe_log(string("commands.extension_error", error=e), "error")

        return
    if command == "extensions":
        if not ui.celune.extension_manager:
            ui.safe_log(string("commands.extension_system_not_initialized"), "warning")
            return

        names = ui.celune.extension_manager.list_extensions()
        if not names:
            ui.safe_log(string("commands.no_extensions_loaded"), "warning")
        else:
            ui.safe_log(string("commands.extensions_list", names=", ".join(names)))
        return
    if command == "voiceprompt":
        voice_prompt_supported = getattr(ui.celune, "voice_prompt_supported", None)
        if callable(voice_prompt_supported) and not voice_prompt_supported():
            ui.celune.voice_prompt = None
            ui.safe_log(
                string("commands.voice_prompts_unavailable"),
                "warning",
            )
            return

        if not args:
            ui.safe_log(string("commands.usage_voiceprompt"), "warning")
            return

        new_prompt = " ".join(args).strip()
        ui.celune.voice_prompt = new_prompt

        if not new_prompt or new_prompt.lower() == "clear":
            ui.celune.voice_prompt = None
            ui.safe_log(string("commands.voice_prompt_cleared"))
            return

        ui.safe_log(string("commands.voice_prompt_set", prompt=new_prompt))
        return
    if command == "speed":
        if not ui.celune.can_use_rubberband:
            ui.safe_log(
                string("commands.rubber_band_unavailable", app_name=APP_NAME),
                "warning",
            )
            return

        if not args:
            ui.safe_log(string("commands.usage_speed"), "warning")
            return

        try:
            if args[0].endswith("%"):
                args[0] = args[0].rstrip("%")

            speed = float(args[0])
            float_speed = speed / 100.0
            if not 0.8 <= float_speed <= 1.2:
                ui.safe_log(string("commands.value_out_of_range_speed"), "warning")
                return
            ui.celune.speed = float_speed
        except ValueError:
            ui.safe_log(string("commands.invalid_argument", value=args[0]), "warning")
        else:
            ui.safe_log(string("commands.speed_set", value=args[0]))
        return
    if command == "reverb":
        if not args:
            ui.safe_log(string("commands.usage_reverb"), "warning")
            return

        try:
            if args[0].endswith("%"):
                args[0] = args[0].rstrip("%")

            strength = float(args[0])
            float_strength = strength / 100.0
            if not 0.0 <= float_strength <= 1.0:
                ui.safe_log(string("commands.value_out_of_range_reverb"), "warning")
                return
            ui.celune.reverb.strength = float_strength
        except ValueError:
            ui.safe_log(string("commands.invalid_argument", value=args[0]), "warning")
        else:
            ui.safe_log(string("commands.reverb_set", value=args[0]))
        return
    if command == "backend":
        if not args:
            ui.safe_log(string("commands.usage_backend"), "warning")
            return

        backend_name = args[0]

        active_backend = getattr(ui.celune, "_active_runtime_backend_name", None)
        if callable(active_backend):
            active_backend_name = active_backend()
        elif getattr(ui.celune, "input_mode", "text_to_speech") == "voice_conversion":
            backend = getattr(ui.celune, "vc_backend", None)
            active_backend_name = getattr(backend, "name", "")
        else:
            backend = getattr(ui.celune, "backend", None)
            active_backend_name = getattr(backend, "name", "")
        if backend_name == active_backend_name:
            ui.safe_log(string("commands.backend_already_loaded"), "warning")
            return

        def backend_worker() -> None:
            try:
                if ui.celune.set_backend_and_wait(backend_name):
                    ui.celune.try_play_signal("readiness")
                    ui.safe_log(
                        string("commands.backend_switched", backend_name=backend_name)
                    )
                else:
                    ui.safe_log(string("commands.backend_not_switched"), "warning")
            except Exception as exc:
                ui.safe_log(
                    string(
                        "commands.backend_switch_failed",
                        error=format_error(exc, ui.celune.dev),
                    ),
                    "error",
                )

        threading.Thread(target=backend_worker, daemon=True).start()
        return
    if command == "cevoice":
        if not args:
            ui.safe_log(string("commands.usage_cevoice"), "warning")
            return

        bundle = args[0]
        if resolve_bundle_path(bundle) == active_bundle_path():
            ui.safe_log(string("commands.character_already_loaded"), "warning")
            return

        def cevoice_worker() -> None:
            try:
                if ui.celune.set_cevoice_and_wait(bundle):
                    ui.safe_log(string("commands.character_changed", bundle=bundle))
                else:
                    ui.safe_log(
                        string("commands.character_not_switched", bundle=bundle),
                        "warning",
                    )
            except Exception as exc:
                ui.safe_log(
                    string(
                        "commands.character_switch_failed",
                        error=format_error(exc, ui.celune.dev),
                    ),
                    "error",
                )

        threading.Thread(target=cevoice_worker, daemon=True).start()
        return
    if command == "vc":
        if getattr(ui.celune, "input_mode", "text_to_speech") != "voice_conversion":
            ui.safe_log(string("commands.voice_conversion_only"), "warning")
            return

        if not args:
            ui.safe_log(string("commands.usage_vc"), "warning")
            return

        source_path = Path(args[0]).expanduser()
        if not source_path.exists() or not source_path.is_file():
            ui.safe_log(
                string("commands.vc_file_not_found", path=args[0]),
                "warning",
            )
            return

        def vc_worker() -> None:
            try:
                audio, sample_rate = sf.read(
                    str(source_path),
                    dtype="float32",
                    always_2d=False,
                )
            except Exception as exc:
                ui.safe_log(
                    string(
                        "commands.vc_decode_failed",
                        error=format_error(exc, ui.celune.dev),
                    ),
                    "error",
                )
                return

            if ui.celune.submit_audio(
                audio,
                sample_rate,
                label=source_path.name,
            ):
                ui.safe_log(
                    string("commands.vc_submitted", path=str(source_path)),
                )
                return

            ui.safe_log(
                string("commands.vc_submission_failed", path=str(source_path)),
                "warning",
            )

        threading.Thread(target=vc_worker, daemon=True).start()
        return
    if command == "vcmode":
        if getattr(ui.celune, "input_mode", "text_to_speech") != "voice_conversion":
            ui.safe_log(string("commands.voice_conversion_only"), "warning")
            return

        if not args:
            ui.safe_log(string("commands.usage_vcmode"), "warning")
            return

        mode = args[0].lower()
        if mode == "talk":
            set_vc_f0_condition(False)
            return
        if mode == "sing":
            set_vc_f0_condition(True)
            return

        ui.safe_log(string("commands.usage_vcmode"), "warning")
        return
    if command == "vcpitch":
        if getattr(ui.celune, "input_mode", "text_to_speech") != "voice_conversion":
            ui.safe_log(string("commands.voice_conversion_only"), "warning")
            return

        if not args:
            ui.safe_log(string("commands.usage_vcpitch"), "warning")
            return

        raw_value = args[0].lower()
        if raw_value == "clear":
            set_vc_pitch_shift(0)
            return

        try:
            semitones = int(raw_value)
        except ValueError:
            ui.safe_log(string("commands.usage_vcpitch"), "warning")
            return

        if not -12 <= semitones <= 12:
            ui.safe_log(string("commands.vcpitch_range"), "warning")
            return

        set_vc_pitch_shift(semitones)
        return
    if command == "xvectoronly":
        backend = ui.celune.backend
        if not isinstance(backend, Qwen3):
            ui.safe_log(string("commands.xvectoronly_qwen3_only"), "warning")
            return

        if not args:
            ui.safe_log(string("commands.usage_xvectoronly"), "warning")
            return

        value = args[0].lower()
        if value not in {"true", "false"}:
            ui.safe_log(
                string("commands.invalid_true_false_argument", command=command),
                "warning",
            )
            return

        backend.x_vector_only = value == "true"
        state = string(
            "commands.state_enabled"
            if backend.x_vector_only
            else "commands.state_disabled"
        )
        ui.safe_log(string("commands.qwen3_identity_only_cloning", state=state))
        return
    if command == "play":
        if not args:
            ui.safe_log(string("commands.usage_play"), "warning")
            return

        try:
            volume = 1.0
            if len(args) >= 2:
                try:
                    volume = float(args[1])
                except ValueError:
                    ui.safe_log(
                        string("commands.invalid_volume", command=command),
                        "warning",
                    )
                    return

            def worker() -> None:
                try:
                    if not ui.celune.play(args[0], volume=volume):
                        return
                    if args[0].startswith("https://"):
                        ui.safe_log(
                            string(
                                "commands.playing_youtube_audio",
                                volume=format_number(volume * 100),
                            )
                        )
                    else:
                        ui.safe_log(
                            string(
                                "commands.playing_audio",
                                path=args[0],
                                volume=format_number(volume * 100),
                            )
                        )
                except Exception as exc:
                    ui.safe_log(
                        string(
                            "commands.cannot_play_audio",
                            error=format_error(exc, ui.celune.dev),
                        ),
                        "error",
                    )

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            ui.safe_log(
                string(
                    "commands.cannot_play_file",
                    error=format_error(e, ui.celune.dev),
                ),
                "error",
            )
            return
        return
    if command == "attach":
        if not args:
            ui.safe_log(string("commands.usage_attach"), "warning")
            return

        if len(args) == 1 and args[0].lower() in {"clear", "reset", "none"}:
            ui.celune.persona_attachments.clear()
            ui.safe_log(string("commands.attachments_cleared"))
            return

        vision = getattr(ui.celune, "vision", "available")
        if vision is None:
            ui.safe_log(
                string("commands.attachments_speech_only_mode", app_name=APP_NAME),
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
                ui.safe_log(
                    string("commands.attachment_not_found", path=raw_path), "warning"
                )
                continue

            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                kind = "image"
            elif suffix in VIDEO_EXTENSIONS:
                kind = "video"
            else:
                ui.safe_log(
                    string("commands.unsupported_attachment_type", path=raw_path),
                    "warning",
                )
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
            string(
                "commands.attachments_added",
                names=", ".join(added),
                count=count,
            )
        )
        return
    if command == "say":
        if not args:
            ui.safe_log(string("commands.usage_say"), "warning")
            return

        raw_text = " ".join(args).strip()
        if not raw_text:
            ui.safe_log(string("commands.usage_say"), "warning")
            return

        if not ui.celune.vision:
            ui.safe_log(string("commands.say_redundant"))
            ui.safe_log(string("commands.say_submit_normally"))
            return

        if ui.celune.config.get("ipa") is False:
            ipa_decoded, unmatched = replace_ipa(raw_text, strict=True)
            if unmatched > 0:
                safe_log_dev = getattr(ui, "safe_log_dev", None)
                if callable(safe_log_dev):
                    safe_log_dev(
                        string("commands.unmatched_ipa", count=unmatched),
                        "warning",
                    )
                else:
                    ui.safe_log(
                        string("commands.unmatched_ipa", count=unmatched),
                        "warning",
                    )

            ui.celune.say(ipa_decoded, display_text=raw_text)
        else:
            ui.celune.say(raw_text)

        return
    if command == "seed":
        if not args:
            ui.celune.backend.current_seed = None
            ui.celune.backend.random_seed = True
            ui.safe_log(string("commands.custom_seed_removed"))
            return

        if args[0].lower() in ["random", "unset", "none", "off"]:
            ui.celune.backend.current_seed = None
            ui.celune.backend.random_seed = True
            ui.safe_log(string("commands.custom_seed_removed"))
            return

        try:
            value = int(args[0])
        except ValueError:
            ui.safe_log(string("commands.invalid_argument", value=args[0]), "warning")
            return

        if not 0 <= value < 2**32:
            ui.safe_log(string("commands.seed_range", max_value=2**32 - 1), "warning")
            return

        ui.celune.backend.current_seed = value
        ui.celune.backend.random_seed = False
        ui.safe_log(string("commands.seed_set", value=value))
        return
    if command == "tutorial":
        ui.safe_log(string("commands.tutorial_activated", app_name=APP_NAME))
        tutorial(ui)
        return
    if command == "stop":
        if not ui.celune.force_stop_speech():
            ui.safe_log(string("commands.nothing_to_stop"))
            return

        return
    if command == "exit":
        ui.graceful_exit()
        return

    ui.safe_log(string("commands.unknown_command", command=command), "warning")
