# SPDX-License-Identifier: Apache-2.0
"""Slash command handling for the Textual UI."""

from __future__ import annotations

import os
import asyncio
import threading
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
from collections.abc import Callable, Awaitable
from typing import TYPE_CHECKING, Optional, cast

import soundfile as sf

from ..utils import (
    available,
    replace_ipa,
    format_number,
    format_error_message,
)
from ..constants import APP_NAME
from ..i18n import string, tagged_string
from ..typing.modes import OperationState
from ..exceptions import InvalidExtensionError
from ..pipeline.playback import close_stream
from ..audio.server import restart_audio_server
from ..core.concurrency import run_in_daemon_thread
from ..persona.capabilities import PersonaCapabilities
from ..core.lifecycle import active_runtime_backend_name
from ..cevoice import active_bundle_path, resolve_bundle_path
from ..vram import vram_report_int, format_vram_bytes, runtime_vram_report

if TYPE_CHECKING:
    from .app import CeluneUI
    from ..celune import Celune

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
_TUTORIAL_SPEECH_TIMEOUT = 180.0


CommandHandler = Callable[["CeluneUI", list[str]], None]
CommandAvailability = Callable[["CeluneUI"], bool]


@dataclass(frozen=True)
class _Command:
    """Describe one registered UI command and when it appears in help."""

    name: str
    handler: CommandHandler
    help_key: str
    availability: CommandAvailability
    help_values: tuple[tuple[str, str], ...]
    warning_keys: tuple[str, ...]


_COMMANDS: dict[str, _Command] = {}


def _always_available(_ui: CeluneUI) -> bool:
    """Return whether an unconditionally available command can be shown."""
    return True


def _register_command(
    name: str,
    help_key: str,
    *,
    availability: Optional[CommandAvailability] = None,
    warning_keys: tuple[str, ...] = (),
    **help_values: str,
) -> Callable[[CommandHandler], CommandHandler]:
    """Register a handler and its localized help metadata."""

    def register(handler: CommandHandler) -> CommandHandler:
        """Add one command to the registry and return its handler unchanged."""
        if name in _COMMANDS:
            raise ValueError(f"Command {name!r} is already registered")
        _COMMANDS[name] = _Command(
            name=name,
            handler=handler,
            help_key=help_key,
            availability=availability or _always_available,
            help_values=tuple(help_values.items()),
            warning_keys=warning_keys,
        )
        return handler

    return register


def _available_commands(ui: CeluneUI) -> tuple[_Command, ...]:
    """Return registered commands available to the current runtime."""
    return tuple(
        command
        for command in sorted(_COMMANDS.values(), key=lambda item: item.name)
        if command.availability(ui)
    )


def _voice_conversion_available(ui: CeluneUI) -> bool:
    """Return whether the active engine is in voice-conversion mode."""
    return getattr(ui.celune, "input_mode", "text_to_speech") == "voice_conversion"


def _vision_available(ui: CeluneUI) -> bool:
    """Return whether the active engine exposes Persona vision input."""
    return getattr(ui.celune, "vision", None) is not None


def _voice_prompt_available(ui: CeluneUI) -> bool:
    """Return whether the current TTS backend supports voice prompts."""
    return getattr(ui.celune.backend, "name", None) != "mini"


def _qwen3_available(ui: CeluneUI) -> bool:
    """Return whether the active TTS backend is Qwen3."""
    return getattr(ui.celune.backend, "name", None) == "qwen3"


def _refresh_vc_controls(ui: CeluneUI) -> None:
    """Refresh the UI controls that reflect voice-conversion settings."""
    refresh = getattr(ui, "refresh_vc_controls", None)
    if callable(refresh):
        refresh()


def _set_vc_f0_condition(ui: CeluneUI, enabled: bool) -> None:
    """Set Seed-VC F0 conditioning through the UI or active backend."""
    setter = getattr(ui, "set_vc_f0_condition", None)
    if callable(setter):
        setter(enabled)
        return

    ui.celune.vc_f0_condition = enabled
    vc_backend = getattr(ui.celune, "vc_backend", None)
    if vc_backend is not None and available("f0_condition", obj=vc_backend):
        vc_backend.f0_condition = enabled
    _refresh_vc_controls(ui)
    ui.safe_log(
        string(
            "commands.vcmode_set",
            mode=string("ui.vc_mode_sing" if enabled else "ui.vc_mode_talk"),
        )
    )


def _run_runtime_async(
    target: Celune,
    async_name: str,
    sync_name: str,
    *method_args: str,
) -> bool:
    """Run one Celune runtime action, preferring the async implementation."""
    async_method = getattr(target, async_name, None)
    if callable(async_method):
        method = cast(Callable[..., Awaitable[bool]], async_method)

        async def run_method() -> bool:
            return await method(*method_args)

        return asyncio.run(run_method())
    return bool(getattr(target, sync_name)(*method_args))


async def _run_runtime_async_on_loop(
    target: Celune,
    async_name: str,
    sync_name: str,
    *method_args: str,
) -> bool:
    """Run one runtime action on the caller's existing event loop."""
    async_method = getattr(target, async_name, None)
    if callable(async_method):
        method = cast(Callable[..., Awaitable[bool]], async_method)
        return await method(*method_args)
    return bool(await run_in_daemon_thread(getattr(target, sync_name), *method_args))


def _vram_component_label(name: str) -> str:
    """Return the localized display label for one runtime component."""
    category, separator, backend = name.partition("/")
    if category == "tts" and separator:
        return string("commands.vram_text_to_speech", backend=backend)
    if category == "vc" and separator:
        return string("commands.vram_speech_input", backend=backend)
    if category == "speech" and separator:
        return string("commands.vram_speech_input", backend=backend)
    if name == "persona":
        return string("commands.vram_language_model")
    if name == "normalizer":
        return string("commands.vram_normalizer")
    if name == "agent":
        return string("commands.vram_agent")
    return name


def _vram_memory_values(
    allocated: int,
    reserved: int,
    peak: int,
) -> tuple[str, str, str]:
    """Format one component's memory values for the VRAM report."""
    allocated_value = format_vram_bytes(allocated)
    if allocated == reserved == peak:
        not_applicable = string("commands.vram_not_applicable")
        return allocated_value, not_applicable, not_applicable
    return (
        allocated_value,
        format_vram_bytes(reserved),
        format_vram_bytes(peak),
    )


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


def _run_tutorial_sequence(
    ui: CeluneUI,
    token: int,
    steps: tuple[
        tuple[str, Optional[Callable[[], None]], Optional[str]],
        ...,
    ],
) -> None:
    """Speak tutorial sections in sequence and synchronize their UI actions.

    Args:
        ui: UI that owns the tutorial state and actions.
        token: Cancellation token captured when this tutorial began.
        steps: Spoken utterances, actions, and optional caption text overrides.
    """

    def is_active() -> bool:
        return token == ui.tutorial_token() and ui.tutorial_active()

    try:
        from ..pipeline.submission import (
            play_tutorial_sections,
            prepare_tutorial_sections,
        )

        sections = tuple(text for text, _action, _display_text in steps)
        display_sections = tuple(
            text if display_text is None else display_text
            for text, _action, display_text in steps
        )
        audio_sections = prepare_tutorial_sections(
            ui.celune,
            sections,
            is_active,
            _TUTORIAL_SPEECH_TIMEOUT * max(1, len(sections)),
            display_sections,
        )
        if not is_active():
            return
        if audio_sections is None:
            raise TimeoutError("tutorial speech preparation did not finish")

        def start_section(index: int) -> None:
            action = steps[index][1]
            if action is not None:
                ui.call_from_thread(action)

        if not play_tutorial_sections(
            ui.celune,
            sections,
            audio_sections,
            is_active,
            start_section,
            _TUTORIAL_SPEECH_TIMEOUT,
            display_sections=display_sections,
        ):
            if not is_active():
                return
            raise TimeoutError("tutorial speech did not finish")
        if ui.celune.cur_state == "error":
            raise RuntimeError("tutorial speech playback failed")
        if is_active():
            ui.call_from_thread(ui.finish_tutorial)
    except Exception as exc:
        if not is_active():
            return
        ui.safe_log(
            format_error_message(
                string("commands.tutorial_failed"),
                exc,
                getattr(ui.celune, "log_level", "info"),
            ),
            "warning",
        )
        ui.call_from_thread(ui.cancel_tutorial, True)


def tutorial(ui: CeluneUI) -> None:
    """Speak the localized tutorial through the active voice.

    Args:
        ui: The instance of CeluneUI that the tutorial will interact with.
    """

    def send_help() -> None:
        """Submit the tutorial help command after its typing animation."""
        ui.type_and_send("/help", process_commands=True)

    steps = (
        (string("commands.tutorial_intro"), None, None),
        (string("commands.tutorial_input"), lambda: ui.pulse_border("#input"), None),
        (string("commands.tutorial_voice"), lambda: ui.pulse_border("#style"), None),
        (string("commands.tutorial_help"), send_help, None),
        (string("commands.tutorial_help_simple"), None, None),
        (string("commands.tutorial_help_vibe"), None, None),
        (string("commands.tutorial_extensions"), None, None),
        (string("commands.tutorial_extension_example"), None, None),
        (string("commands.tutorial_extension_code"), None, None),
        (string("commands.tutorial_local_api"), None, None),
        (string("commands.tutorial_local_api_usage"), None, None),
        (string("commands.tutorial_voice_self", app_name=APP_NAME), None, None),
        (string("commands.tutorial_voice_default", app_name=APP_NAME), None, None),
        (
            string("commands.tutorial_voice_pack"),
            None,
            string("commands.tutorial_voice_pack_display"),
        ),
        (string("commands.tutorial_voice_pack_continued"), None, None),
        (string("commands.tutorial_persona"), None, None),
        (string("commands.tutorial_persona_chat"), None, None),
        (string("commands.tutorial_persona_speech"), None, None),
        (string("commands.tutorial_persona_invitation"), None, None),
        (string("commands.tutorial_agent"), None, None),
        (string("commands.tutorial_agent_abilities"), None, None),
        (string("commands.tutorial_agent_actions"), None, None),
        (string("commands.tutorial_can_do_more"), None, None),
        (string("commands.tutorial_variety", app_name=APP_NAME), None, None),
        (string("commands.tutorial_variety_many"), None, None),
        (string("commands.tutorial_supported"), None, None),
        (string("commands.tutorial_wrap_up"), None, None),
        (string("commands.tutorial_wait"), None, None),
    )

    ui.begin_tutorial()
    threading.Thread(
        target=_run_tutorial_sequence,
        args=(ui, ui.tutorial_token(), steps),
        daemon=True,
    ).start()


@_register_command("help", "commands.help_help")
def _handle_help(ui: CeluneUI, _args: list[str]) -> None:
    """Show localized help for commands available in the active runtime."""
    ui.safe_log(string("commands.help_header", app_name=APP_NAME))
    ui.safe_log(string("commands.help_available"))
    ui.safe_log(string("commands.help_arguments"))
    for definition in _available_commands(ui):
        help_text = string(definition.help_key, **dict(definition.help_values))
        ui.safe_log(f"  {help_text}")
        for warning_key in definition.warning_keys:
            ui.safe_log(string(warning_key), "warning")

    if getattr(ui.celune.backend, "singing", False):
        ui.safe_log(string("commands.help_cesn_header"))
        ui.safe_log(string("commands.help_cesn_order"))
        ui.safe_log(string("commands.help_cesn_language"))
        ui.safe_log(string("commands.help_cesn_music"))
        ui.safe_log(string("commands.help_cesn_notes"))
        ui.safe_log(string("commands.help_cesn_rest"))
        ui.safe_log(string("commands.help_cesn_ipa"))
        ui.safe_log(string("commands.help_cesn_slide"))
        ui.safe_log(string("commands.help_cesn_slur"))
        ui.safe_log(string("commands.help_cesn_tuplet"))
        ui.safe_log(string("commands.help_cesn_singer"))
        ui.safe_log(string("commands.help_cesn_tutorial"))


@_register_command("vram", "commands.help_vram")
def _handle_vram(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /vram command."""
    speech_transcriber = getattr(ui, "_speech_transcriber", None)
    speech_model = getattr(speech_transcriber, "loaded_model", None)
    report = runtime_vram_report(
        ui.celune,
        {"speech/whisper": speech_model},
    )
    ui.safe_log(string("commands.vram_header"))
    components = report.get("components")
    entries: list[tuple[str, bool, int, int, int, str]] = []
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            raw_name = component.get("name")
            if not isinstance(raw_name, str) or component.get("loaded") is not True:
                continue
            name = _vram_component_label(raw_name)
            component_available = component.get("available") is True
            device_value = component.get("device")
            device = (
                device_value
                if isinstance(device_value, str)
                else string("commands.vram_device_unknown")
            )
            entries.append(
                (
                    name,
                    component_available,
                    vram_report_int(component, "allocated_bytes"),
                    vram_report_int(component, "reserved_bytes"),
                    vram_report_int(component, "peak_allocated_bytes"),
                    device,
                )
            )

    label_width = max((len(f"{name}:") for name, *_ in entries), default=0)
    for name, component_available, allocated, reserved, peak, device in entries:
        padded_name = f"{name}:".ljust(label_width)
        if not component_available:
            ui.safe_log(string("commands.vram_component_unavailable", name=padded_name))
            continue
        allocated_value, reserved_value, peak_value = _vram_memory_values(
            allocated,
            reserved,
            peak,
        )
        ui.safe_log(
            string(
                "commands.vram_component",
                name=padded_name,
                allocated=allocated_value,
                reserved=reserved_value,
                peak=peak_value,
                device=device,
            )
        )
    ui.safe_log(string("commands.vram_legend"))


@_register_command("settings", "commands.help_settings")
def _handle_settings(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /settings command."""
    open_settings = getattr(ui, "open_settings_menu", None)
    if callable(open_settings):
        open_settings()
    else:
        ui.safe_log(string("commands.settings_unavailable"), "warning")


@_register_command("restartaudio", "commands.help_restart_audio")
def _handle_restartaudio(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /restartaudio command."""

    def restart_audio() -> None:
        try:
            restart_audio_server()
        except RuntimeError:
            ui.safe_log(string("commands.audio_restart_failed"), "error")
            return

        celune = getattr(ui, "celune", None)
        audio_test_succeeded = False
        if celune is not None:
            close_stream(celune, abort=True)
            try:
                test_audio = getattr(celune, "test_audio", None)
                audio_test_succeeded = callable(test_audio) and bool(test_audio())
            except Exception:
                audio_test_succeeded = False
            finally:
                close_stream(celune, abort=True)

        try_play_signal = getattr(celune, "try_play_signal", None)
        if not audio_test_succeeded:
            ui.safe_log(string("commands.audio_restart_test_failed"), "warning")

        signal_succeeded = callable(try_play_signal) and try_play_signal("readiness")
        if not signal_succeeded:
            ui.safe_log(string("commands.audio_restart_signal_failed"), "warning")
            return
        if not audio_test_succeeded:
            return

        ui.safe_log(string("commands.audio_restart_success"))

    threading.Thread(
        target=restart_audio,
        name="celune-audio-restart",
        daemon=True,
    ).start()


@_register_command(
    "consumebuffer",
    "commands.help_consumebuffer",
    warning_keys=("commands.help_consumebuffer_caution",),
    app_name=APP_NAME,
)
def _handle_consumebuffer(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /consumebuffer command."""
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
        string("commands.invalid_true_false_argument", command="consumebuffer"),
        "warning",
    )
    return


@_register_command("invoke", "commands.help_invoke", app_name=APP_NAME)
def _handle_invoke(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /invoke command."""
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
    except Exception as error:
        ui.safe_log(
            format_error_message(
                tagged_string("commands.extension_error", "EXT ERROR"),
                error,
                getattr(ui.celune, "log_level", "info"),
            ),
            "error",
        )

    return


@_register_command("extensions", "commands.help_extensions", app_name=APP_NAME)
def _handle_extensions(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /extensions command."""
    if not ui.celune.extension_manager:
        ui.safe_log(string("commands.extension_system_not_initialized"), "warning")
        return

    names = ui.celune.extension_manager.list_extensions()
    if not names:
        ui.safe_log(string("commands.no_extensions_loaded"), "warning")
    else:
        ui.safe_log(string("commands.extensions_list", names=", ".join(names)))
    return


@_register_command(
    "voiceprompt",
    "commands.help_voiceprompt",
    availability=_voice_prompt_available,
    warning_keys=("commands.help_voiceprompt_caution",),
    app_name=APP_NAME,
)
def _handle_voiceprompt(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /voiceprompt command."""
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


@_register_command("speed", "commands.help_speed")
def _handle_speed(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /speed command."""
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


@_register_command("reverb", "commands.help_reverb")
def _handle_reverb(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /reverb command."""
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


@_register_command("backend", "commands.help_backend")
def _handle_backend(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /backend command."""
    if not args:
        ui.safe_log(string("commands.usage_backend"), "warning")
        return

    backend_name = args[0]

    active_backend_name = active_runtime_backend_name(ui.celune)
    if backend_name == active_backend_name:
        ui.safe_log(string("commands.backend_already_loaded"), "warning")
        return

    def backend_worker() -> None:
        try:
            if _run_runtime_async(
                ui.celune,
                "set_backend_async",
                "set_backend_and_wait",
                backend_name,
            ):
                ui.celune.try_play_signal("readiness")
                ui.safe_log(
                    string("commands.backend_switched", backend_name=backend_name)
                )
            else:
                ui.safe_log(string("commands.backend_not_switched"), "warning")
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.backend_switch_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    async def backend_worker_async() -> None:
        try:
            if await _run_runtime_async_on_loop(
                ui.celune,
                "set_backend_async",
                "set_backend_and_wait",
                backend_name,
            ):
                ui.celune.try_play_signal("readiness")
                ui.safe_log(
                    string("commands.backend_switched", backend_name=backend_name)
                )
            else:
                ui.safe_log(string("commands.backend_not_switched"), "warning")
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.backend_switch_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(target=backend_worker, daemon=True).start()
    else:
        asyncio.create_task(backend_worker_async())
    return


@_register_command("cevoice", "commands.help_cevoice")
def _handle_cevoice(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /cevoice command."""
    if not args:
        ui.safe_log(string("commands.usage_cevoice"), "warning")
        return

    bundle = args[0]
    if resolve_bundle_path(bundle) == active_bundle_path():
        ui.safe_log(string("commands.character_already_loaded"), "warning")
        return

    def cevoice_worker() -> None:
        try:
            if _run_runtime_async(
                ui.celune,
                "set_cevoice_async",
                "set_cevoice_and_wait",
                bundle,
            ):
                ui.safe_log(string("commands.character_changed", bundle=bundle))
            else:
                ui.safe_log(
                    string("commands.character_not_switched", bundle=bundle),
                    "warning",
                )
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.character_switch_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    async def cevoice_worker_async() -> None:
        try:
            if await _run_runtime_async_on_loop(
                ui.celune,
                "set_cevoice_async",
                "set_cevoice_and_wait",
                bundle,
            ):
                ui.safe_log(string("commands.character_changed", bundle=bundle))
            else:
                ui.safe_log(
                    string("commands.character_not_switched", bundle=bundle),
                    "warning",
                )
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.character_switch_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(target=cevoice_worker, daemon=True).start()
    else:
        asyncio.create_task(cevoice_worker_async())
    return


@_register_command("vc", "commands.help_vc", availability=_voice_conversion_available)
def _handle_vc(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /vc command."""
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
                format_error_message(
                    string("commands.vc_decode_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
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


@_register_command(
    "vcmode", "commands.help_vcmode", availability=_voice_conversion_available
)
def _handle_vcmode(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /vcmode command."""
    if getattr(ui.celune, "input_mode", "text_to_speech") != "voice_conversion":
        ui.safe_log(string("commands.voice_conversion_only"), "warning")
        return

    if not args:
        ui.safe_log(string("commands.usage_vcmode"), "warning")
        return

    mode = args[0].lower()
    if mode == "talk":
        _set_vc_f0_condition(ui, False)
        return
    if mode == "sing":
        _set_vc_f0_condition(ui, True)
        return

    ui.safe_log(string("commands.usage_vcmode"), "warning")
    return


@_register_command(
    "vcdebug", "commands.help_vcdebug", availability=_voice_conversion_available
)
def _handle_vcdebug(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /vcdebug command."""
    if getattr(ui.celune, "input_mode", "text_to_speech") != "voice_conversion":
        ui.safe_log(string("commands.voice_conversion_only"), "warning")
        return

    if len(args) != 1 or args[0].lower() not in {"on", "off"}:
        ui.safe_log(string("commands.usage_vcdebug"), "warning")
        return

    if args[0].lower() == "on":
        if getattr(ui, "_vc_debug_capture", None) is not None:
            ui.safe_log(string("commands.vc_debug_already_enabled"), "warning")
            return
        try:
            from ..vc import VCDebugCapture

            capture = VCDebugCapture()
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.vc_debug_start_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )
            return
        ui._vc_debug_capture = capture
        ui.safe_log(string("commands.vc_debug_started"))
        return

    if getattr(ui, "_vc_debug_capture", None) is None:
        ui.safe_log(string("commands.vc_debug_not_enabled"), "warning")
        return
    ui.safe_log(string("commands.vc_debug_stopping"))
    ui.stop_vc_debug_capture(announce=True)
    return


@_register_command(
    "xvectoronly", "commands.help_xvectoronly", availability=_qwen3_available
)
def _handle_xvectoronly(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /xvectoronly command."""
    backend = ui.celune.backend
    # uses the backend's name, preventing a core dependency on Qwen
    if backend.name != "qwen3":
        ui.safe_log(string("commands.xvectoronly_qwen3_only"), "warning")
        return

    if not args:
        ui.safe_log(string("commands.usage_xvectoronly"), "warning")
        return

    value = args[0].lower()
    if value not in {"true", "false"}:
        ui.safe_log(
            string("commands.invalid_true_false_argument", command="xvectoronly"),
            "warning",
        )
        return

    if available("x_vector_only", obj=backend):
        backend.x_vector_only = value == "true"
        state = string(
            "commands.state_enabled"
            if backend.x_vector_only
            else "commands.state_disabled"
        )
        ui.safe_log(string("commands.qwen3_identity_only_cloning", state=state))
        return
    ui.safe_log(string("commands.xvectoronly_unavailable"), "warning")
    return


@_register_command("play", "commands.help_play")
def _handle_play(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /play command."""
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
                    string("commands.invalid_volume", command="play"),
                    "warning",
                )
                return

        def worker() -> None:
            try:
                is_youtube = args[0].startswith("https://")

                def report_playback_started() -> None:
                    if is_youtube:
                        ui.safe_log(
                            string(
                                "commands.playing_youtube_audio",
                                volume=format_number(volume * 100),
                            )
                        )

                if is_youtube:
                    if not ui.celune.play(
                        args[0],
                        volume=volume,
                        on_started=report_playback_started,
                    ):
                        return
                elif not ui.celune.play(args[0], volume=volume):
                    return
                if not is_youtube:
                    ui.safe_log(
                        string(
                            "commands.playing_audio",
                            path=args[0],
                            volume=format_number(volume * 100),
                        )
                    )
            except Exception as exc:
                ui.safe_log(
                    format_error_message(
                        string("commands.cannot_play_audio"),
                        exc,
                        getattr(ui.celune, "log_level", "info"),
                    ),
                    "error",
                )

        threading.Thread(target=worker, daemon=True).start()
    except Exception as e:
        ui.safe_log(
            format_error_message(
                string("commands.cannot_play_file"),
                e,
                getattr(ui.celune, "log_level", "info"),
            ),
            "error",
        )
        return
    return


@_register_command("attach", "commands.help_attach", availability=_vision_available)
def _handle_attach(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /attach command."""
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
    get_capabilities = getattr(vision, "capabilities", None)
    if callable(get_capabilities):
        capabilities = get_capabilities()
        if (
            isinstance(capabilities, PersonaCapabilities)
            and not capabilities.image_uploads
        ):
            ui.safe_log(
                string(
                    "commands.attachments_speech_only_mode",
                    app_name=APP_NAME,
                ),
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


@_register_command("say", "commands.help_say", availability=_vision_available)
def _handle_say(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /say command."""
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
            ui.safe_log(
                string("commands.unmatched_ipa", count=unmatched),
                "warning",
                loglevel="verbose",
            )

        ui.celune.say(ipa_decoded, display_text=raw_text)
    else:
        ui.celune.say(raw_text)

    return


@_register_command("seed", "commands.help_seed")
def _handle_seed(ui: CeluneUI, args: list[str]) -> None:
    """Handle the /seed command."""
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


@_register_command("tutorial", "commands.help_tutorial", app_name=APP_NAME)
def _handle_tutorial(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /tutorial command."""
    singing = getattr(ui.celune.backend, "singing", False)
    if singing:
        ui.safe_log(string("commands.tutorial_unavailable_singing"))
        return

    ui.safe_log(string("commands.tutorial_activated", app_name=APP_NAME))
    tutorial(ui)
    return


@_register_command("stop", "commands.help_stop")
def _handle_stop(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /stop command."""

    def stop_worker() -> None:
        try:
            if not _run_runtime_async(
                ui.celune,
                "force_stop_speech_async",
                "force_stop_speech",
            ):
                ui.safe_log(string("commands.nothing_to_stop"))
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.stop_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    async def stop_worker_async() -> None:
        try:
            if not await _run_runtime_async_on_loop(
                ui.celune,
                "force_stop_speech_async",
                "force_stop_speech",
            ):
                ui.safe_log(string("commands.nothing_to_stop"))
        except Exception as exc:
            ui.safe_log(
                format_error_message(
                    string("commands.stop_failed"),
                    exc,
                    getattr(ui.celune, "log_level", "info"),
                ),
                "error",
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        threading.Thread(target=stop_worker, daemon=True).start()
    else:
        asyncio.create_task(stop_worker_async())


@_register_command("exit", "commands.help_exit", app_name=APP_NAME)
def _handle_exit(ui: CeluneUI, _args: list[str]) -> None:
    """Handle the /exit command."""
    ui.graceful_exit()


def process_command(ui: CeluneUI, command: str, args: list[str]) -> None:
    """Dispatch a slash command registered for Celune's active UI.

    Args:
        ui: The instance of CeluneUI that owns the command context.
        command: The slash-command name without its leading slash.
        args: The parsed command arguments.
    """
    ui.input_box.load_text("")
    operation_state = getattr(ui.celune, "operation_state", OperationState.NORMAL)
    if operation_state is OperationState.SEMI_FATAL and command != "settings":
        ui.safe_log(string("operation_state.recovery_settings_only"), "warning")
        return
    if operation_state in {OperationState.FATAL, OperationState.STOPPED}:
        ui.safe_log(string("operation_state.unavailable"), "warning")
        return

    definition = _COMMANDS.get(command)
    if definition is None:
        ui.safe_log(string("commands.unknown_command", command=command), "warning")
        return
    definition.handler(ui, args)
