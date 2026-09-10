# SPDX-License-Identifier: Apache-2.0
"""WebUI action helpers."""

from __future__ import annotations

import io
import json
import textwrap
from collections.abc import Awaitable, Callable, Iterator
from typing import Optional, cast

import gradio as gr
import numpy as np
import soundfile as sf
from fastapi.responses import JSONResponse

from . import api as _api
from .constants import APP_NAME, BASE_SR
from .i18n import string, tagged_string
from .persona.impl import persona_talkback_enabled
from .speech import prepare_playback_audio
from .typing.aliases import AudioChunk, AudioChunks
from .typing.api import WebUiAudioValue, WebUiInputAudioValue, WebUiUpdate
from .ui.app import CeluneUI
from .vc import VC_PITCH_SHIFT_MAX, VC_PITCH_SHIFT_MIN
from .webui import (
    _append_webui_error,
    _append_webui_log,
    _normalized_audio,
    _set_webui_status,
    _webui_audio_waveform_options,
    _webui_logs_html,
    _webui_resources_html,
    _webui_snapshot,
    _webui_status_html,
    _webui_submit_snapshot,
    _webui_vc_controls_update,
    _webui_vc_mode_active,
)
from .binding import install_module_functions

__all__ = (
    "_build_webui",
    "_decode_uploaded_audio",
    "_normalize_webui_audio_input",
    "_voice_conversion_unavailable_response",
    "_webui_convert_audio",
    "_webui_cycle_voice",
    "_webui_record_button_update",
    "_webui_run_command",
    "_webui_select_voice",
    "_webui_speak",
    "_webui_stop",
    "_webui_stop_button_update",
    "_webui_toggle_recording",
    "_webui_voice_catalog",
    "_webui_voice_choices",
)


def _webui_run_command(text: str) -> bool:
    """Run one slash command through the main UI command path when available."""
    try:
        parts = CeluneUI.split_command_input(text[1:])
    except ValueError:
        _append_webui_log(string("webui.command_parsing_error"), "error")
        return False

    if not parts:
        return False

    command = parts[0].lower()
    command_args = parts[1:]
    celune = _api.bound_celune
    if command == "settings" and celune is not None:
        from .ui.commands import process_command as process_ui_command

        process_ui_command(
            cast(CeluneUI, _api._WebUiCommandHost(celune)), command, command_args
        )
        return True

    # noinspection PyProtectedMember
    ui = CeluneUI._instance
    if ui is not None:
        ui.call_from_thread(ui.process_command, command, command_args)
        return True

    if celune is None:
        _append_webui_log(string("webui.not_available"), "error")
        return False
    from .ui.commands import process_command as process_ui_command

    process_ui_command(
        cast(CeluneUI, _api._WebUiCommandHost(celune)), command, command_args
    )
    return True


def _decode_uploaded_audio(
    data: bytes,
) -> tuple[AudioChunk, int]:
    """Decode uploaded audio bytes into float32 audio and a source sample rate."""
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    return np.asarray(audio, dtype=np.float32), sample_rate


def _normalize_webui_audio_input(
    source_audio: WebUiInputAudioValue,
) -> WebUiAudioValue:
    """Normalize one Gradio audio value to Celune's float32 waveform contract."""
    if source_audio is None:
        return None

    sample_rate, audio = source_audio
    normalized = np.asarray(audio)
    if normalized.dtype == np.int16:
        normalized = normalized.astype(np.float32) / 32768.0
    else:
        normalized = normalized.astype(np.float32, copy=False)

    return sample_rate, np.ascontiguousarray(normalized, dtype=np.float32)


def _voice_conversion_unavailable_response() -> JSONResponse:
    """Return a standard API error for VC-only endpoints in TTS mode."""
    return JSONResponse(
        status_code=409,
        content={
            "error": "wrong_mode",
            "message": string("webui.wrong_mode"),
        },
    )


def _webui_speak(
    content: str,
) -> Iterator[
    tuple[
        WebUiUpdate,
        WebUiAudioValue,
        str,
        str,
        str,
        WebUiUpdate,
        WebUiUpdate,
    ]
]:
    """Speak text through the browser UI and return browser audio playback."""
    text = content.strip()
    if not text:
        snapshot = _webui_submit_snapshot("")
        yield snapshot[0], None, *snapshot[1:]
        return

    if text.startswith("/"):
        _webui_run_command(text)
        snapshot = _webui_submit_snapshot("")
        yield snapshot[0], None, *snapshot[1:]
        return

    celune = _api.require_celune()
    if _webui_vc_mode_active(celune):
        _append_webui_log(string("webui.text_unavailable_in_vc_mode"), "warning")
        snapshot = _webui_submit_snapshot(text)
        yield snapshot[0], None, *snapshot[1:]
        return
    _api.api_log("SPEAK(WEBUI)", text)

    current_state = (celune.cur_state or "").strip().lower()
    if current_state == "waking":
        _append_webui_log(
            string("webui.not_returned_from_sleep", app_name=APP_NAME), "warning"
        )
        snapshot = _webui_submit_snapshot(text)
        yield snapshot[0], None, *snapshot[1:]
        return

    if getattr(celune, "sleeping", False):
        _set_webui_status(string("status.waking_up"))
        snapshot = _webui_submit_snapshot(text)
        yield snapshot[0], None, *snapshot[1:]
        wake_async = cast(
            Optional[Callable[[], Awaitable[bool]]],
            getattr(celune, "wake_from_sleep_async", None),
        )
        if wake_async is not None:
            woke = bool(_api._run_async_runtime_call(wake_async()))
        else:
            woke = celune.wake_from_sleep()
        if not woke:
            snapshot = _webui_submit_snapshot(text)
            yield snapshot[0], None, *snapshot[1:]
            return

    if persona_talkback_enabled(getattr(celune, "config", {})):
        if not celune.think(text):
            _append_webui_log(string("webui.busy_try_again"), "warning")
        snapshot = _webui_submit_snapshot("")
        yield snapshot[0], None, *snapshot[1:]
        return

    chunks = celune.say_stream(text, save=True)
    if chunks is None:
        _append_webui_log(string("webui.busy_try_again"), "warning")
        snapshot = _webui_submit_snapshot(text)
        yield snapshot[0], None, *snapshot[1:]
        return

    snapshot = _webui_submit_snapshot("")
    yield snapshot[0], None, *snapshot[1:]

    audio_chunks: AudioChunks = []

    try:
        while True:
            item = chunks.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            audio_chunks.append(_normalized_audio(item))
        audio_value: WebUiAudioValue
        if audio_chunks:
            audio_value = (BASE_SR, np.concatenate(audio_chunks))
        else:
            audio_value = None
        snapshot = _webui_submit_snapshot("")
        yield snapshot[0], audio_value, *snapshot[1:]
    except Exception as error:
        _append_webui_error(
            tagged_string("webui.error", "WEBUI ERROR"),
            error,
            celune=celune,
        )
        snapshot = _webui_submit_snapshot("")
        yield snapshot[0], None, *snapshot[1:]


def _webui_convert_audio(
    source_audio: WebUiInputAudioValue,
    pitch_shift: float = 0.0,
    conversion_mode: str = "talk",
) -> tuple[
    WebUiInputAudioValue,
    WebUiAudioValue,
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Convert uploaded audio through the active VC backend for browser playback."""
    if source_audio is None:
        _append_webui_log(
            string("webui.upload_audio_first"),
            "warning",
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            _webui_snapshot()
        )
        return (
            None,
            None,
            logs_html,
            status_html,
            resources_html,
            voice_update,
            send_update,
        )

    celune = _api.require_celune()
    if not _webui_vc_mode_active(celune):
        _append_webui_log(
            string("webui.conversion_only_in_vc_mode"),
            "warning",
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            _webui_snapshot()
        )
        return (
            source_audio,
            None,
            logs_html,
            status_html,
            resources_html,
            voice_update,
            send_update,
        )

    normalized_source_audio = _normalize_webui_audio_input(source_audio)
    assert normalized_source_audio is not None
    sample_rate, audio = normalized_source_audio
    _api.api_log("CONVERT(WEBUI)", "uploaded audio")
    try:
        output = celune.convert_audio(
            audio,
            sample_rate,
            label="browser audio input",
            pitch_shift=round(pitch_shift),
            f0_condition=conversion_mode.strip().lower() == "sing",
        )
    except Exception as error:
        _append_webui_error(
            tagged_string("webui.error", "WEBUI ERROR"),
            error,
            celune=celune,
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            _webui_snapshot()
        )
        return (
            source_audio,
            None,
            logs_html,
            status_html,
            resources_html,
            voice_update,
            send_update,
        )

    if output is None:
        _append_webui_log(string("webui.cannot_convert_right_now"), "warning")
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            _webui_snapshot()
        )
        return (
            source_audio,
            None,
            logs_html,
            status_html,
            resources_html,
            voice_update,
            send_update,
        )

    prepared_audio = prepare_playback_audio(output.audio, output.sample_rate)
    converted_audio = (BASE_SR, prepared_audio)
    logs_html, status_html, resources_html, voice_update, send_update, _input = (
        _webui_snapshot()
    )
    return (
        None,
        converted_audio,
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
    )


def _webui_cycle_voice() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Cycle to the next available Celune voice from the browser UI."""
    celune = _api.require_celune()
    if len(celune.voices) < 2 or not bool(celune.current_voice or celune.voices):
        return _webui_snapshot()

    current_voice = celune.current_voice or celune.voices[0]
    current_index = (
        celune.voices.index(current_voice) if current_voice in celune.voices else -1
    )
    next_voice = celune.voices[(current_index + 1) % len(celune.voices)]
    _api.api_log("VOICE(WEBUI)", next_voice)

    set_voice_async = cast(
        Optional[Callable[[str], Awaitable[bool]]],
        getattr(celune, "set_voice_async", None),
    )
    if set_voice_async is not None:
        switched = bool(_api._run_async_runtime_call(set_voice_async(next_voice)))
    else:
        switched = celune.set_voice_and_wait(next_voice)

    if not switched:
        _append_webui_log(string("webui.cannot_change_voice_right_now"), "error")

    return _webui_snapshot()


def _webui_voice_catalog(celune: _api.Celune) -> tuple[tuple[str, str], ...]:
    """Return readable browser choices for every available voice-pack entry."""
    from .cevoice import (
        CEVoice,
        CEVoiceError,
        bundled_voices_dir,
        bundle_character_name,
    )

    try:
        pack_paths = sorted(
            path
            for path in bundled_voices_dir().iterdir()
            if path.is_file() and path.suffix.casefold() in {".cevoice", ".cechar"}
        )
    except OSError:
        pack_paths = []

    choices: list[tuple[str, str]] = []
    used_pack_names: set[str] = set()
    for path in pack_paths:
        try:
            bundle = CEVoice.open(path)
        except (OSError, CEVoiceError):
            continue
        pack_name = bundle_character_name(bundle) or path.stem
        if pack_name in used_pack_names:
            pack_name = f"{pack_name} ({path.stem})"
        used_pack_names.add(pack_name)
        for voice_entry in bundle.voice_order:
            value = json.dumps(
                {"bundle": str(path), "entry": voice_entry},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            choices.append((f"{pack_name}: {voice_entry}", value))

    if choices:
        return tuple(choices)

    return tuple(
        (
            str(voice),
            json.dumps(
                {"entry": voice},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        for voice in getattr(celune, "voices", ())
    )


def _webui_voice_choices() -> WebUiUpdate:
    """Return the current readable voice list for the browser selector."""
    celune = _api.bound_celune
    if celune is None:
        return gr.update(choices=[], value=None, interactive=False)
    from .cevoice import active_bundle_path

    choices = _webui_voice_catalog(celune)
    current_voice = getattr(celune, "current_voice", None)
    active_bundle = str(active_bundle_path())
    current = None
    for _label, value in choices:
        try:
            selection = json.loads(value)
        except json.JSONDecodeError:
            continue
        if selection.get("entry") == current_voice and (
            not selection.get("bundle") or selection["bundle"] == active_bundle
        ):
            current = value
            break
    if current is None and choices:
        current = choices[0][1]
    return gr.update(
        choices=list(choices),
        value=current,
        interactive=not _api.webui_input_locked and len(choices) >= 1,
    )


def _webui_select_voice(
    name: Optional[str],
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Select one voice through the same asynchronous core switch as the TUI."""
    celune = _api.bound_celune
    if celune is None or not name:
        return _webui_snapshot()
    try:
        selection = json.loads(name)
    except json.JSONDecodeError:
        selection = {"entry": name}
    entry = selection.get("entry") if isinstance(selection, dict) else None
    bundle = selection.get("bundle") if isinstance(selection, dict) else None
    if not isinstance(entry, str) or (
        bundle is not None and not isinstance(bundle, str)
    ):
        _append_webui_log(string("webui.cannot_change_voice_right_now"), "error")
        return _webui_snapshot()

    set_voice_async = cast(
        Optional[Callable[[str], Awaitable[bool]]],
        getattr(celune, "set_voice_async", None),
    )
    set_bundle_async = cast(
        Optional[Callable[[str], Awaitable[bool]]],
        getattr(celune, "set_cevoice_async", None),
    )

    async def select_voice() -> bool:
        if (
            bundle is not None
            and set_bundle_async is not None
            and not await set_bundle_async(bundle)  # pylint: disable=not-callable
        ):
            return False
        if set_voice_async is not None:
            return await set_voice_async(entry)
        if bundle is not None and not celune.set_cevoice_and_wait(bundle):
            return False
        return celune.set_voice_and_wait(entry)

    switched = _api._run_async_runtime_call(select_voice())
    if not switched:
        _append_webui_log(string("webui.cannot_change_voice_right_now"), "error")
    return _webui_snapshot()


def _webui_stop() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Stop the active speech request through the shared runtime lifecycle."""
    celune = _api.bound_celune
    if celune is None:
        return _webui_snapshot()
    ui = CeluneUI._instance
    if ui is not None:
        recording = (
            ui._vc_recording_active
            if _webui_vc_mode_active(celune)
            else ui._persona_recording_active
        )
        if callable(recording) and recording():
            toggle = (
                ui.toggle_vc_recording
                if _webui_vc_mode_active(celune)
                else ui.toggle_persona_recording
            )
            ui.call_from_thread(toggle)
            return _webui_snapshot()
    stop_async = cast(
        Optional[Callable[[], Awaitable[bool]]],
        getattr(celune, "force_stop_speech_async", None),
    )
    if stop_async is not None:
        stopped = bool(_api._run_async_runtime_call(stop_async()))
    else:
        stop_sync = getattr(celune, "force_stop_speech", None)
        stopped = (
            cast(Callable[[], bool], stop_sync)()  # pylint: disable=not-callable
            if callable(stop_sync)
            else False
        )
    if stopped:
        _set_webui_status(string("status.stopped"), "sleeping", source="callback")
    else:
        _append_webui_log(string("commands.nothing_to_stop"), "warning")
    return _webui_snapshot()


def _webui_stop_button_update() -> WebUiUpdate:
    """Return whether the browser stop control should be interactive."""
    celune = _api.bound_celune
    if celune is None:
        return gr.update(interactive=False)
    state = str(getattr(celune, "cur_state", "")).casefold()
    return gr.update(
        interactive=state in {"speaking", "generating", "thinking"}
        or _api.active_speech_task_id is not None
    )


def _webui_record_button_update() -> WebUiUpdate:
    """Return whether the browser can delegate live capture to the TUI runtime."""
    celune = _api.bound_celune
    ui = CeluneUI._instance
    if celune is None or ui is None or _api.webui_input_locked:
        return gr.update(interactive=False)
    if getattr(celune, "is_in_tutorial", False):
        return gr.update(interactive=False)
    return gr.update(interactive=True)


def _webui_toggle_recording() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Toggle the same Persona or live VC capture path used by ``CTRL+R``."""
    celune = _api.bound_celune
    ui = CeluneUI._instance
    if celune is None or ui is None:
        _append_webui_log(string("webui.recording_requires_tui"), "warning")
        return _webui_snapshot()
    if getattr(celune, "sleeping", False):
        ui.call_from_thread(ui.wake_from_sleep)
        return _webui_snapshot()
    if getattr(celune, "cur_state", "") == "waking":
        return _webui_snapshot()

    toggle = (
        ui.toggle_vc_recording
        if _webui_vc_mode_active(celune)
        else ui.toggle_persona_recording
    )
    ui.call_from_thread(toggle)
    return _webui_snapshot()


def _build_webui() -> gr.Blocks:
    # pylint: disable=E1101
    """Create the browser UI mounted by the API."""
    _api._configure_webui_theme()
    with gr.Blocks(
        title=APP_NAME,
        fill_height=True,
    ) as demo:
        gr.HTML(_api.webui_theme_style)
        with gr.Column(elem_id="celune-shell"):
            with gr.Tabs():
                with gr.Tab(string("webui.tts_tab_label")):
                    gr.HTML(
                        textwrap.dedent(
                            f"""
                            <div id="celune-header">
                                <div class="line"></div>
                                <div class="title">{APP_NAME}</div>
                                <div class="line"></div>
                            </div>
                            """
                        )
                    )
                    logs = gr.HTML(_webui_logs_html())
                    with gr.Row(elem_id="celune-input-row"):
                        input_box = gr.Textbox(
                            value="",
                            lines=1,
                            max_lines=4,
                            show_label=False,
                            placeholder=string("ui.wait_placeholder"),
                            container=False,
                            elem_id="celune-input",
                            scale=8,
                            interactive=False,
                        )
                        with gr.Row(elem_id="celune-actions", scale=2):
                            voice_button = gr.Button(
                                value=string("webui.default_voice_button"),
                                elem_id="celune-style",
                                scale=1,
                                min_width=0,
                                interactive=False,
                            )
                            send_button = gr.Button(
                                value=string("webui.send_button"),
                                elem_id="celune-send",
                                scale=1,
                                min_width=0,
                                interactive=False,
                            )
                    record_hotkey = gr.Button(
                        value="",
                        elem_id="celune-record-hotkey",
                        visible=True,
                        interactive=False,
                    )
                    with gr.Row(elem_id="celune-footer"):
                        status = gr.HTML(_webui_status_html(), elem_id="celune-status")
                        resources = gr.HTML(
                            _webui_resources_html(),
                            elem_id="celune-resources",
                        )
                    gr.HTML(
                        textwrap.dedent(f"""
                            <p style="color: var(--celune-primary); text-align: center;">
                                {string("webui.features_may_differ", app_name=APP_NAME)}
                            </p>
                        """)
                    )
                with (
                    gr.Tab(string("webui.vc_tab_label")),
                    gr.Column(elem_id="celune-convert-panel"),
                ):
                    source_audio = gr.Audio(
                        value=None,
                        type="numpy",
                        sources=["upload", "microphone"],
                        autoplay=False,
                        show_label=True,
                        label=string("webui.source_audio_label"),
                        interactive=False,
                        waveform_options=_webui_audio_waveform_options(),
                        elem_id="celune-source-audio",
                    )
                    vc_pitch_shift = gr.Slider(
                        minimum=VC_PITCH_SHIFT_MIN,
                        maximum=VC_PITCH_SHIFT_MAX,
                        step=1,
                        value=0,
                        label=string("webui.pitch_shift_label"),
                        info=string("webui.pitch_shift_info"),
                        interactive=False,
                    )
                    vc_mode = gr.Radio(
                        choices=[
                            ("Talk", "talk"),
                            ("Sing", "sing"),
                        ],
                        value="talk",
                        label=string("webui.conversion_mode_label"),
                        info=string("webui.conversion_mode_info"),
                        interactive=False,
                    )
                    convert_button = gr.Button(
                        value=string("webui.convert_button"),
                        elem_id="celune-convert",
                        interactive=False,
                    )
                    converted_audio = gr.Audio(
                        value=None,
                        type="numpy",
                        autoplay=True,
                        show_label=False,
                        interactive=False,
                        visible="hidden",
                        waveform_options=_webui_audio_waveform_options(),
                        elem_id="celune-converted-audio",
                    )
            audio = gr.Audio(
                value=None,
                type="numpy",
                autoplay=True,
                show_label=False,
                interactive=False,
                visible="hidden",
                waveform_options=_webui_audio_waveform_options(),
                elem_id="celune-audio",
            )
            timer = gr.Timer(value=_api.WEBUI_POLL_INTERVAL_SECONDS)

        timer.tick(  # type: ignore[missing-attribute]
            _webui_snapshot,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            show_progress="hidden",
        )
        timer.tick(  # type: ignore[missing-attribute]
            _webui_vc_controls_update,
            outputs=[source_audio, vc_pitch_shift, vc_mode, convert_button],
            show_progress="hidden",
        )
        timer.tick(  # type: ignore[missing-attribute]
            _webui_record_button_update,
            outputs=[record_hotkey],
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            _webui_snapshot,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            _webui_vc_controls_update,
            outputs=[source_audio, vc_pitch_shift, vc_mode, convert_button],
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            _webui_record_button_update,
            outputs=[record_hotkey],
            show_progress="hidden",
        )
        input_box.submit(  # type: ignore[missing-attribute]
            _webui_speak,
            inputs=[input_box],
            outputs=[
                input_box,
                audio,
                logs,
                status,
                resources,
                voice_button,
                send_button,
            ],
            show_progress="hidden",
        )
        send_button.click(  # type: ignore[missing-attribute]
            _webui_speak,
            inputs=[input_box],
            outputs=[
                input_box,
                audio,
                logs,
                status,
                resources,
                voice_button,
                send_button,
            ],
            show_progress="hidden",
        )
        voice_button.click(  # type: ignore[missing-attribute]
            _webui_cycle_voice,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            show_progress="hidden",
        )
        record_hotkey.click(  # type: ignore[missing-attribute]
            _webui_toggle_recording,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            show_progress="hidden",
        )
        convert_button.click(  # type: ignore[missing-attribute]
            _webui_convert_audio,
            inputs=[source_audio, vc_pitch_shift, vc_mode],
            outputs=[
                source_audio,
                converted_audio,
                logs,
                status,
                resources,
                voice_button,
                send_button,
            ],
            show_progress="hidden",
        )

    return demo


def install(target):
    """Install extracted definitions in the original module."""
    install_module_functions(target, {name: globals()[name] for name in __all__})
