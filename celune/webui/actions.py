# SPDX-License-Identifier: Apache-2.0
"""WebUI action helpers."""

from __future__ import annotations

import io
import json
import textwrap
from typing import TYPE_CHECKING, Optional, cast
from functools import partial
from collections.abc import Callable, Iterator, Awaitable

import numpy as np
import gradio as gr
import soundfile as sf
from fastapi.responses import JSONResponse

from ..cesn import contains_cesn_note_event
from ..i18n import string, tagged_string
from ..ui.app import CeluneUI
from .runtime import (
    api_log,
    require_celune,
    webui_snapshot,
    webui_logs_html,
    append_webui_log,
    normalized_audio,
    set_webui_status,
    webui_status_html,
    append_webui_error,
    webui_resources_html,
    webui_vc_mode_active,
    configure_webui_theme,
    webui_submit_snapshot,
    webui_vc_controls_update,
    webui_audio_waveform_options,
    configure_webui_generation_fn_indices,
)
from .commands import WebUiCommandHost
from ..constants import BASE_SR, APP_NAME
from ..typing.api import WebUiUpdate, WebUiAudioValue, WebUiInputAudioValue
from ..persona.impl import persona_talkback_enabled
from ..server.runtime import run_async_runtime_call
from ..server.session import ServerSession, server_session
from ..typing.aliases import AudioChunk, AudioChunks
from ..server.constants import (
    WEBUI_CONVERT_API_NAME,
    WEBUI_LOAD_VC_API_NAME,
    WEBUI_POLL_VC_API_NAME,
    WEBUI_CYCLE_VOICE_API_NAME,
    WEBUI_SPEAK_INPUT_API_NAME,
    WEBUI_POLL_INTERVAL_SECONDS,
    WEBUI_SPEAK_BUTTON_API_NAME,
    WEBUI_LOAD_SNAPSHOT_API_NAME,
    WEBUI_POLL_SNAPSHOT_API_NAME,
    WEBUI_LOAD_RECORDING_API_NAME,
    WEBUI_POLL_RECORDING_API_NAME,
    WEBUI_TOGGLE_RECORDING_API_NAME,
)

if TYPE_CHECKING:
    from ..celune import Celune

__all__ = (
    "_normalize_webui_audio_input",
    "_webui_record_button_update",
    "_webui_select_voice",
    "_webui_stop",
    "_webui_stop_button_update",
    "_webui_toggle_recording",
    "_webui_voice_catalog",
    "_webui_voice_choices",
    "build_webui",
    "decode_uploaded_audio",
    "voice_conversion_unavailable_response",
    "webui_convert_audio",
    "webui_cycle_voice",
    "webui_run_command",
    "webui_speak",
)


def webui_run_command(
    text: str,
    session: ServerSession = server_session,
) -> bool:
    """Run one slash command through the main UI command path when available."""
    try:
        parts = CeluneUI.split_command_input(text[1:])
    except ValueError:
        append_webui_log(string("webui.command_parsing_error"), "error", session)
        return False

    if not parts:
        return False

    command = parts[0].lower()
    command_args = parts[1:]
    celune = session.bound_celune
    if command == "settings" and celune is not None:
        from ..ui.commands import process_command as process_ui_command

        process_ui_command(
            cast(
                CeluneUI,
                WebUiCommandHost(
                    celune,
                    partial(webui_run_command, session=session),
                ),
            ),
            command,
            command_args,
        )
        return True

    ui = CeluneUI.get_instance()
    if ui is not None:
        ui.call_from_thread(ui.process_command, command, command_args)
        return True

    if celune is None:
        append_webui_log(string("webui.not_available"), "error", session)
        return False
    from ..ui.commands import process_command as process_ui_command

    process_ui_command(
        cast(
            CeluneUI,
            WebUiCommandHost(
                celune,
                partial(webui_run_command, session=session),
            ),
        ),
        command,
        command_args,
    )
    return True


def decode_uploaded_audio(
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


def _webui_playback_audio(
    audio: AudioChunk,
    sample_rate: int,
) -> WebUiAudioValue:
    """Encode normalized playback audio as PCM16 for Gradio."""
    from ..pipeline.submission import prepare_playback_audio

    prepared_audio = prepare_playback_audio(audio, sample_rate)
    peak = float(np.max(np.abs(prepared_audio))) if prepared_audio.size else 0.0
    if peak == 0.0:
        pcm_audio = np.zeros(prepared_audio.shape, dtype=np.int16)
    else:
        pcm_audio = np.asarray(prepared_audio / peak * 32767.0, dtype=np.int16)
    return BASE_SR, pcm_audio


def voice_conversion_unavailable_response() -> JSONResponse:
    """Return a standard API error for VC-only endpoints in TTS mode."""
    return JSONResponse(
        status_code=409,
        content={
            "error": "wrong_mode",
            "message": string("webui.wrong_mode"),
        },
    )


def webui_speak(
    content: str,
    session: ServerSession = server_session,
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
        snapshot = webui_submit_snapshot("", session)
        yield snapshot[0], None, *snapshot[1:]
        return

    if text.startswith("/"):
        webui_run_command(text, session)
        snapshot = webui_submit_snapshot("", session)
        yield snapshot[0], None, *snapshot[1:]
        return

    celune = require_celune(session)
    if webui_vc_mode_active(celune):
        append_webui_log(
            string("webui.text_unavailable_in_vc_mode"), "warning", session
        )
        snapshot = webui_submit_snapshot(text, session)
        yield snapshot[0], None, *snapshot[1:]
        return
    api_log("SPEAK(WEBUI)", text, session=session)

    current_state = (celune.cur_state or "").strip().lower()
    if current_state == "waking":
        append_webui_log(
            string("webui.not_returned_from_sleep", app_name=APP_NAME),
            "warning",
            session,
        )
        snapshot = webui_submit_snapshot(text, session)
        yield snapshot[0], None, *snapshot[1:]
        return

    if getattr(celune, "sleeping", False):
        set_webui_status(string("status.waking_up"), session=session)
        snapshot = webui_submit_snapshot(text, session)
        yield snapshot[0], None, *snapshot[1:]
        wake_async = cast(
            Optional[Callable[[], Awaitable[bool]]],
            getattr(celune, "wake_from_sleep_async", None),
        )
        if wake_async is not None:
            woke = bool(run_async_runtime_call(wake_async()))
        else:
            woke = celune.wake_from_sleep()
        if not woke:
            snapshot = webui_submit_snapshot(text, session)
            yield snapshot[0], None, *snapshot[1:]
            return

    singing_backend = bool(getattr(getattr(celune, "backend", None), "singing", False))
    if persona_talkback_enabled(getattr(celune, "config", {})) and not singing_backend:
        if not celune.think(text):
            append_webui_log(string("webui.busy_try_again"), "warning", session)
        snapshot = webui_submit_snapshot("", session)
        yield snapshot[0], None, *snapshot[1:]
        return

    if singing_backend and not contains_cesn_note_event(text):
        append_webui_log(
            string("celune.singing_backend_requires_cesn"), "warning", session
        )
        snapshot = webui_submit_snapshot(text, session)
        yield snapshot[0], None, *snapshot[1:]
        return

    chunks = celune.say_stream(text, save=True)
    if chunks is None:
        append_webui_log(string("webui.busy_try_again"), "warning", session)
        snapshot = webui_submit_snapshot(text, session)
        yield snapshot[0], None, *snapshot[1:]
        return

    snapshot = webui_submit_snapshot("", session)
    yield snapshot[0], None, *snapshot[1:]

    audio_chunks: AudioChunks = []

    try:
        while True:
            item = chunks.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            audio_chunks.append(normalized_audio(item))
        audio_value: WebUiAudioValue
        if audio_chunks:
            audio_value = _webui_playback_audio(
                np.concatenate(audio_chunks),
                BASE_SR,
            )
        else:
            audio_value = None
        snapshot = webui_submit_snapshot("", session)
        yield snapshot[0], audio_value, *snapshot[1:]
    except Exception as error:
        append_webui_error(
            tagged_string("webui.error", "WEBUI ERROR"),
            error,
            celune=celune,
            session=session,
        )
        snapshot = webui_submit_snapshot("", session)
        yield snapshot[0], None, *snapshot[1:]


def webui_convert_audio(
    source_audio: WebUiInputAudioValue,
    conversion_mode: str = "talk",
    session: ServerSession = server_session,
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
        append_webui_log(
            string("webui.upload_audio_first"),
            "warning",
            session,
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            webui_snapshot(session)
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

    celune = require_celune(session)
    if not webui_vc_mode_active(celune):
        append_webui_log(
            string("webui.conversion_only_in_vc_mode"),
            "warning",
            session,
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            webui_snapshot(session)
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
    api_log("CONVERT(WEBUI)", "uploaded audio", session=session)
    try:
        output = celune.convert_audio(
            audio,
            sample_rate,
            label="browser audio input",
            f0_condition=conversion_mode.strip().lower() == "sing",
        )
    except Exception as error:
        append_webui_error(
            tagged_string("webui.error", "WEBUI ERROR"),
            error,
            celune=celune,
            session=session,
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            webui_snapshot(session)
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
        append_webui_log(
            string("webui.cannot_convert_right_now"),
            "warning",
            session,
        )
        logs_html, status_html, resources_html, voice_update, send_update, _input = (
            webui_snapshot(session)
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

    converted_audio = _webui_playback_audio(output.audio, output.sample_rate)
    logs_html, status_html, resources_html, voice_update, send_update, _input = (
        webui_snapshot(session)
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


def webui_cycle_voice(
    session: ServerSession = server_session,
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Cycle to the next available Celune voice from the browser UI."""
    celune = require_celune(session)
    if getattr(getattr(celune, "backend", None), "singing", False):
        return webui_snapshot(session)
    if len(celune.voices) < 2 or not bool(celune.current_voice or celune.voices):
        return webui_snapshot(session)

    current_voice = celune.current_voice or celune.voices[0]
    current_index = (
        celune.voices.index(current_voice) if current_voice in celune.voices else -1
    )
    next_voice = celune.voices[(current_index + 1) % len(celune.voices)]
    api_log("VOICE(WEBUI)", next_voice, session=session)

    set_voice_async = cast(
        Optional[Callable[[str], Awaitable[bool]]],
        getattr(celune, "set_voice_async", None),
    )
    if set_voice_async is not None:
        switched = bool(run_async_runtime_call(set_voice_async(next_voice)))
    else:
        switched = celune.set_voice_and_wait(next_voice)

    if not switched:
        append_webui_log(
            string("webui.cannot_change_voice_right_now"), "error", session
        )

    return webui_snapshot(session)


def _webui_voice_catalog(celune: Celune) -> tuple[tuple[str, str], ...]:
    """Return readable browser choices for every available voice-pack entry."""
    if getattr(getattr(celune, "backend", None), "singing", False):
        return ()

    from ..cevoice import (
        CEVoice,
        CEVoiceError,
        bundled_voices_dir,
        bundle_display_name,
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
        pack_name = bundle_display_name(bundle)
        if pack_name in used_pack_names:
            pack_name = string(
                "cevoice.official_bundle_label",
                character=pack_name,
                variant=path.stem,
            )
        used_pack_names.add(pack_name)
        for voice_entry in bundle.voice_order:
            value = json.dumps(
                {"bundle": str(path), "entry": voice_entry},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            choices.append(
                (f"{pack_name}: {bundle.voice_display_name(voice_entry)}", value)
            )

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


def _webui_voice_choices(
    session: ServerSession = server_session,
) -> WebUiUpdate:
    """Return the current readable voice list for the browser selector."""
    celune = session.bound_celune
    if celune is None:
        return gr.update(choices=[], value=None, interactive=False)
    from ..cevoice import active_bundle_path

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
        interactive=not session.webui_input_locked and len(choices) >= 1,
    )


def _webui_select_voice(
    name: Optional[str],
    session: ServerSession = server_session,
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Select one voice through the same asynchronous core switch as the TUI."""
    celune = session.bound_celune
    if celune is None or not name:
        return webui_snapshot(session)
    if getattr(getattr(celune, "backend", None), "singing", False):
        append_webui_log(
            string("celune.singing_voice_change_unavailable"), "warning", session
        )
        return webui_snapshot(session)
    try:
        selection = json.loads(name)
    except json.JSONDecodeError:
        selection = {"entry": name}
    entry = selection.get("entry") if isinstance(selection, dict) else None
    bundle = selection.get("bundle") if isinstance(selection, dict) else None
    if not isinstance(entry, str) or (
        bundle is not None and not isinstance(bundle, str)
    ):
        append_webui_log(
            string("webui.cannot_change_voice_right_now"), "error", session
        )
        return webui_snapshot(session)

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

    switched = run_async_runtime_call(select_voice())
    if not switched:
        append_webui_log(
            string("webui.cannot_change_voice_right_now"), "error", session
        )
    return webui_snapshot(session)


def _webui_stop(
    session: ServerSession = server_session,
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Stop the active speech request through the shared runtime lifecycle."""
    celune = session.bound_celune
    if celune is None:
        return webui_snapshot(session)
    ui = CeluneUI.get_instance()
    if ui is not None:
        recording = (
            ui.vc_recording_active
            if webui_vc_mode_active(celune)
            else ui.persona_recording_active
        )
        if callable(recording) and recording():
            toggle = (
                ui.toggle_vc_recording
                if webui_vc_mode_active(celune)
                else ui.toggle_persona_recording
            )
            ui.call_from_thread(toggle)
            return webui_snapshot(session)
    stop_async = cast(
        Optional[Callable[[], Awaitable[bool]]],
        getattr(celune, "force_stop_speech_async", None),
    )
    if stop_async is not None:
        stopped = bool(run_async_runtime_call(stop_async()))
    else:
        stop_sync = getattr(celune, "force_stop_speech", None)
        stopped = (
            cast(Callable[[], bool], stop_sync)()  # pylint: disable=not-callable
            if callable(stop_sync)
            else False
        )
    if stopped:
        set_webui_status(
            string("status.stopped"),
            "sleeping",
            source="callback",
            session=session,
        )
    else:
        append_webui_log(string("commands.nothing_to_stop"), "warning", session)
    return webui_snapshot(session)


def _webui_stop_button_update(
    session: ServerSession = server_session,
) -> WebUiUpdate:
    """Return whether the browser stop control should be interactive."""
    celune = session.bound_celune
    if celune is None:
        return gr.update(interactive=False)
    state = str(getattr(celune, "cur_state", "")).casefold()
    return gr.update(
        interactive=state in {"speaking", "generating", "thinking"}
        or session.active_speech_task_id is not None
    )


def _webui_record_button_update(
    session: ServerSession = server_session,
) -> WebUiUpdate:
    """Return whether the browser can delegate live capture to the TUI runtime."""
    celune = session.bound_celune
    ui = CeluneUI.get_instance()
    if celune is None or ui is None or session.webui_input_locked:
        return gr.update(interactive=False)
    if getattr(celune, "is_in_tutorial", False):
        return gr.update(interactive=False)
    return gr.update(interactive=True)


def _webui_toggle_recording(
    session: ServerSession = server_session,
) -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Toggle the same Persona or live VC capture path used by ``CTRL+R``."""
    celune = session.bound_celune
    ui = CeluneUI.get_instance()
    if celune is None or ui is None:
        append_webui_log(string("webui.recording_requires_tui"), "warning", session)
        return webui_snapshot(session)
    if getattr(celune, "sleeping", False):
        ui.call_from_thread(ui.wake_from_sleep)
        return webui_snapshot(session)
    if getattr(celune, "cur_state", "") == "waking":
        return webui_snapshot(session)

    toggle = (
        ui.toggle_vc_recording
        if webui_vc_mode_active(celune)
        else ui.toggle_persona_recording
    )
    ui.call_from_thread(toggle)
    return webui_snapshot(session)


def build_webui(session: ServerSession = server_session) -> gr.Blocks:
    # pylint: disable=E1101
    """Create the browser UI mounted by the API."""
    configure_webui_theme(session)
    snapshot_callback = partial(webui_snapshot, session=session)
    vc_controls_callback = partial(webui_vc_controls_update, session=session)
    record_button_callback = partial(_webui_record_button_update, session=session)
    speak_callback = partial(webui_speak, session=session)
    cycle_voice_callback = partial(webui_cycle_voice, session=session)
    toggle_recording_callback = partial(_webui_toggle_recording, session=session)
    convert_audio_callback = partial(webui_convert_audio, session=session)
    with gr.Blocks(
        title=APP_NAME,
        fill_height=True,
    ) as demo:
        with gr.Column(elem_id="celune-shell"):
            with gr.Tabs():
                with gr.Tab(string("webui.tts_tab_label")):
                    gr.HTML(
                        textwrap.dedent(
                            f"""
                            <div id="celune-header">
                                <div class="line"></div>
                                <div class="title">{"&nbsp;&nbsp;".join(APP_NAME.upper())}</div>
                                <div class="line"></div>
                            </div>
                            """
                        )
                    )
                    logs = gr.HTML(webui_logs_html(session))
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
                        status = gr.HTML(
                            webui_status_html(session), elem_id="celune-status"
                        )
                        resources = gr.HTML(
                            webui_resources_html(session),
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
                        waveform_options=webui_audio_waveform_options(),
                        elem_id="celune-source-audio",
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
                        waveform_options=webui_audio_waveform_options(),
                        elem_id="celune-converted-audio",
                    )
            audio = gr.Audio(
                value=None,
                type="numpy",
                autoplay=True,
                show_label=False,
                interactive=False,
                visible="hidden",
                waveform_options=webui_audio_waveform_options(),
                elem_id="celune-audio",
            )
            timer = gr.Timer(value=WEBUI_POLL_INTERVAL_SECONDS)

        timer.tick(  # type: ignore[missing-attribute]
            snapshot_callback,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            api_name=WEBUI_POLL_SNAPSHOT_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        timer.tick(  # type: ignore[missing-attribute]
            vc_controls_callback,
            outputs=[source_audio, vc_mode, convert_button],
            api_name=WEBUI_POLL_VC_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        timer.tick(  # type: ignore[missing-attribute]
            record_button_callback,
            outputs=[record_hotkey],
            api_name=WEBUI_POLL_RECORDING_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            snapshot_callback,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            api_name=WEBUI_LOAD_SNAPSHOT_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            vc_controls_callback,
            outputs=[source_audio, vc_mode, convert_button],
            api_name=WEBUI_LOAD_VC_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        demo.load(  # type: ignore[missing-attribute]
            record_button_callback,
            outputs=[record_hotkey],
            api_name=WEBUI_LOAD_RECORDING_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        input_box.submit(  # type: ignore[missing-attribute]
            speak_callback,
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
            api_name=WEBUI_SPEAK_INPUT_API_NAME,
            show_progress="hidden",
        )
        send_button.click(  # type: ignore[missing-attribute]
            speak_callback,
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
            api_name=WEBUI_SPEAK_BUTTON_API_NAME,
            show_progress="hidden",
        )
        voice_button.click(  # type: ignore[missing-attribute]
            cycle_voice_callback,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            api_name=WEBUI_CYCLE_VOICE_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        record_hotkey.click(  # type: ignore[missing-attribute]
            toggle_recording_callback,
            outputs=[logs, status, resources, voice_button, send_button, input_box],
            api_name=WEBUI_TOGGLE_RECORDING_API_NAME,
            queue=False,
            show_progress="hidden",
        )
        convert_button.click(  # type: ignore[missing-attribute]
            convert_audio_callback,
            inputs=[source_audio, vc_mode],
            outputs=[
                source_audio,
                converted_audio,
                logs,
                status,
                resources,
                voice_button,
                send_button,
            ],
            api_name=WEBUI_CONVERT_API_NAME,
            show_progress="hidden",
        )

    configure_webui_generation_fn_indices(demo.config.get("dependencies", []), session)
    return demo
