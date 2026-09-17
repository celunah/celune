# SPDX-License-Identifier: Apache-2.0
"""Extracted CeluneUI interaction methods."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import queue as queue_module
import re
import shlex
import sys
import threading
import types
from copy import deepcopy
from collections.abc import Callable, Iterator
from typing import Optional, cast
from uuid import uuid4

from ..constants import SIGTSTP
from . import app as _app
from ..binding import install_class_functions

__all__ = (
    "_close_menu",
    "_config_autocomplete",
    "_config_explanation",
    "_config_label",
    "_graceful_exit",
    "_hide_scrollbars_for_exit",
    "_install_windows_signal_handler",
    "_iter_config_values",
    "_menu_footer",
    "_report_shutdown_error",
    "_run_shutdown_step",
    "_save_settings",
    "_set_config_value",
    "_show_menu",
    "_shutdown_live_vc_recording",
    "_shutdown_runtime",
    "_signal_handler",
    "_signal_handler_windows",
    "_split_command_input",
    "_start_vc_recording",
    "_submit_text",
    "action_quit",
    "begin_tutorial",
    "cancel_tutorial",
    "consume_buffer",
    "error",
    "finish_tutorial",
    "graceful_exit",
    "on_button_pressed",
    "on_key",
    "on_select_menu_widget_cancelled",
    "on_select_menu_widget_confirmed",
    "on_text_area_changed",
    "on_unmount",
    "on_voice_button_held",
    "open_settings_menu",
    "open_voice_menu",
    "process_command",
    "split_command_input",
    "toggle_vc_recording",
    "tts_idle",
    "tts_log",
    "tts_queue_avail",
    "tts_voice_changed",
    "tutorial_active",
    "tutorial_after",
    "tutorial_token",
)


def _start_vc_recording(self) -> bool:
    """Start recording from the active system input device for VC."""
    _app._load_ui_runtime_dependencies()
    if self.celune is None or not self._is_voice_conversion_mode():
        return False
    if (
        getattr(self.celune, "sleeping", False)
        or getattr(
            self.celune,
            "cur_state",
            "",
        )
        == "waking"
    ):
        return False

    if self._vc_recording_active():
        return True

    input_config = getattr(self.celune, "config", None)
    input_device_key = (
        "input_recording_device"
        if isinstance(input_config, dict) and "input_recording_device" in input_config
        else "input_device"
    )
    try:
        input_device, direct_device_info = _app.resolve_audio_device_with_info(
            input_config,
            input_device_key,
            "input",
        )
    except ValueError as exc:
        self.safe_log(str(exc), "warning")
        return False

    try:
        device_info = (
            cast(dict[str, _app.AudioDeviceInfoValue], dict(direct_device_info))
            if direct_device_info is not None
            else cast(
                dict[str, _app.AudioDeviceInfoValue],
                _app.sd.query_devices(device=input_device, kind="input"),
            )
        )
    except Exception as e:
        self.safe_log(
            _app.format_error_message(
                _app.string("ui.recording_open_input_failed"),
                e,
                getattr(self.celune, "log_level", "info"),
            ),
            "error",
        )
        return False

    channels = _app._device_scalar_int(device_info.get("max_input_channels"), 0)
    if channels <= 0:
        self.safe_log(_app.string("pipeline.no_audio_device"), "warning")
        return False

    sample_rate = _app._device_scalar_int(
        device_info.get("default_samplerate"),
        48000,
    )
    label = _app.format_audio_device_name(device_info) or str(
        device_info.get("name", _app.string("ui.audio_input_label"))
    )
    channel_count = 2 if channels >= 2 else 1
    vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate)
    vad_preroll_frames = self._vc_vad_preroll_frames(sample_rate)
    live_chunk_frames = self._vc_live_chunk_frames(sample_rate)
    ai_vad = _app.create_live_voice_activity_detector()
    submission_queue: queue_module.Queue[
        Optional[tuple[_app.AudioChunk, int, str, bool]]
    ] = queue_module.Queue(maxsize=_app._VC_LIVE_SUBMISSION_QUEUE_SIZE)

    def submit_live_audio() -> None:
        live_source_id: Optional[int] = None
        live_playback_generation: Optional[int] = None

        def queue_playback_segment(
            audio_chunk: _app.AudioChunk,
            sr: int,
            audio_label: str,
        ) -> None:
            nonlocal live_playback_generation, live_source_id
            if len(audio_chunk) <= 0 or self.celune is None:
                return

            # noinspection PyBroadException
            try:
                if live_source_id is None:
                    live_playback_generation = getattr(
                        self.celune,
                        "_playback_generation",
                        0,
                    )
                live_source_id = _app.queue_streaming_sfx_audio(
                    self.celune,
                    _app.np.asarray(audio_chunk, dtype=_app.np.float32),
                    sr,
                    audio_label,
                    source_id=live_source_id,
                    generation=live_playback_generation,
                    status_label_key="pipeline.revoicing_label",
                    reset_ready_announcement=live_source_id is None,
                )
                if live_source_id is None:
                    live_playback_generation = None
            except Exception as exc:
                self.safe_log(
                    _app.format_error_message(
                        _app.string("ui.recording_stream_submit_failed"),
                        exc,
                        _app.resolve_log_level(
                            getattr(self.celune, "log_level", None),
                            self._startup_log_level,
                        ),
                    ),
                    "warning",
                )
                return

        def finish_playback_segment() -> None:
            nonlocal live_playback_generation, live_source_id
            if self.celune is None:
                return
            _app.finish_streaming_sfx_audio(self.celune, live_source_id)
            live_source_id = None
            live_playback_generation = None

        while True:
            item = submission_queue.get()
            if self.celune is None or getattr(self.celune, "exit_requested", False):
                return
            if item is None:
                finish_playback_segment()
                return

            audio, queued_sample_rate, queued_label, is_final_chunk = item
            try:
                if self.celune is None:
                    continue
                converter = getattr(self.celune, "convert_live_audio", None)
                if not callable(converter):
                    converter = self.celune.convert_audio
                converted = cast(
                    Callable[..., Optional[_app.AudioOutput]],
                    converter,
                )(
                    audio,
                    queued_sample_rate,
                    label=queued_label,
                )
                if converted is None:
                    self.safe_log(
                        _app.string("ui.recording_stream_submit_failed"),
                        "warning",
                    )
                    continue

                converted_audio = _app.np.asarray(
                    converted.audio, dtype=_app.np.float32
                )
                playback_sample_rate = converted.sample_rate
                playback_label = converted.label
                if len(converted_audio) > 0:
                    queue_playback_segment(
                        converted_audio,
                        playback_sample_rate,
                        playback_label,
                    )

                if is_final_chunk:
                    finish_playback_segment()
            except Exception as exc:
                if self.celune is None:
                    continue
                self.safe_log(
                    _app.format_error_message(
                        _app.string(
                            "ui.recording_stream_chunk_failed",
                            label=queued_label,
                        ),
                        exc,
                        getattr(self.celune, "log_level", "info"),
                    ),
                    "warning",
                )
                if isinstance(exc, _app.CEDTSError):
                    self._cancel_vc_recording(announce=False)

    worker = threading.Thread(target=submit_live_audio, daemon=True)

    def callback(
        indata: _app.npt.NDArray[_app.np.float32],
        frames: int,
        time_info: Optional[tuple[float, float, float]],
        status: Optional[_app.sd.CallbackFlags],
    ) -> None:
        from ..utils import discard

        discard(frames)
        discard(time_info)
        discard(status)

        callback_audio = _app.np.asarray(indata, dtype=_app.np.float32).copy()
        current_rms = self._vc_input_rms(callback_audio)
        should_stop_for_feedback = False
        feedback_min_capture_frames = self._vc_feedback_min_capture_frames(sample_rate)
        if ai_vad is not None:
            try:
                voice_detected = ai_vad.has_voice(callback_audio, sample_rate)
            except (RuntimeError, AssertionError, ValueError):
                ai_vad.reset()
                voice_detected = self._vc_input_has_voice(callback_audio)
        else:
            voice_detected = self._vc_input_has_voice(callback_audio)

        with self._vc_recording_lock:
            if self._vc_recording_stream is None:
                return
            if self._vc_recording_feedback_detected:
                return

            previous_rms = self._vc_recording_previous_rms
            self._vc_recording_previous_rms = current_rms
            self._vc_recording_captured_frames += len(callback_audio)

            suspicious_feedback = (
                self._vc_recording_captured_frames >= feedback_min_capture_frames
                and self._vc_feedback_rise_detected(previous_rms, current_rms)
            )
            if suspicious_feedback:
                self._vc_recording_feedback_spike_count += 1
            else:
                self._vc_recording_feedback_spike_count = 0

            if (
                self._vc_recording_feedback_spike_count
                >= _app._VC_FEEDBACK_REQUIRED_CONSECUTIVE_SPIKES
            ):
                self._vc_recording_feedback_detected = True
                should_stop_for_feedback = True

            if should_stop_for_feedback:
                self._request_vc_recording_feedback_stop()
                return

            live_audio: Optional[_app.AudioChunk] = None
            if voice_detected:
                if not self._vc_recording_speech_started:
                    self._prepend_vc_preroll_locked()
                    self._vc_recording_chunks.append(callback_audio)
                    self._vc_recording_buffered_frames += len(callback_audio)
                    live_audio = self._flush_vc_recording_buffer_locked()
                else:
                    live_audio = callback_audio
                self._vc_recording_speech_started = True
                self._vc_recording_silence_frames = 0
            elif self._vc_recording_speech_started:
                self._vc_recording_silence_frames += len(callback_audio)
                live_audio = _app.np.zeros_like(callback_audio)
                if self._vc_recording_silence_frames > vad_hangover_frames:
                    self._vc_recording_speech_started = False
                    self._vc_recording_silence_frames = 0
                    if ai_vad is not None:
                        ai_vad.reset()
                    self._append_vc_preroll_audio_locked(
                        callback_audio,
                        vad_preroll_frames,
                    )
            else:
                live_audio = _app.np.zeros_like(callback_audio)
                self._append_vc_preroll_audio_locked(
                    callback_audio,
                    vad_preroll_frames,
                )

            if (
                live_audio is not None
                and self._vc_recording_submission_queue is not None
            ):
                self._enqueue_vc_submission_chunk(
                    self._vc_recording_submission_queue,
                    (live_audio, sample_rate, label, False),
                )

    stream: Optional[_app.sd.InputStream] = None
    acquired, component_lease = self._acquire_recording_component_lease(
        f"vc-recording:{uuid4()}",
        (_app.ComponentLockName.MICROPHONE,),
    )
    if not acquired:
        return False
    try:
        stream = _app.sd.InputStream(
            samplerate=sample_rate,
            channels=channel_count,
            dtype="float32",
            callback=callback,
            device=input_device,
            blocksize=live_chunk_frames,
        )

        with self._vc_recording_lock:
            self._vc_recording_stream = stream
            self._vc_recording_chunks = []
            self._vc_recording_buffered_frames = 0
            self._vc_recording_captured_frames = 0
            self._vc_recording_feedback_detected = False
            self._vc_recording_sample_rate = sample_rate
            self._vc_recording_label = label
            self._vc_recording_preroll_chunks = []
            self._vc_recording_preroll_frames = 0
            self._vc_recording_previous_rms = 0.0
            self._vc_recording_silence_frames = 0
            self._vc_recording_speech_started = False
            self._vc_recording_submission_queue = submission_queue
            self._vc_recording_stop_thread = None
            self._vc_recording_worker = worker
            self._vc_recording_vad = ai_vad
            self._vc_recording_component_lease = component_lease

        stream.start()
    except Exception as e:
        with self._vc_recording_lock:
            if self._vc_recording_stream is stream:
                self._clear_vc_recording_state()
        self._finish_vc_submission_queue(submission_queue)
        self._shutdown_vc_stream(stream)
        if component_lease is not None:
            component_lease.release()
        self.safe_log(
            _app.format_error_message(
                _app.string("ui.recording_start_failed", label=label),
                e,
                getattr(self.celune, "log_level", "info"),
            ),
            "error",
        )
        return False

    worker.start()
    self.safe_log(_app.string("ui.recording_started", label=label), "info")
    self._set_terminal_status(
        "recording",
        _app.string("osc.action_listening_microphone"),
    )
    self.update_resources()
    return True


def toggle_vc_recording(self) -> bool:
    """Toggle live VC recording for the current input device.

    Returns:
        bool: ``True`` when recording started or stopped successfully.
    """
    if self._vc_recording_active():
        with self._vc_recording_lock:
            (
                stream,
                buffered_audio,
                sample_rate,
                label,
                submission_queue,
                captured_frames,
                stop_thread,
                worker,
            ) = self._stop_vc_recording_stream()
            self._finish_vc_submission_queue(
                submission_queue,
                (
                    _app.np.asarray(buffered_audio, dtype=_app.np.float32),
                    sample_rate,
                    label,
                    True,
                )
                if buffered_audio is not None
                else None,
            )
        self._shutdown_vc_stream(stream)
        self._stop_live_vc_backend()
        self._join_vc_recording_threads(stop_thread, worker)

        self.safe_log(_app.string("ui.recording_stopped", label=label), "info")
        self._set_terminal_status("ready", _app.string("osc.action_idle"))
        self.update_resources()
        if captured_frames <= 0:
            self.safe_log(_app.string("ui.recording_empty"), "warning")
            return False
        return True

    return self._start_vc_recording()


def _shutdown_live_vc_recording(self) -> None:
    """Stop live VC recording immediately for application shutdown."""
    self._shutdown_persona_recording()
    if self.celune is not None:
        self.celune._exit_requested = True

    if not self._vc_recording_active():
        return

    with self._vc_recording_lock:
        (
            stream,
            _buffered_audio,
            _sample_rate,
            _label,
            submission_queue,
            _captured_frames,
            stop_thread,
            worker,
        ) = self._stop_vc_recording_stream()
        self._finish_vc_submission_queue(submission_queue)
    self._shutdown_vc_stream(stream)
    self._stop_live_vc_backend()
    self._join_vc_recording_threads(stop_thread, worker)


def tts_voice_changed(self, name: str) -> None:
    """Set UI state after changing Celune's voice.

    Args:
        name: The newly active voice name.
    """
    if self.cur_state == "exiting":
        return

    if name in self.celune_styles:
        self.style_index = self.celune_styles.index(name)

    label = name.capitalize()

    if threading.current_thread() is threading.main_thread():
        self.style_button.label = label
        self.refresh_vc_controls()
        self.update_resources()
    else:

        def update() -> None:
            self.style_button.label = label
            self.refresh_vc_controls()
            self.update_resources()

        self.call_from_thread(update)


def tts_log(
    self,
    msg: str,
    severity: str = "info",
    *,
    loglevel: _app.LogLevel = "info",
) -> None:
    """Handle log messages coming from Celune.

    Args:
        msg: The log message emitted by Celune.
        severity: The log severity level.
        loglevel: The minimum configured log level required to display the message.
    """
    if self.cur_state == "exiting":
        return

    self.safe_log(msg, severity, loglevel=loglevel)


def process_command(self, command: str, args: list[str]) -> None:
    """Process Celune control commands.

    Args:
        command: The control command to run.
        args: The command arguments to use.
    """
    _app.process_ui_command(self, command, args)


def open_settings_menu(self) -> None:
    """Open the configuration manager and prepare a restart on save."""
    if self.celune is None or self._active_menu is not None:
        return

    options: list[_app.SelectMenuOption] = []
    paths: list[tuple[str, ...]] = []
    for path, value in self._iter_config_values(self.celune.config):
        label = self._config_label(path)
        options.append(
            _app.SelectMenuOption(
                label=label,
                value=value,
                autocomplete_values=self._config_autocomplete(path, value),
                explanation=self._config_explanation(path),
            )
        )
        paths.append(path)

    if not options:
        self.safe_log(_app.string("ui.settings_empty"), "warning")
        return

    self._settings_paths = tuple(paths)
    self._show_menu(
        _app.SelectMenuWidget(
            _app.string("ui.settings_title"),
            options,
            value_display="all",
            return_value=False,
            footer_builder=lambda option: self._menu_footer(
                option,
                option_count=len(options),
                confirm_hint=_app.string("ui.settings_confirm_hint"),
                include_search=False,
            ),
        ),
        "settings",
    )


def _iter_config_values(
    config: dict[str, _app.JSONSerializable],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], _app.JSONSerializable]]:
    """Yield editable leaf values from the nested configuration mapping."""
    for key, value in config.items():
        path = (*prefix, key)
        if isinstance(value, dict) and value:
            yield from _app.CeluneUI._iter_config_values(value, path)
        else:
            yield path, value


def _config_autocomplete(
    path: tuple[str, ...], value: _app.JSONSerializable
) -> Optional[tuple[_app.JSONSerializable, ...]]:
    """Return useful autocomplete candidates for one configuration value."""
    if value is None:
        return (None,)
    if isinstance(value, bool):
        return (True, False)

    candidates: dict[tuple[str, ...], tuple[_app.JSONSerializable, ...]] = {
        ("backend",): (
            None,
            "mini",
            "qwen3",
            "luxtts",
            "dots.tts",
            "voxcpm2",
            "seed-vc",
        ),
        ("log_level",): ("info", "verbose", "debug"),
        ("mode",): ("speak", "converse", "agent"),
        ("theme",): ("dark", "light"),
        ("vram",): ("low", "medium", "high", "xhigh"),
        ("audio_api",): (None, "wasapi", "directsound"),
        ("persona", "speech_language"): ("auto",),
    }
    return candidates.get(path)


def _config_label(path: tuple[str, ...]) -> str:
    """Convert a dotted configuration path into a readable setting label."""
    label = " ".join(path).replace("_", " ")
    for source, replacement in (("t2s", "T2S"),):
        label = re.sub(
            rf"\b{re.escape(source)}\b",
            replacement,
            label,
            flags=re.IGNORECASE,
        )

    protected_names = {
        "api": "API",
        "asr": "ASR",
        "celune": "Celune",
        "cpu": "CPU",
        "gpu": "GPU",
        "luxtts": "LuxTTS",
        "ipa": "IPA",
        "persona": "Persona",
        "qwen3": "Qwen3",
        "t2s": "T2S",
        "tts": "TTS",
        "vc": "VC",
        "vram": "VRAM",
    }
    words = label.split()
    formatted = [
        protected_names.get(word.casefold(), word.casefold()) for word in words
    ]
    if formatted and words[0].casefold() not in protected_names:
        formatted[0] = formatted[0].capitalize()
    return " ".join(formatted)


def _config_explanation(path: tuple[str, ...]) -> str:
    """Return a localized explanation for one configuration value."""
    aliases: dict[tuple[str, ...], str] = {
        ("persona", "speech_end_delay_seconds"): "persona.speech_delay",
        (
            "persona",
            "memory",
            "max_short_term_messages",
        ): "persona.memory.short_term",
        ("persona", "memory", "auto_classifier"): "persona.memory.auto",
        ("persona", "memory", "auto_classifier_min_confidence"): "mem.auto_conf",
        (
            "persona",
            "memory",
            "auto_classifier_max_candidates",
        ): "mem.auto_candidates",
        (
            "persona",
            "memory",
            "context_compaction_enabled",
        ): "persona.memory.compaction",
        (
            "persona",
            "memory",
            "context_compaction_keep_recent_messages",
        ): "mem.compact_recent",
        ("persona", "memory", "context_summary_max_characters"): "mem.summary_len",
        ("persona", "memory", "semantic_similarity_threshold"): "mem.similarity",
        (
            "persona",
            "memory",
            "fallback_token_overlap_threshold",
        ): "mem.token_overlap",
        ("persona", "memory", "semantic_embedding_model"): "mem.embedding",
    }
    explanation_path = aliases.get(path, ".".join(path))
    explanation_key = "ui.settings_explanation." + explanation_path
    explanation = _app.string(explanation_key)
    if explanation != explanation_key:
        return explanation
    return _app.string(
        "ui.settings_explanation_generic",
        setting=_app.CeluneUI._config_label(path),
    )


def _menu_footer(
    option: _app.SelectMenuOption,
    *,
    option_count: int,
    confirm_hint: str,
    include_search: bool,
) -> str:
    """Build localized hints for the selected menu row."""
    hints: list[str] = []
    if option_count > 1:
        hints.append(_app.string("ui.menu_hint_select"))
    if (
        option.editable
        and option.autocomplete_values is not None
        and len(option.autocomplete_values) > 1
    ):
        hints.append(_app.string("ui.menu_hint_choose"))
        if include_search:
            hints.append(_app.string("ui.menu_hint_search"))
    hints.extend((confirm_hint, _app.string("ui.menu_hint_cancel")))
    return _app.string("ui.menu_hint_separator").join(hints)


def open_voice_menu(self) -> None:
    """Open a voice menu containing every available CEVOICE/CECHAR entry."""
    if self.celune is None or self._active_menu is not None:
        return

    from ..cevoice import (
        CEVoice,
        CEVoiceError,
        active_bundle_path,
        bundle_display_name,
        bundled_voices_dir,
    )

    options: list[_app.SelectMenuOption] = []
    self._voice_menu_paths = {}
    active_bundle = active_bundle_path()
    active_voice = getattr(self.celune, "current_voice", None)
    voice_directory = bundled_voices_dir()
    try:
        pack_paths = sorted(
            path
            for path in voice_directory.iterdir()
            if path.is_file() and path.suffix.casefold() in {".cevoice", ".cechar"}
        )
    except OSError:
        pack_paths = []
    for path in pack_paths:
        try:
            bundle = CEVoice.open(path)
        except (OSError, CEVoiceError):
            continue

        pack_name = bundle_display_name(bundle)
        if pack_name in self._voice_menu_paths:
            pack_name = _app.string(
                "cevoice.official_bundle_label",
                character=pack_name,
                variant=path.stem,
            )
        voices = bundle.voice_order
        if not voices:
            continue
        self._voice_menu_paths[pack_name] = path
        selected_voice = voices[0]
        if (
            path == active_bundle
            and isinstance(active_voice, str)
            and active_voice in voices
        ):
            selected_voice = active_voice
        options.append(
            _app.SelectMenuOption(
                label=pack_name,
                value=selected_voice,
                editable=len(voices) > 1,
                autocomplete_values=voices if len(voices) > 1 else None,
                confirm_value=voices[0] if len(voices) == 1 else None,
            )
        )

    if not options:
        self.safe_log(_app.string("ui.no_voices_loaded"), "warning")
        return

    for index, option in enumerate(options):
        if (
            self._voice_menu_paths.get(option.label) == active_bundle
            and option.value == active_voice
        ):
            options.insert(0, options.pop(index))
            break

    self._show_menu(
        _app.SelectMenuWidget(
            _app.string("ui.voice_menu_title"),
            options,
            value_display="all",
            footer_builder=lambda option: self._menu_footer(
                option,
                option_count=len(options),
                confirm_hint=_app.string("ui.voice_confirm_hint"),
                include_search=True,
            ),
        ),
        "voice",
    )


def _show_menu(self, menu: _app.SelectMenuWidget, menu_kind: str) -> None:
    """Mount and focus one application menu overlay."""
    overlay = _app.SelectMenuOverlay(menu)
    self._active_menu = menu
    self._active_menu_overlay = overlay
    self._active_menu_kind = menu_kind
    self.push_screen(overlay)


def on_voice_button_held(self, event: _app.VoiceButton.Held) -> None:
    """Open voice selection after the held voice button is released."""
    if event.button is self.style_button:
        self.open_voice_menu()


def on_select_menu_widget_confirmed(
    self, event: _app.SelectMenuWidget.Confirmed
) -> None:
    """Apply a menu confirmation and close the active menu."""
    if event.menu is not self._active_menu:
        return

    menu = event.menu
    menu_kind = self._active_menu_kind
    self._close_menu()
    if menu_kind == "settings":
        self._save_settings(menu)
    elif menu_kind == "voice" and isinstance(event.value, str):
        self._apply_voice_selection({"pack": event.option.label, "entry": event.value})


def on_select_menu_widget_cancelled(
    self, event: _app.SelectMenuWidget.Cancelled
) -> None:
    """Close a menu without applying its changes."""
    if event.menu is not self._active_menu:
        return
    self._close_menu()


def _close_menu(self) -> None:
    """Remove a menu overlay after restoring focus to the input box."""
    self._active_menu = None
    overlay = self._active_menu_overlay
    self._active_menu_overlay = None
    self._active_menu_kind = None
    if overlay is not None and self.screen is overlay:
        overlay.dismiss()
    self.set_focus(self.input_box)


def _save_settings(self, menu: _app.SelectMenuWidget) -> None:
    """Persist the edited configuration and request a launcher restart."""
    if self.celune is None or len(self._settings_paths) != len(menu.options):
        return

    updated = deepcopy(self.celune.config)
    for path, option in zip(self._settings_paths, menu.options):
        self._set_config_value(updated, path, option.value)

    try:
        self.celune.config = updated
        with _app.config_path(create_parent=True).open("w", encoding="utf-8") as file:
            _app.yaml.safe_dump(updated, file, sort_keys=False)
    except OSError as exc:
        self.safe_log(
            _app.format_error_message(
                _app.string("ui.settings_save_failed"),
                exc,
                _app.resolve_log_level(
                    getattr(self.celune, "log_level", None),
                    self._startup_log_level,
                ),
            ),
            "error",
        )
        return

    self.cur_state = "restarting"
    self._run_shutdown_step(
        lambda: self._set_terminal_status(
            "restarting",
            _app.string("osc.action_restarting"),
        )
    )
    self._graceful_exit(return_code=_app.ExitCodes.EXIT_PENDING_RESTART.value)


def _set_config_value(
    config: dict[str, _app.JSONSerializable],
    path: tuple[str, ...],
    value: _app.JSONSerializable,
) -> None:
    """Replace one flattened configuration value in its nested mapping."""
    target = config
    for key in path[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            return
        target = child
    target[path[-1]] = value


def consume_buffer(self, text_len: int) -> None:
    """Consume a sentence from live input and say it.

    Args:
        text_len: The number of characters to consume from the input buffer.
    """
    to_say = self.input_box.text[:text_len].strip()

    self._suppress_input_change = True
    try:
        self.input_box.load_text(self.input_box.text[text_len:])
    # yes, no except:
    # that is valid python
    finally:
        self._suppress_input_change = False

    if not to_say:
        return

    if all(char in ".!?;:, " for char in to_say):
        return

    if self.celune.config.get("ipa") is False:
        from ..utils import replace_ipa

        ipa_decoded, unmatched = replace_ipa(to_say, strict=True)
        if unmatched > 0:
            self.safe_log(
                _app.string("commands.unmatched_ipa", count=unmatched),
                "warning",
                loglevel="verbose",
            )

        self.celune.say(ipa_decoded, display_text=to_say)
    else:
        self.celune.say(to_say)


def _submit_text(self, text: str, process_commands: bool = True) -> bool:
    """Submit text through the same path as the input box."""
    text = text.strip()

    if not text:
        return False

    celune = self.celune
    if celune is None:
        return False

    if getattr(celune, "test_finished", False):
        self._suppress_input_change = True
        try:
            self.input_box.load_text("")
        finally:
            self._suppress_input_change = False
        return True

    if self._is_ui_test_mode():
        self._suppress_input_change = True
        try:
            self.input_box.load_text("")
        finally:
            self._suppress_input_change = False
        self.safe_status(_app.string("ui.test_mode_active"))
        return True

    if self._is_agent_test_mode():
        self._suppress_input_change = True
        try:
            self.input_box.load_text("")
        finally:
            self._suppress_input_change = False
        self.safe_status(_app.string("ui.agent_test_mode_active"))
        return True

    if celune.cur_state == "waking":
        self._cancel_sleep_timer()
        self.safe_status(_app.string("status.waking_up"))
        self.change_input_state(locked=True)
        return True

    if celune.sleeping:
        self._cancel_sleep_timer()
        self.safe_status(_app.string("status.waking_up"))
        self._suppress_input_change = True
        try:
            self.input_box.load_text("")
        finally:
            self._suppress_input_change = False
        self.change_input_state(locked=True)
        self.wake_from_sleep()
        return True

    if process_commands and text.startswith("/"):
        try:
            parts = self.split_command_input(text[1:])
        except ValueError:
            self.safe_log(
                _app.string("ui.command_parsing_error"),
                "error",
            )
            return False

        if not parts:
            return False

        command = parts[0].lower()
        command_args = parts[1:]
        self.process_command(command, command_args)
        return True

    if _app.persona_talkback_enabled(celune.config):
        handled = celune.think(text)
    else:
        if celune.config.get("ipa") is False:
            from ..utils import replace_ipa

            ipa_decoded, unmatched = replace_ipa(text, strict=True)
            if unmatched > 0:
                self.safe_log(
                    f"Found {unmatched} unmatched IPA characters, output may be inaccurate.",
                    "warning",
                    loglevel="verbose",
                )
            handled = celune.say(ipa_decoded, display_text=text)
        else:
            handled = celune.say(text)

    if not handled:
        return False

    self._cancel_sleep_timer()
    self.style_button.actions = _app.ButtonActions(press=False, hold=False)
    self.input_box.placeholder = _app.string("ui.wait_placeholder")
    self.input_box.load_text("")
    self.update_resources()
    return True


def tutorial_after(self, delay: float, callback: Callable[[], None]) -> None:
    """Schedule a cancellable tutorial callback.

    Args:
        delay: Delay in seconds before running the callback.
        callback: Callback to run if the tutorial has not been canceled.
    """
    token = self._tutorial_token

    def run() -> None:
        if token != self._tutorial_token:
            return
        callback()

    if delay <= 0:
        self.call_later(run)
        return

    timer = self.set_timer(delay, run)
    self._tutorial_timers.append(timer)


def begin_tutorial(self) -> None:
    """Start a new cancellable tutorial action sequence."""
    self.cancel_tutorial(stop_audio=True)
    self._tutorial_active = True
    self.change_input_state(locked=True)
    self.input_box.placeholder = "Currently in tutorial mode"
    self.celune.is_in_tutorial = True
    self.change_voice_lock_state(locked=True)


def finish_tutorial(self) -> None:
    """Mark the current tutorial sequence as complete."""
    self._tutorial_active = False
    self._tutorial_timers.clear()
    self.celune.is_in_tutorial = False
    self.change_input_state(locked=False)
    self.change_voice_lock_state(locked=len(self.celune.voices) < 2)


def cancel_tutorial(self, stop_audio: bool = True) -> bool:
    """Cancel pending tutorial actions and any active tutorial typing.

    Args:
        stop_audio: Whether active tutorial playback should be interrupted.

    Returns:
        bool: ``True`` when tutorial work was canceled.
    """
    was_active = self._tutorial_active or bool(self._tutorial_timers)
    if not was_active:
        return False

    self._tutorial_token += 1
    self._tutorial_active = False
    self.celune.is_in_tutorial = False

    for timer in self._tutorial_timers:
        timer.stop()
    self._tutorial_timers.clear()

    if stop_audio and was_active and self.celune is not None:

        def stop_tutorial_audio() -> None:
            try:
                asyncio.run(self.celune.force_stop_speech_async())
            except Exception as exc:
                self.safe_log(
                    _app.format_error_message(
                        _app.string("ui.tutorial_stop_failed"),
                        exc,
                        getattr(self.celune, "log_level", "info"),
                    ),
                    "error",
                )

        threading.Thread(
            target=stop_tutorial_audio,
            daemon=True,
        ).start()

    self._suppress_input_change = True
    try:
        self.input_box.load_text("")
    finally:
        self._suppress_input_change = False
    self.change_input_state(locked=False)
    self.change_voice_lock_state(locked=len(self.celune.voices) < 2)

    return True


async def action_quit(self) -> None:
    """Exit through the startup-aware graceful shutdown path."""
    self._graceful_exit(return_code=self._startup_error_exit_code)


def on_key(self, event: _app.events.Key) -> None:
    """Accept input and send text to Celune.

    Args:
        event: The key event received from Textual.
    """
    with contextlib.suppress(EOFError):
        if self.cur_state == "exiting":
            return

        if event.key == "ctrl+q":
            event.prevent_default()
            event.stop()
            self._graceful_exit(return_code=self._startup_error_exit_code)
            return

        if (
            getattr(self.celune, "test_finished", False)
            or getattr(self.celune, "cur_state", None) == "stopped"
        ):
            event.prevent_default()
            event.stop()
            return

        if event.key in {"ctrl+j", "ctrl+enter"} and self.cancel_tutorial():
            event.prevent_default()
            event.stop()
            return

        if event.key == "ctrl+t":
            if self.active_theme_name == "celune_april_fools":
                event.prevent_default()
                return

            next_theme = (
                self.themes[1]
                if self.active_theme_name == self.themes[0]
                else self.themes[0]
            )
            self._apply_theme(next_theme)
            self.celune.config["theme"] = (
                "dark" if self.theme == self.themes[0] else "light"
            )
            with open(_app.config_path(create_parent=True), "w", encoding="utf-8") as f:
                _app.yaml.dump(self.celune.config, f)
            self.update_resources()

            event.prevent_default()
            return

        if event.key == "ctrl+r":
            if self._is_agent_test_mode():
                event.prevent_default()
                event.stop()
                return
            if getattr(self.celune, "sleeping", False):
                self._cancel_sleep_timer()
                self.safe_status(_app.string("status.waking_up"))
                self.change_input_state(locked=True)
                self.wake_from_sleep()
                event.prevent_default()
                event.stop()
                return
            if getattr(self.celune, "cur_state", "") == "waking":
                event.prevent_default()
                event.stop()
                return
            if self._is_voice_conversion_mode():
                recording_toggled = self.toggle_vc_recording()
            else:
                recording_toggled = self.toggle_persona_recording()
            if recording_toggled:
                event.prevent_default()
                event.stop()
            return

        if event.key == "ctrl+j" and self._submit_text(self.input_box.text):
            event.prevent_default()


def on_button_pressed(self, event: _app.Button.Pressed) -> None:
    """Change Celune's tone.

    Args:
        event: The button press event emitted by Textual.
    """
    if self.cur_state == "exiting":
        return

    celune = self.celune
    if celune is None or getattr(celune, "test_finished", False):
        return

    if self._is_agent_test_mode():
        return

    if celune.is_in_tutorial:
        return

    if event.button == self.vc_mode_button:
        if self._is_voice_conversion_mode():
            self.set_vc_f0_condition(
                not bool(getattr(celune, "vc_f0_condition", False))
            )
        return

    if event.button == self.vc_pitch_button:
        if self._is_voice_conversion_mode():
            current_value = int(getattr(celune, "vc_pitch_shift", 0))
            next_value = current_value + 1
            if next_value > _app.VC_PITCH_SHIFT_MAX:
                next_value = _app.VC_PITCH_SHIFT_MIN
            self.set_vc_pitch_shift(next_value)
        return

    if event.button != self.style_button:
        return

    if len(celune.voices) == 0 or not self.celune_styles:
        self.safe_log(_app.string("ui.no_voices_loaded"), "warning")
        self.change_voice_lock_state(locked=True)
        return

    if not self.celune_ready and not celune.backend.is_fake:
        self.safe_log(_app.string("ui.core_engine_not_loaded"), "warning")
        self.change_voice_lock_state(locked=True)
        return

    self.style_index = (self.style_index + 1) % len(self.celune_styles)
    next_voice = self.celune_styles[self.style_index]
    threading.Thread(
        target=celune.set_voice,
        args=(next_voice,),
        daemon=True,
    ).start()


def on_unmount(self) -> None:
    """Unload Celune."""
    restarting = self.cur_state == "restarting"
    if not restarting:
        self.cur_state = "exiting"
    self._run_shutdown_step(self._cancel_sleep_timer)
    self._run_shutdown_step(self._clear_caption_timers)
    self._run_shutdown_step(
        lambda: self._set_terminal_status(
            "restarting" if restarting else "exiting",
            _app.string(
                "osc.action_restarting" if restarting else "osc.action_exiting"
            ),
        )
    )
    if _app.ui_resources is not None:
        self._run_shutdown_step(_app.ui_resources.stop_gpu_usage_worker)
    self._run_shutdown_step(self._shutdown_runtime)

    if self._runtime_log_capture_enabled:
        self._run_shutdown_step(self._disable_runtime_log_capture)

    _app.CeluneUI._instance = None


def tts_idle(self) -> None:
    """Reset UI state after Celune stops talking."""
    self._hide_caption_widgets()
    celune = self.celune
    if celune is None:
        return
    if getattr(celune, "test_finished", False) or self._is_agent_test_mode():
        self.change_input_state(locked=True)
        self.change_voice_lock_state(locked=True)
        if getattr(celune, "test_finished", False):
            if self.input_box is not None:
                self.input_box.placeholder = _app.string("ui.stopped_placeholder")
            self.safe_status(_app.string("status.stopped"), "sleeping")
        return
    if self.cur_state in {"exiting", "error"} or not self.celune_ready:
        if self.input_box is not None:
            self.input_box.placeholder = _app.string("ui.wait_placeholder")
        self.change_voice_lock_state(locked=True)
        return
    if celune.cur_state in {"reloading", "waking"}:
        self.change_input_state(locked=True)
        self.change_voice_lock_state(locked=True)
        if celune.cur_state == "waking":
            self.safe_status(_app.string("status.waking_up"))
        return
    celune.locked = False
    if celune.sleeping:
        self.safe_status(_app.string("status.sleeping"), "sleeping")
        return
    celune.cur_state = "idle"
    if celune.is_in_tutorial:
        self.input_box.placeholder = _app.string("ui.tutorial_placeholder")
        self.change_voice_lock_state(locked=True)
    else:
        self.change_input_state(locked=False)
        self.change_voice_lock_state(locked=len(celune.voices) < 2)
    self.safe_status(_app.string("status.idle"))
    self._schedule_sleep_timer()


def tts_queue_avail(
    self,
) -> None:  # allow enqueuing new inputs while speaking but after generation
    """Unlock input queueing after Celune completes generation."""
    celune = self.celune
    if (
        celune is None
        or getattr(celune, "test_finished", False)
        or self._is_agent_test_mode()
    ):
        return
    if self.cur_state in {"exiting", "error"} or not self.celune_ready:
        return
    celune.locked = False
    self._cancel_sleep_timer()
    self.safe_status(_app.string("status.speaking"))
    if celune.is_in_tutorial:
        self.input_box.placeholder = _app.string("ui.tutorial_placeholder")
        self.change_voice_lock_state(locked=True)
    else:
        self.change_input_state(locked=False)
        self.change_voice_lock_state(locked=len(celune.voices) < 2)


def error(self, message: str) -> None:
    """Set the UI status to the error message.

    Args:
        message: The error text to display.
    """
    if self.cur_state == "exiting":
        return
    self._hide_caption_widgets()
    self.safe_status(message, "error")


def on_text_area_changed(self, event: _app.TextArea.Changed) -> None:
    """Monitor text area changes and perform actions.

    Args:
        event: The Textual text-area change event.
    """
    if self.cur_state == "exiting":
        return

    if self._suppress_input_change:
        return

    if event.text_area.id != "input":
        return

    text = event.text_area.text
    if self.celune.sleeping and text.strip():
        self._submit_text(text, process_commands=False)
        return

    line_count = text.count("\n") + 1
    min_lines = 1
    max_lines = 8

    visible_lines = max(min_lines, min(line_count, max_lines))
    event.text_area.styles.height = visible_lines + 2

    if self.consume_on_boundary and text and text[-1] in ".!?":
        if text in ".!?":
            return
        self.consume_buffer(len(text))


def _signal_handler(self, sig: int, frame: Optional[types.FrameType]) -> None:
    """Handle incoming signals."""
    from ..utils import discard

    discard(frame)

    if SIGTSTP is not None and sig == SIGTSTP:
        return

    self._graceful_exit()


def _install_windows_signal_handler(self) -> None:
    """Install Windows console shutdown handler."""
    winfunctype = getattr(ctypes, "WINFUNCTYPE", None)
    windll = getattr(ctypes, "windll", None)

    if winfunctype is None or windll is None:
        return

    handler_type = winfunctype(ctypes.c_bool, ctypes.c_uint)
    self._windows_signal_handler = handler_type(self._signal_handler_windows)

    windll.kernel32.SetConsoleCtrlHandler(
        self._windows_signal_handler,
        True,
    )


def _signal_handler_windows(self, sig: int) -> bool:
    """Handle incoming Windows signals."""
    if sig in (2, 5, 6):
        self._graceful_exit()
        return True
    return False


def _hide_scrollbars_for_exit(self) -> None:
    """Hide mounted scrollbars before painting the final transparent frame."""
    try:
        screen = self.screen
        widgets = (screen, *screen.query(_app.Widget))
    except Exception:
        return

    for widget in widgets:
        with contextlib.suppress(Exception):
            widget.styles.scrollbar_size_vertical = 0
            widget.styles.scrollbar_size_horizontal = 0
            widget.show_vertical_scrollbar = False
            widget.show_horizontal_scrollbar = False
            for scrollbar_name in (
                "_vertical_scrollbar",
                "_horizontal_scrollbar",
                "_scrollbar_corner",
            ):
                scrollbar = getattr(widget, scrollbar_name, None)
                if scrollbar is not None:
                    scrollbar.display = False
            widget.refresh(layout=True, repaint=True)


def _graceful_exit(self, return_code: Optional[int] = None) -> None:
    """Exit from Celune gracefully.

    Args:
        return_code: Optional value for Textual to return after shutdown.
    """
    if self.cur_state == "exiting":
        return
    if self.cur_state != "restarting":
        self.cur_state = "exiting"

    def finish_exit() -> None:
        """Finish shutdown after the visible UI has faded away."""
        self._run_shutdown_step(self._shutdown_runtime)
        if return_code is None:
            self.exit()
        else:
            self.exit(return_code=return_code)

    def fade_out() -> None:
        """Fade the mounted Textual screen before requesting unmount."""

        def finish_fade() -> None:
            """Paint one final fully transparent frame before unmounting."""
            try:
                self._hide_scrollbars_for_exit()
                self.screen.styles.opacity = 0.0
                self.screen.refresh(repaint=True)
                self.call_after_refresh(finish_exit)
            except Exception:
                finish_exit()

        try:
            self._animate_opacity(
                self.screen,
                0.0,
                on_complete=finish_fade,
                duration=_app._EXIT_FADE_SECONDS,
            )
        except Exception:
            finish_exit()

    if threading.current_thread() is threading.main_thread():
        fade_out()
        return

    try:
        self.call_from_thread(fade_out)
    except RuntimeError:
        finish_exit()


def _run_shutdown_step(self, callback: Callable[[], None]) -> None:
    """Run one shutdown action without allowing cleanup to crash the UI."""
    try:
        callback()
    except Exception as exc:
        self._report_shutdown_error(exc)


def _report_shutdown_error(self, exc: Exception) -> None:
    """Write a shutdown error to both the log and original terminal stream."""
    log_level = getattr(
        self.celune,
        "log_level",
        self._startup_log_level,
    )
    message = _app.format_error_message(
        _app.string("celune.internal_error"),
        exc,
        _app.resolve_log_level(log_level, self._startup_log_level),
    )
    with contextlib.suppress(Exception):
        self._persist_log_entry(message, "error")

    stream = self._old_stderr or sys.__stderr__
    if stream is None:
        return
    with contextlib.suppress(OSError, ValueError):
        stream.write(f"{message}\n")
        stream.flush()


def _shutdown_runtime(self) -> None:
    """Stop live input and close the core at most once."""
    with self._interaction_state.runtime_shutdown_lock:
        if self._interaction_state.runtime_shutdown_complete:
            return
        try:
            try:
                self._shutdown_live_vc_recording()
            except Exception as exc:
                self._report_shutdown_error(exc)
            try:
                self._unbind_agent_events()
            except Exception as exc:
                self._report_shutdown_error(exc)
            try:
                if self.celune is not None:
                    self.celune.close()
            except Exception as exc:
                self._report_shutdown_error(exc)
        finally:
            self._interaction_state.runtime_shutdown_complete = True


def graceful_exit(self) -> None:
    """Exit the UI through the same graceful shutdown path as internal callers."""
    self._graceful_exit()


def tutorial_token(self) -> int:
    """Return the active tutorial cancellation token.

    Returns:
        int: The tutorial token currently used to invalidate pending tutorial work.
    """
    return self._tutorial_token


def tutorial_active(self) -> bool:
    """Return whether a tutorial flow is currently active.

    Returns:
        bool: ``True`` when tutorial work is active, otherwise ``False``.
    """
    return self._tutorial_active


def _split_command_input(text: str) -> list[str]:
    """Split one slash-command string into a command name and arguments."""
    posix = os.name != "nt"
    parts = shlex.split(text, posix=posix)
    if posix:
        return parts

    normalized: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            normalized.append(part[1:-1])
        else:
            normalized.append(part)
    return normalized


def split_command_input(text: str) -> list[str]:
    """Split one slash-command string into a command name and arguments.

    Args:
        text: The command input to split.

    Returns:
        list[str]: The parsed command name followed by its arguments.
    """
    posix = os.name != "nt"
    parts = shlex.split(text, posix=posix)
    if posix:
        return parts

    normalized: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {"'", '"'}:
            normalized.append(part[1:-1])
        else:
            normalized.append(part)
    return normalized


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(target, {name: globals()[name] for name in __all__})
