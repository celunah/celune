# SPDX-License-Identifier: Apache-2.0
"""Extracted CeluneUI capture methods."""

from __future__ import annotations

import contextlib
import queue as queue_module
import re
import threading
import time
from collections.abc import Callable
from typing import Optional, Union, cast
from uuid import uuid4

from textual.color import Color

from . import app as _app
from ..binding import install_class_functions
from .constants import _CAPTION_FADE_SECONDS

__all__ = (
    "_acquire_recording_component_lease",
    "_animate_opacity",
    "_append_vc_preroll_audio_locked",
    "_cancel_vc_recording",
    "_clear_caption_state",
    "_clear_caption_timers",
    "_clear_vc_preroll_locked",
    "_clear_vc_recording_state",
    "_close_live_vad",
    "_complete_persona_transcription",
    "_crossfade_vc_overlap",
    "_enqueue_vc_submission_chunk",
    "_finish_vc_submission_queue",
    "_flush_vc_recording_buffer_locked",
    "_flush_vc_recording_chunk_locked",
    "_format_vc_pitch_shift",
    "_hide_caption_widgets",
    "_is_voice_conversion_mode",
    "_join_vc_recording_threads",
    "_normalize_vc_overlap_audio",
    "_persona_recording_active",
    "_persona_recording_audio_locked",
    "_persona_speech_end_delay_seconds",
    "_persona_speech_language",
    "_persona_speech_model_id",
    "_persona_transcription_worker",
    "_prepend_vc_preroll_locked",
    "_queue_persona_recording_item_locked",
    "_refresh_persona_availability",
    "_request_persona_recording_stop",
    "_request_vc_recording_feedback_stop",
    "_reset_playback_widgets",
    "_set_persona_recording_text",
    "_set_vc_controls_visibility",
    "_show_caption_widgets",
    "_shutdown_persona_recording",
    "_shutdown_vc_stream",
    "_start_persona_recording",
    "_stop_live_vc_backend",
    "_stop_vc_recording_for_feedback",
    "_stop_vc_recording_stream",
    "_vc_feedback_min_capture_frames",
    "_vc_feedback_rise_detected",
    "_vc_input_has_voice",
    "_vc_input_rms",
    "_vc_live_chunk_frames",
    "_vc_live_chunk_overlap_frames",
    "_vc_recording_active",
    "_vc_vad_hangover_frames",
    "_vc_vad_preroll_frames",
    "_with_darkened_brightness",
    "change_input_state",
    "change_voice_lock_state",
    "on_uilog_message",
    "pulse_border",
    "refresh_vc_controls",
    "safe_caption_progress",
    "safe_log",
    "safe_log_dev",
    "safe_status",
    "set_vc_f0_condition",
    "set_vc_pitch_shift",
    "toggle_persona_recording",
    "tts_caption",
)


def _reset_playback_widgets(self) -> None:
    """Clear transient caption state and restore the normal playback bar."""
    self._clear_caption_timers()
    self._caption_transition_token += 1
    self._caption_active = False
    self._caption_transitioning = False
    if self.caption is not None:
        self.caption.display = False
        self.caption.styles.height = 0
        self.caption.styles.opacity = 0.0
    if self.progress_label is not None:
        self.progress_label.set_progress(None, None)
    self._clear_caption_state()
    self._restore_progress_bar(force=True)


def safe_caption_progress(
    self, progress: Optional[float], total: Optional[float] = None
) -> None:
    """Update the active caption from speech-only playback progress."""
    if self.cur_state == "exiting" or not self._caption_active:
        return

    def update_caption() -> None:
        if not self._caption_active or total is None or total <= 0:
            return
        current = 0.0 if progress is None else progress
        fraction = max(0.0, min(1.0, current / total))
        if fraction < self._caption_progress:
            return
        caption_finished = fraction >= 1.0
        self._caption_progress = fraction
        visible_sentence, visible_words = self._caption_words_for_progress(
            self._caption_progress
        )
        rendered_text = " ".join(visible_sentence)
        if (
            visible_words == self._caption_visible_words
            and rendered_text == self._caption_rendered_text
        ):
            if caption_finished:
                self._hide_caption_widgets()
            return
        self._caption_visible_words = visible_words
        self._caption_rendered_text = rendered_text
        if self.caption is not None:
            self.caption.update(rendered_text)
        if caption_finished:
            self._hide_caption_widgets()

    self._run_on_ui_thread(update_caption)


def _animate_opacity(
    self,
    widget: _app.Widget,
    opacity: float,
    on_complete: Optional[Callable[[], None]] = None,
    token: Optional[int] = None,
    duration: float = _CAPTION_FADE_SECONDS,
) -> None:
    """Fade one widget through its mutable CSS opacity property.

    Args:
        widget: Widget whose opacity should be animated.
        opacity: Target opacity for the widget.
        on_complete: Optional callback after the final animation frame.
        token: Optional caption transition token that cancels stale fades.
        duration: Total fade duration in seconds.
    """
    callback = on_complete or (lambda: None)
    if not getattr(widget, "is_attached", False):
        widget.styles.opacity = opacity
        callback()
        return

    start_opacity = widget.styles.opacity
    steps = 6
    frame_delay = duration / steps

    def animate_frame(index: int) -> None:
        if token is not None and token != self._caption_transition_token:
            return
        progress = min(1.0, index / steps)
        widget.styles.opacity = start_opacity + (opacity - start_opacity) * progress
        if index >= steps:
            callback()
            return
        timer = self.set_timer(frame_delay, lambda: animate_frame(index + 1))
        if timer is not None:
            self._caption_timers.append(timer)

    animate_frame(0)


def _clear_caption_timers(self) -> None:
    """Stop pending caption fade timers before starting a new transition."""
    for timer in self._caption_timers:
        timer.stop()
    self._caption_timers.clear()


def _clear_caption_state(self) -> None:
    """Clear caption content and its progress bookkeeping."""
    self._caption_text = ""
    self._caption_words = ()
    self._caption_sentences = ()
    self._caption_word_timings = ()
    self._caption_audio_duration = 0.0
    self._caption_rendered_text = ""
    self._caption_visible_words = 0
    self._caption_progress = 0.0


def _show_caption_widgets(self) -> None:
    """Fade the caption in while fading the playback bar out."""
    if self.caption is None or self.progress_bar is None:
        return

    token = self._caption_transition_token
    self._set_progress_row_display(False)
    if self.progress_label is not None:
        self.progress_label.set_progress(None, None)
    self.caption.display = True
    self.caption.styles.height = 1
    self.caption.styles.opacity = 0.0
    self.progress_bar.display = True
    self.progress_bar.styles.opacity = 1.0

    def hide_progress_bar() -> None:
        if self._caption_transition_token == token and self._caption_active:
            self.progress_bar.display = False
            self.caption.styles.height = 1
            self._animate_opacity(self.caption, 1.0, token=token)

    self._animate_opacity(
        self.progress_bar,
        0.0,
        hide_progress_bar,
        token=token,
    )


def _hide_caption_widgets(self) -> None:
    """Fade the completed caption away and restore the playback bar."""
    if self.cur_state == "exiting":
        return
    if threading.current_thread() is not threading.main_thread():
        with contextlib.suppress(LookupError, RuntimeError, _app.ScreenStackError):
            self.call_from_thread(self._hide_caption_widgets)
        return

    caption_active = self._caption_active
    if not caption_active and self._caption_transitioning:
        return
    self._clear_caption_timers()
    self._caption_transition_token += 1
    self._caption_transitioning = True
    self._set_progress_row_display(False)
    if self.progress_label is not None:
        self.progress_label.set_progress(None, None)
    self._caption_active = False
    if self.caption is None or self.progress_bar is None:
        self._caption_active = False
        self._caption_transitioning = False
        self._clear_caption_state()
        return

    if not caption_active:
        self.caption.display = False
        self.caption.styles.height = 0
        self.caption.styles.opacity = 0.0
        self._caption_transitioning = False
        self._restore_progress_bar(force=True)
        self._clear_caption_state()
        return

    token = self._caption_transition_token
    self.caption.styles.height = 1

    def restore_progress_bar() -> None:
        if self._caption_transition_token != token:
            return
        self.caption.display = False
        self.caption.styles.height = 0
        self.caption.styles.opacity = 0.0
        self._caption_transitioning = False
        self._restore_progress_bar(force=True)
        self.progress_bar.styles.opacity = 0.0
        self._animate_opacity(self.progress_bar, 1.0, token=token)
        self._clear_caption_state()

    self._animate_opacity(
        self.caption,
        0.0,
        restore_progress_bar,
        token=token,
    )


def tts_caption(self, caption: Optional[str]) -> None:
    """Show a speech caption and reveal its words with played-audio progress."""
    if (
        self.cur_state == "exiting"
        or getattr(self.celune, "test_finished", False)
        or not caption
    ):
        return

    sentences = tuple(
        tuple(sentence.split())
        for sentence in re.split(
            r"(?:(?<=[.!?])\s+|\n+)",
            caption.strip(),
        )
        if sentence.strip()
    )
    words = tuple(word for sentence in sentences for word in sentence)
    if not words:
        return

    def update() -> None:
        self._clear_caption_timers()
        self._caption_transition_token += 1
        self._caption_text = caption
        self._caption_words = words
        self._caption_sentences = sentences
        self._caption_word_timings = ()
        self._caption_audio_duration = 0.0
        self._caption_rendered_text = ""
        self._caption_visible_words = 0
        self._caption_progress = 0.0
        self._caption_active = True
        self._caption_transitioning = False
        if self.caption is not None:
            self.caption.update("")
        self._show_caption_widgets()

    self._run_on_ui_thread(update)


def _with_darkened_brightness(color: Color) -> Color:
    """Return ``color`` with a visibly darker brightness."""
    target_brightness = max(0.0, color.brightness * 0.6)
    return _app.CeluneUI.with_brightness(color, target_brightness)


def pulse_border(self, target: Union[str, _app.Widget]) -> None:
    """Softly pulse a widget border darker and back.

    Args:
        target: Widget or Textual selector for the target widget.
    """
    if threading.current_thread() is not threading.main_thread():
        self.call_from_thread(lambda: self.pulse_border(target))
        return

    duration = 2.06
    steps = 10

    widget = self.query_one(target) if isinstance(target, str) else target
    original_border: tuple[_app.EdgeStyle, ...] = tuple(widget.styles.border)

    if not any(edge_type for edge_type, _ in original_border):
        return

    widget_key = id(widget)
    token = self._border_pulse_tokens.get(widget_key, 0) + 1
    self._border_pulse_tokens[widget_key] = token
    self._border_pulse_widgets[widget_key] = widget

    target_border: tuple[_app.EdgeStyle, ...] = tuple(
        (
            edge_type,
            self._with_darkened_brightness(color) if edge_type else color,
        )
        for edge_type, color in original_border
    )
    steps = max(1, steps)
    duration = max(0.0, duration)
    hold_duration = min(0.2, duration / 3)
    transition_duration = max(0.0, duration - hold_duration)
    frame_delay = transition_duration / (steps * 2) if transition_duration else 0.0

    def set_border(border: tuple[_app.EdgeStyle, ...]) -> None:
        (
            widget.styles.border_top,
            widget.styles.border_right,
            widget.styles.border_bottom,
            widget.styles.border_left,
        ) = border
        widget.refresh(layout=False)

    def apply_blend(progress: float) -> None:
        if self._border_pulse_tokens.get(widget_key) != token:
            return

        eased = progress * progress * (3 - 2 * progress)
        set_border(
            tuple(
                (
                    edge_type,
                    start_color.blend(end_color, eased) if edge_type else start_color,
                )
                for (edge_type, start_color), (_, end_color) in zip(
                    original_border, target_border
                )
            )
        )

    def restore() -> None:
        if self._border_pulse_tokens.get(widget_key) != token:
            return
        widget.styles.border = None
        widget.refresh(layout=False)
        self._border_pulse_tokens.pop(widget_key, None)
        self._border_pulse_widgets.pop(widget_key, None)

    def schedule_frame(index: int, delay: float) -> None:
        if self._border_pulse_tokens.get(widget_key) != token:
            return

        if index >= steps * 2:
            restore()
            return

        def run_frame() -> None:
            if self._border_pulse_tokens.get(widget_key) != token:
                return

            if index < steps:
                apply_blend((index + 1) / steps)
                next_delay = (
                    hold_duration + frame_delay if index + 1 == steps else frame_delay
                )
            else:
                apply_blend(1 - ((index - steps + 1) / steps))
                next_delay = frame_delay

            schedule_frame(index + 1, next_delay)

        if delay <= 0:
            run_frame()
        else:
            self.set_timer(delay, run_frame)

    schedule_frame(0, frame_delay)


def change_voice_lock_state(
    self,
    locked: bool,
    *,
    can_open_menu: Optional[bool] = None,
) -> None:
    """Set voice-cycle and voice-menu availability independently.

    Args:
        locked: Whether clicking to cycle voices should be disabled.
        can_open_menu: Whether holding the button can open the voice menu.
            When omitted, the menu follows the click availability.
    """
    if can_open_menu is None:
        can_open_menu = not locked

    def update() -> None:
        self.style_button.disabled = locked
        if isinstance(self.style_button, _app.VoiceButton):
            self.style_button.hold_enabled = can_open_menu
        self.update_resources()

    self._run_on_ui_thread(update)


def _refresh_persona_availability(self) -> None:
    """Refresh Persona availability in the background for placeholder text."""
    if self._persona_probe_running:
        return

    self._persona_probe_running = True

    def probe() -> None:
        available = self._persona_loaded()

        def apply_result() -> None:
            self._persona_probe_running = False
            if self.cur_state == "exiting":
                return

            changed = self._persona_available != available
            self._persona_available = available
            if changed and not self._input_locked:
                self.input_box.placeholder = self._normal_input_placeholder()

        self._run_on_ui_thread(apply_result)

    threading.Thread(target=probe, daemon=True).start()


def change_input_state(self, locked: bool) -> None:
    """Lock or unlock Celune's UI layer.

    Args:
        locked: Whether user input should be disabled.
    """

    if not locked:
        self._schedule_sleep_timer()

    def update() -> None:
        self._input_locked = locked
        stopped = bool(
            self.celune is not None
            and (
                getattr(self.celune, "test_finished", False)
                or getattr(self.celune, "cur_state", None) == "stopped"
            )
        )
        self.input_box.placeholder = (
            _app.string("ui.stopped_placeholder")
            if stopped
            else _app.string("ui.wait_placeholder")
            if locked
            else self._normal_input_placeholder()
        )
        self.style_button.disabled = locked
        self.refresh_vc_controls()
        self.update_resources()

    self._run_on_ui_thread(update)
    if not locked:
        self._refresh_persona_availability()


def safe_status(self, msg: str, severity: str = "info") -> None:
    """Update current status.

    Args:
        msg: The status text to display.
        severity: The status severity level.
    """
    if self.cur_state == "exiting" or self.status is None:
        return

    if (
        self.celune is not None
        and getattr(self.celune, "test_finished", False)
        and msg != _app.string("status.stopped")
    ):
        return

    if not _app._RUNTIME_DEPENDENCIES_LOADED:
        _app._load_ui_runtime_dependencies()

    if severity not in _app.colors.SEVERITY_COLORS["celune"]:
        self.safe_log(
            f"[WARNING] Unknown severity '{severity}', defaulting to info",
            "warning",
        )
        severity = "info"

    if self._fatal_error_active and severity != "error":
        return

    self.status_severity = severity
    terminal_state, terminal_action = self._terminal_status_for(msg, severity)

    def update() -> None:
        self._status_text = msg
        self._status_marquee_offset = 0
        self._refresh_theme_text()
        self._update_status_label()
        if self._loading_screen is not None:
            self._loading_screen.set_status_message(msg)
        self._set_terminal_status(terminal_state, terminal_action)
        self.update_resources()
        self._publish_webui_timed_update()

    self._run_on_ui_thread(update)


def safe_log(
    self,
    msg: str,
    severity: str = "info",
    *,
    loglevel: _app.LogLevel = "info",
) -> None:
    """Log a message.

    Args:
        msg: The log line to append.
        severity: The log severity level.
        loglevel: The minimum configured log level required to append the line.
    """
    if self.cur_state == "exiting":
        return

    levels = {"info": 0, "verbose": 1, "debug": 2}
    active_log_level = _app.resolve_log_level(
        getattr(self.celune, "log_level", None),
        self._startup_log_level,
    )
    if levels.get(active_log_level, 0) < levels.get(loglevel, 0):
        return

    if not _app._RUNTIME_DEPENDENCIES_LOADED:
        _app._load_ui_runtime_dependencies()

    if severity not in _app.colors.SEVERITY_COLORS["celune"]:
        severity = "info"

    with self._log_history_lock:
        self.log_history.append((msg, severity))
    self._persist_log_entry(msg, severity)
    if loglevel == "info" and self._loading_screen is not None:
        self._run_on_ui_thread(lambda: self._update_loading_log(msg))
    if self.logs is None:
        return

    entry = _app.Text(msg, style=self._severity_color(severity))

    if threading.current_thread() is threading.main_thread():
        self.logs.write(entry)
        self._rendered_log_count += 1
    else:
        self.post_message(_app.UILogMessage(msg, severity))


def on_uilog_message(self, message: _app.UILogMessage) -> None:
    """Reconcile background log history on Textual's application thread.

    Args:
        message: Background log message that woke the reconciliation handler.
    """
    del message
    self._refresh_logs()


def safe_log_dev(self, msg: str, severity: str = "info") -> None:
    """Log a message.

    Args:
        msg: The log line to append.
        severity: The log severity level.
    """
    self.safe_log(msg, severity, loglevel="verbose")


def _is_voice_conversion_mode(self) -> bool:
    """Return whether the attached Celune instance is running in VC mode."""
    return bool(
        self.celune is not None and getattr(self.celune, "vc_backend", None) is not None
    )


def _format_vc_pitch_shift(value: int) -> str:
    """Return one signed semitone label for the VC pitch control."""
    return f"{value:+d}"


def _set_vc_controls_visibility(self, visible: bool) -> None:
    """Show or hide the VC-only controls in the bottom input row."""
    if self.vc_mode_button is not None:
        self.vc_mode_button.display = visible
    if self.vc_pitch_button is not None:
        self.vc_pitch_button.display = visible


def refresh_vc_controls(self) -> None:
    """Refresh VC control labels and enabled state from the current engine state."""
    if (
        self.vc_mode_button is None
        or self.vc_pitch_button is None
        or self.celune is None
    ):
        return

    is_vc_mode = self._is_voice_conversion_mode()
    self._set_vc_controls_visibility(is_vc_mode)
    if not is_vc_mode:
        self._cancel_vc_recording(announce=False)
    f0_condition = bool(getattr(self.celune, "vc_f0_condition", False))
    pitch_shift = int(getattr(self.celune, "vc_pitch_shift", 0))
    self.vc_mode_button.label = _app.string(
        "ui.vc_mode_sing" if f0_condition else "ui.vc_mode_talk"
    )
    self.vc_pitch_button.label = _app.string(
        "ui.vc_pitch_button",
        value=self._format_vc_pitch_shift(pitch_shift),
    )
    self.vc_mode_button.disabled = (not is_vc_mode) or self._input_locked
    self.vc_pitch_button.disabled = (not is_vc_mode) or self._input_locked


def set_vc_f0_condition(self, enabled: bool, announce: bool = True) -> None:
    """Update the active VC talk-vs-sing mode in the UI and backend state.

    Args:
        enabled: Whether to enable sing-mode F0 conditioning.
        announce: Whether to log the new mode to the user.
    """
    if self.celune is None:
        return

    self.celune.vc_f0_condition = enabled
    backend = getattr(self.celune, "vc_backend", None)
    if backend is not None:
        from ..utils import available

        if available("f0_condition", obj=backend):
            backend.f0_condition = enabled
    self.refresh_vc_controls()

    if announce:
        self.safe_log(
            _app.string(
                "ui.vc_mode_changed",
                mode=_app.string("ui.vc_mode_sing" if enabled else "ui.vc_mode_talk"),
            )
        )


def set_vc_pitch_shift(self, value: int, announce: bool = True) -> None:
    """Update the active VC pitch-shift value in the UI and backend state.

    Args:
        value: The requested pitch shift in semitones before clamping.
        announce: Whether to log the new pitch shift to the user.
    """
    if self.celune is None:
        return

    clamped = _app.clamp_vc_pitch_shift(value)
    self.celune.vc_pitch_shift = clamped
    backend = getattr(self.celune, "vc_backend", None)
    if backend is not None:
        from ..utils import available

        if available("pitch_shift", obj=backend):
            backend.pitch_shift = clamped
    self.refresh_vc_controls()

    if announce:
        self.safe_log(
            _app.string(
                "ui.vc_pitch_changed",
                value=self._format_vc_pitch_shift(clamped),
            )
        )


def _persona_recording_active(self) -> bool:
    """Return whether Persona microphone capture is active."""
    return self._persona_recording_stream is not None


def _acquire_recording_component_lease(
    self,
    operation_id: str,
    components: tuple[_app.ComponentLockName, ...],
) -> tuple[bool, Optional[_app.ComponentLockLease]]:
    """Reserve the resources required by one microphone operation."""
    if self.celune is None:
        return False, None
    manager = getattr(self.celune, "component_locks", None)
    if manager is None:
        return True, None

    owner = _app.ComponentLockOwner(operation_id=operation_id)
    acquisition, lease = manager.try_acquire_lease(
        tuple(_app.ComponentLockRequirement(component) for component in components),
        owner,
    )
    if lease is not None:
        return True, lease

    busy = acquisition.busy
    if busy is not None:
        self.celune._last_component_busy = busy
        labels = ", ".join(component.name for component in busy.components)
        self.safe_log(
            _app.string("pipeline.busy_components", components=labels),
            "warning",
        )
    return False, None


def _persona_speech_model_id(self) -> str:
    """Return the configured Hugging Face Whisper model ID."""
    configured = _app.persona_config(self.celune.config).get("speech_model_id")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return _app.DEFAULT_PERSONA_SPEECH_MODEL_ID


def _persona_speech_language(self) -> Optional[str]:
    """Return a configured Whisper language, or ``None`` for auto-detection."""
    configured = _app.persona_config(self.celune.config).get("speech_language")
    if not isinstance(configured, str) or configured.strip().lower() in {
        "",
        "auto",
    }:
        return None
    return configured.strip()


def _persona_speech_end_delay_seconds(self) -> float:
    """Return the extra VAD silence delay before Persona submission."""
    configured = _app.persona_config(self.celune.config).get("speech_end_delay_seconds")
    if (
        isinstance(configured, (int, float))
        and not isinstance(configured, bool)
        and configured >= 0
    ):
        return float(configured)
    return _app.PERSONA_SPEECH_END_DELAY_SECONDS


def _persona_recording_audio_locked(self) -> _app.npt.NDArray[_app.np.float32]:
    """Return captured Persona audio while holding its lock."""
    if not self._persona_recording_chunks:
        return _app.np.zeros(0, dtype=_app.np.float32)
    return _app.np.concatenate(self._persona_recording_chunks, axis=0).astype(
        _app.np.float32,
        copy=False,
    )


def _queue_persona_recording_item_locked(self, final_value: bool) -> None:
    """Queue a partial or final Persona transcription snapshot."""
    recording_queue = self._persona_recording_queue
    if recording_queue is None:
        return

    audio = self._persona_recording_audio_locked().copy()
    if final_value:
        while True:
            try:
                recording_queue.get_nowait()
            except queue_module.Empty:
                break
        recording_queue.put_nowait((audio, True))
        return

    try:
        recording_queue.put_nowait((audio, False))
    except queue_module.Full:
        with contextlib.suppress(queue_module.Empty):
            recording_queue.get_nowait()
        with contextlib.suppress(queue_module.Full):
            recording_queue.put_nowait((audio, False))


def _set_persona_recording_text(self, transcript: str) -> None:
    """Display a live Whisper transcript in the main input box."""
    prefix = self._persona_recording_text_prefix
    text = f"{prefix} {transcript}".strip() if prefix else transcript.strip()

    def update() -> None:
        if self.cur_state == "exiting" or self.input_box is None:
            return
        self._suppress_input_change = True
        try:
            self.input_box.load_text(text)
        finally:
            self._suppress_input_change = False

    self._run_on_ui_thread(update)


def _complete_persona_transcription(
    self,
    transcript: str,
    prefix: str,
    error: Optional[Exception] = None,
    error_already_reported: bool = False,
) -> None:
    """Submit the final Persona transcript or report its transcription error."""
    if error is not None and not error_already_reported:
        self.safe_log(
            _app.format_error_message(
                _app.string("ui.persona_transcription_failed"),
                error,
                getattr(self.celune, "log_level", "info"),
            ),
            "error",
        )
    if error is not None or error_already_reported:
        self.safe_status(_app.string("ui.idle_status"))
        if self.style_button is not None:
            self.style_button.disabled = self._input_locked
        self.update_resources()
        return

    text = f"{prefix} {transcript}".strip() if prefix else transcript.strip()
    if text:
        self._set_persona_recording_text(transcript)
        self._submit_text(text, process_commands=False)
    else:
        self.safe_log(_app.string("ui.recording_empty"), "warning")
    self.safe_status(_app.string("ui.idle_status"))
    if self.style_button is not None:
        self.style_button.disabled = self._input_locked
    self.update_resources()


def _persona_transcription_worker(
    self,
    recording_queue: queue_module.Queue[tuple[_app.AudioChunk, bool]],
    transcriber: _app.WhisperTranscriber,
    sample_rate: int,
    prefix: str,
) -> None:
    """Transcribe Persona microphone snapshots off the UI thread."""
    partial_error_reported = False
    while True:
        audio, final_value = recording_queue.get()
        transcript = ""
        error: Optional[Exception] = None
        if audio.size:
            try:
                transcript = transcriber.transcribe(audio, sample_rate)
            except Exception as exc:
                error = exc

        if transcript:
            self._set_persona_recording_text(transcript)

        if (
            error is not None
            and (final_value or not partial_error_reported)
            and not final_value
        ):
            partial_error_reported = True
            self.safe_log(
                _app.format_error_message(
                    _app.string("ui.persona_transcription_failed"),
                    error,
                    getattr(self.celune, "log_level", "info"),
                ),
                "warning",
            )

        if not final_value:
            continue

        with self._persona_recording_lock:
            stream = self._persona_recording_stream
            vad = self._persona_recording_vad
            component_lease = self._persona_recording_component_lease
            self._persona_recording_stream = None
            self._persona_recording_queue = None
            self._persona_recording_worker = None
            self._persona_recording_vad = None
            self._persona_recording_transcriber = None
            self._persona_recording_chunks = []
            self._persona_recording_stop_requested = False
            self._persona_recording_speech_started = False
            self._persona_recording_silence_frames = 0
            self._persona_recording_component_lease = None

        self._shutdown_vc_stream(stream)
        self._close_live_vad(vad)
        if component_lease is not None:
            component_lease.release()

        def complete_transcription(
            transcript: str = transcript,
            prefix: str = prefix,
            error: Optional[Exception] = error,
            partial_error_reported: bool = partial_error_reported,
        ) -> None:
            """Complete the captured Persona transcription on the UI thread."""
            self._complete_persona_transcription(
                transcript,
                prefix,
                error,
                error_already_reported=partial_error_reported and error is not None,
            )

        self._run_on_ui_thread(complete_transcription)
        return


def _request_persona_recording_stop(self) -> bool:
    """Queue final Persona audio for transcription and automatic submission."""
    with self._persona_recording_lock:
        if self._persona_recording_stream is None:
            return False
        if self._persona_recording_stop_requested:
            return True
        self._persona_recording_stop_requested = True
        self._queue_persona_recording_item_locked(final_value=True)

    self.safe_status(_app.string("ui.persona_transcribing"))
    return True


def _start_persona_recording(self) -> bool:
    """Start push-to-talk microphone capture for the active Persona."""
    _app._load_ui_runtime_dependencies()
    if (
        self.celune is None
        or self._is_voice_conversion_mode()
        or not self._persona_loaded()
        or not _app.persona_talkback_enabled(self.celune.config)
    ):
        return False
    if self._persona_recording_active():
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
        device_info = (
            cast(dict[str, _app.AudioDeviceInfoValue], dict(direct_device_info))
            if direct_device_info is not None
            else cast(
                dict[str, _app.AudioDeviceInfoValue],
                _app.sd.query_devices(device=input_device, kind="input"),
            )
        )
    except Exception as exc:
        self.safe_log(
            _app.format_error_message(
                _app.string("ui.recording_open_input_failed"),
                exc,
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
    channel_count = 2 if channels >= 2 else 1
    vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate) + int(
        sample_rate * self._persona_speech_end_delay_seconds()
    )
    ai_vad = _app.create_live_voice_activity_detector()
    recording_queue: queue_module.Queue[tuple[_app.AudioChunk, bool]] = (
        queue_module.Queue(maxsize=1)
    )
    transcriber = _app.WhisperTranscriber(
        self._persona_speech_model_id(),
        language=self._persona_speech_language(),
    )
    prefix = self.input_box.text.strip() if self.input_box is not None else ""
    recording_started_at = time.monotonic()
    should_stop = False

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
        nonlocal should_stop

        callback_audio = _app.np.asarray(indata, dtype=_app.np.float32).copy()
        if ai_vad is not None:
            try:
                voice_detected = ai_vad.has_voice(callback_audio, sample_rate)
            except (RuntimeError, AssertionError, ValueError):
                ai_vad.reset()
                voice_detected = self._vc_input_has_voice(callback_audio)
        else:
            voice_detected = self._vc_input_has_voice(callback_audio)

        with self._persona_recording_lock:
            if (
                self._persona_recording_stream is None
                or self._persona_recording_stop_requested
            ):
                return

            if voice_detected:
                self._persona_recording_speech_started = True
                self._persona_recording_silence_frames = 0
            elif self._persona_recording_speech_started:
                self._persona_recording_silence_frames += len(callback_audio)

            if self._persona_recording_speech_started:
                self._persona_recording_chunks.append(callback_audio)
                if time.monotonic() - self._persona_recording_last_partial_at >= 0.8:
                    self._queue_persona_recording_item_locked(final_value=False)
                    self._persona_recording_last_partial_at = time.monotonic()

            if (
                self._persona_recording_speech_started
                and self._persona_recording_silence_frames >= vad_hangover_frames
            ) or (
                not self._persona_recording_speech_started
                and time.monotonic() - recording_started_at
                >= _app.PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS
            ):
                self._persona_recording_stop_requested = True
                self._queue_persona_recording_item_locked(final_value=True)
                should_stop = True

        if should_stop:
            self.safe_status(_app.string("ui.persona_transcribing"))

    worker: Optional[threading.Thread] = None
    stream: Optional[_app.sd.InputStream] = None
    acquired, component_lease = self._acquire_recording_component_lease(
        f"persona-recording:{uuid4()}",
        (_app.ComponentLockName.MICROPHONE, _app.ComponentLockName.ASR),
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
        )
        worker = threading.Thread(
            target=self._persona_transcription_worker,
            args=(recording_queue, transcriber, sample_rate, prefix),
            daemon=True,
        )
        with self._persona_recording_lock:
            self._persona_recording_stream = stream
            self._persona_recording_queue = recording_queue
            self._persona_recording_worker = worker
            self._persona_recording_vad = ai_vad
            self._persona_recording_transcriber = transcriber
            self._persona_recording_sample_rate = sample_rate
            self._persona_recording_chunks = []
            self._persona_recording_silence_frames = 0
            self._persona_recording_speech_started = False
            self._persona_recording_stop_requested = False
            self._persona_recording_text_prefix = prefix
            self._persona_recording_last_partial_at = time.monotonic()
            self._persona_recording_component_lease = component_lease
        stream.start()
        worker.start()
    except Exception as exc:
        with self._persona_recording_lock:
            stream = self._persona_recording_stream
            vad = self._persona_recording_vad or ai_vad
            self._persona_recording_stream = None
            self._persona_recording_queue = None
            self._persona_recording_worker = None
            self._persona_recording_vad = None
            self._persona_recording_transcriber = None
            self._persona_recording_chunks = []
            self._persona_recording_stop_requested = True
            self._persona_recording_component_lease = None
        self._shutdown_vc_stream(stream)
        self._close_live_vad(vad)
        if component_lease is not None:
            component_lease.release()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        self.safe_log(
            _app.format_error_message(
                _app.string(
                    "ui.recording_start_failed",
                    label=_app.string("ui.audio_input_label"),
                ),
                exc,
                getattr(self.celune, "log_level", "info"),
            ),
            "error",
        )
        return False

    self.safe_log(_app.string("ui.persona_recording_started"), "info")
    self.safe_status(_app.string("ui.persona_recording_listening"))
    self.update_resources()
    return True


def toggle_persona_recording(self) -> bool:
    """Toggle Persona microphone capture and final transcription.

    Returns:
        ``True`` when recording was stopped or started successfully.
    """
    if self._persona_recording_active():
        return self._request_persona_recording_stop()
    return self._start_persona_recording()


def _shutdown_persona_recording(self) -> None:
    """Stop Persona microphone capture without submitting a final utterance."""
    with self._persona_recording_lock:
        stream = self._persona_recording_stream
        vad = self._persona_recording_vad
        component_lease = self._persona_recording_component_lease
        recording_queue = self._persona_recording_queue
        worker = self._persona_recording_worker
        self._persona_recording_stream = None
        self._persona_recording_queue = None
        self._persona_recording_worker = None
        self._persona_recording_vad = None
        self._persona_recording_transcriber = None
        self._persona_recording_chunks = []
        self._persona_recording_stop_requested = True
        self._persona_recording_component_lease = None
        if recording_queue is not None:
            while True:
                try:
                    recording_queue.get_nowait()
                except queue_module.Empty:
                    break
            with contextlib.suppress(queue_module.Full):
                recording_queue.put_nowait(
                    (_app.np.zeros(0, dtype=_app.np.float32), True)
                )
    self._shutdown_vc_stream(stream)
    self._close_live_vad(vad)
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=2.0)
    if component_lease is not None:
        component_lease.release()


def _vc_recording_active(self) -> bool:
    """Return whether live VC recording is active in the TUI."""
    return self._vc_recording_stream is not None


def _vc_input_rms(audio: _app.npt.NDArray[_app.np.float32]) -> float:
    """Return RMS energy for one microphone callback buffer."""
    return _app.vc_input_rms(audio)


def _vc_feedback_rise_detected(
    previous_rms: float,
    current_rms: float,
) -> bool:
    """Return whether the latest RMS jump looks like runaway feedback."""
    if previous_rms < _app._VC_FEEDBACK_RMS_MIN_PREVIOUS:
        return False
    if current_rms < _app._VC_FEEDBACK_RMS_MIN_CURRENT:
        return False
    if current_rms < previous_rms * _app._VC_FEEDBACK_RMS_RISE_RATIO:
        return False
    return (current_rms - previous_rms) >= _app._VC_FEEDBACK_RMS_RISE_DELTA


def _vc_feedback_min_capture_frames(sample_rate: int) -> int:
    """Return the minimum capture length before feedback auto-stop is allowed."""
    return max(1, int(sample_rate * _app._VC_FEEDBACK_MIN_CAPTURE_SECONDS))


def _request_vc_recording_feedback_stop(self) -> None:
    """Request a feedback-triggered recording stop on a dedicated thread."""
    stop_thread = threading.Thread(
        target=self._stop_vc_recording_for_feedback,
        daemon=True,
    )
    self._vc_recording_stop_thread = stop_thread
    stop_thread.start()


def _flush_vc_recording_buffer_locked(self) -> Optional[_app.AudioChunk]:
    """Return and clear the buffered microphone chunk accumulator."""
    if not self._vc_recording_chunks:
        return None
    audio = _app.np.concatenate(self._vc_recording_chunks, axis=0)
    self._vc_recording_chunks = []
    self._vc_recording_buffered_frames = 0
    return audio


def _flush_vc_recording_chunk_locked(
    self,
    keep_tail_frames: int = 0,
) -> Optional[_app.AudioChunk]:
    """Return one buffered VC chunk while optionally retaining a tail overlap."""
    audio = self._flush_vc_recording_buffer_locked()
    if audio is None or keep_tail_frames <= 0:
        return audio

    if len(audio) <= keep_tail_frames:
        self._vc_recording_chunks = [audio]
        self._vc_recording_buffered_frames = len(audio)
        return None

    retained = _app.np.asarray(audio[-keep_tail_frames:], dtype=_app.np.float32).copy()
    flushed = _app.np.asarray(audio[:-keep_tail_frames], dtype=_app.np.float32).copy()
    self._vc_recording_chunks = [retained]
    self._vc_recording_buffered_frames = len(retained)
    return flushed


def _vc_vad_hangover_frames(sample_rate: int) -> int:
    """Return how many trailing silent frames to tolerate before flushing."""
    return _app.vc_vad_hangover_frames(sample_rate)


def _vc_vad_preroll_frames(sample_rate: int) -> int:
    """Return how much recent pre-speech audio to keep before VAD triggers."""
    return _app.vc_vad_preroll_frames(sample_rate)


def _vc_live_chunk_frames(sample_rate: int) -> int:
    """Return how much active speech to collect before a live VC flush."""
    return _app.vc_live_chunk_frames(sample_rate)


def _vc_live_chunk_overlap_frames(sample_rate: int) -> int:
    """Return how much tail audio to retain between live VC chunks."""
    return _app.vc_live_chunk_overlap_frames(sample_rate)


def _append_vc_preroll_audio_locked(
    self,
    audio: _app.npt.NDArray[_app.np.float32],
    max_frames: int,
) -> None:
    """Retain only the newest pre-speech audio frames for VAD onset recovery."""
    copied = _app.np.asarray(audio, dtype=_app.np.float32).copy()
    self._vc_recording_preroll_chunks.append(copied)
    self._vc_recording_preroll_frames += len(copied)

    while (
        self._vc_recording_preroll_chunks
        and self._vc_recording_preroll_frames > max_frames
    ):
        overflow_frames = self._vc_recording_preroll_frames - max_frames
        oldest = self._vc_recording_preroll_chunks[0]
        if len(oldest) <= overflow_frames:
            self._vc_recording_preroll_chunks.pop(0)
            self._vc_recording_preroll_frames -= len(oldest)
            continue
        trimmed = _app.np.asarray(
            oldest[overflow_frames:], dtype=_app.np.float32
        ).copy()
        self._vc_recording_preroll_chunks[0] = trimmed
        self._vc_recording_preroll_frames -= overflow_frames
        break


def _prepend_vc_preroll_locked(self) -> None:
    """Move retained pre-speech audio into the active VC speech buffer."""
    if not self._vc_recording_preroll_chunks:
        return
    self._vc_recording_chunks = [
        *self._vc_recording_preroll_chunks,
        *self._vc_recording_chunks,
    ]
    self._vc_recording_buffered_frames += self._vc_recording_preroll_frames
    self._vc_recording_preroll_chunks = []
    self._vc_recording_preroll_frames = 0


def _clear_vc_preroll_locked(self) -> None:
    """Discard any retained pre-speech VC audio."""
    self._vc_recording_preroll_chunks = []
    self._vc_recording_preroll_frames = 0


def _vc_input_has_voice(audio: _app.npt.NDArray[_app.np.float32]) -> bool:
    """Return whether one microphone callback likely contains voice activity."""
    _app._load_ui_runtime_dependencies()
    return _app.vc_input_has_voice(audio)


def _normalize_vc_overlap_audio(
    audio: _app.npt.NDArray[_app.np.float32],
) -> _app.npt.NDArray[_app.np.float32]:
    """Normalize one VC overlap chunk into valid mono or stereo time-first audio."""
    _app._load_ui_runtime_dependencies()
    normalized = _app.np.asarray(audio, dtype=_app.np.float32)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim != 2:
        raise ValueError(f"expected 1D or 2D VC overlap audio, got {normalized.shape}")
    if normalized.shape[1] == 1:
        return normalized[:, 0]
    if normalized.shape[1] == 2:
        return normalized
    raise ValueError(
        f"expected mono or stereo VC overlap audio, got {normalized.shape}"
    )


def _crossfade_vc_overlap(
    self,
    previous_tail: _app.npt.NDArray[_app.np.float32],
    current_head: _app.npt.NDArray[_app.np.float32],
) -> _app.npt.NDArray[_app.np.float32]:
    """Crossfade two same-rate VC overlap regions into one seamless bridge."""
    _app._load_ui_runtime_dependencies()
    overlap_frames = min(len(previous_tail), len(current_head))
    if overlap_frames <= 0:
        return _app.np.zeros((0, 2), dtype=_app.np.float32)

    previous = self._normalize_vc_overlap_audio(previous_tail[-overlap_frames:])
    current = self._normalize_vc_overlap_audio(current_head[:overlap_frames])

    if previous.ndim != current.ndim:
        if previous.ndim == 1:
            previous = _app.np.column_stack((previous, previous))
        if current.ndim == 1:
            current = _app.np.column_stack((current, current))

    fade = _app.np.linspace(0.0, 1.0, overlap_frames, dtype=_app.np.float32)
    if previous.ndim == 2:
        fade = fade[:, None]

    return _app.np.asarray(
        (previous * (1.0 - fade)) + (current * fade),
        dtype=_app.np.float32,
    )


def _enqueue_vc_submission_chunk(
    submission_queue: queue_module.Queue[
        Optional[tuple[_app.AudioChunk, int, str, bool]]
    ],
    item: tuple[_app.AudioChunk, int, str, bool],
) -> None:
    """Queue one live VC chunk while dropping only the stalest backlog item."""
    try:
        submission_queue.put_nowait(item)
        return
    except queue_module.Full:
        pass

    with contextlib.suppress(queue_module.Empty):
        submission_queue.get_nowait()

    with contextlib.suppress(queue_module.Full):
        submission_queue.put_nowait(item)


def _finish_vc_submission_queue(
    submission_queue: Optional[
        queue_module.Queue[Optional[tuple[_app.AudioChunk, int, str, bool]]]
    ],
    final_item: Optional[tuple[_app.AudioChunk, int, str, bool]] = None,
) -> None:
    """Flush stale live VC chunks and end the submission worker."""
    if submission_queue is None:
        return

    with contextlib.suppress(queue_module.Empty):
        while True:
            submission_queue.get_nowait()

    if final_item is not None:
        with contextlib.suppress(queue_module.Full):
            submission_queue.put_nowait(final_item)

    with contextlib.suppress(queue_module.Full):
        submission_queue.put_nowait(None)


def _clear_vc_recording_state(self) -> None:
    """Clear transient VC recording buffers after stop or cancel."""
    vad = self._vc_recording_vad
    component_lease = self._vc_recording_component_lease
    self._close_live_vad(vad)
    self._vc_recording_stream = None
    self._vc_recording_chunks = []
    self._vc_recording_buffered_frames = 0
    self._vc_recording_captured_frames = 0
    self._vc_recording_feedback_detected = False
    self._vc_recording_feedback_spike_count = 0
    self._vc_recording_sample_rate = 0
    self._vc_recording_label = _app.string("ui.audio_input_label")
    self._vc_recording_preroll_chunks = []
    self._vc_recording_preroll_frames = 0
    self._vc_recording_previous_rms = 0.0
    self._vc_recording_silence_frames = 0
    self._vc_recording_speech_started = False
    self._vc_recording_submission_queue = None
    self._vc_recording_stop_thread = None
    self._vc_recording_worker = None
    self._vc_recording_vad = None
    self._vc_recording_component_lease = None
    if component_lease is not None:
        component_lease.release()


def _stop_vc_recording_stream(
    self,
) -> tuple[
    Optional[_app.sd.InputStream],
    Optional[_app.AudioChunk],
    int,
    str,
    Optional[queue_module.Queue[Optional[tuple[_app.AudioChunk, int, str, bool]]]],
    int,
    Optional[threading.Thread],
    Optional[threading.Thread],
]:
    """Stop the active VC recording stream and return any pending live-state data."""
    stream = self._vc_recording_stream
    buffered_audio = self._flush_vc_recording_buffer_locked()
    sample_rate = self._vc_recording_sample_rate
    label = self._vc_recording_label
    submission_queue = self._vc_recording_submission_queue
    captured_frames = self._vc_recording_captured_frames
    stop_thread = self._vc_recording_stop_thread
    worker = self._vc_recording_worker
    self._clear_vc_recording_state()
    return (
        stream,
        buffered_audio,
        sample_rate,
        label,
        submission_queue,
        captured_frames,
        stop_thread,
        worker,
    )


def _shutdown_vc_stream(stream: Optional[_app.sd.InputStream]) -> None:
    """Stop and close one VC input stream outside the recording lock."""
    if stream is None:
        return

    with contextlib.suppress(Exception):
        stream.stop()
    with contextlib.suppress(Exception):
        stream.close()


def _close_live_vad(vad: Optional[_app.LiveVoiceActivityDetector]) -> None:
    """Stop one optional live VAD while preserving lightweight test doubles."""
    close = getattr(vad, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _stop_live_vc_backend(self) -> None:
    """Reset the active backend's native live conversion state."""
    celune = self.celune
    if celune is None:
        return
    backend = getattr(celune, "vc_backend", None)
    stop_live = getattr(backend, "stop_live", None)
    if not callable(stop_live):
        return

    def reset_backend() -> None:
        with contextlib.suppress(Exception):
            stop_live()

    threading.Thread(
        target=reset_backend,
        name="celune-live-vc-reset",
        daemon=True,
    ).start()


def _join_vc_recording_threads(
    stop_thread: Optional[threading.Thread],
    worker: Optional[threading.Thread],
    timeout: float = 2.0,
) -> None:
    """Wait briefly for live VC helper threads to finish."""
    for thread in (stop_thread, worker):
        if thread is None or thread is threading.current_thread():
            continue
        with contextlib.suppress(Exception):
            thread.join(timeout=timeout)


def _cancel_vc_recording(self, announce: bool = True) -> bool:
    """Stop VC recording without submitting audio for conversion."""
    if not self._vc_recording_active():
        return False

    with self._vc_recording_lock:
        (
            stream,
            _audio,
            _sample_rate,
            label,
            submission_queue,
            _captured_frames,
            stop_thread,
            worker,
        ) = self._stop_vc_recording_stream()
        self._finish_vc_submission_queue(submission_queue)
    self._shutdown_vc_stream(stream)
    self._stop_live_vc_backend()
    self._join_vc_recording_threads(stop_thread, worker)

    if announce:
        self.safe_log(_app.string("ui.recording_stopped", label=label), "info")
    self._set_terminal_status("ready", _app.string("osc.action_idle"))
    return True


def _stop_vc_recording_for_feedback(self) -> None:
    """Stop live VC recording after detecting a sudden feedback-like RMS spike."""
    if not self._vc_recording_active():
        return

    with self._vc_recording_lock:
        (
            stream,
            buffered_audio,
            sample_rate,
            label,
            submission_queue,
            _captured_frames,
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

    self.safe_log(_app.string("ui.recording_stopped_feedback", label=label), "warning")
    self._set_terminal_status("ready", _app.string("osc.action_idle"))
    self.update_resources()


def install(target):
    """Install extracted definitions in the original module."""
    install_class_functions(target, {name: globals()[name] for name in __all__})
