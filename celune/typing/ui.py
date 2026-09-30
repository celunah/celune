# SPDX-License-Identifier: Apache-2.0
"""UI protocol definitions."""

# pylint: disable=unused-argument,unnecessary-ellipsis

from __future__ import annotations

import queue as queue_module
import threading
from collections.abc import Callable, Iterator
from io import TextIOWrapper
from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import sounddevice as sd
    from textual.app import ComposeResult, ReturnType
    from textual.color import Color

    from .agent import AgentTask
    from ..celune import Celune
    from ..locks import ComponentLockLease
    from .aliases import AudioChunk, LogLevel
    from .common import JSONSerializable


class CeluneBaseUI(Protocol):
    """Celune base UI protocols."""

    celune: Celune

    def run(self) -> None:
        """Run the UI's main loop."""


class CeluneUIMethodSurface:
    """Typed surface for methods installed from split implementation modules."""

    _startup_log_level: LogLevel
    _run_on_ui_thread: Callable[..., None]
    _severity_color: Callable[..., str]
    _runtime_theme_name: Callable[..., str]
    _ensure_themes_registered: Callable[..., None]
    _ensure_startup_error_themes_registered: Callable[..., None]
    _prepare_loading_theme: Callable[..., None]
    _apply_theme: Callable[..., None]
    _has_celune: Callable[..., bool]
    prepare_theme: Callable[..., None]
    _clear_border_pulses: Callable[..., None]
    _refresh_theme_text: Callable[..., None]
    _bind_runtime_callbacks: Callable[..., None]
    _chain_runtime_callback: Callable[..., None]
    _publish_webui_timed_update: Callable[..., None]
    _bind_agent_events: Callable[..., None]
    _unbind_agent_events: Callable[..., None]
    _on_agent_task_state_changed: Callable[..., None]
    _on_agent_approval_requested: Callable[..., None]
    _on_agent_choice_requested: Callable[..., None]
    _on_agent_task_finished: Callable[..., None]
    _agent_task_for_display: Callable[..., Optional[AgentTask]]
    _agent_status_text: Callable[..., Optional[str]]
    _refresh_agent_status: Callable[..., None]
    _is_ui_test_mode: Callable[..., bool]
    _is_agent_test_mode: Callable[..., bool]
    _finish_test_startup: Callable[..., None]
    _apply_test_finished_state: Callable[..., None]
    _refresh_status: Callable[..., None]
    _status_view_width: Callable[..., int]
    _render_status_text: Callable[..., str]
    _update_status_label: Callable[..., None]
    on_resize: Callable[..., None]
    _refresh_logs: Callable[..., None]
    _persist_log_entry: Callable[..., None]
    _prepare_terminal_output_stream: Callable[..., Optional[TextIOWrapper]]
    run: Callable[..., Optional[ReturnType]]
    compose: Callable[..., ComposeResult]
    on_mount: Callable[..., None]
    _start_deferred_runtime: Callable[..., None]
    _startup_terminal_status_for: Callable[..., Optional[tuple[str, str]]]
    receive_startup_diagnostic: Callable[..., None]
    _emit_startup_diagnostic: Callable[..., None]
    _load_deferred_runtime: Callable[..., None]

    _handle_deferred_runtime_error: Callable[[BaseException], None]

    attach_celune: Callable[..., None]
    _check_launcher_loss: Callable[..., None]
    update_resources: Callable[..., None]
    _write_terminal_escape: Callable[..., None]
    _write_terminal_title: Callable[..., None]
    _set_terminal_status: Callable[..., None]
    _terminal_status_for: Callable[..., tuple[str, str]]
    _redirect_dunder_stdio: Callable[..., None]
    _restore_dunder_stdio: Callable[..., None]
    _install_low_level_stderr_capture: Callable[..., None]
    _is_textual_terminal_frame: Callable[..., bool]
    _forward_low_level_stderr: Callable[..., None]
    _remove_low_level_stderr_capture: Callable[..., None]
    _remove_runtime_log_redirects: Callable[..., None]
    advance_resources: Callable[..., None]
    _cancel_sleep_timer: Callable[..., None]
    _schedule_sleep_timer: Callable[..., None]
    _enter_sleep_mode: Callable[..., None]
    start_background_init: Callable[..., None]
    _show_loading_screen: Callable[..., None]
    _update_loading_log: Callable[..., None]
    _show_loading_error: Callable[..., None]
    _dismiss_loading_screen: Callable[..., None]
    _caption_word_timing_ranges: Callable[..., tuple[tuple[float, float], ...]]
    tts_caption_timing: Callable[..., None]
    _caption_words_for_progress: Callable[..., tuple[tuple[str, ...], int]]
    safe_progress: Callable[..., None]
    _set_progress_row_display: Callable[..., None]
    _restore_progress_bar: Callable[..., None]
    _reset_playback_widgets: Callable[..., None]
    safe_caption_progress: Callable[..., None]
    _animate_opacity: Callable[..., None]
    _clear_caption_timers: Callable[..., None]
    _clear_caption_state: Callable[..., None]
    _show_caption_widgets: Callable[..., None]
    _hide_caption_widgets: Callable[..., None]
    tts_caption: Callable[..., None]
    _with_darkened_brightness: Callable[..., Color]
    pulse_border: Callable[..., None]
    change_voice_lock_state: Callable[..., None]
    _refresh_persona_availability: Callable[..., None]
    change_input_state: Callable[..., None]
    safe_status: Callable[..., None]
    safe_log: Callable[..., None]
    on_uilog_message: Callable[..., None]
    safe_log_dev: Callable[..., None]
    _is_voice_conversion_mode: Callable[..., bool]
    _format_vc_pitch_shift: Callable[..., str]
    _set_vc_controls_visibility: Callable[..., None]
    refresh_vc_controls: Callable[..., None]
    set_vc_f0_condition: Callable[..., None]
    set_vc_pitch_shift: Callable[..., None]
    _persona_recording_active: Callable[..., bool]
    _acquire_recording_component_lease: Callable[
        ..., tuple[bool, Optional[ComponentLockLease]]
    ]
    _persona_speech_model_id: Callable[..., str]
    _persona_speech_language: Callable[..., Optional[str]]
    _persona_speech_end_delay_seconds: Callable[..., float]
    _persona_recording_audio_locked: Callable[..., npt.NDArray[np.float32]]
    _queue_persona_recording_item_locked: Callable[..., None]
    _set_persona_recording_text: Callable[..., None]
    _complete_persona_transcription: Callable[..., None]
    _persona_transcription_worker: Callable[..., None]
    _request_persona_recording_stop: Callable[..., bool]
    _start_persona_recording: Callable[..., bool]
    toggle_persona_recording: Callable[..., bool]
    _shutdown_persona_recording: Callable[..., None]
    _vc_recording_active: Callable[..., bool]
    _vc_input_rms: Callable[..., float]
    _vc_feedback_rise_detected: Callable[..., bool]
    _vc_feedback_min_capture_frames: Callable[..., int]
    _request_vc_recording_feedback_stop: Callable[..., None]
    _flush_vc_recording_buffer_locked: Callable[..., Optional[AudioChunk]]
    _flush_vc_recording_chunk_locked: Callable[..., Optional[AudioChunk]]
    _vc_vad_hangover_frames: Callable[..., int]
    _vc_vad_preroll_frames: Callable[..., int]
    _vc_live_chunk_frames: Callable[..., int]
    _vc_live_chunk_overlap_frames: Callable[..., int]
    _append_vc_preroll_audio_locked: Callable[..., None]
    _prepend_vc_preroll_locked: Callable[..., None]
    _clear_vc_preroll_locked: Callable[..., None]
    _vc_input_has_voice: Callable[..., bool]
    _normalize_vc_overlap_audio: Callable[..., npt.NDArray[np.float32]]
    _crossfade_vc_overlap: Callable[..., npt.NDArray[np.float32]]
    _enqueue_vc_submission_chunk: Callable[..., None]
    _finish_vc_submission_queue: Callable[..., None]
    _clear_vc_recording_state: Callable[..., None]
    _stop_vc_recording_stream: Callable[
        ...,
        tuple[
            Optional[sd.InputStream],
            Optional[AudioChunk],
            int,
            str,
            Optional[queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]],
            int,
            Optional[threading.Thread],
            Optional[threading.Thread],
        ],
    ]
    _shutdown_vc_stream: Callable[..., None]
    _close_live_vad: Callable[..., None]
    _stop_live_vc_backend: Callable[..., None]
    _join_vc_recording_threads: Callable[..., None]
    _cancel_vc_recording: Callable[..., bool]
    _stop_vc_recording_for_feedback: Callable[..., None]
    _start_vc_recording: Callable[..., bool]
    toggle_vc_recording: Callable[..., bool]
    _shutdown_live_vc_recording: Callable[..., None]
    tts_voice_changed: Callable[..., None]
    tts_log: Callable[..., None]
    process_command: Callable[..., None]
    open_settings_menu: Callable[..., None]
    _iter_config_values: Callable[
        ..., Iterator[tuple[tuple[str, ...], JSONSerializable]]
    ]
    _config_autocomplete: Callable[..., Optional[tuple[JSONSerializable, ...]]]
    _config_label: Callable[..., str]
    _config_explanation: Callable[..., str]
    _menu_footer: Callable[..., str]
    open_voice_menu: Callable[..., None]
    _show_menu: Callable[..., None]
    on_voice_button_long_pressed: Callable[..., None]
    on_select_menu_widget_confirmed: Callable[..., None]
    on_select_menu_widget_cancelled: Callable[..., None]
    _close_menu: Callable[..., None]
    _save_settings: Callable[..., None]
    _set_config_value: Callable[..., None]
    consume_buffer: Callable[..., None]
    _submit_text: Callable[..., bool]
    tutorial_after: Callable[..., None]
    begin_tutorial: Callable[..., None]
    finish_tutorial: Callable[..., None]
    cancel_tutorial: Callable[..., bool]
    on_key: Callable[..., None]
    on_button_pressed: Callable[..., None]
    on_unmount: Callable[..., None]
    tts_idle: Callable[..., None]
    tts_queue_avail: Callable[..., None]
    error: Callable[..., None]
    on_text_area_changed: Callable[..., None]
    _signal_handler: Callable[..., None]
    _install_windows_signal_handler: Callable[..., None]
    _signal_handler_windows: Callable[..., bool]
    _hide_scrollbars_for_exit: Callable[..., None]
    _graceful_exit: Callable[..., None]
    _run_shutdown_step: Callable[..., None]
    _report_shutdown_error: Callable[..., None]
    _shutdown_runtime: Callable[..., None]
    graceful_exit: Callable[..., None]
    tutorial_token: int
    tutorial_active: bool
    _split_command_input: Callable[..., list[str]]
    split_command_input: Callable[..., list[str]]


class CeluneTextualUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's interactive Textual UI callbacks."""

    def tts_log(self, msg: str, severity: str = "info") -> None:
        """Handle log messages coming from Celune.

        Args:
            msg: Message text emitted by Celune.
            severity: Message severity label.
        """

    def safe_status(self, msg: str, severity: str = "info") -> None:
        """Update current status.

        Args:
            msg: Status text to display.
            severity: Status severity label.
        """

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Current completed progress amount.
            total: Optional total progress amount.
        """

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: Error text to surface to the user.
        """

    def tts_idle(self) -> None:
        """Reset UI state after Celune stops talking."""

    def tts_queue_avail(self) -> None:
        """Unlock input queueing after Celune completes generation."""

    def tts_voice_changed(self, name: str) -> None:
        """Set UI state after changing Celune's voice.

        Args:
            name: Newly selected voice name.
        """

    def change_input_state(self, locked: bool) -> None:
        """Lock or unlock Celune's UI layer.

        Args:
            locked: Whether input should be locked.
        """

    def change_voice_lock_state(self, locked: bool) -> None:
        """Lock or unlock Celune's voice change button.

        Args:
            locked: Whether voice selection should be locked.
        """


class CeluneHeadlessBaseUI(CeluneBaseUI, Protocol):
    """Protocol for Celune's headless UI callbacks."""

    def headless_log(self, msg: str, severity: str = "info") -> None:
        """Log to the headless interface.

        Args:
            msg: Message text emitted by Celune.
            severity: Message severity label.
        """

    def headless_error(self, error: str) -> None:
        """Log an error to the headless interface.

        Args:
            error: Error text to surface to the operator.
        """
