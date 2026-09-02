# SPDX-License-Identifier: Apache-2.0
"""Frontend layer."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

# UI runtime dependencies are declared under TYPE_CHECKING and populated lazily
# by _load_ui_runtime_dependencies to keep the startup frame lightweight.
# ruff: noqa: TC004

from __future__ import annotations

import os
import re
import sys
import math
import time
from copy import deepcopy
import queue as queue_module
import shlex
import types
import ctypes
import signal
import asyncio
import logging
import datetime
import inspect
import itertools
import threading
import contextlib
from io import TextIOWrapper
from uuid import uuid4
from typing import (
    TYPE_CHECKING,
    Union,
    TextIO,
    Optional,
    Protocol,
    Never,
    ClassVar,
    cast,
    final,
)
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Callable, Iterator

from textual import work, events
from rich.text import Text
from textual.app import (
    App,
    ReturnType,
    ComposeResult,
    ScreenStackError,
    AutopilotCallbackType,
)
from textual.color import Color
from textual.theme import Theme
from textual.timer import Timer
from textual.widget import Widget
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Label, Button, RichLog, TextArea, ProgressBar
from textual.css.query import NoMatches
from textual.css.types import EdgeStyle
from textual.containers import Vertical, Horizontal

from ..i18n import string, tagged_string
from .theme import CELUNE_CSS, severity_color
from .loading import CeluneLoadingScreen
from .terminal import SelectMenuOption, SelectMenuWidget
from ..constants import SIGTSTP, APP_NAME, ExitCodes, BASE_SR
from ..theme.defaults import default_error_theme_family, default_theme_family
from ..watchdog import launcher_loss_requested
from ..typing.agent import AgentTaskState
from ..typing.common import JSONSerializable
from ..typing.config import AudioDeviceInfoValue
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)

if TYPE_CHECKING:
    from ..theme import colors
    from ..config import format_audio_device_name, resolve_audio_device_with_info
    from ..exceptions import CEDTSError
    from ..paths import config_path, main_window_log_path
    from ..utils import (
        discard,
        is_april_fools,
        replace_ipa,
        typing_delay,
    )
    from ..typing.aliases import _VCAudioCallback
    from .terminal import LogRedirect, UILogHandler, is_celune_log_record
    import yaml
    import numpy as np
    import sounddevice as sd
    import numpy.typing as npt

    from . import resources as ui_resources
    from ..vc import (
        VC_PITCH_SHIFT_MAX,
        VC_PITCH_SHIFT_MIN,
        LiveVoiceActivityDetector,
        vc_input_rms,
        vc_input_has_voice,
        clamp_vc_pitch_shift,
        vc_live_chunk_frames,
        vc_vad_preroll_frames,
        vc_vad_hangover_frames,
        vc_live_chunk_overlap_frames,
        create_live_voice_activity_detector,
    )
    from ..locks import ComponentLockLease
    from ..celune import Celune
    from ..cevoice import CEVoiceLoader
    from .commands import process_command as process_ui_command
    from ..pipeline import (
        current_playback_status,
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from .resources import FOOTER_ROTATE_SECONDS
    from ..persona.asr import (
        DEFAULT_PERSONA_SPEECH_MODEL_ID,
        PERSONA_SPEECH_END_DELAY_SECONDS,
        PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
        WhisperSegment,
        WhisperTranscriber,
    )
    from ..persona.impl import (
        persona_config,
        persona_enabled,
        persona_talkback_enabled,
    )
    from ..typing.agent import AgentTask
    from ..typing.aliases import (
        LogLevel,
        AudioChunk,
        AudioChunks,
    )
    from ..extensions.events import EventDispatcher
    from ..dataclasses.events import (
        AgentTaskFinishedEvent,
        AgentChoiceRequestedEvent,
        AgentTaskStateChangedEvent,
        AgentApprovalRequestedEvent,
    )
    from ..dataclasses.pipeline import AudioOutput

    class _UIResources(Protocol):
        """Type-checkable subset of the resource footer module."""

        def prime_usage(self) -> None:
            """Prime resource usage polling."""

        def start_gpu_usage_worker(self) -> None:
            """Start native async GPU usage polling."""

        def stop_gpu_usage_worker(self) -> None:
            """Stop native async GPU usage polling."""

        def resource_pages(
            self,
            celune: Celune,
            theme_name: str,
        ) -> tuple[str, ...]:
            """Return the current resource footer pages."""


default_loader: Optional[Callable[[], Optional[CEVoiceLoader]]] = None
ui_resources: Optional[_UIResources] = None
_RUNTIME_DEPENDENCIES_LOADED = False

_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES: frozenset[str] = frozenset()

if not TYPE_CHECKING:
    _VCAudioCallback = Callable[..., None]


def format_error(error: BaseException, log_level: Union[LogLevel, bool]) -> str:
    """Format an error without importing the heavy utility module at startup."""
    from ..utils import format_error as format_error_helper

    return format_error_helper(error, log_level)


def format_error_message(
    message: str,
    error: BaseException,
    log_level: Union[LogLevel, bool],
) -> str:
    """Append level-appropriate exception detail without eager runtime imports."""
    from ..utils import format_error_message as format_error_message_helper

    return format_error_message_helper(message, error, log_level)


class VoiceButton(Button):
    """Button with independent click and held-release actions."""

    class LongPressed(Message):
        """Message emitted after the primary mouse button is released."""

        def __init__(self, button: VoiceButton) -> None:
            super().__init__()
            self.button = button

    def __init__(
        self,
        label: str,
        *,
        widget_id: Optional[str] = None,
        disabled: bool = False,
        hold_enabled: bool = False,
    ) -> None:
        super().__init__(label, id=widget_id, disabled=disabled)
        self._hold_seconds = 0.55
        self._hold_timer: Optional[Timer] = None
        self._long_pressed = False
        self.hold_enabled = hold_enabled

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """Start the long-press timer for the primary mouse button."""
        if event.button == 1 and self.hold_enabled:
            self._long_pressed = False
            self._stop_hold_timer()
            self._hold_timer = self.set_timer(
                self._hold_seconds,
                self._emit_long_pressed,
            )
        await super()._on_mouse_down(event)

    async def _on_mouse_up(self, event: events.MouseUp) -> None:
        """Emit a held-release message or allow the normal click action."""
        self._stop_hold_timer()
        long_pressed = self._long_pressed
        self._long_pressed = False
        if long_pressed:
            self.suppress_click()
        await super()._on_mouse_up(event)
        if long_pressed:
            self.post_message(self.LongPressed(self))

    def _emit_long_pressed(self) -> None:
        """Mark a held press without opening its modal before release."""
        self._hold_timer = None
        if not self.hold_enabled or not self.is_mouse_over:
            return
        self._long_pressed = True
        self.suppress_click()

    def _stop_hold_timer(self) -> None:
        """Stop the pending long-press timer, if any."""
        if self._hold_timer is not None:
            self._hold_timer.stop()
            self._hold_timer = None


class SelectMenuOverlay(ModalScreen[None]):
    """Center one selection menu over the application content."""

    def __init__(self, menu: SelectMenuWidget) -> None:
        super().__init__()
        self.menu = menu

    def compose(self) -> ComposeResult:
        """Yield the menu that should be centered by this overlay."""
        yield self.menu

    def on_key(self, event: events.Key) -> None:
        """Keep unhandled keys from reaching the application underneath."""
        event.stop()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Consume mouse presses outside the menu popup."""
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Consume mouse releases outside the menu popup."""
        event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        """Consume downward scrolling outside the menu popup."""
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        """Consume upward scrolling outside the menu popup."""
        event.stop()

    def on_mouse_scroll_right(self, event: events.MouseScrollRight) -> None:
        """Consume rightward scrolling outside the menu popup."""
        event.stop()

    def on_mouse_scroll_left(self, event: events.MouseScrollLeft) -> None:
        """Consume leftward scrolling outside the menu popup."""
        event.stop()

    def on_click(self, event: events.Click) -> None:
        """Consume clicks outside the menu popup."""
        event.stop()


def indent(text: str, spaces: int, direction: str = "left") -> str:
    """Indent lightweight UI text without importing the heavy utility module."""
    if direction == "left":
        return " " * spaces + text
    if direction == "right":
        return text + " " * spaces

    raise ValueError("can't indent from this direction")


def supports_ansi(stream: Optional[TextIO] = None) -> bool:
    """Check terminal ANSI support without importing the heavy utility module."""
    from ..terminal import supports_ansi as terminal_supports_ansi

    return terminal_supports_ansi(stream)


def terminal_title_escape(status: tuple[str, str, str]) -> str:
    """Build a terminal-title escape without loading runtime UI dependencies."""
    from ..terminal import terminal_title_escape as build_terminal_title

    return build_terminal_title(status)


def set_terminal_title(
    status: tuple[str, str, str],
    output: Optional[TextIO] = None,
) -> None:
    """Set a terminal title without loading runtime UI dependencies."""
    from ..terminal import set_terminal_title as write_terminal_title

    write_terminal_title(status, output)


def __getattr__(name: str):
    """Resolve legacy runtime globals only when callers explicitly request them."""
    if name == "colors":
        from ..theme import colors

        return colors
    if name in {
        "launcher_loss_requested",
        "resolve_audio_device_with_info",
    }:
        if name == "launcher_loss_requested":
            from ..watchdog import launcher_loss_requested as requested

            return requested
        if name == "resolve_audio_device_with_info":
            from ..config import resolve_audio_device_with_info

            return resolve_audio_device_with_info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_ui_runtime_dependencies() -> None:
    """Load optional UI integrations after the first loading frame is visible."""
    global _RUNTIME_DEPENDENCIES_LOADED
    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    global AudioOutput
    global CEDTSError
    global DEFAULT_PERSONA_SPEECH_MODEL_ID
    global FOOTER_ROTATE_SECONDS
    global LiveVoiceActivityDetector
    global PERSONA_SPEECH_END_DELAY_SECONDS
    global PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS
    global VC_PITCH_SHIFT_MAX
    global VC_PITCH_SHIFT_MIN
    global WhisperSegment
    global WhisperTranscriber
    global clamp_vc_pitch_shift
    global colors
    global config_path
    global discard
    global format_audio_device_name
    global create_live_voice_activity_detector
    global current_playback_status
    global default_loader
    global finish_streaming_sfx_audio
    global indent
    global is_april_fools
    global launcher_loss_requested
    global LogRedirect
    global main_window_log_path
    global np
    global npt
    global persona_config
    global persona_enabled
    global persona_talkback_enabled
    global process_ui_command
    global queue_streaming_sfx_audio
    global replace_ipa
    global resolve_audio_device_with_info
    global sd
    global set_terminal_title
    global supports_ansi
    global terminal_title_escape
    global typing_delay
    global UILogHandler
    global ui_resources
    global vc_input_has_voice
    global vc_input_rms
    global vc_live_chunk_frames
    global vc_live_chunk_overlap_frames
    global vc_vad_hangover_frames
    global vc_vad_preroll_frames
    global yaml
    global is_celune_log_record
    global _RUNTIME_LOG_REDIRECT_FILTER_MESSAGES

    from ..theme import colors
    from ..config import format_audio_device_name, resolve_audio_device_with_info
    from ..exceptions import CEDTSError
    from ..paths import config_path, main_window_log_path
    from ..terminal import (
        RUNTIME_LOG_FILTER_MESSAGES,
        set_terminal_title,
        terminal_title_escape,
    )
    from ..utils import (
        discard,
        indent,
        is_april_fools,
        replace_ipa,
        supports_ansi,
        typing_delay,
    )
    from ..watchdog import launcher_loss_requested as loaded_launcher_loss_requested
    from .terminal import LogRedirect, UILogHandler, is_celune_log_record
    import yaml
    import numpy as np
    import sounddevice as sd
    import numpy.typing as npt

    from . import resources as ui_resources
    from ..vc import (
        VC_PITCH_SHIFT_MAX,
        VC_PITCH_SHIFT_MIN,
        LiveVoiceActivityDetector,
        vc_input_rms,
        vc_input_has_voice,
        clamp_vc_pitch_shift,
        vc_live_chunk_frames,
        vc_vad_preroll_frames,
        vc_vad_hangover_frames,
        vc_live_chunk_overlap_frames,
        create_live_voice_activity_detector,
    )
    from ..cevoice import default_loader
    from .commands import process_command as process_ui_command
    from ..pipeline import (
        current_playback_status,
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from .resources import FOOTER_ROTATE_SECONDS
    from ..persona.asr import (
        DEFAULT_PERSONA_SPEECH_MODEL_ID,
        PERSONA_SPEECH_END_DELAY_SECONDS,
        PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
        WhisperSegment,
        WhisperTranscriber,
    )
    from ..persona.impl import (
        persona_config,
        persona_enabled,
        persona_talkback_enabled,
    )
    from ..dataclasses.pipeline import AudioOutput

    launcher_loss_requested = loaded_launcher_loss_requested
    _RUNTIME_LOG_REDIRECT_FILTER_MESSAGES = RUNTIME_LOG_FILTER_MESSAGES
    _RUNTIME_DEPENDENCIES_LOADED = True


_CAPTION_FADE_SECONDS = 0.36
_LOADING_FADE_SECONDS = 1.0
_MAIN_UI_FADE_SECONDS = 0.6
_EXIT_FADE_SECONDS = 0.6
_VC_FEEDBACK_MIN_CAPTURE_SECONDS = 0.35
_VC_FEEDBACK_REQUIRED_CONSECUTIVE_SPIKES = 2
_VC_FEEDBACK_RMS_MIN_PREVIOUS = 0.05
_VC_FEEDBACK_RMS_MIN_CURRENT = 0.18
_VC_FEEDBACK_RMS_RISE_RATIO = 2.0
_VC_FEEDBACK_RMS_RISE_DELTA = 0.08
_VC_LIVE_SUBMISSION_QUEUE_SIZE = 3
_AGENT_ACTIVE_STATES = frozenset(
    {
        AgentTaskState.QUEUED,
        AgentTaskState.IDLE,
        AgentTaskState.CLASSIFYING,
        AgentTaskState.WORKING,
        AgentTaskState.PLANNING,
        AgentTaskState.EXECUTING_TOOL,
        AgentTaskState.RESPONDING,
        AgentTaskState.CANCELLING,
    }
)
_AGENT_AWAITING_STATES = frozenset(
    {AgentTaskState.AWAITING_APPROVAL, AgentTaskState.AWAITING_CHOICE}
)
_AGENT_PAUSED_STATES = frozenset({AgentTaskState.PAUSED, AgentTaskState.INTERRUPTED})


def _device_scalar_int(value: Optional[AudioDeviceInfoValue], default: int) -> int:
    """Return one audio-device metadata value as an integer when possible."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


class UILogMessage(Message):
    """Deliver one background log entry to the Textual application thread."""

    def __init__(self, message: str, severity: str) -> None:
        super().__init__()
        self.message = message
        self.severity = severity


class ProgressLabel(Label):
    """Display playback time or general progress beside the progress bar."""

    def __init__(self, widget_id: Optional[str] = None) -> None:
        super().__init__("", id=widget_id)
        self.display = False

    def set_progress(
        self,
        progress: Optional[float],
        total: Optional[float],
        *,
        audio_playing: bool = False,
        sample_rate: float = 1.0,
    ) -> None:
        """Update or hide the progress readout.

        Args:
            progress: Current progress in units supplied by the callback.
            total: Total progress in the same units, or ``None`` when unknown.
            audio_playing: Display elapsed audio time instead of a percentage.
            sample_rate: Units per second when ``audio_playing`` is true.
        """
        if (
            progress is None
            or total is None
            or total <= 0
            or progress < 0
            or sample_rate <= 0
        ):
            self.display = False
            self.update("")
            return

        if audio_playing:
            value = self._format_time(progress / sample_rate)
        else:
            percentage = round(max(0.0, min(1.0, progress / total)) * 100)
            value = f"{percentage:3d}%"

        self.update(value)
        self.display = True

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format elapsed audio time as minutes and seconds."""
        whole_seconds = max(0, int(seconds))
        minutes, remaining_seconds = divmod(whole_seconds, 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"


@dataclass
class CeluneUIWidgetState:
    """Resolved widget references owned by the UI."""

    logs: Optional[RichLog] = None
    input_box: Optional[TextArea] = None
    style_button: Optional[Button] = None
    vc_mode_button: Optional[Button] = None
    vc_pitch_button: Optional[Button] = None
    status: Optional[Label] = None
    resources: Optional[Label] = None
    caption: Optional[Label] = None
    progress_bar: Optional[ProgressBar] = None
    progress_label: Optional[ProgressLabel] = None
    header: Optional[Label] = None
    header_lines: tuple[Label, ...] = ()


@dataclass
class CeluneUIThemeState:
    """Theme and status marquee state."""

    themes: tuple[str, str]
    active_theme_name: str
    fatal_error_active: bool = False
    log_history: list[tuple[str, str]] = field(default_factory=list)
    log_history_lock: threading.Lock = field(default_factory=threading.Lock)
    rendered_log_count: int = 0
    status_severity: str = "info"
    status_text: str = ""
    status_marquee_offset: int = 0
    status_marquee_gap: str = "   "
    status_marquee_timer: Optional[Timer] = None


@dataclass
class CeluneUIBindingState:
    """Bindings between the UI and the runtime."""

    celune: Optional[Celune] = None
    celune_ready: bool = False
    celune_styles: tuple[str, ...] = ()
    celune_voices: Optional[Iterator[str]] = None
    style_index: int = 0
    cur_state: str = "active"
    startup_error_exit_code: Optional[int] = None
    consume_on_boundary: bool = False
    suppress_input_change: bool = False
    resource_page: int = 0
    webui_timed_update_sequence: int = 0
    input_locked: bool = True
    persona_available: bool = False
    persona_probe_running: bool = False


@dataclass
class CeluneUILogCaptureState:
    """Stdio/log redirection and persisted log state."""

    old_stdout: TextIO
    old_stderr: TextIO
    log_stdout: Optional[LogRedirect] = None
    log_stderr: Optional[LogRedirect] = None
    runtime_log_capture_enabled: bool = False
    runtime_redirect_handler: Optional[UILogHandler] = None
    runtime_redirect_original_call_handlers: Optional[
        Callable[[logging.Logger, logging.LogRecord], None]
    ] = None
    runtime_redirect_original_last_resort: Optional[logging.Handler] = None
    runtime_redirect_original_raise_exceptions: Optional[bool] = None
    original_dunder_stdout: Optional[TextIO] = None
    original_dunder_stderr: Optional[TextIO] = None
    terminal_output_stream: Optional[TextIOWrapper] = None
    stderr_pipe_read_fd: Optional[int] = None
    stderr_pipe_write_fd: Optional[int] = None
    stderr_original_fd_dup: Optional[int] = None
    stderr_forward_thread: Optional[threading.Thread] = None
    warnings_capture_enabled: bool = False
    log_file_path: Path = field(default_factory=Path)
    log_file_initialized: bool = False


@dataclass
class CeluneUIInteractionState:
    """Transient UI effects, sleep scheduling, and tutorial state."""

    border_pulse_tokens: dict[int, int] = field(default_factory=dict)
    border_pulse_widgets: dict[int, Widget] = field(default_factory=dict)
    runtime_shutdown_complete: bool = False
    runtime_shutdown_lock: threading.Lock = field(default_factory=threading.Lock)
    tutorial_timers: list[Timer] = field(default_factory=list)
    vc_recording_buffered_frames: int = 0
    vc_recording_chunks: AudioChunks = field(default_factory=list)
    vc_recording_captured_frames: int = 0
    vc_recording_feedback_detected: bool = False
    vc_recording_feedback_spike_count: int = 0
    vc_recording_label: str = ""
    vc_recording_lock: threading.Lock = field(default_factory=threading.Lock)
    vc_recording_preroll_chunks: AudioChunks = field(default_factory=list)
    vc_recording_preroll_frames: int = 0
    vc_recording_previous_rms: float = 0.0
    vc_recording_sample_rate: int = 0
    vc_recording_silence_frames: int = 0
    vc_recording_speech_started: bool = False
    vc_recording_submission_queue: Optional[
        queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]
    ] = None
    vc_recording_stream: Optional[sd.InputStream] = None
    vc_recording_stop_thread: Optional[threading.Thread] = None
    vc_recording_worker: Optional[threading.Thread] = None
    vc_recording_vad: Optional[LiveVoiceActivityDetector] = None
    vc_recording_component_lease: Optional[ComponentLockLease] = None
    persona_recording_chunks: AudioChunks = field(default_factory=list)
    persona_recording_lock: threading.Lock = field(default_factory=threading.Lock)
    persona_recording_queue: Optional[queue_module.Queue[tuple[AudioChunk, bool]]] = (
        None
    )
    persona_recording_sample_rate: int = 0
    persona_recording_silence_frames: int = 0
    persona_recording_speech_started: bool = False
    persona_recording_stop_requested: bool = False
    persona_recording_stream: Optional[sd.InputStream] = None
    persona_recording_text_prefix: str = ""
    persona_recording_transcriber: Optional[WhisperTranscriber] = None
    persona_recording_worker: Optional[threading.Thread] = None
    persona_recording_vad: Optional[LiveVoiceActivityDetector] = None
    persona_recording_last_partial_at: float = 0.0
    persona_recording_component_lease: Optional[ComponentLockLease] = None
    caption_text: str = ""
    caption_words: tuple[str, ...] = ()
    caption_sentences: tuple[tuple[str, ...], ...] = ()
    caption_word_timings: tuple[tuple[float, float], ...] = ()
    caption_audio_duration: float = 0.0
    caption_rendered_text: str = ""
    caption_transcriber: Optional[WhisperTranscriber] = None
    caption_visible_words: int = 0
    caption_progress: float = 0.0
    caption_active: bool = False
    caption_transitioning: bool = False
    caption_transition_token: int = 0
    caption_timers: list[Timer] = field(default_factory=list)
    sleep_timer: Optional[Timer] = None
    tutorial_token: int = 0
    tutorial_active: bool = False
    agent_event_dispatcher: Optional[EventDispatcher] = None
    agent_task_id: Optional[str] = None
    agent_task_state: Optional[AgentTaskState] = None
    agent_iterations: int = 0
    agent_max_loops: int = 0
    agent_busy_components: tuple[ComponentLockName, ...] = ()
    agent_status_signature: Optional[tuple[str, ...]] = None


def _forward_ui_property(container_name: str, field_name: str) -> property:
    """Create a property that forwards storage to a grouped UI state container."""

    def getter(instance):
        return getattr(getattr(instance, container_name), field_name)

    def setter(instance, value) -> None:
        setattr(getattr(instance, container_name), field_name, value)

    return property(getter, setter)


@final
class CeluneUI(App):
    """Celune's main user interface."""

    def __init_subclass__(cls, **kwargs: Never) -> Never:
        raise TypeError(f"{__class__.__name__} is final and cannot be subclassed")

    ENABLE_COMMAND_PALETTE = False
    CSS = CELUNE_CSS
    _instance: ClassVar[Optional[CeluneUI]] = None

    def __init__(
        self,
        startup_loader: Optional[Callable[[], Celune]] = None,
        startup_messages: Optional[list[str]] = None,
        startup_log_level: LogLevel = "info",
        test_completion_callback: Optional[
            Callable[[Celune, bool, Optional[str]], None]
        ] = None,
    ) -> None:
        super().__init__()

        if CeluneUI._instance is not None:
            raise RuntimeError(f"can only instantiate {self.__class__.__name__} once")

        themes = ("celune", "celune_light")
        active_theme_name = "celune"

        self._widgets = CeluneUIWidgetState()
        self._theme_state = CeluneUIThemeState(
            themes=themes,
            active_theme_name=active_theme_name,
        )
        self._binding_state = CeluneUIBindingState(celune_voices=itertools.cycle(()))
        self._log_capture_state = CeluneUILogCaptureState(
            old_stdout=sys.stdout,
            old_stderr=sys.stderr,
            log_file_path=Path(),
        )
        self._interaction_state = CeluneUIInteractionState()
        self._terminal_status: Optional[tuple[str, str, str]] = None
        self._loading_screen: Optional[CeluneLoadingScreen] = None
        self._startup_loader = startup_loader
        self._startup_messages = list(startup_messages or [])
        self._startup_log_level = startup_log_level
        self._test_completion_callback = test_completion_callback
        self._runtime_intervals_started = False
        self._windows_signal_handler: Optional[Callable[[int], bool]] = None
        self._active_menu: Optional[SelectMenuWidget] = None
        self._active_menu_overlay: Optional[SelectMenuOverlay] = None
        self._active_menu_kind: Optional[str] = None
        self._settings_paths: tuple[tuple[str, ...], ...] = ()
        self._voice_menu_paths: dict[str, Path] = {}

        CeluneUI._instance = self

    logs = _forward_ui_property("_widgets", "logs")
    input_box = _forward_ui_property("_widgets", "input_box")
    style_button = _forward_ui_property("_widgets", "style_button")
    vc_mode_button = _forward_ui_property("_widgets", "vc_mode_button")
    vc_pitch_button = _forward_ui_property("_widgets", "vc_pitch_button")
    status = _forward_ui_property("_widgets", "status")
    resources = _forward_ui_property("_widgets", "resources")
    caption = _forward_ui_property("_widgets", "caption")
    progress_bar = _forward_ui_property("_widgets", "progress_bar")
    progress_label = _forward_ui_property("_widgets", "progress_label")
    header = _forward_ui_property("_widgets", "header")
    header_lines = _forward_ui_property("_widgets", "header_lines")

    themes = _forward_ui_property("_theme_state", "themes")
    active_theme_name = _forward_ui_property("_theme_state", "active_theme_name")
    _fatal_error_active = _forward_ui_property("_theme_state", "fatal_error_active")
    log_history = _forward_ui_property("_theme_state", "log_history")
    _log_history_lock = _forward_ui_property("_theme_state", "log_history_lock")
    _rendered_log_count = _forward_ui_property("_theme_state", "rendered_log_count")
    status_severity = _forward_ui_property("_theme_state", "status_severity")
    _status_text = _forward_ui_property("_theme_state", "status_text")
    _status_marquee_offset = _forward_ui_property(
        "_theme_state", "status_marquee_offset"
    )
    _status_marquee_gap = _forward_ui_property("_theme_state", "status_marquee_gap")
    _status_marquee_timer = _forward_ui_property("_theme_state", "status_marquee_timer")

    celune = _forward_ui_property("_binding_state", "celune")
    celune_ready = _forward_ui_property("_binding_state", "celune_ready")
    celune_styles = _forward_ui_property("_binding_state", "celune_styles")
    celune_voices = _forward_ui_property("_binding_state", "celune_voices")
    style_index = _forward_ui_property("_binding_state", "style_index")
    cur_state = _forward_ui_property("_binding_state", "cur_state")
    _startup_error_exit_code = _forward_ui_property(
        "_binding_state", "startup_error_exit_code"
    )
    consume_on_boundary = _forward_ui_property("_binding_state", "consume_on_boundary")
    _suppress_input_change = _forward_ui_property(
        "_binding_state", "suppress_input_change"
    )
    _resource_page = _forward_ui_property("_binding_state", "resource_page")
    _webui_timed_update_sequence = _forward_ui_property(
        "_binding_state", "webui_timed_update_sequence"
    )
    _input_locked = _forward_ui_property("_binding_state", "input_locked")
    _persona_available = _forward_ui_property("_binding_state", "persona_available")
    _persona_probe_running = _forward_ui_property(
        "_binding_state", "persona_probe_running"
    )

    _old_stdout = _forward_ui_property("_log_capture_state", "old_stdout")
    _old_stderr = _forward_ui_property("_log_capture_state", "old_stderr")
    _log_stdout = _forward_ui_property("_log_capture_state", "log_stdout")
    _log_stderr = _forward_ui_property("_log_capture_state", "log_stderr")
    _runtime_log_capture_enabled = _forward_ui_property(
        "_log_capture_state", "runtime_log_capture_enabled"
    )
    _runtime_redirect_handler = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_handler"
    )
    _runtime_redirect_original_call_handlers = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_call_handlers"
    )
    _runtime_redirect_original_last_resort = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_last_resort"
    )
    _runtime_redirect_original_raise_exceptions = _forward_ui_property(
        "_log_capture_state", "runtime_redirect_original_raise_exceptions"
    )
    _original_dunder_stdout = _forward_ui_property(
        "_log_capture_state", "original_dunder_stdout"
    )
    _original_dunder_stderr = _forward_ui_property(
        "_log_capture_state", "original_dunder_stderr"
    )
    _terminal_output_stream = _forward_ui_property(
        "_log_capture_state", "terminal_output_stream"
    )
    _stderr_pipe_read_fd = _forward_ui_property(
        "_log_capture_state", "stderr_pipe_read_fd"
    )
    _stderr_pipe_write_fd = _forward_ui_property(
        "_log_capture_state", "stderr_pipe_write_fd"
    )
    _stderr_original_fd_dup = _forward_ui_property(
        "_log_capture_state", "stderr_original_fd_dup"
    )
    _stderr_forward_thread = _forward_ui_property(
        "_log_capture_state", "stderr_forward_thread"
    )
    _warnings_capture_enabled = _forward_ui_property(
        "_log_capture_state", "warnings_capture_enabled"
    )
    _log_file_path = _forward_ui_property("_log_capture_state", "log_file_path")
    _log_file_initialized = _forward_ui_property(
        "_log_capture_state", "log_file_initialized"
    )

    _border_pulse_tokens = _forward_ui_property(
        "_interaction_state", "border_pulse_tokens"
    )
    _border_pulse_widgets = _forward_ui_property(
        "_interaction_state", "border_pulse_widgets"
    )
    _tutorial_timers = _forward_ui_property("_interaction_state", "tutorial_timers")
    _vc_recording_chunks = _forward_ui_property(
        "_interaction_state", "vc_recording_chunks"
    )
    _vc_recording_buffered_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_buffered_frames"
    )
    _vc_recording_label = _forward_ui_property(
        "_interaction_state", "vc_recording_label"
    )
    _vc_recording_captured_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_captured_frames"
    )
    _vc_recording_feedback_detected = _forward_ui_property(
        "_interaction_state", "vc_recording_feedback_detected"
    )
    _vc_recording_feedback_spike_count = _forward_ui_property(
        "_interaction_state", "vc_recording_feedback_spike_count"
    )
    _vc_recording_lock = _forward_ui_property("_interaction_state", "vc_recording_lock")
    _vc_recording_preroll_chunks = _forward_ui_property(
        "_interaction_state", "vc_recording_preroll_chunks"
    )
    _vc_recording_preroll_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_preroll_frames"
    )
    _vc_recording_previous_rms = _forward_ui_property(
        "_interaction_state", "vc_recording_previous_rms"
    )
    _vc_recording_sample_rate = _forward_ui_property(
        "_interaction_state", "vc_recording_sample_rate"
    )
    _vc_recording_silence_frames = _forward_ui_property(
        "_interaction_state", "vc_recording_silence_frames"
    )
    _vc_recording_speech_started = _forward_ui_property(
        "_interaction_state", "vc_recording_speech_started"
    )
    _vc_recording_submission_queue = _forward_ui_property(
        "_interaction_state", "vc_recording_submission_queue"
    )
    _vc_recording_stream = _forward_ui_property(
        "_interaction_state", "vc_recording_stream"
    )
    _vc_recording_stop_thread = _forward_ui_property(
        "_interaction_state", "vc_recording_stop_thread"
    )
    _vc_recording_worker = _forward_ui_property(
        "_interaction_state", "vc_recording_worker"
    )
    _vc_recording_vad = _forward_ui_property("_interaction_state", "vc_recording_vad")
    _vc_recording_component_lease = _forward_ui_property(
        "_interaction_state", "vc_recording_component_lease"
    )
    _persona_recording_chunks = _forward_ui_property(
        "_interaction_state", "persona_recording_chunks"
    )
    _persona_recording_lock = _forward_ui_property(
        "_interaction_state", "persona_recording_lock"
    )
    _persona_recording_queue = _forward_ui_property(
        "_interaction_state", "persona_recording_queue"
    )
    _persona_recording_sample_rate = _forward_ui_property(
        "_interaction_state", "persona_recording_sample_rate"
    )
    _persona_recording_silence_frames = _forward_ui_property(
        "_interaction_state", "persona_recording_silence_frames"
    )
    _persona_recording_speech_started = _forward_ui_property(
        "_interaction_state", "persona_recording_speech_started"
    )
    _persona_recording_stop_requested = _forward_ui_property(
        "_interaction_state", "persona_recording_stop_requested"
    )
    _persona_recording_stream = _forward_ui_property(
        "_interaction_state", "persona_recording_stream"
    )
    _persona_recording_text_prefix = _forward_ui_property(
        "_interaction_state", "persona_recording_text_prefix"
    )
    _persona_recording_transcriber = _forward_ui_property(
        "_interaction_state", "persona_recording_transcriber"
    )
    _persona_recording_worker = _forward_ui_property(
        "_interaction_state", "persona_recording_worker"
    )
    _persona_recording_vad = _forward_ui_property(
        "_interaction_state", "persona_recording_vad"
    )
    _persona_recording_last_partial_at = _forward_ui_property(
        "_interaction_state", "persona_recording_last_partial_at"
    )
    _persona_recording_component_lease = _forward_ui_property(
        "_interaction_state", "persona_recording_component_lease"
    )
    _caption_text = _forward_ui_property("_interaction_state", "caption_text")
    _caption_words = _forward_ui_property("_interaction_state", "caption_words")
    _caption_sentences = _forward_ui_property("_interaction_state", "caption_sentences")
    _caption_word_timings = _forward_ui_property(
        "_interaction_state", "caption_word_timings"
    )
    _caption_audio_duration = _forward_ui_property(
        "_interaction_state", "caption_audio_duration"
    )
    _caption_rendered_text = _forward_ui_property(
        "_interaction_state", "caption_rendered_text"
    )
    _caption_transcriber = _forward_ui_property(
        "_interaction_state", "caption_transcriber"
    )
    _caption_visible_words = _forward_ui_property(
        "_interaction_state", "caption_visible_words"
    )
    _caption_progress = _forward_ui_property("_interaction_state", "caption_progress")
    _caption_active = _forward_ui_property("_interaction_state", "caption_active")
    _caption_transitioning = _forward_ui_property(
        "_interaction_state", "caption_transitioning"
    )
    _caption_transition_token = _forward_ui_property(
        "_interaction_state", "caption_transition_token"
    )
    _caption_timers = _forward_ui_property("_interaction_state", "caption_timers")
    _sleep_timer = _forward_ui_property("_interaction_state", "sleep_timer")
    _tutorial_token = _forward_ui_property("_interaction_state", "tutorial_token")
    _tutorial_active = _forward_ui_property("_interaction_state", "tutorial_active")
    _agent_event_dispatcher = _forward_ui_property(
        "_interaction_state", "agent_event_dispatcher"
    )
    _agent_task_id = _forward_ui_property("_interaction_state", "agent_task_id")
    _agent_task_state = _forward_ui_property("_interaction_state", "agent_task_state")
    _agent_iterations = _forward_ui_property("_interaction_state", "agent_iterations")
    _agent_max_loops = _forward_ui_property("_interaction_state", "agent_max_loops")
    _agent_busy_components = _forward_ui_property(
        "_interaction_state", "agent_busy_components"
    )
    _agent_status_signature = _forward_ui_property(
        "_interaction_state", "agent_status_signature"
    )

    def _run_on_ui_thread(self, callback: Callable[[], None]) -> None:
        if threading.current_thread() is threading.main_thread():
            callback()
        else:
            try:
                self.call_from_thread(callback)
            except RuntimeError:
                pass

    def _severity_color(self, severity: str = "info") -> str:
        """Return the current theme color for a log severity."""
        return severity_color(self.active_theme_name, severity)

    def _runtime_theme_name(self) -> str:
        """Return the current Textual theme name, including runtime error overrides."""
        if self._fatal_error_active:
            if self.active_theme_name == "celune_light":
                return "celune_light_error"
            return "celune_error"
        return self.active_theme_name

    def _register_runtime_error_themes(self) -> None:
        """Register error themes used for runtime failure states."""
        dark_foreground = colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_BACKGROUND,
            7.0,
        )
        light_foreground = colors.ensure_contrast(
            colors.ERROR_HIGHLIGHT,
            colors.ERROR_LIGHT_BACKGROUND,
            7.0,
        )
        dark_theme = Theme(
            name="celune_error",
            primary=colors.ERROR_DARK_ACCENT,
            secondary=colors.ERROR_DARK_ACCENT,
            accent=colors.THEME.error,
            foreground=dark_foreground,
            background=colors.ERROR_BACKGROUND,
            surface=colors.ERROR_BACKGROUND,
            warning=colors.THEME.warning,
            error=colors.THEME.error,
            dark=True,
        )
        light_theme = Theme(
            name="celune_light_error",
            primary=colors.ERROR_DARK_ACCENT,
            secondary=colors.ERROR_DARK_ACCENT,
            accent=colors.THEME_LIGHT.error,
            foreground=light_foreground,
            background=colors.ERROR_LIGHT_BACKGROUND,
            surface=colors.ERROR_LIGHT_BACKGROUND,
            warning=colors.THEME_LIGHT.warning,
            error=colors.THEME_LIGHT.error,
            dark=False,
        )
        if dark_theme.name not in self.available_themes:
            self.register_theme(dark_theme)
        if light_theme.name not in self.available_themes:
            self.register_theme(light_theme)

    def _ensure_themes_registered(self) -> None:
        """Register Celune's built-in themes when the app is not fully mounted yet."""
        if colors.THEME.name not in self.available_themes:
            self.register_theme(colors.THEME)
        if colors.THEME_LIGHT.name not in self.available_themes:
            self.register_theme(colors.THEME_LIGHT)
        if colors.THEME_APRIL_FOOLS.name not in self.available_themes:
            self.register_theme(colors.THEME_APRIL_FOOLS)
        self._register_runtime_error_themes()

    def _ensure_startup_error_themes_registered(self) -> None:
        """Register error themes without importing full runtime dependencies."""
        for theme in default_error_theme_family():
            if theme.name not in self.available_themes:
                self.register_theme(theme)

    def _prepare_loading_theme(self) -> None:
        """Apply Celune's palette before the first loading frame is rendered."""
        dark_theme, light_theme = default_theme_family()
        self.register_theme(dark_theme)
        self.register_theme(light_theme)
        self.theme = self.active_theme_name
        self.refresh_css(animate=False)

    def _apply_theme(self, theme_name: str) -> None:
        """Apply theme and repaint theme-sensitive widgets."""
        self._clear_border_pulses()
        self.active_theme_name = theme_name
        self.theme = self._runtime_theme_name()
        self._refresh_status()
        self._refresh_theme_text()
        self._refresh_logs(recolor=True)

    def _has_celune(self) -> bool:
        """Is the app attached to this UI instance?"""
        return self.celune is not None

    def prepare_theme(self) -> None:
        """Prepare the selected Celune theme before the first rendered frame."""
        if not _RUNTIME_DEPENDENCIES_LOADED:
            _load_ui_runtime_dependencies()
        colors.configure_theme()

        if self._has_celune():
            loader_factory = default_loader
            if loader_factory is None:
                _load_ui_runtime_dependencies()
                loader_factory = default_loader
            loader = loader_factory() if loader_factory is not None else None
            if loader is not None:
                theme = loader.bundle.metadata.get("theme")
                if isinstance(theme, dict):
                    background = theme.get("background")
                    accent = theme.get("accent")
                    faded_accent = theme.get("faded_accent")
                    if faded_accent is None:
                        faded_accent = theme.get("sleeping_color")
                    if (
                        isinstance(background, str)
                        and isinstance(accent, str)
                        and (faded_accent is None or isinstance(faded_accent, str))
                    ):
                        colors.configure_theme(
                            background,
                            accent,
                            faded_accent,
                        )

        self._ensure_themes_registered()
        if is_april_fools() and os.getenv("CELUNE_DISABLE_APRIL_FOOLS") not in {
            "1",
            "true",
            "on",
            "yes",
            "enabled",
        }:
            self.active_theme_name = "celune_april_fools"
        else:
            theme = os.getenv("CELUNE_THEME") or (
                self.celune.config.get("theme", "dark")
                if self.celune is not None
                else "dark"
            )

            if theme == "dark":
                self.active_theme_name = "celune"
            elif theme == "light":
                self.active_theme_name = "celune_light"
            else:
                self.active_theme_name = "celune"

        self.theme = self.active_theme_name
        self.refresh_css(animate=False)

    def _clear_border_pulses(self) -> None:
        """Remove temporary border pulse overrides so CSS can theme them."""
        for widget_key, widget in list(self._border_pulse_widgets.items()):
            self._border_pulse_tokens[widget_key] = (
                self._border_pulse_tokens.get(widget_key, 0) + 1
            )
            widget.styles.border = None
            widget.refresh(layout=False)

        self._border_pulse_widgets.clear()
        self._border_pulse_tokens.clear()

    def _refresh_theme_text(self) -> None:
        """Refresh widgets after a runtime theme change."""

        def repaint(widget: Widget) -> None:
            refresh = getattr(widget, "refresh", None)
            if refresh is None:
                return
            try:
                refresh(layout=False)
            except TypeError:
                refresh()

        runtime_theme_name = self._runtime_theme_name()
        self._ensure_themes_registered()
        if self.theme != runtime_theme_name:
            self.theme = runtime_theme_name
        try:
            screen = self.screen
        except ScreenStackError:
            screen = None
        if screen is not None and hasattr(screen, "styles"):
            screen.styles.background = None
            repaint(screen)
        if self.logs is not None:
            self.logs.styles.color = None
            self.logs.styles.border = None
            self.logs.styles.background = None
            self.logs.styles.scrollbar_color = None
            self.logs.styles.scrollbar_color_hover = None
            self.logs.styles.scrollbar_color_active = None
            self.logs.styles.scrollbar_background = None
            self.logs.styles.scrollbar_background_hover = None
            self.logs.styles.scrollbar_background_active = None
            repaint(self.logs)
        if self.input_box is not None:
            self.input_box.styles.color = None
            self.input_box.styles.border = None
            self.input_box.styles.background = None
            self.input_box.styles.scrollbar_color = None
            self.input_box.styles.scrollbar_color_hover = None
            self.input_box.styles.scrollbar_color_active = None
            self.input_box.styles.scrollbar_background = None
            self.input_box.styles.scrollbar_background_hover = None
            self.input_box.styles.scrollbar_background_active = None
            repaint(self.input_box)
        if self.style_button is not None:
            self.style_button.styles.color = None
            self.style_button.styles.border = None
            self.style_button.styles.background = None
            repaint(self.style_button)
        if self.resources is not None:
            self.resources.styles.color = None
            repaint(self.resources)
        if self.header is not None:
            self.header.styles.color = None
            repaint(self.header)
        for line in self.header_lines:
            line.styles.border_top = None
            repaint(line)
        if self.progress_bar is not None and hasattr(self.progress_bar, "styles"):
            self.progress_bar.styles.color = None
            self.progress_bar.styles.background = None
            repaint(self.progress_bar)
        if self.caption is not None and hasattr(self.caption, "styles"):
            self.caption.styles.color = None
            self.caption.styles.background = None
            repaint(self.caption)

    def _wrap_runtime_fatal_glow(self) -> None:
        """Mirror runtime fatal glow events into the UI fatal theme flag."""
        if self.celune is None or getattr(self.celune, "_ui_fatal_glow_wrapped", False):
            return

        glow = getattr(self.celune, "glow", None)
        if glow is None or not hasattr(glow, "fatal"):
            return
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            self._fatal_error_active = True
            self._run_on_ui_thread(self._refresh_theme_text)
            original_fatal()

        glow.fatal = wrapped_fatal
        self.celune._ui_fatal_glow_wrapped = True

    def _bind_runtime_callbacks(self) -> None:
        """Bind one attached Celune instance back into this UI."""
        if self.celune is None:
            return

        callbacks: tuple[tuple[str, Callable[..., None]], ...] = (
            ("log_callback", self.tts_log),
            ("status_callback", self.safe_status),
            ("error_callback", self.error),
            ("idle_callback", self.tts_idle),
            ("queue_avail_callback", self.tts_queue_avail),
            ("voice_changed_callback", self.tts_voice_changed),
            ("change_input_state_callback", self.change_input_state),
            ("change_voice_lock_state_callback", self.change_voice_lock_state),
            ("progress_callback", self.safe_progress),
            ("caption_progress_callback", self.safe_caption_progress),
            ("caption_callback", self.tts_caption),
            ("caption_timing_callback", self.tts_caption_timing),
        )
        for attribute, callback in callbacks:
            self._chain_runtime_callback(attribute, callback)

    def _chain_runtime_callback(
        self,
        attribute: str,
        callback: Callable[..., None],
    ) -> None:
        """Add one UI callback without overwriting another frontend callback."""
        if self.celune is None:
            return

        current_value = getattr(self.celune, attribute, None)
        if not callable(current_value):
            setattr(self.celune, attribute, callback)
            return
        current = cast(Callable[..., None], current_value)
        if current == callback:
            return
        if attribute == "log_callback" and callback == getattr(
            self.celune, "_startup_log_sink", None
        ):
            return
        chained_callbacks = getattr(current, "_celune_callback_chain", ())
        if callback in chained_callbacks:
            return

        def invoke(
            target: Callable[..., None],
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> None:
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                target(*args, **kwargs)
                return
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                target(*args)
            else:
                target(*args, **kwargs)

        def chained(*args: object, **kwargs: object) -> None:
            invoke(callback, args, kwargs)
            invoke(current, args, kwargs)

        chained._celune_callback_chain = (  # type: ignore[attr-defined]
            *chained_callbacks,
            callback,
        )
        setattr(self.celune, attribute, chained)

    def _publish_webui_timed_update(self) -> None:
        """Publish the current TUI timing state through the CEDTS UI channel."""
        if self.celune is None:
            return

        try:
            from ..cedts.ui import UiTimedUpdate, ui_timed_update_channel
        except ImportError:
            return

        sequence = getattr(self, "_webui_timed_update_sequence", 0) + 1
        self._webui_timed_update_sequence = sequence
        ui_timed_update_channel.publish(
            UiTimedUpdate(
                runtime_id=str(id(self.celune)),
                sequence=sequence,
                emitted_at=time.monotonic(),
                resource_page=self._resource_page,
                theme_name=self.active_theme_name,
                status_text=self._status_text,
                status_severity=self.status_severity,
                status_marquee_offset=self._status_marquee_offset,
            )
        )

    def _bind_agent_events(self) -> None:
        """Subscribe the UI to the existing typed agent lifecycle events."""
        self._unbind_agent_events()
        if self.celune is None:
            return
        dispatcher = getattr(self.celune, "_event_dispatcher", None)
        if dispatcher is None:
            return

        dispatcher.subscribe(
            "agent_task_state_changed",
            self._on_agent_task_state_changed,
            "CeluneUI",
        )
        dispatcher.subscribe(
            "agent_approval_requested",
            self._on_agent_approval_requested,
            "CeluneUI",
        )
        dispatcher.subscribe(
            "agent_choice_requested",
            self._on_agent_choice_requested,
            "CeluneUI",
        )
        dispatcher.subscribe(
            "agent_task_finished",
            self._on_agent_task_finished,
            "CeluneUI",
        )
        self._agent_event_dispatcher = dispatcher

    def _unbind_agent_events(self) -> None:
        """Unsubscribe UI lifecycle callbacks before replacing or closing Celune."""
        dispatcher = self._agent_event_dispatcher
        if dispatcher is None:
            return

        dispatcher.unsubscribe(
            "agent_task_state_changed", self._on_agent_task_state_changed
        )
        dispatcher.unsubscribe(
            "agent_approval_requested", self._on_agent_approval_requested
        )
        dispatcher.unsubscribe(
            "agent_choice_requested", self._on_agent_choice_requested
        )
        dispatcher.unsubscribe("agent_task_finished", self._on_agent_task_finished)
        self._agent_event_dispatcher = None

    def _on_agent_task_state_changed(
        self,
        event: AgentTaskStateChangedEvent,
    ) -> None:
        """Refresh the UI after a typed agent task transition."""
        self._agent_task_id = event.task_id
        self._run_on_ui_thread(self._refresh_agent_status)

    def _on_agent_approval_requested(
        self,
        event: AgentApprovalRequestedEvent,
    ) -> None:
        """Refresh the UI when a task pauses for approval."""
        self._agent_task_id = event.task_id
        self._run_on_ui_thread(self._refresh_agent_status)

    def _on_agent_choice_requested(
        self,
        event: AgentChoiceRequestedEvent,
    ) -> None:
        """Refresh the UI when a task pauses for a user choice."""
        self._agent_task_id = event.task_id
        self._run_on_ui_thread(self._refresh_agent_status)

    def _on_agent_task_finished(self, event: AgentTaskFinishedEvent) -> None:
        """Refresh the UI once a task reaches its terminal lifecycle state."""
        self._agent_task_id = event.task_id
        self._run_on_ui_thread(self._refresh_agent_status)

    def _agent_task_for_display(self) -> Optional[AgentTask]:
        """Return the active or most recently evented task for status rendering."""
        celune = self.celune
        if celune is None:
            return None
        runtime = getattr(celune, "agent_runtime", None)
        if runtime is None:
            return None

        active_task = runtime.get_active_task("default")
        if active_task is not None:
            self._agent_task_id = active_task.task_id
            return active_task
        if self._agent_task_id is None:
            return None
        with contextlib.suppress(ValueError):
            return runtime.get_task(self._agent_task_id)
        return None

    def _agent_status_text(
        self,
        task: Optional[AgentTask],
        busy_components: tuple[ComponentLockName, ...],
    ) -> Optional[str]:
        """Resolve one localized status message from typed task and lock state."""
        if busy_components:
            labels = ", ".join(
                string(f"agent.component_{component.value}")
                for component in busy_components
            )
            return string("agent.status.busy_components", components=labels)
        if task is None:
            return None
        if task.needs_context_compaction and task.state in {
            AgentTaskState.PLANNING,
            AgentTaskState.WORKING,
        }:
            return string("agent.status.compacting")
        if task.state in {
            AgentTaskState.WORKING,
            AgentTaskState.PLANNING,
            AgentTaskState.EXECUTING_TOOL,
            AgentTaskState.RESPONDING,
        }:
            return string(
                "agent.status.working",
                iteration=task.iterations,
                maximum=task.config.max_loops,
            )
        status_keys = {
            AgentTaskState.QUEUED: "agent.status.queued",
            AgentTaskState.IDLE: "agent.status.idle",
            AgentTaskState.CLASSIFYING: "agent.status.classifying",
            AgentTaskState.AWAITING_APPROVAL: "agent.status.awaiting_approval",
            AgentTaskState.AWAITING_CHOICE: "agent.status.awaiting_choice",
            AgentTaskState.PAUSED: "agent.status.paused",
            AgentTaskState.INTERRUPTED: "agent.status.interrupted",
            AgentTaskState.CANCELLING: "agent.status.cancelling",
            AgentTaskState.COMPLETED: "agent.status.completed",
            AgentTaskState.FAILED: "agent.status.failed",
            AgentTaskState.CANCELLED: "agent.status.cancelled",
            AgentTaskState.ABORTED: "agent.status.aborted",
        }
        key = status_keys.get(task.state)
        return string(key) if key is not None else None

    def _refresh_agent_status(self) -> None:
        """Project typed agent progress and component contention into the UI status."""
        celune = self.celune
        if celune is None or getattr(celune, "test_finished", False):
            return
        if getattr(celune, "cur_state", None) == "stopped":
            return

        task = self._agent_task_for_display()
        busy = getattr(celune, "last_component_busy", None)
        busy_components = tuple(getattr(busy, "components", ()))
        message = self._agent_status_text(task, busy_components)
        if message is None:
            return

        task_id = task.task_id if task is not None else ""
        task_state = task.state if task is not None else None
        iterations = task.iterations if task is not None else 0
        maximum = task.config.max_loops if task is not None else 0
        signature = (
            task_id,
            task_state.value if task_state is not None else "",
            str(iterations),
            str(maximum),
            *(component.value for component in busy_components),
            message,
        )
        self._agent_task_state = task_state
        self._agent_iterations = iterations
        self._agent_max_loops = maximum
        self._agent_busy_components = busy_components
        if self._agent_status_signature == signature:
            return
        self._agent_status_signature = signature
        self.safe_status(message, "warning" if busy_components else "info")

    def _is_ui_test_mode(self) -> bool:
        """Return whether the attached runtime is the interactive fake-backend UI test mode."""
        if self.celune is None:
            return False

        backend_mode = getattr(self.celune, "backend_mode", None)
        if isinstance(backend_mode, str):
            return backend_mode == "ui_test"
        backend = getattr(self.celune, "backend", None)
        return bool(getattr(backend, "is_fake", False)) and "pytest" not in sys.modules

    def _is_agent_test_mode(self) -> bool:
        """Return whether the attached runtime is the restricted agent test mode."""
        return bool(
            self.celune is not None
            and getattr(self.celune, "backend_mode", None) == "agent_test"
        )

    def _finish_test_startup(
        self,
        success: bool,
        detail: Optional[str] = None,
    ) -> None:
        """Report explicit test-mode startup completion once to the runner."""
        callback = self._test_completion_callback
        if callback is None or self.celune is None:
            return
        try:
            callback(self.celune, success, detail)
        except Exception as error:
            self.safe_log(
                format_error_message(
                    string("test.callback_failed"),
                    error,
                    getattr(self.celune, "log_level", self._startup_log_level),
                ),
                "error",
            )
        self._run_on_ui_thread(self._apply_test_finished_state)

    def _apply_test_finished_state(self) -> None:
        """Reconcile the visible UI with a completed explicit test."""
        celune = self.celune
        if celune is None or not getattr(celune, "test_finished", False):
            return

        self._hide_caption_widgets()
        self.change_input_state(locked=True)
        self.change_voice_lock_state(locked=True)

        voice = getattr(celune, "current_voice", None)
        if not isinstance(voice, str) or not voice:
            voices = getattr(celune, "voices", ())
            voice = voices[0] if voices else None
        if isinstance(voice, str) and voice:
            self.tts_voice_changed(voice)

        if self.input_box is not None:
            self.input_box.placeholder = string("ui.stopped_placeholder")
        self.safe_status(string("status.stopped"), "sleeping")
        self._refresh_logs()

    def _refresh_status(self) -> None:
        """Refresh the status color for the active theme."""
        if self.status is None:
            return
        self.status.styles.color = self._severity_color(self.status_severity)

    def _status_view_width(self) -> int:
        """Estimate how many status characters can fit without clipping."""
        if self.status is None:
            return 32

        size = getattr(self.status, "size", None)
        width = getattr(size, "width", 0) if size is not None else 0
        if isinstance(width, int) and width > 6:
            return max(8, width - 2)
        return 32

    def _render_status_text(self) -> str:
        """Return the current status text, marqueeing when it exceeds the label width."""
        width = self._status_view_width()
        if len(self._status_text) <= width:
            self._status_marquee_offset = 0
            return indent(self._status_text, spaces=2)

        loop = f"{self._status_text}{self._status_marquee_gap}"
        offset = self._status_marquee_offset % len(loop)
        window = (loop * 2)[offset : offset + width]
        return indent(window, spaces=2)

    def _update_status_label(self) -> None:
        """Push the current status text into the label."""
        if self.status is None:
            return
        self.status.update(self._render_status_text())
        self._refresh_status()

    def _advance_status_marquee(self) -> None:
        """Advance the marquee one character for long status messages."""
        if self.status is None:
            return
        if self.celune is not None and getattr(self.celune, "test_finished", False):
            return
        self._refresh_agent_status()
        playback_status = (
            current_playback_status(self.celune) if self.celune is not None else None
        )
        agent_is_active = self._agent_task_state in (
            _AGENT_ACTIVE_STATES | _AGENT_AWAITING_STATES | _AGENT_PAUSED_STATES
        )
        if (
            playback_status is not None
            and playback_status != self._status_text
            and not agent_is_active
        ):
            self._status_text = playback_status
            self.status_severity = "info"
            self._status_marquee_offset = 0
            self._refresh_theme_text()
        if len(self._status_text) <= self._status_view_width():
            self._update_status_label()
            self._publish_webui_timed_update()
            return
        self._status_marquee_offset += 1
        self._update_status_label()
        self._publish_webui_timed_update()

    def on_resize(self, _event: events.Resize) -> None:
        """Re-render width-sensitive widgets after the window size changes.

        Args:
            _event: Textual resize event that triggered the redraw.
        """
        if self.status is not None:
            self._update_status_label()

    def _refresh_logs(self, *, recolor: bool = False) -> None:
        """Reconcile visible log entries with the retained UI log history.

        Args:
            recolor: Whether existing entries should be rebuilt with the active
                theme colors.
        """
        if self.logs is None:
            return

        with self._log_history_lock:
            history = tuple(self.log_history)

        if not recolor:
            if self._rendered_log_count > len(history):
                self.logs.clear()
                self._rendered_log_count = 0
            elif self._rendered_log_count == len(history):
                if not history or not getattr(self.logs, "_size_known", True):
                    return
                if self.logs.lines:
                    if self.logs.auto_scroll:
                        self.logs.scroll_end(
                            animate=False,
                            immediate=True,
                            force=True,
                        )
                    return
                self.logs.clear()
                self._rendered_log_count = 0

            for message, severity in history[self._rendered_log_count :]:
                self.logs.write(
                    Text(message, style=self._severity_color(severity)),
                )
                self._rendered_log_count += 1
            return

        scroll_offset = self.logs.scroll_offset
        auto_scroll = self.logs.auto_scroll
        self.logs.auto_scroll = False
        self.logs.clear()

        for message, severity in history:
            self.logs.write(
                Text(message, style=self._severity_color(severity)),
                scroll_end=False,
            )

        self.logs.auto_scroll = auto_scroll
        self.logs.scroll_to(
            scroll_offset.x,
            scroll_offset.y,
            animate=False,
            immediate=True,
            force=True,
        )
        self._rendered_log_count = len(history)

    def _persist_log_entry(self, msg: str, severity: str) -> None:
        """Append one UI log entry to the persisted main-window log file."""
        with contextlib.suppress(OSError):
            if not self._log_file_initialized:
                self._log_file_path.write_text("", encoding="utf-8")
                self._log_file_initialized = True

            timestamp = datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            )
            with self._log_file_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] [{severity.upper()}] {msg}\n")

    def _prepare_terminal_output_stream(self) -> Optional[TextIOWrapper]:
        """Give Textual an independent terminal stream before stderr capture starts."""
        if os.name == "nt" or self._terminal_output_stream is not None:
            return self._terminal_output_stream

        original_stderr = sys.__stderr__
        if original_stderr is None:
            return None

        fileno = getattr(original_stderr, "fileno", None)
        if not callable(fileno):
            return None

        output_fd: Optional[int] = None
        output_stream: Optional[TextIOWrapper] = None
        discard(output_stream)
        try:
            output_fd = os.dup(cast(Callable[[], int], fileno)())
            output_stream = os.fdopen(
                output_fd,
                "w",
                encoding=getattr(original_stderr, "encoding", None) or "utf-8",
                errors=getattr(original_stderr, "errors", None) or "replace",
                buffering=1,
            )
        except (OSError, TypeError, ValueError):
            if output_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(output_fd)
            return None

        if output_stream is None:
            return None

        self._terminal_output_stream = output_stream
        sys.__stderr__ = output_stream
        return output_stream

    def run(
        self,
        *,
        headless: bool = False,
        inline: bool = False,
        inline_no_clear: bool = False,
        mouse: bool = True,
        size: Optional[tuple[int, int]] = None,
        auto_pilot: Optional[AutopilotCallbackType] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> Optional[ReturnType]:
        """Run Textual with an output stream independent of low-level stderr capture.

        Args:
            headless: Whether to run without interactive terminal input.
            inline: Whether to use Textual's inline terminal mode.
            inline_no_clear: Whether inline mode should preserve existing terminal output.
            mouse: Whether mouse input should be enabled.
            size: Optional terminal size override.
            auto_pilot: Optional callback used to drive automated interaction.
            loop: Optional event loop used by the Textual application.

        Returns:
            Optional return value produced by the Textual application.
        """
        original_stderr = sys.__stderr__
        output_stream = self._prepare_terminal_output_stream()

        try:
            return super().run(
                headless=headless,
                inline=inline,
                inline_no_clear=inline_no_clear,
                mouse=mouse,
                size=size,
                auto_pilot=auto_pilot,
                loop=loop,
            )
        finally:
            if output_stream is not None:
                sys.__stderr__ = original_stderr
                self._terminal_output_stream = None
                with contextlib.suppress(OSError, ValueError):
                    output_stream.close()

    def compose(self) -> ComposeResult:
        """Define the UI.

        Returns:
            ComposeResult: The root widget tree for the interface.
        """
        with Vertical(id="container"):
            with Horizontal(id="header-container"):
                yield Label("", classes="line")
                yield Label(APP_NAME, id="header")
                yield Label("", classes="line")
            yield RichLog(id="logs", wrap=True, markup=False)
            yield Label("", id="caption", markup=False)
            with Horizontal(id="progress-container"):
                yield ProgressBar(
                    id="progress", show_percentage=False, show_eta=False, total=1
                )
                yield ProgressLabel(widget_id="progress-label")
            with Horizontal(id="controls"):
                yield TextArea(id="input", placeholder=string("ui.wait_placeholder"))
                yield VoiceButton(
                    string("ui.no_voice_set"),
                    widget_id="style",
                    disabled=True,
                )
                yield Button(string("ui.vc_mode_talk"), id="vc-mode", disabled=True)
                yield Button(
                    string("ui.vc_pitch_button", value="+0"),
                    id="vc-pitch",
                    disabled=True,
                )
            with Horizontal(id="bottom"):
                yield Label("", id="status")
                yield Label("", id="resources")
        yield CeluneLoadingScreen(widget_id="loading-overlay")

    def on_mount(self) -> None:
        """Prepare the UI and start deferred runtime initialization."""
        if os.name == "nt":
            self._install_windows_signal_handler()
        else:
            if SIGTSTP is not None:
                signal.signal(SIGTSTP, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self.set_interval(0.1, self._check_launcher_loss)

        self._loading_screen = self.query_one("#loading-overlay", CeluneLoadingScreen)
        self._loading_screen.set_startup_messages(self._startup_messages)
        self.logs = self.query_one("#logs", RichLog)
        self.input_box = self.query_one("#input", TextArea)
        self.status = self.query_one("#status", Label)
        self.resources = self.query_one("#resources", Label)
        self.caption = self.query_one("#caption", Label)
        self.style_button = self.query_one("#style", Button)
        self.vc_mode_button = self.query_one("#vc-mode", Button)
        self.vc_pitch_button = self.query_one("#vc-pitch", Button)
        self.progress_bar = self.query_one("#progress", ProgressBar)
        self.progress_label = self.query_one("#progress-label", ProgressLabel)
        self.header = self.query_one("#header", Label)
        self.header_lines = tuple(cast(Label, widget) for widget in self.query(".line"))
        self._refresh_logs()

        self.set_focus(None)
        self._prepare_loading_theme()
        self._show_loading_screen()
        if self._loading_screen is not None:
            self._loading_screen.set_status_message(string("status.initializing"))
        self._status_text = string("status.initializing")
        if self._startup_messages:
            terminal_status = self._startup_terminal_status_for(
                self._startup_messages[-1]
            )
            if terminal_status is not None:
                self._set_terminal_status(*terminal_status)
            else:
                self._set_terminal_status("initializing", string("osc.action_starting"))
        else:
            self._set_terminal_status("initializing", string("osc.action_starting"))
        if self._startup_loader is not None or self.celune is not None:
            if self._startup_loader is not None:
                self.call_after_refresh(self._start_deferred_runtime)
            else:
                self.attach_celune(self.celune)

    def _start_deferred_runtime(self) -> None:
        """Start constructing Celune after the initial loading frame renders."""
        self.run_worker(self._load_deferred_runtime, thread=True, exclusive=True)

    def _startup_terminal_status_for(self, message: str) -> Optional[tuple[str, str]]:
        """Resolve a loading-screen diagnostic to its terminal title transition."""
        startup_actions = {
            string("ui.startup_checking_dependencies"): (
                "initializing",
                string("osc.action_checking_dependencies"),
            ),
            string("ui.startup_loading_core"): (
                "initializing",
                string("osc.action_loading_core"),
            ),
            string("ui.startup_initializing_core"): (
                "initializing",
                string("osc.action_initializing_core"),
            ),
        }
        return startup_actions.get(message)

    def receive_startup_diagnostic(self, message: str) -> None:
        """Display one early startup diagnostic on the loading screen.

        Args:
            message: Diagnostic emitted while the runtime is being prepared.
        """
        self._startup_messages.append(message)
        terminal_status = self._startup_terminal_status_for(message)
        if terminal_status is not None:
            self._set_terminal_status(*terminal_status)

        def update() -> None:
            if self._loading_screen is not None:
                self._loading_screen.append_startup_message(message)

        self._run_on_ui_thread(update)

    def _emit_startup_diagnostic(self, message: str) -> None:
        """Display a verbose diagnostic for a stage of deferred startup.

        Args:
            message: Diagnostic text describing the current startup stage.
        """
        terminal_status = self._startup_terminal_status_for(message)
        if terminal_status is not None:
            self._set_terminal_status(*terminal_status)
        if self._startup_log_level != "info":
            self.receive_startup_diagnostic(message)

    def _load_deferred_runtime(self) -> None:
        """Construct the engine and load optional UI integrations off the UI thread."""
        if self._startup_loader is None and self.celune is None:
            return
        try:
            celune = self.celune
            if self._startup_loader is not None:
                celune = self._startup_loader()
            _load_ui_runtime_dependencies()
        except BaseException as exc:
            self.call_from_thread(self._handle_deferred_runtime_error, exc)
            return
        if celune is not None:
            self.call_from_thread(self.attach_celune, celune)

    def _handle_deferred_runtime_error(self, error: BaseException) -> None:
        """Show a deferred startup failure without tearing down the UI.

        Args:
            error: Exception raised while constructing the deferred runtime.
        """
        missing_dependency = isinstance(error, ModuleNotFoundError) or (
            isinstance(error, SystemExit)
            and error.code == ExitCodes.EXIT_MISSING_DEPENDENCIES.value
        )
        self.cur_state = "error"
        self._startup_error_exit_code = (
            ExitCodes.EXIT_MISSING_DEPENDENCIES.value
            if missing_dependency
            else ExitCodes.EXIT_FAILURE.value
        )
        self._fatal_error_active = True
        self._ensure_startup_error_themes_registered()
        self.theme = self._runtime_theme_name()
        self.refresh_css(animate=False)
        if isinstance(error, ModuleNotFoundError) and error.name is not None:
            failure_status = string("status.missing_dependency")
            terminal_action = string("osc.action_missing_dependency")
        else:
            failure_status = string("status.early_initialization_failed")
            terminal_action = string("osc.action_early_initialization_failed")
        self._terminal_status = (
            APP_NAME,
            string("osc.state_error"),
            terminal_action,
        )
        self._write_terminal_title(self._terminal_status)
        message = format_error_message(
            tagged_string("ui.init_error", "INIT ERROR"),
            error,
            self._startup_log_level,
        )
        self._show_loading_error(
            message,
            status_message=string("status.early_initialization_failed"),
            footer_message=failure_status,
        )

    def attach_celune(self, celune: Celune) -> None:
        """Attach the constructed engine and begin its normal initialization."""
        if threading.current_thread() is not threading.main_thread():
            self.call_from_thread(self.attach_celune, celune)
            return
        if self.cur_state == "exiting":
            celune.close()
            return

        if default_loader is None or ui_resources is None:
            _load_ui_runtime_dependencies()
        self._log_file_path = main_window_log_path(create_parent=True)
        self._unbind_agent_events()
        self.celune = celune
        self._bind_runtime_callbacks()
        self._bind_agent_events()
        self.prepare_theme()
        self._wrap_runtime_fatal_glow()
        configured_theme = os.getenv("CELUNE_THEME") or self.celune.config.get(
            "theme", "dark"
        )
        if self.active_theme_name == "celune" and configured_theme not in {
            "dark",
            "light",
        }:
            self.safe_log(string("ui.invalid_theme_defaulting_dark"), "warning")
        self._refresh_theme_text()
        self.refresh_vc_controls()
        if not self.celune.backend.is_fake or "pytest" in sys.modules:
            self._enable_runtime_log_capture()
        resources = ui_resources
        if resources is None:
            return
        resources.prime_usage()
        resources.start_gpu_usage_worker()
        if not self._runtime_intervals_started:
            self.set_interval(FOOTER_ROTATE_SECONDS, self.advance_resources)
            self._status_marquee_timer = self.set_interval(
                0.18, self._advance_status_marquee
            )
            self._runtime_intervals_started = True
        self.update_resources()
        self.call_after_refresh(self.start_background_init)

    def _check_launcher_loss(self) -> None:
        """Run the normal UI shutdown path after the launcher disconnects."""
        if launcher_loss_requested() and self.cur_state != "exiting":
            self._graceful_exit()

    def update_resources(self) -> None:
        """Refresh the currently selected resource footer page."""
        if self.cur_state == "exiting" or self.resources is None or self.celune is None:
            return
        resources = ui_resources
        if resources is None:
            return

        def update() -> None:
            pages = resources.resource_pages(self.celune, self.active_theme_name)
            text = pages[self._resource_page % len(pages)]

            self.resources.update(indent(text, spaces=2, direction="right"))

        self._run_on_ui_thread(update)

    def _enable_runtime_log_capture(self) -> None:
        """Capture Celune runtime output after the Textual app has started cleanly."""
        if self._runtime_log_capture_enabled:
            return

        self._log_stdout = LogRedirect(
            write_callback=self.safe_log,
            default_severity="info",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )
        self._log_stderr = LogRedirect(
            write_callback=self.safe_log,
            default_severity="warning",
            stdout=self._old_stdout,
            stderr=self._old_stderr,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )

        sys.stdout = self._log_stdout
        sys.stderr = self._log_stderr
        self._redirect_dunder_stdio()
        self._install_runtime_log_redirects()
        self._install_low_level_stderr_capture()
        self._runtime_log_capture_enabled = True

    def _write_terminal_escape(self, escape: str) -> None:
        """Write one ANSI escape sequence to the real terminal when available."""
        if self._log_stdout is not None:
            self._log_stdout.ansi(escape)
            return

        if self._old_stdout is not None:
            self._old_stdout.write(escape)
            self._old_stdout.flush()

    def _write_terminal_title(self, status: tuple[str, str, str]) -> None:
        """Write one structured state title to the real terminal."""
        if self._log_stdout is not None:
            self._log_stdout.ansi(terminal_title_escape(status))
            return

        if self._old_stdout is not None:
            set_terminal_title(status, self._old_stdout)

    def _set_terminal_status(self, state: str, action: str) -> None:
        """Publish a stable state and action in the terminal title."""
        status = (APP_NAME, string(f"osc.state_{state}"), action)
        if getattr(self, "_terminal_status", None) == status:
            return

        self._terminal_status = status

        def update() -> None:
            if supports_ansi(self._old_stdout):
                self._write_terminal_title(status)

        self._run_on_ui_thread(update)

    def _terminal_status_for(self, msg: str, severity: str) -> tuple[str, str]:
        """Resolve the terminal glossary state and action for one UI status."""
        status_actions = {
            string("status.api_starting"): (
                "initializing",
                string("osc.action_starting"),
            ),
            string("status.could_not_continue"): (
                "error",
                string("osc.action_failed"),
            ),
            string("status.could_not_reload"): (
                "error",
                string("osc.action_failed"),
            ),
            string("status.could_not_start"): (
                "error",
                string("osc.action_failed"),
            ),
            string("status.could_not_wake"): (
                "error",
                string("osc.action_failed"),
            ),
            string("status.downloading_audio"): (
                "speaking",
                string("osc.action_downloading"),
            ),
            string("status.early_initialization_failed"): (
                "error",
                string("osc.action_early_initialization_failed"),
            ),
            string("status.failed_to_start"): (
                "error",
                string("osc.action_failed"),
            ),
            string("status.generating"): (
                "speaking",
                string("osc.action_generating_audio"),
            ),
            string("status.idle"): ("ready", string("osc.action_idle")),
            string("ui.idle_status"): ("ready", string("osc.action_idle")),
            string("status.initializing"): (
                "initializing",
                string("osc.action_starting"),
            ),
            string("status.missing_dependency"): (
                "error",
                string("osc.action_missing_dependency"),
            ),
            string("status.normalizing"): (
                "thinking",
                string("osc.action_normalizing"),
            ),
            string("status.reloading"): (
                "reloading",
                string("osc.action_reloading"),
            ),
            string("status.reloading_backend"): (
                "reloading",
                string("osc.action_loading_backend"),
            ),
            string("status.reloading_character"): (
                "reloading",
                string("osc.action_loading_voice"),
            ),
            string("status.restoring_backend"): (
                "reloading",
                string("osc.action_restoring"),
            ),
            string("status.sleeping"): ("sleeping", string("osc.action_idle")),
            string("ui.sleeping_status"): ("sleeping", string("osc.action_idle")),
            string("status.speaking"): (
                "speaking",
                string("osc.action_playing_audio"),
            ),
            string("status.stopped"): ("stopped", string("osc.action_stopped")),
            string("status.thinking"): ("thinking", string("osc.action_thinking")),
            string("status.waiting_for_model"): (
                "initializing",
                string("osc.action_waiting_for_model"),
            ),
            string("status.waking_up"): (
                "initializing",
                string("osc.action_waking_up"),
            ),
            string("status.warming_up"): (
                "initializing",
                string("osc.action_warming_up"),
            ),
        }
        if severity == "error":
            if msg in status_actions:
                return status_actions[msg]
            return "error", string("osc.action_error")
        if severity == "warning":
            return "warning", string("osc.action_warning")

        if self._persona_recording_active() or self._vc_recording_active():
            if self._persona_recording_stop_requested:
                return "recording", string("osc.action_transcribing_speech")
            return "recording", string("osc.action_listening_microphone")

        runtime_state = getattr(self.celune, "cur_state", "idle")
        if runtime_state == "stopped":
            return "stopped", string("osc.action_stopped")
        if msg in status_actions:
            return status_actions[msg]
        if msg.startswith(string("pipeline.playing_label", label="")):
            return "speaking", string("osc.action_playing_audio")
        if msg.startswith(string("pipeline.revoicing_label", label="")):
            return "speaking", string("osc.action_playing_audio")
        if not self.celune_ready:
            return "initializing", string("osc.action_loading_voice_pack")

        if self._agent_task_state in _AGENT_AWAITING_STATES:
            return "awaiting", msg
        if self._agent_task_state in _AGENT_PAUSED_STATES:
            return "paused", msg
        if self._agent_task_state in _AGENT_ACTIVE_STATES:
            return "thinking", msg
        state_actions = {
            "idle": ("ready", string("osc.action_idle")),
            "thinking": ("thinking", string("osc.action_thinking")),
            "generating": ("speaking", string("osc.action_generating_audio")),
            "speaking": ("speaking", string("osc.action_playing_audio")),
            "sleeping": ("sleeping", string("osc.action_idle")),
            "stopped": ("stopped", string("osc.action_stopped")),
            "waking": ("initializing", string("osc.action_waking_up")),
            "reloading": ("reloading", string("osc.action_reloading")),
            "error": ("error", string("osc.action_error")),
            "restarting": ("restarting", string("osc.action_restarting")),
        }
        return state_actions.get(
            runtime_state,
            ("ready", string("osc.action_idle")),
        )

    def _install_runtime_log_redirects(self) -> None:
        """Route non-Celune Python logging output into Celune's UI log widget."""
        if self._runtime_redirect_handler is not None:
            return

        handler = UILogHandler(
            self.safe_log,
            filter_messages=_RUNTIME_LOG_REDIRECT_FILTER_MESSAGES,
        )
        original_call_handlers = logging.Logger.callHandlers

        def call_handlers(self: logging.Logger, record: logging.LogRecord) -> None:
            if is_celune_log_record(record):
                original_call_handlers(self, record)
                return

            handler.handle(record)

        self._runtime_redirect_handler = handler
        self._runtime_redirect_original_call_handlers = original_call_handlers
        self._runtime_redirect_original_last_resort = logging.lastResort
        self._runtime_redirect_original_raise_exceptions = logging.raiseExceptions
        logging.Logger.callHandlers = call_handlers
        logging.lastResort = None
        logging.raiseExceptions = False

        logging.captureWarnings(True)
        self._warnings_capture_enabled = True

    def _redirect_dunder_stdio(self) -> None:
        """Redirect ``sys.__stdout__`` and ``sys.__stderr__`` when possible."""
        if self._original_dunder_stdout is None:
            self._original_dunder_stdout = sys.__stdout__
        if self._original_dunder_stderr is None:
            self._original_dunder_stderr = sys.__stderr__

        if self._log_stdout is not None:
            sys.__stdout__ = self._log_stdout
        if self._log_stderr is not None:
            sys.__stderr__ = self._log_stderr

    def _restore_dunder_stdio(self) -> None:
        """Restore ``sys.__stdout__`` and ``sys.__stderr__`` after capture ends."""
        if self._original_dunder_stdout is not None:
            sys.__stdout__ = self._original_dunder_stdout
        if self._original_dunder_stderr is not None:
            sys.__stderr__ = self._original_dunder_stderr

        self._original_dunder_stdout = None
        self._original_dunder_stderr = None

    def _install_low_level_stderr_capture(self) -> None:
        """Capture writes that bypass Python and go straight to stderr."""
        if self._stderr_forward_thread is not None:
            return

        stderr_stream = self._old_stderr
        if stderr_stream is None or not hasattr(stderr_stream, "fileno"):
            return

        original_fd_dup: Optional[int] = None
        pipe_read_fd: Optional[int] = None
        pipe_write_fd: Optional[int] = None

        try:
            stderr_fd = stderr_stream.fileno()
            if not isinstance(stderr_fd, int):
                return
            original_fd_dup = os.dup(stderr_fd)
            pipe_read_fd, pipe_write_fd = os.pipe()
            os.dup2(pipe_write_fd, stderr_fd)
        except (AttributeError, OSError, TypeError, ValueError):
            with contextlib.suppress(OSError):
                if original_fd_dup is not None:
                    os.close(original_fd_dup)
            with contextlib.suppress(OSError):
                if pipe_read_fd is not None:
                    os.close(pipe_read_fd)
            with contextlib.suppress(OSError):
                if pipe_write_fd is not None:
                    os.close(pipe_write_fd)
            return

        self._stderr_original_fd_dup = original_fd_dup
        self._stderr_pipe_read_fd = pipe_read_fd
        self._stderr_pipe_write_fd = pipe_write_fd

        forward_thread = threading.Thread(
            target=self._forward_low_level_stderr,
            name="celune-stderr-capture",
            daemon=True,
        )
        self._stderr_forward_thread = forward_thread
        forward_thread.start()

    @staticmethod
    def _is_textual_terminal_frame(payload: bytes) -> bool:
        """Return whether bytes contain a Textual synchronized terminal frame."""
        return b"\x1b[?2026h" in payload or b"\x1b[?2026l" in payload

    def _forward_low_level_stderr(self) -> None:
        """Forward low-level stderr bytes back to the terminal and UI log."""
        read_fd = self._stderr_pipe_read_fd
        original_fd_dup = self._stderr_original_fd_dup
        redirect = self._log_stderr

        if read_fd is None or original_fd_dup is None or redirect is None:
            return

        encoding = getattr(self._old_stderr, "encoding", None) or "utf-8"
        errors = getattr(self._old_stderr, "errors", None) or "replace"

        while True:
            try:
                payload = os.read(read_fd, 4096)
            except OSError:
                break

            if not payload:
                break

            if self._is_textual_terminal_frame(payload):
                try:
                    os.write(original_fd_dup, payload)
                except OSError:
                    break
                continue

            redirect.write(payload.decode(encoding, errors=errors))

        redirect.flush()

    def _remove_low_level_stderr_capture(self) -> None:
        """Restore stderr after low-level capture was installed."""
        stderr_stream = self._old_stderr
        original_fd_dup = self._stderr_original_fd_dup
        pipe_write_fd = self._stderr_pipe_write_fd
        pipe_read_fd = self._stderr_pipe_read_fd

        if (
            stderr_stream is not None
            and hasattr(stderr_stream, "fileno")
            and original_fd_dup is not None
        ):
            with contextlib.suppress(OSError, ValueError):
                stderr_stream.flush()
            with contextlib.suppress(OSError, ValueError):
                os.dup2(original_fd_dup, stderr_stream.fileno())

        if pipe_write_fd is not None:
            with contextlib.suppress(OSError):
                os.close(pipe_write_fd)
        if pipe_read_fd is not None:
            with contextlib.suppress(OSError):
                os.close(pipe_read_fd)
        if original_fd_dup is not None:
            with contextlib.suppress(OSError):
                os.close(original_fd_dup)

        self._stderr_pipe_read_fd = None
        self._stderr_pipe_write_fd = None
        self._stderr_original_fd_dup = None
        self._stderr_forward_thread = None

    def _remove_runtime_log_redirects(self) -> None:
        """Restore Python logging dispatch after UI shutdown."""
        handler = self._runtime_redirect_handler
        original_call_handlers = self._runtime_redirect_original_call_handlers
        if handler is not None and original_call_handlers is not None:
            logging.Logger.callHandlers = original_call_handlers
            logging.lastResort = self._runtime_redirect_original_last_resort
            if self._runtime_redirect_original_raise_exceptions is not None:
                logging.raiseExceptions = (
                    self._runtime_redirect_original_raise_exceptions
                )
            handler.close()

        if self._warnings_capture_enabled:
            logging.captureWarnings(False)
            self._warnings_capture_enabled = False

        self._runtime_redirect_handler = None
        self._runtime_redirect_original_call_handlers = None
        self._runtime_redirect_original_last_resort = None
        self._runtime_redirect_original_raise_exceptions = None

    def _disable_runtime_log_capture(self) -> None:
        """Restore global stdio once the UI is shutting down."""
        if self._log_stdout is not None:
            self._log_stdout.flush()
        if self._log_stderr is not None:
            self._log_stderr.flush()

        self._remove_low_level_stderr_capture()
        self._remove_runtime_log_redirects()
        self._restore_dunder_stdio()

        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._runtime_log_capture_enabled = False

    def advance_resources(self) -> None:
        """Advance the resource footer to the next page and refresh it."""
        if self.cur_state == "exiting" or self.resources is None:
            return
        resources = ui_resources
        if resources is None or self.celune is None:
            return

        self._resource_page = (self._resource_page + 1) % len(
            resources.resource_pages(self.celune, self.active_theme_name)
        )
        self.update_resources()
        self._publish_webui_timed_update()

    def _cancel_sleep_timer(self) -> None:
        """Cancel a pending automatic sleep transition."""
        if threading.current_thread() is not threading.main_thread():
            self._run_on_ui_thread(self._cancel_sleep_timer)
            return

        if self._sleep_timer is not None:
            self._sleep_timer.stop()
            self._sleep_timer = None

    def _schedule_sleep_timer(self) -> None:
        """Schedule automatic sleep after the configured idle timeout."""
        if threading.current_thread() is not threading.main_thread():
            self._run_on_ui_thread(self._schedule_sleep_timer)
            return

        self._cancel_sleep_timer()
        if (
            self.cur_state == "exiting"
            or not self.celune_ready
            or not hasattr(self.celune, "sleep_enabled")
            or not self.celune.sleep_enabled()
            or self.celune.sleeping
            or self.celune.is_in_tutorial
        ):
            return

        self._sleep_timer = self.set_timer(
            self.celune.sleep_timeout_seconds(),
            self._enter_sleep_mode,
        )

    def _enter_sleep_mode(self) -> None:
        """Put the app to sleep from the UI idle timer."""
        self._sleep_timer = None
        if self.cur_state == "exiting" or self.celune is None:
            return

        self.enter_sleep_mode()

    @work(exclusive=True)
    async def enter_sleep_mode(self) -> None:
        """Put the app to sleep without blocking the UI event loop."""
        if await self.celune.enter_sleep_mode_async():
            if self.cur_state == "exiting":
                return
            self.safe_log(
                string("ui.sleeping_log", app_name=APP_NAME),
                "sleeping",
            )
            self.safe_status(string("ui.sleeping_status"), "sleeping")
            self.change_voice_lock_state(
                locked=True,
                can_open_menu=bool(self.celune_styles),
            )

    @work(exclusive=True)
    async def wake_from_sleep(self) -> None:
        """Wake the app after the user types into the sleeping UI."""
        if self.cur_state != "exiting":
            self._run_on_ui_thread(self._reset_playback_widgets)
        try:
            if (
                await self.celune.wake_from_sleep_async()
                and self.cur_state != "exiting"
            ):
                self._schedule_sleep_timer()
        finally:
            if self.cur_state != "exiting" and self.celune.sleeping:
                self.safe_status(string("ui.sleeping_status"), "sleeping")

    def start_background_init(self) -> None:
        """Run the initialization function."""
        self._show_loading_screen()
        self.load_tts()

    def _show_loading_screen(self) -> None:
        """Reveal the startup overlay already mounted above the main UI."""
        if self._loading_screen is None:
            return
        try:
            main_container = self.query_one("#container", Vertical)
        except (NoMatches, ScreenStackError):
            main_container = None
        if main_container is not None:
            main_container.styles.opacity = 0.0
            main_container.display = False
        self._loading_screen.styles.opacity = 1.0
        self._loading_screen.display = True

    def _update_loading_log(self, message: str) -> None:
        """Forward one useful startup log line to the loading screen.

        Args:
            message: Non-verbose, non-debug log message to display.
        """
        if self._loading_screen is not None:
            self._loading_screen.set_latest_log_message(message)

    def _show_loading_error(
        self,
        message: str,
        *,
        status_message: Optional[str] = None,
        footer_message: Optional[str] = None,
    ) -> None:
        """Keep the loading screen visible while showing an initialization error.

        Args:
            message: Initialization error to display.
            status_message: Optional replacement for the failure heading.
            footer_message: Optional status to show in the lower-left footer.
        """

        def update() -> None:
            if self._loading_screen is not None:
                self._loading_screen.show_error(
                    message,
                    status_message=status_message,
                    footer_message=footer_message,
                )

        self._run_on_ui_thread(update)

    def _dismiss_loading_screen(self) -> None:
        """Fade out and remove the startup screen after successful loading."""

        def dismiss() -> None:
            overlay = self._loading_screen
            if overlay is None:
                return
            main_container: Optional[Vertical] = None

            def reveal_main_ui() -> None:
                nonlocal main_container
                if self._loading_screen is not overlay:
                    return
                try:
                    main_container = self.query_one("#container", Vertical)
                except (NoMatches, ScreenStackError):
                    self.call_after_refresh(fade_overlay)
                    return
                main_container.styles.opacity = 0.0
                main_container.display = True
                main_container.refresh(layout=True, repaint=True)
                self.refresh(layout=True, repaint=True)
                self.call_after_refresh(fade_overlay)

            def show_main_ui() -> None:
                if self._loading_screen is not overlay:
                    return
                overlay.display = False
                if main_container is not None:
                    self._animate_opacity(
                        main_container,
                        1.0,
                        duration=_MAIN_UI_FADE_SECONDS,
                    )
                self.call_after_refresh(self._refresh_logs)

            def fade_overlay() -> None:
                if self._loading_screen is not overlay:
                    return
                overlay.animate(
                    "opacity",
                    0.0,
                    duration=_LOADING_FADE_SECONDS,
                    easing="out_cubic",
                    on_complete=show_main_ui,
                )

            self.call_after_refresh(reveal_main_ui)

        self._run_on_ui_thread(dismiss)

    @work(thread=True, exclusive=True)
    def load_tts(self) -> None:
        """Load the app runtime."""
        try:
            if self.celune.load():
                self.celune_styles = self.celune.voices
                if not self.celune_styles:
                    if self._is_ui_test_mode():
                        if not self.celune.use_normalization:
                            self.safe_progress(1, 1)
                        self.change_input_state(locked=True)
                        self.change_voice_lock_state(locked=True)
                        self.safe_status(string("ui.test_mode_active"))
                        self._dismiss_loading_screen()
                        self._finish_test_startup(True)
                        return

                    self.change_input_state(locked=True)
                    self.change_voice_lock_state(locked=True)
                    self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                    self.cur_state = "error"
                    self._show_loading_error(string("ui.no_voices_loaded"))
                    return
                if self._is_agent_test_mode():
                    self.celune_ready = True
                    active_voice = self.celune.current_voice or self.celune_styles[0]
                    self.tts_voice_changed(active_voice)
                    self.change_input_state(locked=True)
                    self.change_voice_lock_state(locked=True)
                    self.safe_status(string("ui.agent_test_mode_active"))
                    self._dismiss_loading_screen()
                    self._finish_test_startup(True)
                    return
                self.celune_voices = itertools.cycle(self.celune_styles)
                if self.celune.current_voice in self.celune_styles:
                    self.style_index = self.celune_styles.index(
                        self.celune.current_voice
                    )
                else:
                    self.style_index = 0
                self.celune_ready = True
                self.safe_status(string("ui.idle_status"))
                self.tts_voice_changed(
                    self.celune.current_voice or self.celune.voices[0]
                )
                self.safe_progress(1, 1)
                self.change_input_state(locked=False)
                self.change_voice_lock_state(locked=len(self.celune.voices) < 2)
                self.safe_log(string("ui.tutorial_prompt", app_name=APP_NAME))
                self._schedule_sleep_timer()
                self._set_terminal_status("ready", string("osc.action_idle"))
                self._dismiss_loading_screen()
            else:
                self.cur_state = "error"
                self.change_input_state(locked=True)
                self.change_voice_lock_state(locked=True)
                self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                self._show_loading_error(
                    string("ui.app_could_not_start", app_name=APP_NAME)
                )
                self._finish_test_startup(
                    False,
                    string("ui.app_could_not_start", app_name=APP_NAME),
                )
        except Exception as e:
            self.cur_state = "error"
            error_message = format_error_message(
                tagged_string("ui.init_error", "INIT ERROR"),
                e,
                getattr(self.celune, "log_level", self._startup_log_level),
            )
            self.safe_log(error_message, "error")
            self.celune.fatal()
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            self.error(string("ui.app_could_not_start", app_name=APP_NAME))
            self._show_loading_error(error_message)
            self._finish_test_startup(False, error_message)

    @staticmethod
    def _caption_word_timing_ranges(
        words: tuple[str, ...],
        segments: tuple[WhisperSegment, ...],
        audio_duration: float,
        timing_words: Optional[tuple[str, ...]] = None,
    ) -> tuple[tuple[float, float], ...]:
        """Map normalized speech timestamps onto displayed caption words."""
        if not words or not segments or audio_duration <= 0.0:
            return ()

        whisper_words = [
            word
            for segment in segments
            for word in segment.words
            if word.text and word.end >= word.start
        ]
        if not whisper_words:
            return ()

        def normalize(value: str) -> str:
            return re.sub(r"[^\w]+", "", value.casefold())

        matching_words = timing_words if timing_words else words
        caption_keys = [normalize(word) for word in matching_words]
        whisper_keys = [normalize(word.text) for word in whisper_words]
        assigned: list[Optional[int]] = [None] * len(matching_words)
        whisper_index = 0
        for caption_index, caption_key in enumerate(caption_keys):
            if not caption_key:
                continue
            for candidate in range(
                whisper_index,
                min(len(whisper_words), whisper_index + 5),
            ):
                if caption_key == whisper_keys[candidate]:
                    assigned[caption_index] = candidate
                    whisper_index = candidate + 1
                    break

        timing_ranges: list[tuple[float, float]] = []
        for index, assigned_index in enumerate(assigned):
            if assigned_index is None:
                assigned_index = round(
                    index * (len(whisper_words) - 1) / max(len(matching_words) - 1, 1)
                )
            assigned_index = max(0, min(len(whisper_words) - 1, assigned_index))
            word = whisper_words[assigned_index]
            start = max(0.0, min(audio_duration, word.start))
            end = max(start, min(audio_duration, word.end))
            timing_ranges.append((start, end))

        previous_end = 0.0
        normalized_ranges: list[tuple[float, float]] = []
        for start, end in timing_ranges:
            start = max(previous_end, start)
            end = max(start, end)
            normalized_ranges.append((start, end))
            previous_end = end
        if len(matching_words) == len(words):
            return tuple(normalized_ranges)

        displayed_ranges: list[tuple[float, float]] = []
        for index in range(len(words)):
            start_index = min(
                len(normalized_ranges) - 1,
                math.floor(index * len(normalized_ranges) / len(words)),
            )
            end_index = math.ceil((index + 1) * len(normalized_ranges) / len(words))
            end_index = max(start_index + 1, end_index)
            end_index = min(end_index, len(normalized_ranges))
            displayed_ranges.append(
                (
                    normalized_ranges[start_index][0],
                    normalized_ranges[end_index - 1][1],
                )
            )
        return tuple(displayed_ranges)

    def tts_caption_timing(
        self,
        caption: str,
        audio: AudioChunk,
        sample_rate: int,
        timing_text: Optional[str] = None,
    ) -> None:
        """Refine displayed caption timing from normalized speech timestamps."""
        if (
            self.cur_state == "exiting"
            or getattr(self.celune, "test_finished", False)
            or not caption
            or len(audio) <= 0
        ):
            return

        audio_copy = np.asarray(audio, dtype=np.float32).copy()
        token = self._caption_transition_token
        duration = len(audio_copy) / max(sample_rate, 1)
        normalized_timing_text = timing_text if timing_text is not None else caption
        timing_words = tuple(normalized_timing_text.split())

        def analyze() -> None:
            try:
                transcriber = (
                    self._caption_transcriber or self._persona_recording_transcriber
                )
                if transcriber is None:
                    if self.celune is None or not persona_enabled(self.celune.config):
                        return
                    model_id = getattr(self, "_persona_speech_model_id", None)
                    language_getter = getattr(self, "_persona_speech_language", None)
                    if not callable(model_id) or not callable(language_getter):
                        return
                    model_id_getter = cast(Callable[[], str], model_id)
                    language_value_getter = cast(
                        Callable[[], Optional[str]], language_getter
                    )
                    transcriber = WhisperTranscriber(
                        model_id_getter(),
                        language=language_value_getter(),
                        progress_callback=self.safe_progress,
                    )
                    self._caption_transcriber = transcriber
                segments = transcriber.transcribe_segments(audio_copy, sample_rate)
                word_timings = self._caption_word_timing_ranges(
                    self._caption_words,
                    segments,
                    duration,
                    timing_words,
                )
            except Exception as error:
                self.safe_log(
                    format_error_message(
                        string("ui.caption_transcription_failed"),
                        error,
                        getattr(self.celune, "log_level", self._startup_log_level),
                    ),
                    "warning",
                )
                return
            if not word_timings:
                return

            def update() -> None:
                if (
                    token != self._caption_transition_token
                    or not self._caption_active
                    or caption != self._caption_text
                ):
                    return
                self._caption_word_timings = word_timings
                self._caption_audio_duration = duration
                visible_sentence, visible_words = self._caption_words_for_progress(
                    self._caption_progress
                )
                rendered_text = " ".join(visible_sentence)
                self._caption_visible_words = visible_words
                self._caption_rendered_text = rendered_text
                if self.caption is not None:
                    self.caption.update(rendered_text)

            with contextlib.suppress(LookupError, RuntimeError, ScreenStackError):
                self._run_on_ui_thread(update)

        threading.Thread(target=analyze, daemon=True).start()

    def _caption_words_for_progress(
        self,
        fraction: float,
    ) -> tuple[tuple[str, ...], int]:
        """Return the current sentence words and total revealed word count."""
        if (
            self._caption_word_timings
            and len(self._caption_word_timings) == len(self._caption_words)
            and self._caption_audio_duration > 0.0
        ):
            elapsed = fraction * self._caption_audio_duration
            revealed_words = sum(
                elapsed >= start for start, _end in self._caption_word_timings
            )
            revealed_words = max(self._caption_visible_words, revealed_words)
            remaining_words = revealed_words
            for sentence in self._caption_sentences:
                if remaining_words <= len(sentence):
                    return sentence[:remaining_words], revealed_words
                remaining_words -= len(sentence)
            return (
                self._caption_sentences[-1] if self._caption_sentences else (),
                revealed_words,
            )

        visible_words = min(
            len(self._caption_words),
            math.ceil(fraction * len(self._caption_words)),
        )
        visible_words = max(self._caption_visible_words, visible_words)
        remaining_words = visible_words
        visible_sentence: tuple[str, ...] = ()
        for sentence in self._caption_sentences:
            if remaining_words <= len(sentence):
                visible_sentence = sentence[:remaining_words]
                break
            remaining_words -= len(sentence)
        return visible_sentence, visible_words

    def safe_progress(
        self, progress: Optional[float], total: Optional[float] = None
    ) -> None:
        """Update current progress.

        Args:
            progress: Current progress, or ``None`` for an indeterminate bar.
            total: Total progress, or ``None`` for an indeterminate bar.
        """
        if self.cur_state == "exiting":
            return

        if self.progress_bar is None:
            return

        def update() -> None:
            celune = self.celune
            idle_after_startup = (
                progress is None
                and total is None
                and self.celune_ready
                and self.cur_state not in {"error", "exiting"}
                and celune is not None
                and getattr(celune, "cur_state", None) == "idle"
                and not getattr(celune, "persona_loading", False)
            )
            resolved_progress = 1 if idle_after_startup else progress
            resolved_total = 1 if idle_after_startup else total
            self.progress_bar.update(
                total=resolved_total,
                progress=0 if resolved_progress is None else resolved_progress,
            )
            if self.progress_label is not None:
                audio_playing = (
                    celune is not None
                    and getattr(celune, "cur_state", None) == "speaking"
                )
                self.progress_label.set_progress(
                    resolved_progress,
                    resolved_total,
                    audio_playing=audio_playing,
                    sample_rate=BASE_SR,
                )
                if self._caption_active:
                    self.progress_label.display = False
            if not self._caption_active and not self._caption_transitioning:
                self._restore_progress_bar()

        self._run_on_ui_thread(update)

    def _set_progress_row_display(self, visible: bool) -> None:
        """Show or hide the progress row that contains the bar and readout."""
        if self.progress_bar is None:
            return
        progress_row = getattr(self.progress_bar, "parent", None)
        if progress_row is not None:
            progress_row.display = visible

    def _restore_progress_bar(self, *, force: bool = False) -> None:
        """Restore the progress row and bar after a transient caption state."""
        if (
            self.progress_bar is None
            or self._caption_active
            or (self._caption_transitioning and not force)
        ):
            return
        self._set_progress_row_display(True)
        self.progress_bar.display = True
        self.progress_bar.styles.opacity = 1.0

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
        widget: Widget,
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
            with contextlib.suppress(LookupError, RuntimeError, ScreenStackError):
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

    @staticmethod
    def _with_brightness(color: Color, brightness: float) -> Color:
        """Return ``color`` blended toward the requested brightness."""
        brightness = max(0.0, min(1.0, brightness))
        current = color.brightness

        if abs(current - brightness) < 0.01:
            return color

        destination = Color(255, 255, 255) if current < brightness else Color(0, 0, 0)
        destination_brightness = destination.brightness
        factor = (brightness - current) / (destination_brightness - current)
        return color.blend(destination, max(0.0, min(1.0, factor)))

    @staticmethod
    def _with_darkened_brightness(color: Color) -> Color:
        """Return ``color`` with a visibly darker brightness."""
        target_brightness = max(0.0, color.brightness * 0.6)
        return CeluneUI.with_brightness(color, target_brightness)

    def pulse_border(self, target: Union[str, Widget]) -> None:
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
        original_border: tuple[EdgeStyle, ...] = tuple(widget.styles.border)

        if not any(edge_type for edge_type, _ in original_border):
            return

        widget_key = id(widget)
        token = self._border_pulse_tokens.get(widget_key, 0) + 1
        self._border_pulse_tokens[widget_key] = token
        self._border_pulse_widgets[widget_key] = widget

        target_border: tuple[EdgeStyle, ...] = tuple(
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

        def set_border(border: tuple[EdgeStyle, ...]) -> None:
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
                        start_color.blend(end_color, eased)
                        if edge_type
                        else start_color,
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
                        hold_duration + frame_delay
                        if index + 1 == steps
                        else frame_delay
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
            if isinstance(self.style_button, VoiceButton):
                self.style_button.hold_enabled = can_open_menu
            self.update_resources()

        self._run_on_ui_thread(update)

    def _normal_input_placeholder(self) -> str:
        """Return the unlocked input placeholder without blocking the UI."""
        if self._is_voice_conversion_mode():
            return string("ui.voice_changer_placeholder")

        if (
            self._persona_loaded()
            and self._persona_available
            and persona_enabled(self.celune.config)
            and persona_talkback_enabled(self.celune.config)
        ):
            return string("ui.say_placeholder")

        return string("ui.input_placeholder")

    def _persona_loaded(self) -> bool:
        """Return whether the attached Celune instance currently has Persona."""
        persona_ready = getattr(self.celune, "persona_ready", None)
        if persona_ready is None:
            return bool(getattr(self.celune, "vision", None))
        return bool(persona_ready)

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
                string("ui.stopped_placeholder")
                if stopped
                else string("ui.wait_placeholder")
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
            and msg != string("status.stopped")
        ):
            return

        if not _RUNTIME_DEPENDENCIES_LOADED:
            _load_ui_runtime_dependencies()

        if severity not in colors.SEVERITY_COLORS["celune"]:
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
        loglevel: LogLevel = "info",
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
        active_log_level = getattr(
            self.celune,
            "log_level",
            self._startup_log_level,
        )
        if levels.get(active_log_level, 0) < levels.get(loglevel, 0):
            return

        if not _RUNTIME_DEPENDENCIES_LOADED:
            _load_ui_runtime_dependencies()

        if severity not in colors.SEVERITY_COLORS["celune"]:
            severity = "info"

        with self._log_history_lock:
            self.log_history.append((msg, severity))
        self._persist_log_entry(msg, severity)
        if loglevel == "info" and self._loading_screen is not None:
            self._run_on_ui_thread(lambda: self._update_loading_log(msg))
        if self.logs is None:
            return

        entry = Text(msg, style=self._severity_color(severity))

        if threading.current_thread() is threading.main_thread():
            self.logs.write(entry)
            self._rendered_log_count += 1
        else:
            self.post_message(UILogMessage(msg, severity))

    def on_uilog_message(self, message: UILogMessage) -> None:
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
            self.celune is not None
            and getattr(self.celune, "vc_backend", None) is not None
        )

    @staticmethod
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
        self.vc_mode_button.label = string(
            "ui.vc_mode_sing" if f0_condition else "ui.vc_mode_talk"
        )
        self.vc_pitch_button.label = string(
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
        if backend is not None and hasattr(backend, "f0_condition"):
            backend.f0_condition = enabled
        self.refresh_vc_controls()

        if announce:
            self.safe_log(
                string(
                    "ui.vc_mode_changed",
                    mode=string("ui.vc_mode_sing" if enabled else "ui.vc_mode_talk"),
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

        clamped = clamp_vc_pitch_shift(value)
        self.celune.vc_pitch_shift = clamped
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and hasattr(backend, "pitch_shift"):
            backend.pitch_shift = clamped
        self.refresh_vc_controls()

        if announce:
            self.safe_log(
                string(
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
        components: tuple[ComponentLockName, ...],
    ) -> tuple[bool, Optional[ComponentLockLease]]:
        """Reserve the resources required by one microphone operation."""
        if self.celune is None:
            return False, None
        manager = getattr(self.celune, "component_locks", None)
        if manager is None:
            return True, None

        owner = ComponentLockOwner(operation_id=operation_id)
        acquisition, lease = manager.try_acquire_lease(
            tuple(ComponentLockRequirement(component) for component in components),
            owner,
        )
        if lease is not None:
            return True, lease

        busy = acquisition.busy
        if busy is not None:
            self.celune._last_component_busy = busy
            labels = ", ".join(component.name for component in busy.components)
            self.safe_log(
                string("pipeline.busy_components", components=labels),
                "warning",
            )
        return False, None

    def _persona_speech_model_id(self) -> str:
        """Return the configured Hugging Face Whisper model ID."""
        configured = persona_config(self.celune.config).get("speech_model_id")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return DEFAULT_PERSONA_SPEECH_MODEL_ID

    def _persona_speech_language(self) -> Optional[str]:
        """Return a configured Whisper language, or ``None`` for auto-detection."""
        configured = persona_config(self.celune.config).get("speech_language")
        if not isinstance(configured, str) or configured.strip().lower() in {
            "",
            "auto",
        }:
            return None
        return configured.strip()

    def _persona_speech_end_delay_seconds(self) -> float:
        """Return the extra VAD silence delay before Persona submission."""
        configured = persona_config(self.celune.config).get("speech_end_delay_seconds")
        if (
            isinstance(configured, (int, float))
            and not isinstance(configured, bool)
            and configured >= 0
        ):
            return float(configured)
        return PERSONA_SPEECH_END_DELAY_SECONDS

    def _persona_recording_audio_locked(self) -> npt.NDArray[np.float32]:
        """Return captured Persona audio while holding its lock."""
        if not self._persona_recording_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._persona_recording_chunks, axis=0).astype(
            np.float32,
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
                format_error_message(
                    string("ui.persona_transcription_failed"),
                    error,
                    getattr(self.celune, "log_level", "info"),
                ),
                "error",
            )
        if error is not None or error_already_reported:
            self.safe_status(string("ui.idle_status"))
            if self.style_button is not None:
                self.style_button.disabled = self._input_locked
            self.update_resources()
            return

        text = f"{prefix} {transcript}".strip() if prefix else transcript.strip()
        if text:
            self._set_persona_recording_text(transcript)
            self._submit_text(text, process_commands=False)
        else:
            self.safe_log(string("ui.recording_empty"), "warning")
        self.safe_status(string("ui.idle_status"))
        if self.style_button is not None:
            self.style_button.disabled = self._input_locked
        self.update_resources()

    def _persona_transcription_worker(
        self,
        recording_queue: queue_module.Queue[tuple[AudioChunk, bool]],
        transcriber: WhisperTranscriber,
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
                    format_error_message(
                        string("ui.persona_transcription_failed"),
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

        self.safe_status(string("ui.persona_transcribing"))
        return True

    def _start_persona_recording(self) -> bool:
        """Start push-to-talk microphone capture for the active Persona."""
        _load_ui_runtime_dependencies()
        if (
            self.celune is None
            or self._is_voice_conversion_mode()
            or not self._persona_loaded()
            or not persona_talkback_enabled(self.celune.config)
        ):
            return False
        if self._persona_recording_active():
            return True

        input_config = getattr(self.celune, "config", None)
        input_device_key = (
            "input_recording_device"
            if isinstance(input_config, dict)
            and "input_recording_device" in input_config
            else "input_device"
        )
        try:
            input_device, direct_device_info = resolve_audio_device_with_info(
                input_config,
                input_device_key,
                "input",
            )
            device_info = (
                cast(dict[str, AudioDeviceInfoValue], dict(direct_device_info))
                if direct_device_info is not None
                else cast(
                    dict[str, AudioDeviceInfoValue],
                    sd.query_devices(device=input_device, kind="input"),
                )
            )
        except Exception as exc:
            self.safe_log(
                format_error_message(
                    string("ui.recording_open_input_failed"),
                    exc,
                    getattr(self.celune, "log_level", "info"),
                ),
                "error",
            )
            return False

        channels = _device_scalar_int(device_info.get("max_input_channels"), 0)
        if channels <= 0:
            self.safe_log(string("pipeline.no_audio_device"), "warning")
            return False

        sample_rate = _device_scalar_int(
            device_info.get("default_samplerate"),
            48000,
        )
        channel_count = 2 if channels >= 2 else 1
        vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate) + int(
            sample_rate * self._persona_speech_end_delay_seconds()
        )
        ai_vad = create_live_voice_activity_detector()
        recording_queue: queue_module.Queue[tuple[AudioChunk, bool]] = (
            queue_module.Queue(maxsize=1)
        )
        transcriber = WhisperTranscriber(
            self._persona_speech_model_id(),
            language=self._persona_speech_language(),
            progress_callback=self.safe_progress,
        )
        prefix = self.input_box.text.strip() if self.input_box is not None else ""
        recording_started_at = time.monotonic()
        should_stop = False

        def callback(
            indata: npt.NDArray[np.float32],
            frames: int,
            time_info: Optional[tuple[float, float, float]],
            status: Optional[sd.CallbackFlags],
        ) -> None:
            from ..utils import discard

            discard(frames)
            discard(time_info)
            discard(status)
            nonlocal should_stop

            callback_audio = np.asarray(indata, dtype=np.float32).copy()
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
                    if (
                        time.monotonic() - self._persona_recording_last_partial_at
                        >= 0.8
                    ):
                        self._queue_persona_recording_item_locked(final_value=False)
                        self._persona_recording_last_partial_at = time.monotonic()

                if (
                    self._persona_recording_speech_started
                    and self._persona_recording_silence_frames >= vad_hangover_frames
                ) or (
                    not self._persona_recording_speech_started
                    and time.monotonic() - recording_started_at
                    >= PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS
                ):
                    self._persona_recording_stop_requested = True
                    self._queue_persona_recording_item_locked(final_value=True)
                    should_stop = True

            if should_stop:
                self.safe_status(string("ui.persona_transcribing"))

        worker: Optional[threading.Thread] = None
        stream: Optional[sd.InputStream] = None
        acquired, component_lease = self._acquire_recording_component_lease(
            f"persona-recording:{uuid4()}",
            (ComponentLockName.MICROPHONE, ComponentLockName.ASR),
        )
        if not acquired:
            return False
        try:
            stream = sd.InputStream(
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
                format_error_message(
                    string(
                        "ui.recording_start_failed",
                        label=string("ui.audio_input_label"),
                    ),
                    exc,
                    getattr(self.celune, "log_level", "info"),
                ),
                "error",
            )
            return False

        self.safe_log(string("ui.persona_recording_started"), "info")
        self.safe_status(string("ui.persona_recording_listening"))
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
                    recording_queue.put_nowait((np.zeros(0, dtype=np.float32), True))
        self._shutdown_vc_stream(stream)
        self._close_live_vad(vad)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        if component_lease is not None:
            component_lease.release()

    def _vc_recording_active(self) -> bool:
        """Return whether live VC recording is active in the TUI."""
        return self._vc_recording_stream is not None

    @staticmethod
    def _vc_input_rms(audio: npt.NDArray[np.float32]) -> float:
        """Return RMS energy for one microphone callback buffer."""
        return vc_input_rms(audio)

    @staticmethod
    def _vc_feedback_rise_detected(
        previous_rms: float,
        current_rms: float,
    ) -> bool:
        """Return whether the latest RMS jump looks like runaway feedback."""
        if previous_rms < _VC_FEEDBACK_RMS_MIN_PREVIOUS:
            return False
        if current_rms < _VC_FEEDBACK_RMS_MIN_CURRENT:
            return False
        if current_rms < previous_rms * _VC_FEEDBACK_RMS_RISE_RATIO:
            return False
        return (current_rms - previous_rms) >= _VC_FEEDBACK_RMS_RISE_DELTA

    @staticmethod
    def _vc_feedback_min_capture_frames(sample_rate: int) -> int:
        """Return the minimum capture length before feedback auto-stop is allowed."""
        return max(1, int(sample_rate * _VC_FEEDBACK_MIN_CAPTURE_SECONDS))

    def _request_vc_recording_feedback_stop(self) -> None:
        """Request a feedback-triggered recording stop on a dedicated thread."""
        stop_thread = threading.Thread(
            target=self._stop_vc_recording_for_feedback,
            daemon=True,
        )
        self._vc_recording_stop_thread = stop_thread
        stop_thread.start()

    def _flush_vc_recording_buffer_locked(self) -> Optional[AudioChunk]:
        """Return and clear the buffered microphone chunk accumulator."""
        if not self._vc_recording_chunks:
            return None
        audio = np.concatenate(self._vc_recording_chunks, axis=0)
        self._vc_recording_chunks = []
        self._vc_recording_buffered_frames = 0
        return audio

    def _flush_vc_recording_chunk_locked(
        self,
        keep_tail_frames: int = 0,
    ) -> Optional[AudioChunk]:
        """Return one buffered VC chunk while optionally retaining a tail overlap."""
        audio = self._flush_vc_recording_buffer_locked()
        if audio is None or keep_tail_frames <= 0:
            return audio

        if len(audio) <= keep_tail_frames:
            self._vc_recording_chunks = [audio]
            self._vc_recording_buffered_frames = len(audio)
            return None

        retained = np.asarray(audio[-keep_tail_frames:], dtype=np.float32).copy()
        flushed = np.asarray(audio[:-keep_tail_frames], dtype=np.float32).copy()
        self._vc_recording_chunks = [retained]
        self._vc_recording_buffered_frames = len(retained)
        return flushed

    @staticmethod
    def _vc_vad_hangover_frames(sample_rate: int) -> int:
        """Return how many trailing silent frames to tolerate before flushing."""
        return vc_vad_hangover_frames(sample_rate)

    @staticmethod
    def _vc_vad_preroll_frames(sample_rate: int) -> int:
        """Return how much recent pre-speech audio to keep before VAD triggers."""
        return vc_vad_preroll_frames(sample_rate)

    @staticmethod
    def _vc_live_chunk_frames(sample_rate: int) -> int:
        """Return how much active speech to collect before a live VC flush."""
        return vc_live_chunk_frames(sample_rate)

    @staticmethod
    def _vc_live_chunk_overlap_frames(sample_rate: int) -> int:
        """Return how much tail audio to retain between live VC chunks."""
        return vc_live_chunk_overlap_frames(sample_rate)

    def _append_vc_preroll_audio_locked(
        self,
        audio: npt.NDArray[np.float32],
        max_frames: int,
    ) -> None:
        """Retain only the newest pre-speech audio frames for VAD onset recovery."""
        copied = np.asarray(audio, dtype=np.float32).copy()
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
            trimmed = np.asarray(oldest[overflow_frames:], dtype=np.float32).copy()
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

    @staticmethod
    def _vc_input_has_voice(audio: npt.NDArray[np.float32]) -> bool:
        """Return whether one microphone callback likely contains voice activity."""
        _load_ui_runtime_dependencies()
        return vc_input_has_voice(audio)

    @staticmethod
    def _normalize_vc_overlap_audio(
        audio: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Normalize one VC overlap chunk into valid mono or stereo time-first audio."""
        _load_ui_runtime_dependencies()
        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.ndim == 1:
            return normalized
        if normalized.ndim != 2:
            raise ValueError(
                f"expected 1D or 2D VC overlap audio, got {normalized.shape}"
            )
        if normalized.shape[1] == 1:
            return normalized[:, 0]
        if normalized.shape[1] == 2:
            return normalized
        raise ValueError(
            f"expected mono or stereo VC overlap audio, got {normalized.shape}"
        )

    def _crossfade_vc_overlap(
        self,
        previous_tail: npt.NDArray[np.float32],
        current_head: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Crossfade two same-rate VC overlap regions into one seamless bridge."""
        _load_ui_runtime_dependencies()
        overlap_frames = min(len(previous_tail), len(current_head))
        if overlap_frames <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        previous = self._normalize_vc_overlap_audio(previous_tail[-overlap_frames:])
        current = self._normalize_vc_overlap_audio(current_head[:overlap_frames])

        if previous.ndim != current.ndim:
            if previous.ndim == 1:
                previous = np.column_stack((previous, previous))
            if current.ndim == 1:
                current = np.column_stack((current, current))

        fade = np.linspace(0.0, 1.0, overlap_frames, dtype=np.float32)
        if previous.ndim == 2:
            fade = fade[:, None]

        return np.asarray(
            (previous * (1.0 - fade)) + (current * fade),
            dtype=np.float32,
        )

    @staticmethod
    def _enqueue_vc_submission_chunk(
        submission_queue: queue_module.Queue[
            Optional[tuple[AudioChunk, int, str, bool]]
        ],
        item: tuple[AudioChunk, int, str, bool],
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

    @staticmethod
    def _finish_vc_submission_queue(
        submission_queue: Optional[
            queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]
        ],
        final_item: Optional[tuple[AudioChunk, int, str, bool]] = None,
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
        self._vc_recording_label = string("ui.audio_input_label")
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
        Optional[sd.InputStream],
        Optional[AudioChunk],
        int,
        str,
        Optional[queue_module.Queue[Optional[tuple[AudioChunk, int, str, bool]]]],
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

    @staticmethod
    def _shutdown_vc_stream(stream: Optional[sd.InputStream]) -> None:
        """Stop and close one VC input stream outside the recording lock."""
        if stream is None:
            return

        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    @staticmethod
    def _close_live_vad(vad: Optional[LiveVoiceActivityDetector]) -> None:
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

    @staticmethod
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
            self.safe_log(string("ui.recording_stopped", label=label), "info")
        self._set_terminal_status("ready", string("osc.action_idle"))
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
                    np.asarray(buffered_audio, dtype=np.float32),
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

        self.safe_log(string("ui.recording_stopped_feedback", label=label), "warning")
        self._set_terminal_status("ready", string("osc.action_idle"))
        self.update_resources()

    def _start_vc_recording(self) -> bool:
        """Start recording from the active system input device for VC."""
        _load_ui_runtime_dependencies()
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
            if isinstance(input_config, dict)
            and "input_recording_device" in input_config
            else "input_device"
        )
        try:
            input_device, direct_device_info = resolve_audio_device_with_info(
                input_config,
                input_device_key,
                "input",
            )
        except ValueError as error:
            self.safe_log(str(error), "warning")
            return False

        try:
            device_info = (
                cast(dict[str, AudioDeviceInfoValue], dict(direct_device_info))
                if direct_device_info is not None
                else cast(
                    dict[str, AudioDeviceInfoValue],
                    sd.query_devices(device=input_device, kind="input"),
                )
            )
        except Exception as e:
            self.safe_log(
                format_error_message(
                    string("ui.recording_open_input_failed"),
                    e,
                    getattr(self.celune, "log_level", "info"),
                ),
                "error",
            )
            return False

        channels = _device_scalar_int(device_info.get("max_input_channels"), 0)
        if channels <= 0:
            self.safe_log(string("pipeline.no_audio_device"), "warning")
            return False

        sample_rate = _device_scalar_int(
            device_info.get("default_samplerate"),
            48000,
        )
        label = format_audio_device_name(device_info) or str(
            device_info.get("name", string("ui.audio_input_label"))
        )
        channel_count = 2 if channels >= 2 else 1
        vad_hangover_frames = self._vc_vad_hangover_frames(sample_rate)
        vad_preroll_frames = self._vc_vad_preroll_frames(sample_rate)
        live_chunk_frames = self._vc_live_chunk_frames(sample_rate)
        ai_vad = create_live_voice_activity_detector()
        submission_queue: queue_module.Queue[
            Optional[tuple[AudioChunk, int, str, bool]]
        ] = queue_module.Queue(maxsize=_VC_LIVE_SUBMISSION_QUEUE_SIZE)

        def submit_live_audio() -> None:
            live_source_id: Optional[int] = None
            live_playback_generation: Optional[int] = None

            def queue_playback_segment(
                audio_chunk: AudioChunk,
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
                    live_source_id = queue_streaming_sfx_audio(
                        self.celune,
                        np.asarray(audio_chunk, dtype=np.float32),
                        sr,
                        audio_label,
                        source_id=live_source_id,
                        generation=live_playback_generation,
                        status_label_key="pipeline.revoicing_label",
                        reset_ready_announcement=live_source_id is None,
                    )
                    if live_source_id is None:
                        live_playback_generation = None
                except Exception as error:
                    self.safe_log(
                        format_error_message(
                            string("ui.recording_stream_submit_failed"),
                            error,
                            getattr(
                                self.celune,
                                "log_level",
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
                finish_streaming_sfx_audio(self.celune, live_source_id)
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
                        Callable[..., Optional[AudioOutput]],
                        converter,
                    )(
                        audio,
                        queued_sample_rate,
                        label=queued_label,
                    )
                    if converted is None:
                        self.safe_log(
                            string("ui.recording_stream_submit_failed"),
                            "warning",
                        )
                        continue

                    converted_audio = np.asarray(converted.audio, dtype=np.float32)
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
                        format_error_message(
                            string(
                                "ui.recording_stream_chunk_failed",
                                label=queued_label,
                            ),
                            exc,
                            getattr(self.celune, "log_level", "info"),
                        ),
                        "warning",
                    )
                    if isinstance(exc, CEDTSError):
                        self._cancel_vc_recording(announce=False)

        worker = threading.Thread(target=submit_live_audio, daemon=True)

        def callback(
            indata: npt.NDArray[np.float32],
            frames: int,
            time_info: Optional[tuple[float, float, float]],
            status: Optional[sd.CallbackFlags],
        ) -> None:
            from ..utils import discard

            discard(frames)
            discard(time_info)
            discard(status)

            callback_audio = np.asarray(indata, dtype=np.float32).copy()
            current_rms = self._vc_input_rms(callback_audio)
            should_stop_for_feedback = False
            feedback_min_capture_frames = self._vc_feedback_min_capture_frames(
                sample_rate
            )
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
                    >= _VC_FEEDBACK_REQUIRED_CONSECUTIVE_SPIKES
                ):
                    self._vc_recording_feedback_detected = True
                    should_stop_for_feedback = True

                if should_stop_for_feedback:
                    self._request_vc_recording_feedback_stop()
                    return

                live_audio: Optional[AudioChunk] = None
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
                    live_audio = np.zeros_like(callback_audio)
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
                    live_audio = np.zeros_like(callback_audio)
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

        stream: Optional[sd.InputStream] = None
        acquired, component_lease = self._acquire_recording_component_lease(
            f"vc-recording:{uuid4()}",
            (ComponentLockName.MICROPHONE,),
        )
        if not acquired:
            return False
        try:
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channel_count,
                dtype="float32",
                callback=callback,
                device=input_device,
                blocksize=live_chunk_frames,
            )

            # Publish the recording state before starting the stream. PortAudio
            # may invoke the callback during ``start()``, and that first buffer
            # must not be discarded as an inactive recording.
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
                format_error_message(
                    string("ui.recording_start_failed", label=label),
                    e,
                    getattr(self.celune, "log_level", "info"),
                ),
                "error",
            )
            return False

        worker.start()
        self.safe_log(string("ui.recording_started", label=label), "info")
        self._set_terminal_status(
            "recording",
            string("osc.action_listening_microphone"),
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
                        np.asarray(buffered_audio, dtype=np.float32),
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

            self.safe_log(string("ui.recording_stopped", label=label), "info")
            self._set_terminal_status("ready", string("osc.action_idle"))
            self.update_resources()
            if captured_frames <= 0:
                self.safe_log(string("ui.recording_empty"), "warning")
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
        loglevel: LogLevel = "info",
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
        process_ui_command(self, command, args)

    def open_settings_menu(self) -> None:
        """Open the configuration manager and prepare a restart on save."""
        if self.celune is None or self._active_menu is not None:
            return

        options: list[SelectMenuOption] = []
        paths: list[tuple[str, ...]] = []
        for path, value in self._iter_config_values(self.celune.config):
            label = self._config_label(path)
            options.append(
                SelectMenuOption(
                    label=label,
                    value=value,
                    autocomplete_values=self._config_autocomplete(path, value),
                    explanation=self._config_explanation(path),
                )
            )
            paths.append(path)

        if not options:
            self.safe_log(string("ui.settings_empty"), "warning")
            return

        self._settings_paths = tuple(paths)
        self._show_menu(
            SelectMenuWidget(
                string("ui.settings_title"),
                options,
                value_display="all",
                return_value=False,
                footer_builder=lambda option: self._menu_footer(
                    option,
                    option_count=len(options),
                    confirm_hint=string("ui.settings_confirm_hint"),
                    include_search=False,
                ),
            ),
            "settings",
        )

    @staticmethod
    def _iter_config_values(
        config: dict[str, JSONSerializable],
        prefix: tuple[str, ...] = (),
    ) -> Iterator[tuple[tuple[str, ...], JSONSerializable]]:
        """Yield editable leaf values from the nested configuration mapping."""
        for key, value in config.items():
            path = (*prefix, key)
            if isinstance(value, dict) and value:
                yield from CeluneUI._iter_config_values(value, path)
            else:
                yield path, value

    @staticmethod
    def _config_autocomplete(
        path: tuple[str, ...], value: JSONSerializable
    ) -> Optional[tuple[JSONSerializable, ...]]:
        """Return useful autocomplete candidates for one configuration value."""
        if value is None:
            return (None,)
        if isinstance(value, bool):
            return (True, False)

        candidates: dict[tuple[str, ...], tuple[JSONSerializable, ...]] = {
            ("backend",): (None, "mini", "qwen3", "dots.tts", "voxcpm2", "seed-vc"),
            ("gpt_sovits_variant",): (
                "auto",
                "v1",
                "v2",
                "v2Pro",
                "v2ProPlus",
                "v3",
                "v4",
            ),
            ("log_level",): ("info", "verbose", "debug"),
            ("mode",): ("speak", "converse", "agent"),
            ("theme",): ("dark", "light"),
            ("vram",): ("low", "medium", "high", "xhigh"),
            ("audio_api",): (None, "wasapi", "directsound"),
            ("persona", "speech_language"): ("auto",),
        }
        return candidates.get(path)

    @staticmethod
    def _config_label(path: tuple[str, ...]) -> str:
        """Convert a dotted configuration path into a readable setting label."""
        label = " ".join(path).replace("_", " ")
        for source, replacement in (
            ("gpt sovits", "GPT-SoVITS"),
            ("t2s", "T2S"),
        ):
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
            "gpt-sovits": "GPT-SoVITS",
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

    @staticmethod
    def _config_explanation(path: tuple[str, ...]) -> str:
        """Return a localized explanation for one configuration value."""
        aliases: dict[tuple[str, ...], str] = {
            ("gpt_sovits_t2s_weights_path",): "gpt_weights",
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
        explanation = string(explanation_key)
        if explanation != explanation_key:
            return explanation
        return string(
            "ui.settings_explanation_generic",
            setting=CeluneUI._config_label(path),
        )

    @staticmethod
    def _menu_footer(
        option: SelectMenuOption,
        *,
        option_count: int,
        confirm_hint: str,
        include_search: bool,
    ) -> str:
        """Build localized hints for the selected menu row."""
        hints: list[str] = []
        if option_count > 1:
            hints.append(string("ui.menu_hint_select"))
        if (
            option.editable
            and option.autocomplete_values is not None
            and len(option.autocomplete_values) > 1
        ):
            hints.append(string("ui.menu_hint_choose"))
            if include_search:
                hints.append(string("ui.menu_hint_search"))
        hints.extend((confirm_hint, string("ui.menu_hint_cancel")))
        return string("ui.menu_hint_separator").join(hints)

    def open_voice_menu(self) -> None:
        """Open a voice menu containing every available CEVOICE/CECHAR entry."""
        if self.celune is None or self._active_menu is not None:
            return

        from ..cevoice import (
            CEVoice,
            CEVoiceError,
            active_bundle_path,
            bundle_character_name,
            bundled_voices_dir,
        )

        options: list[SelectMenuOption] = []
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

            pack_name = bundle_character_name(bundle) or path.stem
            if pack_name in self._voice_menu_paths:
                pack_name = f"{pack_name} ({path.stem})"
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
                SelectMenuOption(
                    label=pack_name,
                    value=selected_voice,
                    editable=len(voices) > 1,
                    autocomplete_values=voices if len(voices) > 1 else None,
                    confirm_value=voices[0] if len(voices) == 1 else None,
                )
            )

        if not options:
            self.safe_log(string("ui.no_voices_loaded"), "warning")
            return

        for index, option in enumerate(options):
            if (
                self._voice_menu_paths.get(option.label) == active_bundle
                and option.value == active_voice
            ):
                options.insert(0, options.pop(index))
                break

        self._show_menu(
            SelectMenuWidget(
                string("ui.voice_menu_title"),
                options,
                value_display="all",
                footer_builder=lambda option: self._menu_footer(
                    option,
                    option_count=len(options),
                    confirm_hint=string("ui.voice_confirm_hint"),
                    include_search=True,
                ),
            ),
            "voice",
        )

    def _show_menu(self, menu: SelectMenuWidget, menu_kind: str) -> None:
        """Mount and focus one application menu overlay."""
        overlay = SelectMenuOverlay(menu)
        self._active_menu = menu
        self._active_menu_overlay = overlay
        self._active_menu_kind = menu_kind
        self.push_screen(overlay)

    def on_voice_button_long_pressed(self, event: VoiceButton.LongPressed) -> None:
        """Open voice selection after the held voice button is released."""
        if event.button is self.style_button:
            self.open_voice_menu()

    def on_select_menu_widget_confirmed(
        self, event: SelectMenuWidget.Confirmed
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
            self._apply_voice_selection(
                {"pack": event.option.label, "entry": event.value}
            )

    def on_select_menu_widget_cancelled(
        self, event: SelectMenuWidget.Cancelled
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

    def _save_settings(self, menu: SelectMenuWidget) -> None:
        """Persist the edited configuration and request a launcher restart."""
        if self.celune is None or len(self._settings_paths) != len(menu.options):
            return

        updated = deepcopy(self.celune.config)
        for path, option in zip(self._settings_paths, menu.options):
            self._set_config_value(updated, path, option.value)

        try:
            self.celune.config = updated
            with config_path(create_parent=True).open("w", encoding="utf-8") as file:
                yaml.safe_dump(updated, file, sort_keys=False)
        except OSError as error:
            self.safe_log(
                format_error_message(
                    string("ui.settings_save_failed"),
                    error,
                    getattr(self.celune, "log_level", self._startup_log_level),
                ),
                "error",
            )
            return

        self.cur_state = "restarting"
        self._run_shutdown_step(
            lambda: self._set_terminal_status(
                "restarting",
                string("osc.action_restarting"),
            )
        )
        self._graceful_exit(return_code=ExitCodes.EXIT_PENDING_RESTART.value)

    @staticmethod
    def _set_config_value(
        config: dict[str, JSONSerializable],
        path: tuple[str, ...],
        value: JSONSerializable,
    ) -> None:
        """Replace one flattened configuration value in its nested mapping."""
        target = config
        for key in path[:-1]:
            child = target.get(key)
            if not isinstance(child, dict):
                return
            target = child
        target[path[-1]] = value

    @work(exclusive=True)
    async def _apply_voice_selection(self, value: dict[str, JSONSerializable]) -> None:
        """Load the selected pack and voice without blocking the Textual loop."""
        if self.celune is None:
            return
        from ..cevoice import active_bundle_path

        pack = value.get("pack")
        entry = value.get("entry")
        bundle_path = (
            self._voice_menu_paths.get(pack) if isinstance(pack, str) else None
        )
        if bundle_path is None or not isinstance(entry, str):
            return

        try:
            if (
                getattr(self.celune, "sleeping", False)
                and not await self.celune.wake_from_sleep_async()
            ):
                self.safe_log(string("ui.voice_change_failed"), "warning")
                return

            if active_bundle_path() == bundle_path:
                loaded = await asyncio.to_thread(
                    self.celune.set_voice_and_wait,
                    entry,
                )
            else:
                loaded = await asyncio.to_thread(
                    self.celune.set_cevoice_and_wait,
                    bundle_path,
                )
                if loaded:
                    loaded = await asyncio.to_thread(
                        self.celune.set_voice_and_wait,
                        entry,
                    )
            if not loaded:
                self.safe_log(string("ui.voice_change_failed"), "warning")
                return
            self.celune_styles = self.celune.voices
            self.style_index = self.celune_styles.index(entry)
            self.tts_voice_changed(entry)
            self.change_voice_lock_state(locked=len(self.celune_styles) < 2)
        except Exception as error:
            self.safe_log(
                format_error_message(
                    string("ui.voice_change_failed_error"),
                    error,
                    getattr(self.celune, "log_level", self._startup_log_level),
                ),
                "error",
            )

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
            ipa_decoded, unmatched = replace_ipa(to_say, strict=True)
            if unmatched > 0:
                self.safe_log(
                    string("commands.unmatched_ipa", count=unmatched),
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
            self.safe_status(string("ui.test_mode_active"))
            return True

        if self._is_agent_test_mode():
            self._suppress_input_change = True
            try:
                self.input_box.load_text("")
            finally:
                self._suppress_input_change = False
            self.safe_status(string("ui.agent_test_mode_active"))
            return True

        if celune.cur_state == "waking":
            self._cancel_sleep_timer()
            self.safe_status(string("status.waking_up"))
            self.change_input_state(locked=True)
            return True

        if celune.sleeping:
            self._cancel_sleep_timer()
            self.safe_status(string("status.waking_up"))
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
                    string("ui.command_parsing_error"),
                    "error",
                )
                return False

            if not parts:
                return False

            command = parts[0].lower()
            command_args = parts[1:]
            self.process_command(command, command_args)
            return True

        if persona_talkback_enabled(celune.config):
            handled = celune.think(text)
        else:
            if celune.config.get("ipa") is False:
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
        self.style_button.disabled = True
        self.input_box.placeholder = string("ui.wait_placeholder")
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
                        format_error_message(
                            string("ui.tutorial_stop_failed"),
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

    @work(group="tutorial", exclusive=True)
    async def type_and_send(
        self,
        text: str,
        process_commands: bool = False,
        cancellable: bool = True,
    ) -> None:
        """Type text into the input box using Celune's typing animation and submit it.

        Args:
            text: The text to type into the input box.
            process_commands: Whether typed slash commands should be executed.
            cancellable: Whether tutorial cancellation should stop this typing.
        """
        token = self._tutorial_token
        typed = ""

        def replace_input(value: str) -> None:
            self._suppress_input_change = True
            try:
                self.input_box.load_text(value)
            finally:
                self._suppress_input_change = False

        replace_input("")

        for char in text:
            if cancellable and token != self._tutorial_token:
                return
            if self.cur_state == "exiting":
                return

            await asyncio.sleep(typing_delay(char))

            if cancellable and token != self._tutorial_token:
                return
            if self.cur_state == "exiting":
                return
            typed += char
            replace_input(typed)

        final_char = text[-1] if text else " "
        await asyncio.sleep(typing_delay(final_char))

        if self.cur_state != "exiting" and (
            not cancellable or token == self._tutorial_token
        ):
            self._submit_text(typed, process_commands)

    async def action_quit(self) -> None:
        """Exit through the startup-aware graceful shutdown path."""
        self._graceful_exit(return_code=self._startup_error_exit_code)

    def on_key(self, event: events.Key) -> None:
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
                with open(config_path(create_parent=True), "w", encoding="utf-8") as f:
                    yaml.dump(self.celune.config, f)
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
                    self.safe_status(string("status.waking_up"))
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
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
                if next_value > VC_PITCH_SHIFT_MAX:
                    next_value = VC_PITCH_SHIFT_MIN
                self.set_vc_pitch_shift(next_value)
            return

        if event.button != self.style_button:
            return

        if len(celune.voices) == 0 or not self.celune_styles:
            self.safe_log(string("ui.no_voices_loaded"), "warning")
            self.change_voice_lock_state(locked=True)
            return

        if not self.celune_ready and not celune.backend.is_fake:
            self.safe_log(string("ui.core_engine_not_loaded"), "warning")
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
                string("osc.action_restarting" if restarting else "osc.action_exiting"),
            )
        )
        if ui_resources is not None:
            self._run_shutdown_step(ui_resources.stop_gpu_usage_worker)
        self._run_shutdown_step(self._shutdown_runtime)

        if self._runtime_log_capture_enabled:
            self._run_shutdown_step(self._disable_runtime_log_capture)

        CeluneUI._instance = None

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
                    self.input_box.placeholder = string("ui.stopped_placeholder")
                self.safe_status(string("status.stopped"), "sleeping")
            return
        if self.cur_state in {"exiting", "error"} or not self.celune_ready:
            if self.input_box is not None:
                self.input_box.placeholder = string("ui.wait_placeholder")
            self.change_voice_lock_state(locked=True)
            return
        if celune.cur_state in {"reloading", "waking"}:
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            if celune.cur_state == "waking":
                self.safe_status(string("status.waking_up"))
            return
        celune.locked = False
        if celune.sleeping:
            self.safe_status(string("status.sleeping"), "sleeping")
            return
        celune.cur_state = "idle"
        if celune.is_in_tutorial:
            self.input_box.placeholder = string("ui.tutorial_placeholder")
            self.change_voice_lock_state(locked=True)
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(celune.voices) < 2)
        self.safe_status(string("status.idle"))
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
        self.safe_status(string("status.speaking"))
        if celune.is_in_tutorial:
            self.input_box.placeholder = string("ui.tutorial_placeholder")
            self.change_voice_lock_state(locked=True)
        else:
            self.change_input_state(locked=False)
            self.change_voice_lock_state(locked=len(celune.voices) < 2)

    def error(self, error: str) -> None:
        """Set the UI status to the error message.

        Args:
            error: The error text to display.
        """
        if self.cur_state == "exiting":
            return
        self._hide_caption_widgets()
        self.safe_status(error, "error")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
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
            widgets = (screen, *screen.query(Widget))
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
                    duration=_EXIT_FADE_SECONDS,
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

    def _report_shutdown_error(self, error: Exception) -> None:
        """Write a shutdown error to both the log and original terminal stream."""
        log_level = getattr(
            self.celune,
            "log_level",
            self._startup_log_level,
        )
        message = format_error_message(
            string("celune.internal_error"),
            error,
            log_level,
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

    @property
    def tutorial_token(self) -> int:
        """Return the active tutorial cancellation token.

        Returns:
            int: The tutorial token currently used to invalidate pending tutorial work.
        """
        return self._tutorial_token

    @property
    def tutorial_active(self) -> bool:
        """Return whether a tutorial flow is currently active.

        Returns:
            bool: ``True`` when tutorial work is active, otherwise ``False``.
        """
        return self._tutorial_active

    @staticmethod
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

    @staticmethod
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

    register_runtime_error_themes = _register_runtime_error_themes
    wrap_runtime_fatal_glow = _wrap_runtime_fatal_glow
    advance_status_marquee = _advance_status_marquee
    enable_runtime_log_capture = _enable_runtime_log_capture
    install_runtime_log_redirects = _install_runtime_log_redirects
    disable_runtime_log_capture = _disable_runtime_log_capture
    with_brightness = _with_brightness
    normal_input_placeholder = _normal_input_placeholder
    persona_loaded = _persona_loaded
