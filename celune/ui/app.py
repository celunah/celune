# SPDX-License-Identifier: Apache-2.0
"""Frontend layer."""

# pylint: disable=ungrouped-imports

from __future__ import annotations

import sys
import queue as queue_module
import asyncio
import logging
import itertools
import threading
from io import TextIOWrapper
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Never,
    Union,
    TextIO,
    Literal,
    ClassVar,
    Optional,
    Protocol,
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
    RenderableType,
    ScreenStackError,
    AutopilotCallbackType,
)
from textual.color import Color
from textual.theme import Theme
from textual.timer import Timer
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.message import Message
from textual.widgets import (
    Label,
    RichLog,
    TextArea,
    ProgressBar,
)
from textual.widgets import (
    Button as TextualButton,
)
from textual.css.query import NoMatches
from textual.css.types import EdgeStyle
from textual.containers import Vertical, Horizontal

from ..i18n import string, tagged_string
from .theme import CELUNE_CSS, severity_color
from .loading import CeluneLoadingScreen
from ..threads import run_in_daemon_thread
from .terminal import SelectMenuOption, SelectMenuWidget
from ..watchdog import launcher_loss_requested
from ..constants import BASE_SR, SIGTSTP, APP_NAME, ExitCodes
from ..typing.ui import CeluneUIMethodSurface
from ..typing.agent import AgentTaskState
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..typing.common import JSONSerializable
from ..typing.config import AudioDeviceInfoValue
from ..theme.defaults import default_theme_family, default_error_theme_family

if TYPE_CHECKING:
    import sounddevice as sd

    from .terminal import LogRedirect, UILogHandler
    from ..typing.agent import AgentTask
    from ..typing.aliases import LogLevel, AudioChunk, AudioChunks, _VCAudioCallback

    colors: ModuleType

    import yaml
    import numpy as np
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
    from ..paths import config_path, main_window_log_path
    from ..utils import discard, replace_ipa, is_april_fools
    from ..celune import Celune
    from ..config import format_audio_device_name, resolve_audio_device_with_info
    from ..speech import queue_streaming_sfx_audio, finish_streaming_sfx_audio
    from ..cevoice import CEVoiceLoader
    from .commands import process_command as process_ui_command
    from .terminal import is_celune_log_record
    from ..playback import current_playback_status
    from .resources import FOOTER_ROTATE_SECONDS
    from ..exceptions import CEDTSError
    from ..persona.asr import (
        DEFAULT_PERSONA_SPEECH_MODEL_ID,
        PERSONA_SPEECH_END_DELAY_SECONDS,
        PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
        WhisperSegment,
        WhisperTranscriber,
    )
    from ..persona.impl import persona_config, persona_enabled, persona_talkback_enabled
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

_UI_RUNTIME_EXPORTS = (
    Text,
    AutopilotCallbackType,
    ReturnType,
    ScreenStackError,
    NoMatches,
    EdgeStyle,
    Horizontal,
    Vertical,
    severity_color,
    SelectMenuOption,
    BASE_SR,
    ExitCodes,
    SIGTSTP,
    ComponentLockOwner,
    ComponentLockRequirement,
    default_error_theme_family,
    default_theme_family,
    launcher_loss_requested,
)

if TYPE_CHECKING:
    _UI_TYPE_EXPORTS = (
        AgentTask,
        _VCAudioCallback,
        LogRedirect,
        UILogHandler,
        sd,
        LiveVoiceActivityDetector,
        ComponentLockLease,
        Celune,
        CEVoiceLoader,
        WhisperTranscriber,
        EventDispatcher,
        AgentApprovalRequestedEvent,
        AgentChoiceRequestedEvent,
        AgentTaskFinishedEvent,
        AgentTaskStateChangedEvent,
        format_audio_device_name,
        resolve_audio_device_with_info,
        AudioOutput,
        CEDTSError,
        config_path,
        main_window_log_path,
        DEFAULT_PERSONA_SPEECH_MODEL_ID,
        PERSONA_SPEECH_END_DELAY_SECONDS,
        PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS,
        WhisperSegment,
        persona_config,
        persona_enabled,
        persona_talkback_enabled,
        current_playback_status,
        finish_streaming_sfx_audio,
        queue_streaming_sfx_audio,
        discard,
        is_april_fools,
        replace_ipa,
        VC_PITCH_SHIFT_MAX,
        VC_PITCH_SHIFT_MIN,
        clamp_vc_pitch_shift,
        create_live_voice_activity_detector,
        vc_input_has_voice,
        vc_input_rms,
        vc_live_chunk_frames,
        vc_live_chunk_overlap_frames,
        vc_vad_hangover_frames,
        vc_vad_preroll_frames,
        process_ui_command,
        FOOTER_ROTATE_SECONDS,
        is_celune_log_record,
        np,
        npt,
        yaml,
    )

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


def resolve_log_level(
    value: Union[LogLevel, bool, None],
    fallback: LogLevel,
) -> LogLevel:
    """Return a valid configured log level or the startup fallback."""
    if isinstance(value, str) and value in {"debug", "info", "verbose"}:
        return cast("LogLevel", value)
    return fallback


@dataclass(frozen=True)
class ButtonActions:
    """Describe the independent actions currently available on a button."""

    press: bool = True
    hold: bool = False


class Button(TextualButton):
    """Native-looking button with independently gated press and hold actions."""

    class Held(Message):
        """Message emitted after an enabled hold is released."""

        def __init__(self, button: Button) -> None:
            super().__init__()
            self.button = button

    def __init__(
        self,
        label: Optional[str] = None,
        variant: Literal["default", "error", "primary", "success", "warning"] = (
            "default"
        ),
        *,
        # pylint: disable=redefined-builtin
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
        tooltip: Optional[RenderableType] = None,
        action: Optional[str] = None,
        compact: bool = False,
        flat: bool = False,
        actions: Optional[ButtonActions] = None,
    ) -> None:
        super().__init__(
            label=label,
            variant=variant,
            name=name,
            id=id,
            classes=classes,
            tooltip=tooltip,
            action=action,
            compact=compact,
            flat=flat,
        )
        self._hold_seconds = 0.55
        self._hold_timer: Optional[Timer] = None
        self._long_pressed = False
        self._actions = ButtonActions()
        self.actions = actions or ButtonActions()

    @property
    def actions(self) -> ButtonActions:
        """Return the UI-owned capabilities available on this button."""
        return self._actions

    @actions.setter
    def actions(self, value: ButtonActions) -> None:
        """Update capabilities and their disabled appearance."""
        self._actions = value
        self.set_class(not value.press and not value.hold, "-actions-disabled")

    def _clear_focus(self) -> None:
        """Clear focus after an action so the button does not stay highlighted."""
        if self.is_attached:
            self.app.set_focus(None)

    def press(self) -> Button:
        """Perform the configured press action when it is available."""
        if not self.actions.press:
            return self
        super().press()
        self._clear_focus()
        return self

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        """Start the long-press timer for the primary mouse button."""
        if event.button == 1 and self.actions.hold:
            self._long_pressed = False
            self._stop_hold_timer()
            self._hold_timer = self.set_timer(
                self._hold_seconds,
                self._emit_held,
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
        self._clear_focus()
        if long_pressed:
            self.post_message(self.Held(self))

    def _emit_held(self) -> None:
        """Mark an enabled hold without running its action before release."""
        self._hold_timer = None
        if not self.actions.hold or not self.is_mouse_over:
            return
        self._long_pressed = True
        self.suppress_click()

    def _stop_hold_timer(self) -> None:
        """Stop the pending long-press timer, if any."""
        if self._hold_timer is not None:
            self._hold_timer.stop()
            self._hold_timer = None


class VoiceButton(Button):
    """Voice selector button retaining a semantic name for the UI state."""

    class Held(Button.Held):
        """Message emitted after the voice button's enabled hold is released."""


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
    global config_path
    global discard
    global format_audio_device_name
    global create_live_voice_activity_detector
    global current_playback_status
    global finish_streaming_sfx_audio
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
    from ..paths import config_path, main_window_log_path
    from ..theme import colors as loaded_colors
    from ..utils import (
        indent as loaded_indent,
    )
    from ..utils import (
        discard,
        replace_ipa,
        is_april_fools,
    )
    from ..utils import (
        supports_ansi as loaded_supports_ansi,
    )
    from ..config import format_audio_device_name, resolve_audio_device_with_info
    from ..speech import (
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from ..cevoice import default_loader as loaded_default_loader
    from .commands import process_command as process_ui_command
    from .terminal import LogRedirect, UILogHandler, is_celune_log_record
    from ..playback import current_playback_status
    from ..terminal import (
        RUNTIME_LOG_FILTER_MESSAGES,
    )
    from ..terminal import (
        set_terminal_title as loaded_set_terminal_title,
    )
    from ..terminal import (
        terminal_title_escape as loaded_terminal_title_escape,
    )
    from ..watchdog import launcher_loss_requested as loaded_launcher_loss_requested
    from .resources import FOOTER_ROTATE_SECONDS
    from ..exceptions import CEDTSError
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

    globals().update(
        {
            "colors": loaded_colors,
            "default_loader": loaded_default_loader,
            "indent": loaded_indent,
            "set_terminal_title": loaded_set_terminal_title,
            "supports_ansi": loaded_supports_ansi,
            "terminal_title_escape": loaded_terminal_title_escape,
        }
    )
    launcher_loss_requested = loaded_launcher_loss_requested
    _RUNTIME_LOG_REDIRECT_FILTER_MESSAGES = RUNTIME_LOG_FILTER_MESSAGES
    _RUNTIME_DEPENDENCIES_LOADED = True


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
    style_button: Optional[VoiceButton] = None
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
    speech_transcriber: Optional[WhisperTranscriber] = None
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
class CeluneUI(App, CeluneUIMethodSurface):
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
    _speech_transcriber = _forward_ui_property(
        "_interaction_state", "speech_transcriber"
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

    def _register_runtime_error_themes(self) -> None:
        """Register error themes used for runtime failure states."""
        from ..theme import colors

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

    def _wrap_runtime_fatal_glow(self) -> None:
        """Mirror runtime fatal glow events into the UI fatal theme flag."""
        if self.celune is None or getattr(self.celune, "_ui_fatal_glow_wrapped", False):
            return

        glow = getattr(self.celune, "glow", None)
        if glow is None:
            return
        from ..utils import available

        if not available("fatal", obj=glow):
            return
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            self._fatal_error_active = True
            self._run_on_ui_thread(self._refresh_theme_text)
            original_fatal()

        glow.fatal = wrapped_fatal
        self.celune._ui_fatal_glow_wrapped = True

    def _advance_status_marquee(self) -> None:
        """Advance the marquee one character for long status messages."""
        from ..playback import current_playback_status

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

    def _enable_runtime_log_capture(self) -> None:
        """Capture Celune runtime output after the Textual app has started cleanly."""
        if self._runtime_log_capture_enabled:
            return

        from .terminal import LogRedirect

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

    def _install_runtime_log_redirects(self) -> None:
        """Route non-Celune Python logging output into Celune's UI log widget."""
        if self._runtime_redirect_handler is not None:
            return

        from .terminal import UILogHandler, is_celune_log_record

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

    def _normal_input_placeholder(self) -> str:
        """Return the unlocked input placeholder without blocking the UI."""
        from ..persona.impl import persona_enabled, persona_talkback_enabled

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
                loaded = await run_in_daemon_thread(
                    self.celune.set_voice_and_wait,
                    entry,
                )
            else:
                loaded = await run_in_daemon_thread(
                    self.celune.set_cevoice_and_wait,
                    bundle_path,
                )
                if loaded:
                    loaded = await run_in_daemon_thread(
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
        from ..utils import typing_delay as loaded_typing_delay

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

            await asyncio.sleep(loaded_typing_delay(char))

            if cancellable and token != self._tutorial_token:
                return
            if self.cur_state == "exiting":
                return
            typed += char
            replace_input(typed)

        final_char = text[-1] if text else " "
        await asyncio.sleep(loaded_typing_delay(final_char))

        if self.cur_state != "exiting" and (
            not cancellable or token == self._tutorial_token
        ):
            self._submit_text(typed, process_commands)

    register_runtime_error_themes = _register_runtime_error_themes
    wrap_runtime_fatal_glow = _wrap_runtime_fatal_glow
    advance_status_marquee = _advance_status_marquee
    enable_runtime_log_capture = _enable_runtime_log_capture
    install_runtime_log_redirects = _install_runtime_log_redirects
    disable_runtime_log_capture = _disable_runtime_log_capture
    with_brightness = _with_brightness
    normal_input_placeholder = _normal_input_placeholder
    persona_loaded = _persona_loaded


def _install_ui_methods() -> None:
    """Install split UI methods after the concrete UI class exists."""
    from . import capture, runtime, interaction

    runtime.install(CeluneUI)
    capture.install(CeluneUI)
    interaction.install(CeluneUI)


_install_ui_methods()
