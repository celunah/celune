# SPDX-License-Identifier: Apache-2.0
"""Frontend layer."""

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
from collections.abc import Callable, Iterator, Awaitable

from textual import (
    app,
    css,
    work,
    theme,
    timer,
    events,
    binding,
    widgets,
    containers,
)
from textual import (
    color as textual_color,
)
from textual import (
    screen as textual_screen,
)
from textual import (
    widget as textual_widget,
)
from textual import (
    message as textual_message,
)
from rich.text import Text

from . import debug as _debug
from . import capture as _capture
from . import runtime as _runtime
from . import interaction as _interaction
from ..i18n import string, tagged_string
from .theme import CELUNE_CSS, severity_color, with_brightness
from .compat import install_textual_timer_sleep
from .loading import CeluneLoadingScreen
from .terminal import SelectMenuOption, SelectMenuWidget
from ..watchdog import launcher_loss_requested
from ..constants import BASE_SR, SIGTSTP, APP_NAME, ExitCodes
from ..typing.agent import AgentTaskState
from ..typing.locks import (
    ComponentLockName,
    ComponentLockOwner,
    ComponentLockRequirement,
)
from ..typing.modes import OperationState
from ..typing.common import JSONSerializable
from ..typing.config import AudioDeviceInfoValue
from ..registry.types import ModelMetadata
from ..theme.defaults import default_theme_family, default_error_theme_family
from ..core.concurrency import run_in_daemon_thread

App = app.App
ReturnType = app.ReturnType
ComposeResult = app.ComposeResult
RenderableType = app.RenderableType
ScreenStackError = app.ScreenStackError
AutopilotCallbackType = app.AutopilotCallbackType
Color = textual_color.Color
Theme = theme.Theme
Timer = timer.Timer
ModalScreen = textual_screen.ModalScreen
DebugInspectorScreen = _debug.DebugInspectorScreen
Widget = textual_widget.Widget
BindingType = binding.BindingType
Message = textual_message.Message
Input = widgets.Input
Label = widgets.Label
Static = widgets.Static
RichLog = widgets.RichLog
TextArea = widgets.TextArea
ProgressBar = widgets.ProgressBar
TextualButton = widgets.Button
Vertical = containers.Vertical
Horizontal = containers.Horizontal

if TYPE_CHECKING:
    from textual.css.query import NoMatches
    from textual.css.types import EdgeStyle
else:
    NoMatches = css.query.NoMatches
    EdgeStyle = css.types.EdgeStyle

install_textual_timer_sleep()

if TYPE_CHECKING:
    import yaml
    import numpy as np
    import sounddevice as sd
    import numpy.typing as npt

    from ..vc import (
        VC_PLAYBACK_GAIN,
        VCDebugCapture,
        LiveVoiceActivityDetector,
        vc_input_rms,
        vc_input_has_voice,
        vc_live_chunk_frames,
        vc_vad_preroll_frames,
        vc_vad_hangover_frames,
        vc_live_chunk_overlap_frames,
        create_live_voice_activity_detector,
    )
    from ..paths import config_path, main_window_log_path
    from ..utils import discard, replace_ipa, is_april_fools
    from ..celune import Celune
    from ..config import format_audio_device_name, resolve_audio_device_with_info
    from ..cevoice import CEVoiceLoader
    from .commands import process_command as process_ui_command
    from .terminal import (
        LogRedirect,
        UILogHandler,
        is_celune_log_record,
    )
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
    from ..typing.agent import AgentTask
    from ..typing.aliases import LogLevel, AudioChunk, AudioChunks, VCAudioCallback
    from ..core.concurrency import ComponentLockLease
    from ..extensions.events import EventDispatcher
    from ..pipeline.captions import current_playback_status
    from ..dataclasses.events import (
        CharacterLoadedEvent,
        CharacterChangedEvent,
        AgentTaskFinishedEvent,
        CharacterUnloadedEvent,
        AgentChoiceRequestedEvent,
        AgentTaskStateChangedEvent,
        AgentApprovalRequestedEvent,
    )
    from ..pipeline.submission import (
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from ..dataclasses.pipeline import AudioOutput

    colors: ModuleType

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
        VCAudioCallback,
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
        CharacterChangedEvent,
        CharacterLoadedEvent,
        CharacterUnloadedEvent,
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
        VC_PLAYBACK_GAIN,
        discard,
        is_april_fools,
        replace_ipa,
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
    VCAudioCallback = Callable[..., None]


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


def concise_error_message(message: str) -> str:
    """Remove traceback frames from startup text while retaining its final error."""
    _, separator, traceback_text = message.partition(
        "Traceback (most recent call last):"
    )
    if not separator:
        return message

    final_line = next(
        (
            line.strip()
            for line in reversed(traceback_text.splitlines())
            if line.strip()
        ),
        "",
    )
    _, separator, detail = final_line.partition("): ")
    if separator and detail:
        return detail

    _, separator, detail = final_line.partition(": ")
    if separator and detail:
        return detail
    return final_line or message


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


class ModelConsentScreen(ModalScreen[bool]):
    """Require an exact typed acknowledgement before Celune loads a model."""

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "decline", "Decline")]
    DEFAULT_CSS = """
    ModelConsentScreen {
        align: center middle;
        background: transparent;
    }
    #model-consent-panel {
        width: 82%;
        max-width: 88;
        max-height: 90%;
        height: auto;
        padding: 1 2;
        border: round $primary;
        background: $background;
        color: $foreground;
    }
    #model-consent-prompt {
        height: auto;
        margin-bottom: 1;
    }
    #model-consent-detail {
        height: auto;
        color: $text-muted;
    }
    #model-consent-phrase {
        margin-top: 1;
        width: 1fr;
        height: 3;
        border: round $primary;
        background: $background;
        color: $foreground;
    }
    #model-consent-phrase:focus {
        border: round $foreground;
        background: $background;
        background-tint: transparent;
    }
    #model-consent-phrase .input--placeholder {
        color: $secondary;
    }
    #model-consent-actions {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    def __init__(
        self,
        prompt: str,
        instruction: str,
        details: tuple[str, ...],
        entry_hint: str,
        consent_phrase: str,
        accept_label: str,
        decline_label: str,
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.instruction = instruction
        self.details = details
        self.entry_hint = entry_hint
        self.consent_phrase = consent_phrase
        self.accept_label = accept_label
        self.decline_label = decline_label

    def compose(self) -> ComposeResult:
        """Yield the license source, restriction details and decision controls."""
        with Vertical(id="model-consent-panel"):
            yield Static(self.prompt, id="model-consent-prompt")
            yield Static(self.instruction)
            for detail in self.details:
                yield Static(detail, classes="model-consent-detail")
            yield Input(
                placeholder=self.entry_hint,
                id="model-consent-phrase",
            )
            with Horizontal(id="model-consent-actions"):
                yield Button(
                    self.decline_label,
                    id="model-consent-decline",
                    variant="default",
                )
                yield Button(
                    self.accept_label,
                    id="model-consent-accept",
                    variant="primary",
                )

    def on_mount(self) -> None:
        """Focus the exact phrase field when the consent screen opens."""
        self.query_one("#model-consent-accept", Button).disabled = True
        self.query_one(Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Enable consent only after the requested phrase is typed verbatim."""
        self.query_one("#model-consent-accept", Button).disabled = (
            event.value != self.consent_phrase
        )

    def on_button_pressed(self, event: TextualButton.Pressed) -> None:
        """Return the decision selected by the user."""
        if event.button.id == "model-consent-accept":
            if self.query_one(Input).value == self.consent_phrase:
                self.dismiss(True)
            return
        if event.button.id == "model-consent-decline":
            self.dismiss(False)

    def action_decline(self) -> None:
        """Reject the pending model load when the user presses Escape."""
        self.dismiss(False)


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


def ensure_ui_runtime_dependencies() -> None:
    """Load optional UI integrations after the first loading frame is visible."""
    global _RUNTIME_DEPENDENCIES_LOADED
    global colors
    global default_loader
    global indent
    global set_terminal_title
    global supports_ansi
    global terminal_title_escape
    if _RUNTIME_DEPENDENCIES_LOADED:
        return

    global AudioOutput
    global CEDTSError
    global DEFAULT_PERSONA_SPEECH_MODEL_ID
    global FOOTER_ROTATE_SECONDS
    global LiveVoiceActivityDetector
    global PERSONA_SPEECH_END_DELAY_SECONDS
    global PERSONA_SPEECH_NO_INPUT_TIMEOUT_SECONDS
    global WhisperSegment
    global WhisperTranscriber
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
    global VC_PLAYBACK_GAIN
    global yaml
    global is_celune_log_record
    global _RUNTIME_LOG_REDIRECT_FILTER_MESSAGES

    import yaml
    import numpy as np
    import sounddevice as sd
    import numpy.typing as npt

    from . import resources as loaded_ui_resources

    ui_resources = loaded_ui_resources
    from ..vc import (
        VC_PLAYBACK_GAIN,
        LiveVoiceActivityDetector,
        vc_input_rms,
        vc_input_has_voice,
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
    from ..cevoice import default_loader as loaded_default_loader
    from .commands import process_command as process_ui_command
    from .terminal import LogRedirect, UILogHandler, is_celune_log_record
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
    from ..pipeline.captions import current_playback_status
    from ..pipeline.submission import (
        queue_streaming_sfx_audio,
        finish_streaming_sfx_audio,
    )
    from ..dataclasses.pipeline import AudioOutput

    colors = loaded_colors
    default_loader = loaded_default_loader
    indent = loaded_indent
    set_terminal_title = loaded_set_terminal_title
    supports_ansi = loaded_supports_ansi
    terminal_title_escape = loaded_terminal_title_escape
    launcher_loss_requested = loaded_launcher_loss_requested
    _RUNTIME_LOG_REDIRECT_FILTER_MESSAGES = RUNTIME_LOG_FILTER_MESSAGES
    _RUNTIME_DEPENDENCIES_LOADED = True


LOADING_FADE_SECONDS = 1.0
MAIN_UI_FADE_SECONDS = 0.6
EXIT_FADE_SECONDS = 0.6
VC_FEEDBACK_MIN_CAPTURE_SECONDS = 0.35
VC_FEEDBACK_REQUIRED_CONSECUTIVE_SPIKES = 2
VC_FEEDBACK_RMS_MIN_PREVIOUS = 0.05
VC_FEEDBACK_RMS_MIN_CURRENT = 0.18
VC_FEEDBACK_RMS_RISE_RATIO = 2.0
VC_FEEDBACK_RMS_RISE_DELTA = 0.08
VC_LIVE_SUBMISSION_QUEUE_SIZE = 3
AGENT_ACTIVE_STATES = frozenset(
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
AGENT_AWAITING_STATES = frozenset(
    {AgentTaskState.AWAITING_APPROVAL, AgentTaskState.AWAITING_CHOICE}
)
AGENT_PAUSED_STATES = frozenset({AgentTaskState.PAUSED, AgentTaskState.INTERRUPTED})


def device_scalar_int(value: Optional[AudioDeviceInfoValue], default: int) -> int:
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
    loading_ui_reveal_available: bool = False
    loading_ui_revealed: bool = False
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
    latest_startup_error: Optional[str] = None
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
    vc_debug_capture: Optional[VCDebugCapture] = None
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
    caption_rendered_text: str = ""
    caption_visible_words: int = 0
    caption_progress: float = 0.0
    caption_segmented_progress: bool = False
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
class CeluneUI(
    _runtime.RuntimeHooks, _capture.CaptureHooks, _interaction.InteractionHooks, App
):
    """Celune's main user interface."""

    def __init_subclass__(cls, **kwargs: Never) -> Never:
        raise TypeError(f"{__class__.__name__} is final and cannot be subclassed")

    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        *App.BINDINGS,
        binding.Binding(
            "ctrl+b",
            "toggle_debug_inspector",
            string("debug.binding"),
            priority=True,
        ),
    ]
    CSS = CELUNE_CSS
    _instance: ClassVar[Optional[CeluneUI]] = None

    @classmethod
    def get_instance(cls) -> Optional[CeluneUI]:
        """Return the active UI instance, if one has been created."""
        return cls._instance

    def release_instance(self) -> None:
        """Release this UI instance from the process-wide singleton slot."""
        if type(self)._instance is self:
            type(self)._instance = None

    def __init__(
        self,
        startup_loader: Optional[Callable[[], Celune]] = None,
        startup_messages: Optional[list[str]] = None,
        startup_log_level: LogLevel = "info",
        test_completion_callback: Optional[
            Callable[[Celune, bool, Optional[str]], None]
        ] = None,
        startup_dependency_help: Optional[str] = None,
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
        self._agent_test_response_queue: Optional[queue_module.Queue[Optional[str]]] = (
            None
        )
        self._terminal_status: Optional[tuple[str, str, str]] = None
        self._loading_screen: Optional[CeluneLoadingScreen] = None
        self._startup_loader = startup_loader
        self._startup_messages = list(startup_messages or [])
        self._startup_log_level = startup_log_level
        self._debug_call_trace = _debug.DebugCallTrace(self)
        self._test_completion_callback = test_completion_callback
        self._startup_dependency_help = startup_dependency_help
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
    _loading_ui_reveal_available = _forward_ui_property(
        "_binding_state", "loading_ui_reveal_available"
    )
    _loading_ui_revealed = _forward_ui_property("_binding_state", "loading_ui_revealed")
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
    _latest_startup_error = _forward_ui_property(
        "_log_capture_state", "latest_startup_error"
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
    _vc_debug_capture = _forward_ui_property("_interaction_state", "vc_debug_capture")
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
    _caption_rendered_text = _forward_ui_property(
        "_interaction_state", "caption_rendered_text"
    )
    _caption_visible_words = _forward_ui_property(
        "_interaction_state", "caption_visible_words"
    )
    _caption_progress = _forward_ui_property("_interaction_state", "caption_progress")
    _caption_segmented_progress = _forward_ui_property(
        "_interaction_state", "caption_segmented_progress"
    )
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
        self.register_theme(dark_theme)
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
        from ..pipeline.captions import current_playback_status

        if self.status is None:
            return
        if self.celune is not None and getattr(self.celune, "test_finished", False):
            return
        self._refresh_agent_status()
        playback_status = (
            current_playback_status(self.celune) if self.celune is not None else None
        )
        agent_is_active = self._agent_task_state in (
            AGENT_ACTIVE_STATES | AGENT_AWAITING_STATES | AGENT_PAUSED_STATES
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

    def load_tts(self) -> None:
        """Start runtime loading in a daemon thread."""
        threading.Thread(
            target=self._load_tts,
            name="celune-ui-runtime-load",
            daemon=True,
        ).start()

    def _request_model_consent(self, metadata: ModelMetadata) -> bool:
        """Show one blocking consent screen while model work waits in its thread."""
        from ..registry.consent import consent_details, consent_prompt_text

        prompt, instruction = consent_prompt_text(metadata)
        screen = ModelConsentScreen(
            prompt=prompt,
            instruction=instruction,
            details=consent_details(metadata),
            entry_hint=string("registry.consent_entry_hint"),
            consent_phrase=string("registry.consent_phrase"),
            accept_label=string("registry.consent_accept"),
            decline_label=string("registry.consent_decline"),
        )
        response = threading.Event()
        decision: list[Optional[bool]] = [None]
        waiting_status = string("registry.waiting_status")
        from ..registry.consent import consent_status, pending_model_consents

        def show_screen() -> None:
            """Present the modal and route its result to the waiting thread."""
            self.safe_status(waiting_status, "warning")

            def on_consent_result(accepted: Optional[bool]) -> None:
                """Wake the model-loading worker with the user's decision."""
                decision[0] = accepted
                response.set()

            self.push_screen(screen, on_consent_result)

        self._run_on_ui_thread(show_screen)
        browser_accepted = False

        def pop_screen_if_current() -> None:
            if self.screen is screen:
                self.pop_screen()

        while not response.wait(0.2):
            if consent_status(metadata):
                browser_accepted = True
                self._run_on_ui_thread(pop_screen_if_current)
                break
            if not any(
                pending.model_id == metadata.model_id
                and pending.revision == metadata.revision
                for pending in pending_model_consents()
            ):
                self._run_on_ui_thread(pop_screen_if_current)
                break
            if self.cur_state == "exiting":
                return False

        if self._status_text == waiting_status:
            self.safe_status(string("ui.idle_status"))
        return browser_accepted or decision[0] is True

    def _load_tts(self) -> None:
        """Load the app runtime."""
        if self.cur_state == "exiting":
            return
        if self._is_ui_test_mode():
            self._latest_startup_error = None
            try:
                loaded = self.celune.load()
                if self.cur_state == "exiting":
                    return
                if not loaded:
                    failure_message = self._latest_startup_error or string(
                        "ui.app_could_not_start",
                        app_name=APP_NAME,
                    )
                    self.cur_state = "error"
                    self.change_input_state(locked=True)
                    self.change_voice_lock_state(locked=True)
                    self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                    self._show_loading_error(failure_message)
                    self._finish_test_startup(False, failure_message)
                    return

                self.celune_styles = ()
                if not self.celune.use_normalization:
                    self.safe_progress(1, 1)
                self.change_input_state(locked=True)
                self.change_voice_lock_state(locked=True)
                self.safe_status(string("ui.test_mode_active"))
                self._dismiss_loading_screen()
                self._finish_test_startup(True)
            except Exception as error:
                if self.cur_state == "exiting":
                    return
                failure_message = str(error) or string(
                    "ui.app_could_not_start",
                    app_name=APP_NAME,
                )
                self.cur_state = "error"
                self.safe_log(failure_message, "error")
                self.celune.fatal()
                self.change_input_state(locked=True)
                self.change_voice_lock_state(locked=True)
                self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                self._show_loading_error(failure_message)
                self._finish_test_startup(False, failure_message)
            return

        from ..exceptions import ModelConsentDeclined
        from ..registry.consent import set_model_consent_handler

        set_model_consent_handler(self._request_model_consent)
        self._latest_startup_error = None
        try:
            loaded = self.celune.load()
            if self.cur_state == "exiting":
                return
            if loaded:
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
                self.change_input_state(
                    locked=(
                        getattr(
                            self.celune,
                            "operation_state",
                            OperationState.NORMAL,
                        )
                        is not OperationState.SEMI_FATAL
                    )
                )
                self.change_voice_lock_state(locked=True)
                failure_message = self._latest_startup_error or string(
                    "ui.app_could_not_start",
                    app_name=APP_NAME,
                )
                self.error(string("ui.app_could_not_start", app_name=APP_NAME))
                self._show_loading_error(failure_message)
                self._finish_test_startup(False, failure_message)
        except ModelConsentDeclined:
            if self.cur_state == "exiting":
                return
            self.cur_state = "error"
            message = string("registry.consent_declined")
            self.celune.fatal()
            self.change_input_state(locked=True)
            self.change_voice_lock_state(locked=True)
            self.safe_status(message, "warning")
            self._show_loading_error(message)
            self._finish_test_startup(False, message)
        except Exception as e:
            if self.cur_state == "exiting":
                return
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

    def _normal_input_placeholder(self) -> str:
        """Return the unlocked input placeholder without blocking the UI."""
        from ..persona.impl import persona_enabled, persona_talkback_enabled

        if self.is_voice_conversion_mode():
            return string("ui.voice_changer_placeholder")

        if getattr(getattr(self.celune, "backend", None), "singing", False):
            return string("ui.cesn_input_placeholder")

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
        celune = self.celune
        if celune is None:
            return
        singing_backend = getattr(getattr(celune, "backend", None), "singing", False)
        from ..cevoice import SINGING_VOICE_NAME, active_bundle_path

        pack = value.get("pack")
        entry = value.get("entry")
        bundle_path = (
            self._voice_menu_paths.get(pack) if isinstance(pack, str) else None
        )
        if bundle_path is None or not isinstance(entry, str):
            return
        if singing_backend and entry != SINGING_VOICE_NAME:
            return

        async def reload_voice_pack() -> bool:
            """Wait for the pack reload callbacks to finish before changing voice."""
            async_reload = cast(
                Optional[Callable[[Path], Awaitable[bool]]],
                getattr(celune, "set_cevoice_async", None),
            )
            if callable(async_reload):
                return await async_reload(bundle_path)
            return bool(
                await run_in_daemon_thread(
                    celune.set_cevoice_and_wait,
                    bundle_path,
                )
            )

        try:
            if (
                getattr(celune, "sleeping", False)
                and not await celune.wake_from_sleep_async()
            ):
                self.safe_log(string("ui.voice_change_failed"), "warning")
                return

            if singing_backend:
                if active_bundle_path() == bundle_path:
                    return
                loaded = await reload_voice_pack()
            elif active_bundle_path() == bundle_path:
                loaded = await run_in_daemon_thread(
                    celune.set_voice_and_wait,
                    entry,
                )
            else:
                loaded = await reload_voice_pack()
                if loaded:
                    loaded = await run_in_daemon_thread(
                        celune.set_voice_and_wait,
                        entry,
                    )
            if not loaded:
                self.safe_log(string("ui.voice_change_failed"), "warning")
                return
            self.celune_styles = celune.voices
            current_voice = getattr(celune, "current_voice", None)
            if singing_backend:
                if (
                    isinstance(current_voice, str)
                    and current_voice in self.celune_styles
                ):
                    self.style_index = self.celune_styles.index(current_voice)
                    self.tts_voice_changed(current_voice)
                self.change_voice_lock_state(locked=True, can_open_menu=True)
            else:
                self.style_index = self.celune_styles.index(entry)
                self.tts_voice_changed(entry)
                self.change_voice_lock_state(locked=len(self.celune_styles) < 2)
        except Exception as error:
            self.safe_log(
                format_error_message(
                    string("ui.voice_change_failed_error"),
                    error,
                    getattr(celune, "log_level", self._startup_log_level),
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

    def register_runtime_error_themes(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._register_runtime_error_themes()

    def wrap_runtime_fatal_glow(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._wrap_runtime_fatal_glow()

    def advance_status_marquee(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._advance_status_marquee()

    def enable_runtime_log_capture(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._enable_runtime_log_capture()

    def install_runtime_log_redirects(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._install_runtime_log_redirects()

    def disable_runtime_log_capture(self) -> None:
        """Delegate to the existing Textual UI hook."""
        return self._disable_runtime_log_capture()

    @staticmethod
    def with_brightness(color: Color, brightness: float) -> Color:
        """Return ``color`` blended toward the requested brightness."""
        return with_brightness(color, brightness)

    def normal_input_placeholder(self) -> str:
        """Delegate to the existing Textual UI hook."""
        return self._normal_input_placeholder()

    def persona_loaded(self) -> bool:
        """Delegate to the existing Textual UI hook."""
        return self._persona_loaded()
