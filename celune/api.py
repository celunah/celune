# SPDX-License-Identifier: Apache-2.0
"""API layer."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import os
import json
import time
import uuid
import errno
import socket
import asyncio
import inspect
import datetime
import textwrap
import threading
import contextlib
from hmac import compare_digest
from typing import (
    Union,
    Literal,
    Optional,
)
from collections import deque, defaultdict
from dataclasses import field, dataclass
from collections.abc import Callable, Iterator, Awaitable

import numpy as np
import gradio as gr
import uvicorn
from fastapi import (
    File,
    Form,
    FastAPI,
    Request,
    WebSocket,
    UploadFile,
    HTTPException,
    WebSocketDisconnect,
)
from pydantic import Field, BaseModel
from fastapi.responses import (
    Response,
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from . import __version__
from .ui import resources as ui_resources
from .i18n import string
from .paths import project_root
from .theme import colors
from .utils import available, format_error_message
from .celune import Celune
from .cevoice import default_loader
from .speech import (
    prepare_playback_audio,
)
from .audio.dsp import resample_audio
from .constants import BASE_SR, APP_NAME
from .exceptions import TaskSubscriptionClosed
from .typing.api import (
    TaskStatus,
    TaskEventName,
    TaskCommandName,
)
from .typing.common import JSONSerializable
from .typing.events import EventName, EventCallback
from .typing.aliases import LogLevel
from .extensions.events import EventDispatcher
from . import webactions, webui
from .webactions import (
    _build_webui,
    _decode_uploaded_audio,
    _voice_conversion_unavailable_response,
    _webui_run_command,
    _webui_speak,
)
from .webui import (
    _append_webui_log,
    _collect_speech_job,
    _flac_bytes,
    _forget_speech_job,
    _publish_active_task_log,
    _publish_task_event,
    _remember_speech_job,
    _seed_webui_logs,
    _set_active_speech_task,
    _set_webui_status,
    _speech_job_snapshot,
    _strip_webui_log_prefix,
    _subscribe_to_speech_job,
    _subscribe_webui_events,
    _subscribe_webui_timed_updates,
    _task_status,
    _unsubscribe_from_speech_job,
    _update_speech_job,
    _webui_input_placeholder,
    _webui_snapshot,
    _webui_theme_html,
    _webui_vc_mode_active,
    _wrap_celune_callbacks,
    api_log,
    audio_bytes,
    require_celune,
    stream_headers,
)

# Keep the historical API-module facade for WebUI helpers used by integrations.
_receive_webui_timed_update = webui._receive_webui_timed_update
_webui_audio_waveform_options = webui._webui_audio_waveform_options
_input_update = webui._input_update
_webui_resources_html = webui._webui_resources_html
_send_button_update = webui._send_button_update
_webui_status_html = webui._webui_status_html
_webui_vc_controls_update = webui._webui_vc_controls_update
_webui_convert_audio = webactions._webui_convert_audio
_webui_cycle_voice = webactions._webui_cycle_voice
CeluneUI = webui.CeluneUI
UiTimedUpdate = webui.UiTimedUpdate

api = FastAPI(title=f"{APP_NAME}API")
bound_celune: Optional[Celune] = None
auth_token: Optional[str] = None
rate_limit_per_minute = 60
rate_limit_lock = threading.Lock()
rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
max_sfx_upload_bytes = 25 * 1024 * 1024
speech_jobs_lock = threading.Lock()
speech_jobs: dict[str, "SpeechJob"] = {}
speech_job_ttl_seconds = 15 * 60
active_speech_task_id: Optional[str] = None
webui_log_lines: deque[tuple[str, str]] = deque(maxlen=240)
webui_status_text = "Waiting for response"
webui_status_severity = "info"
webui_logs_seeded = False
webui_caption_text = ""
webui_caption_progress = 0.0
webui_caption_active = False
webui_progress_current: Optional[float] = None
webui_progress_total: Optional[float] = None
webui_resource_page = 0
webui_last_resource_advance = 0.0
webui_last_probed_state: Optional[str] = None
webui_active_theme_name = "celune"
webui_timed_update_sequence = 0
webui_timed_update_received_at = 0.0
webui_timed_update_source = "fallback"
webui_timed_update_unsubscribe: Optional[Callable[[], None]] = None
webui_event_dispatcher: Optional[EventDispatcher] = None
webui_event_callbacks: tuple[tuple[EventName, EventCallback], ...] = ()


def _invoke_message_callback(
    callback: Callable[..., None],
    msg: str,
    severity: str,
    loglevel: LogLevel,
) -> None:
    """Invoke a message callback while preserving legacy two-argument callbacks."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        callback(msg, severity, loglevel=loglevel)
        return

    try:
        signature.bind(msg, severity, loglevel=loglevel)
    except TypeError:
        callback(msg, severity)
    else:
        callback(msg, severity, loglevel=loglevel)


webui_input_locked = True
webui_input_placeholder = string("ui.wait_placeholder")
webui_voice_locked = True
webui_theme_style = ""
webui_status_source = "probe"
webui_status_updated_at = 0.0
current_api_server: Optional["StartedServer"] = None
WEBUI_RESOURCE_ROTATE_SECONDS = 2.06
WEBUI_POLL_INTERVAL_SECONDS = WEBUI_RESOURCE_ROTATE_SECONDS / 4
WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS = 0.9
WEBUI_TIMED_UPDATE_STALE_SECONDS = WEBUI_RESOURCE_ROTATE_SECONDS * 2


class TaskEvent(BaseModel):
    """Typed event mirrored to clients watching one API task."""

    task_id: str
    event: TaskEventName
    status: TaskStatus
    message: Optional[str] = None
    severity: Optional[str] = None
    current: Optional[float] = None
    total: Optional[float] = None
    location: Optional[str] = None
    error: Optional[str] = None


class TaskCommand(BaseModel):
    """Typed command accepted by a task WebSocket."""

    command: TaskCommandName


class TaskCommandResponse(BaseModel):
    """Typed response for a command sent through a task WebSocket."""

    type: Literal["command_result"] = "command_result"
    task_id: str
    command: Optional[TaskCommandName] = None
    accepted: bool
    status: str


@dataclass
class TaskSubscription:
    """Event-loop-owned queue with a thread-safe event publisher."""

    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[Optional[TaskEvent]] = field(
        default_factory=asyncio.Queue,
        repr=False,
    )
    closed: threading.Event = field(default_factory=threading.Event)

    def _enqueue(self, event: Optional[TaskEvent]) -> None:
        """Enqueue an event or the close sentinel on the owning loop."""
        if event is None or not self.closed.is_set():
            self.events.put_nowait(event)

    def _call_on_loop(self, event: Optional[TaskEvent]) -> None:
        """Schedule one queue mutation on the subscription's event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self.loop:
            self._enqueue(event)
            return

        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._enqueue, event)

    def put(self, event: TaskEvent) -> None:
        """Queue one event for the subscribed WebSocket.

        Args:
            event: The typed event to deliver to the subscriber.
        """
        if not self.closed.is_set():
            self._call_on_loop(event)

    def close(self) -> None:
        """Stop waiting for events without affecting the underlying task."""
        if self.closed.is_set():
            return
        self.closed.set()
        self._call_on_loop(None)

    async def next_event(self) -> TaskEvent:
        """Wait natively on the subscription's event-loop queue.

        Returns:
            TaskEvent: The next event published for this subscription.

        Raises:
            TaskSubscriptionClosed: If the subscription was closed.
        """
        if self.closed.is_set():
            raise TaskSubscriptionClosed
        event = await self.events.get()
        if event is None:
            raise TaskSubscriptionClosed
        return event


def _run_async_runtime_call(
    awaitable: Awaitable[JSONSerializable],
) -> JSONSerializable:
    """Run one async runtime call from a synchronous API or WebUI callback."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:

        async def await_result() -> JSONSerializable:
            return await awaitable

        return asyncio.run(await_result())
    raise RuntimeError("synchronous runtime calls cannot run on an active event loop")


WEBUI_HEAD = textwrap.dedent(
    """
    <link rel="icon" type="image/x-icon" href="/favicon.ico">
    <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico">
    <script>
    (() => {
      if (window.__celuneLogAutoscrollInstalled) {
        return;
      }
      window.__celuneLogAutoscrollInstalled = true;

      const logScrollThreshold = 24;

      function isNearLogBottom(logElement) {
        return logElement.scrollHeight - logElement.scrollTop - logElement.clientHeight
          <= logScrollThreshold;
      }

      function scrollLogToBottom() {
        const logElement = document.querySelector("#celune-log-panel pre");
        if (!logElement) {
          return;
        }
        logElement.scrollTop = logElement.scrollHeight;
      }

      function updateLogFollowState(event) {
        window.__celuneLogAutoscrollFollow = isNearLogBottom(event.currentTarget);
      }

      function installLogObserver() {
        const logElement = document.querySelector("#celune-log-panel pre");
        if (!logElement) {
          return;
        }

        if (window.__celuneLogAutoscrollTarget === logElement) {
          return;
        }

        const previousLogElement = window.__celuneLogAutoscrollTarget;
        if (previousLogElement) {
          window.__celuneLogAutoscrollFollow = isNearLogBottom(previousLogElement);
        } else if (typeof window.__celuneLogAutoscrollFollow !== "boolean") {
          window.__celuneLogAutoscrollFollow = true;
        }

        window.__celuneLogAutoscrollTarget = logElement;
        logElement.addEventListener("scroll", updateLogFollowState, {
          passive: true,
        });
        if (window.__celuneLogAutoscrollFollow) {
          window.requestAnimationFrame(scrollLogToBottom);
        }

        if (window.__celuneLogAutoscrollObserver) {
          window.__celuneLogAutoscrollObserver.disconnect();
        }

        const observer = new MutationObserver(() => {
          if (window.__celuneLogAutoscrollFollow) {
            scrollLogToBottom();
          }
        });
        observer.observe(logElement, {
          childList: true,
          characterData: true,
          subtree: true,
        });
        window.__celuneLogAutoscrollObserver = observer;
      }

      const pageObserver = new MutationObserver(() => {
        installLogObserver();
      });

      function startLogAutoscroll() {
        installLogObserver();
        pageObserver.observe(document.body, {
          childList: true,
          subtree: true,
        });
        window.setInterval(installLogObserver, 500);
      }

      if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", startLogAutoscroll, {
          once: true,
        });
      } else {
        startLogAutoscroll();
      }
    })();

    (() => {
      function handleRecordingShortcut(event) {
        if (
          !event.altKey
          || event.ctrlKey
          || event.metaKey
          || event.key.toLowerCase() !== "r"
        ) {
          return;
        }

        const recordButton = document.querySelector(
          "#celune-record-hotkey button, button#celune-record-hotkey",
        );
        if (!recordButton || recordButton.disabled) {
          return;
        }

        event.preventDefault();
        recordButton.click();
      }

      document.addEventListener("keydown", handleRecordingShortcut);
    })();
    </script>
    """
)

WEBUI_CSS = textwrap.dedent(
    """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100..800&display=swap');

    html,
    body,
    gradio-app {
        --color-accent: var(--celune-primary, #cebaff) !important;
        --body-text-color: var(--celune-secondary, #a595cc) !important;
        --block-label-text-color: var(--celune-primary, #cebaff) !important;
        --block-title-text-color: var(--celune-primary, #cebaff) !important;
        --block-info-text-color: var(--celune-secondary, #a595cc) !important;
        /* the Celune "accent" color in CSS is actually considered as Celune tertiary */
        --body-text-color-subdued: var(--celune-accent, #7c7099) !important;
        background: var(--celune-ui-bg, var(--celune-background, #1d1826)) !important;
    }

    .column {
        place-content: center;
    }

    .gradio-container {
        background: var(--celune-ui-bg, var(--celune-background, #1d1826));
        font-family: Outfit, sans-serif !important;
        height: 100dvh;
        overflow: hidden;
    }
    
    .gradio-container .tab-container::after {
        display: none;
    }

    .gradio-container > .main,
    .gradio-container .wrap,
    .gradio-container .block,
    .gradio-container .form,
    .gradio-container label,
    .gradio-container label.selected,
    .gradio-container .tab-like-container,
    .gradio-container .tab-like-container input,
    .gradio-container .loading-container,
    .gradio-container .loading-container > div,
    .gradio-container button[role="tab"]:hover {
        background: var(--celune-ui-bg, var(--celune-background, #1d1826)) !important;
    }

    .main {
        flex: 1 1 auto !important;
        min-height: 0;
    }

    body {
        font-family: Outfit, sans-serif;
    }

    #celune-shell {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        height: calc(100dvh - 2rem);
        min-height: 0;
        background: var(--celune-ui-bg, var(--celune-background, #1d1826));
    }

    #celune-header {
        display: flex;
        place-items: center;
    }

    #celune-header .line {
        width: 100%;
        background: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        height: 2px;
    }

    #celune-header .title {
        font-weight: bold;
        padding: 0 2em;
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
    }

    button#celune-send, button#celune-convert {
        background: var(--celune-button-bg, #3a304c);
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        border-radius: 4px;
    }

    button#celune-send:hover, button#celune-convert:hover {
        background: var(--celune-button-hover, #443a56);
    }

    #celune-log-panel {
        border: 2px solid var(--celune-ui-accent, var(--celune-primary, #cebaff));
        background: var(--celune-ui-bg, var(--celune-background, #1d1826));
        padding: 1em;
        border-radius: 8px;
        max-height: min(75dvh, calc(100dvh - 20rem));
        overflow: hidden;
        flex: 1 1 auto;
        min-height: 0;
    }

    #celune-log-panel pre {
        font-family: "JetBrains Mono", monospace;
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        white-space: pre-wrap;
        margin: 0;
        max-height: min(calc(75dvh - 2em), calc(75dvh - 15rem));
        height: 100%;
        overflow-y: auto;
        padding-right: 0.75em;
        scrollbar-gutter: stable both-edges;
        scrollbar-color: var(--celune-ui-accent, var(--celune-primary, #cebaff))
            var(--celune-ui-bg, var(--celune-background, #1d1826));
    }

    #celune-input textarea {
        background: var(--celune-ui-input-bg, var(--celune-input-bg, #3a304c));
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        border-radius: 4px;
        scrollbar-color: var(--celune-ui-accent, var(--celune-primary, #cebaff))
            var(--celune-ui-bg, var(--celune-background, #1d1826));
    }

    #celune-input textarea::placeholder {
        color: var(--celune-placeholder, #9c88ce);
    }

    #celune-resources .footer-block {
        text-align: right;
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
    }

    .webui-recording-hint {
        margin-top: 0.25rem;
        font-size: 0.9rem;
    }

    #celune-record-hotkey {
        display: none !important;
    }

    .webui-desktop-only {
        display: inline !important;
        color: inherit;
    }

    .webui-mobile-only {
        display: none !important;
        color: inherit;
    }

    #celune-actions {
        gap: 0.75rem;
    }

    button#celune-send {
        min-height: 2.75rem;
    }

    #celune-input-row, #celune-footer {
        padding: 0 1em;
    }

    button#celune-send {
        display: none;
    }

    #celune-log-panel pre::-webkit-scrollbar,
    #celune-input textarea::-webkit-scrollbar {
        width: 0.8rem;
    }

    #celune-log-panel pre::-webkit-scrollbar-track,
    #celune-input textarea::-webkit-scrollbar-track {
        background: var(--celune-ui-bg, var(--celune-background, #1d1826));
    }

    #celune-log-panel pre::-webkit-scrollbar-thumb,
    #celune-input textarea::-webkit-scrollbar-thumb {
        background: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        border-radius: 999px;
        border: 2px solid var(--celune-ui-bg, var(--celune-background, #1d1826));
    }

    .gradio-container .minimal-audio-player button:hover,
    .gradio-container .minimal-audio-player button:focus,
    .gradio-container .standard-player button:hover,
    .gradio-container .standard-player button:focus,
    .gradio-container .controls .icon:hover,
    .gradio-container .controls .icon:focus,
    .gradio-container .controls .action:hover,
    .gradio-container .controls .action:focus,
    .gradio-container .controls .playback:hover,
    .gradio-container .controls .playback:focus,
    .gradio-container .controls .text-button:hover,
    .gradio-container .controls .text-button:focus {
        color: var(--celune-primary, #cebaff) !important;
        border-color: var(--celune-primary, #cebaff) !important;
    }

    .gradio-container .standard-player input[type="range"],
    .gradio-container .minimal-audio-player input[type="range"] {
        accent-color: var(--celune-primary, #cebaff) !important;
    }

    .gradio-container .standard-player input[type="range"]::-webkit-slider-thumb,
    .gradio-container .minimal-audio-player input[type="range"]::-webkit-slider-thumb {
        background-color: var(--celune-primary, #cebaff) !important;
    }

    .gradio-container .standard-player input[type="range"]::-moz-range-thumb,
    .gradio-container .minimal-audio-player input[type="range"]::-moz-range-thumb {
        background-color: var(--celune-primary, #cebaff) !important;
    }

    .toast-body.error {
        position: fixed;
        background: color-mix(
            in srgb,
            var(--celune-background, #1d1826) 90%,
            transparent 10%
        ) !important;
        border: none;
        top: 0;
    }

    .toast-header {
        display: none !important;
    }

    .toast-messages {
        height: 100vh;
        width: 100vw;
        place-items: center;
        justify-content: center;
    }

    .toast-message-text.error {
        font-size: 0;
    }

    .toast-message-text.error::before {
        content: __CELUNE_CONNECTION_LOST_MESSAGE__;
        font-size: 16px;
        color: var(--celune-error, #f07178);
    }

    input[type="number"] {
        -webkit-appearance: textfield !important;
        appearance: textfield !important;
    }

    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        appearance: none;
        margin: 0;
    }

    input[type="radio"][aria-checked="false"][disabled] {
        background: color-mix(
            var(--celune-background) 90%,
            var(--celune-primary) 10%
        );
    }

    input[type="range"]::-moz-range-thumb {
        background: var(--celune-primary) !important;
        border-color: var(--celune-primary) !important;
    }

    input[type="range"]::-moz-range-track {
        background: var(--celune-accent) !important;
    }

    input[type="range"]::-webkit-slider-runnable-track {
        background: var(--celune-accent) !important;
    }

    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        background: var(--celune-primary) !important;
        border: none !important;
    }

    input[type="range"] {
        -webkit-appearance: none;
        appearance: none;
    }

    @media (max-width: 768px), (any-pointer: coarse), (hover: none) {
        .gradio-container {
            height: 100dvh;
            overflow: hidden;
        }

        #celune-shell {
            height: calc(100dvh - 8rem);
            min-height: 0;
        }

        #celune-input-row {
            flex-direction: column;
        }

        #celune-actions {
            display: flex;
            flex-direction: row;
            width: 100%;
            flex-wrap: nowrap;
            gap: 0;
        }

        #celune-actions > * {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }

        button#celune-send {
            width: 100%;
        }

        button#celune-send {
            display: flex;
        }

        #celune-input textarea, #celune-input textarea::placeholder {
            text-align: center;
        }

        #celune-log-panel {
            max-height: min(52dvh, calc(100dvh - 12rem));
        }

        #celune-log-panel pre {
            max-height: min(calc(52dvh - 2em), calc(100dvh - 14rem));
        }

        button#celune-send {
            border-radius: 4px;
        }

        .webui-desktop-only {
            display: none !important;
        }

        .webui-mobile-only {
            display: inline !important;
        }
    }
    """
).replace(
    "__CELUNE_CONNECTION_LOST_MESSAGE__",
    json.dumps(string("webui.connection_lost"), ensure_ascii=False),
)


@dataclass
class SpeechJob:
    """In-memory state for an accepted speech request."""

    status: TaskStatus
    created_at: float
    audio: Optional[bytes] = None
    error: Optional[str] = None
    events: deque[TaskEvent] = field(
        default_factory=lambda: deque(maxlen=256),
        repr=False,
    )
    subscriptions: list[TaskSubscription] = field(default_factory=list, repr=False)


def _configure_webui_theme() -> None:
    """Sync the browser UI palette with the active CEVOICE-derived theme."""
    global webui_theme_style

    colors.configure_theme()
    loader = default_loader()
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
                colors.configure_theme(background, accent, faded_accent)

    background = colors.THEME.background or "#1d1826"
    palette = colors.SEVERITY_COLORS["celune"]
    primary = palette["info"]
    error = palette["error"]
    foreground = colors.THEME.foreground or "#ffffff"
    secondary = colors.THEME.secondary or primary
    accent = colors.THEME.accent or primary
    sleeping = palette["sleeping"]
    button_bg = colors.blend(primary, background, 0.72)
    button_hover = colors.blend(primary, background, 0.6)
    input_bg = colors.blend(primary, background, 0.78)

    webui_theme_style = (
        "<style>"
        ":root {"
        f"--celune-background: {background};"
        f"--celune-primary: {primary};"
        f"--celune-error: {error};"
        f"--celune-foreground: {foreground};"
        f"--celune-secondary: {secondary};"
        f"--celune-accent: {accent};"
        f"--celune-ui-accent: {primary};"
        f"--celune-ui-bg: {background};"
        f"--celune-ui-input-bg: {input_bg};"
        f"--celune-sleeping: {sleeping};"
        f"--celune-button-bg: {button_bg};"
        f"--celune-button-hover: {button_hover};"
        f"--celune-input-bg: {input_bg};"
        f"--celune-placeholder: {secondary};"
        "}"
        "</style>"
    )


class StartedServer(uvicorn.Server):
    """Uvicorn server that reports when socket binding actually succeeds."""

    def __init__(
        self,
        config: uvicorn.Config,
        on_started: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(config)
        self.on_started = on_started

    async def startup(self, sockets: Optional[list[socket.socket]] = None) -> None:
        """Run Uvicorn startup and report only after the server is listening.

        Args:
            sockets: A list of sockets to bind the server to.
        """
        await super().startup(sockets=sockets)
        if self.started and self.on_started is not None:
            self.on_started()
        if self.started:
            ui_resources.start_gpu_usage_worker()

    async def shutdown(self, sockets: Optional[list[socket.socket]] = None) -> None:
        """Stop API-owned async resource workers before Uvicorn shuts down."""
        ui_resources.stop_gpu_usage_worker()
        await super().shutdown(sockets=sockets)


def _is_port_in_use_error(error: OSError) -> bool:
    """Return whether an operating-system error indicates an occupied port."""
    return (
        error.errno in {errno.EADDRINUSE, 10048}
        or getattr(error, "winerror", None) == 10048
    )


def _bind_api_socket(host: str, port: int) -> socket.socket:
    """Bind the API socket before handing it to Uvicorn."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    api_socket = socket.socket(family=family)
    api_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        api_socket.bind((host, port))
    except OSError:
        api_socket.close()
        raise

    api_socket.set_inheritable(True)
    return api_socket


def _shutdown_api_for_fatal_error() -> None:
    """Stop the browser/API surface after a fatal Celune runtime failure."""
    global bound_celune
    bound_celune = None
    if current_api_server is not None:
        current_api_server.should_exit = True
        current_api_server.force_exit = True


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Normalize empty token values to ``None``."""
    if token is None:
        return None
    token = token.strip()
    return token or None


def _env_auth_token() -> Optional[str]:
    """Return the API token from the environment, if configured."""
    return _clean_token(os.getenv("CELUNE_API_TOKEN"))


def configure_api_security(
    token: Optional[str] = None,
    requests_per_minute: int = 60,
) -> None:
    """Configure API authentication and rate limiting.

    Args:
        token: A required token to send requests.
        requests_per_minute: The max amount of requests per minute the user is allowed to send.
    """
    global auth_token, rate_limit_per_minute

    auth_token = _clean_token(token) or _env_auth_token()
    rate_limit_per_minute = max(0, requests_per_minute)
    with rate_limit_lock:
        rate_limit_hits.clear()


def resolve_api_host(token: Optional[str] = None, host: Optional[str] = None) -> str:
    """Resolve the API bind host from authentication state.

    Args:
        token: The token set up with the API.
        host: The host name or address explicitly set by the user.

    Returns:
        str: The host name or address the API is using.
    """
    if host:
        return host
    configured_token = _clean_token(token) or _env_auth_token()
    if configured_token is None:
        return "127.0.0.1"
    return "0.0.0.0"


def _request_token(request: Request) -> Optional[str]:
    """Extract the bearer or app token from a request."""
    auth_header = request.headers.get("authorization", "")
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return _clean_token(request.headers.get("x-celune-token"))


def _authenticated(request: Request) -> bool:
    """Return whether the request carries the configured API token."""
    if auth_token is None:
        return True
    given = _request_token(request)
    return given is not None and compare_digest(given, auth_token)


def is_browser_ui_request(request: Request) -> bool:
    """Return whether the request targets the mounted browser UI.

    Args:
        request: Incoming HTTP request to classify.

    Returns:
        bool: ``True`` when the request path points at the mounted WebUI.
    """
    path = request.url.path.rstrip("/")
    return path == "/ui" or path.startswith("/ui/")


def _is_public_api_request(request: Request) -> bool:
    """Return whether the request is safe to serve without an API token."""
    method = request.method.upper()
    path = request.url.path.rstrip("/") or "/"

    if is_browser_ui_request(request):
        return True

    return method == "GET" and path in {
        "/",
        "/favicon.ico",
        "/v1",
        "/v1/version",
    }


def _rate_limit_key(request: Request) -> str:
    """Return the client key used for rate limiting."""
    if request.client is None:
        return "unknown"
    return request.client.host


def _rate_limited(request: Request) -> bool:
    """Return whether the request exceeds the configured rate limit."""
    if rate_limit_per_minute <= 0:
        return False

    now = time.monotonic()
    window_start = now - 60.0
    key = _rate_limit_key(request)

    with rate_limit_lock:
        hits = rate_limit_hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= rate_limit_per_minute:
            return True

        hits.append(now)
        return False


@api.middleware("http")
async def api_security(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Apply token authentication and a simple per-client rate limit.

    Args:
        request: The request that should be protected.
        call_next: What to run if security checks have passed.

    Returns:
        Response: The response returned by the protected route or security layer.
    """
    if _is_public_api_request(request):
        return await call_next(request)

    if not _authenticated(request):
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": string("api.unauthorized"),
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _rate_limited(request):
        current_time = datetime.datetime.now(datetime.UTC)
        next_minute = current_time.replace(
            second=0, microsecond=0
        ) + datetime.timedelta(minutes=1)
        retry_after = (next_minute - current_time).total_seconds()

        return JSONResponse(
            status_code=429,
            content={
                "error": "ratelimit_exceeded",
                "message": string("api.rate_limit"),
            },
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


def bind_celune(celune: Celune) -> None:
    """Bind the running Celune instance to API routes.

    Args:
        celune: The instance of Celune to bind.
    """
    global bound_celune
    bound_celune = celune
    global webui_resource_page, webui_last_resource_advance, webui_last_probed_state
    global webui_input_locked, webui_input_placeholder, webui_voice_locked
    global webui_logs_seeded, webui_active_theme_name
    global webui_timed_update_sequence, webui_timed_update_received_at
    global webui_timed_update_source
    global webui_caption_text, webui_caption_progress, webui_caption_active
    global webui_progress_current, webui_progress_total
    webui_resource_page = 0
    webui_last_resource_advance = 0.0
    webui_last_probed_state = None
    webui_log_lines.clear()
    webui_logs_seeded = False
    webui_active_theme_name = "celune"
    webui_caption_text = ""
    webui_caption_progress = 0.0
    webui_caption_active = False
    webui_progress_current = None
    webui_progress_total = None
    webui_timed_update_sequence = 0
    webui_timed_update_received_at = 0.0
    webui_timed_update_source = "fallback"
    _unsubscribe_webui_events()
    _subscribe_webui_events(celune)
    _subscribe_webui_timed_updates()
    _configure_webui_theme()
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    webui_input_locked = celune.locked or not has_voice
    webui_input_placeholder = _webui_input_placeholder(
        celune,
        webui_input_locked,
        has_voice,
    )
    webui_voice_locked = (
        len(celune.voices) < 2 or celune.is_in_tutorial or not has_voice
    )
    _seed_webui_logs()
    _wrap_celune_callbacks(celune)
    if celune.current_voice:
        _append_webui_log(string("webui.voice_ready", voice=celune.current_voice))
    _set_webui_status(
        string("status.idle")
        if celune.cur_state == "idle"
        else celune.cur_state.title(),
        source="probe",
    )


def _unsubscribe_webui_events() -> None:
    """Remove event subscriptions owned by the browser UI bridge."""
    global webui_event_dispatcher, webui_event_callbacks
    if webui_event_dispatcher is not None:
        for event_name, callback in webui_event_callbacks:
            webui_event_dispatcher.unsubscribe(event_name, callback)
    webui_event_dispatcher = None
    webui_event_callbacks = ()


class _WebUiInputProxy:
    """Minimal input surface required by the shared slash-command handler."""

    @staticmethod
    def load_text(_value: str) -> None:
        """Discard command input after the browser has submitted it."""


class _WebUiCommandHost:
    """Core-backed command host used when the Textual UI is not mounted."""

    def __init__(self, celune: Celune) -> None:
        self.celune = celune
        self.input_box = _WebUiInputProxy()
        self.consume_on_boundary = False
        self.tutorial_token = 0

    @property
    def tutorial_active(self) -> bool:
        """Return whether the core is currently in tutorial mode."""
        return bool(getattr(self.celune, "is_in_tutorial", False))

    def safe_log(
        self,
        message: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        """Forward command output to both WebUI state and active task logs."""
        _publish_active_task_log(message, severity)
        _append_webui_log(message, severity)
        _ = loglevel

    def safe_status(self, message: str, severity: str = "info") -> None:
        """Forward a command status to the browser footer."""
        _set_webui_status(message, severity, source="callback")

    @staticmethod
    def call_from_thread(
        callback: Callable[..., None], *args: object, **kwargs: object
    ) -> None:
        """Run a command callback immediately in the API worker context."""
        callback(*args, **kwargs)

    def refresh_vc_controls(self) -> None:
        """Refresh browser controls on the next snapshot."""

    def set_vc_f0_condition(self, enabled: bool) -> None:
        """Set VC talk or sing conditioning through the core state."""
        self.celune.vc_f0_condition = enabled
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and available("f0_condition", obj=backend):
            backend.f0_condition = enabled

    def set_vc_pitch_shift(self, value: int) -> None:
        """Set the active VC pitch shift through the core state."""
        from .vc import clamp_vc_pitch_shift

        clamped = clamp_vc_pitch_shift(value)
        self.celune.vc_pitch_shift = clamped
        backend = getattr(self.celune, "vc_backend", None)
        if backend is not None and available("pitch_shift", obj=backend):
            backend.pitch_shift = clamped

    def open_settings_menu(self) -> None:
        """Report that configuration editing belongs to the Textual UI."""
        self.safe_log(string("commands.settings_unavailable"), "warning")

    def begin_tutorial(self) -> None:
        """Leave tutorial lifecycle ownership to the core runtime."""

    def finish_tutorial(self) -> None:
        """Leave tutorial lifecycle ownership to the core runtime."""

    def cancel_tutorial(self, _restore_input: bool = False) -> bool:
        """Return whether the core was already outside tutorial mode."""
        return not self.tutorial_active

    def tutorial_after(self, _delay: float, callback: Callable[[], None]) -> None:
        """Run a command tutorial callback without a second timer source."""
        callback()

    def type_and_send(self, text: str, process_commands: bool = True) -> None:
        """Submit a tutorial string through the browser command path."""
        if process_commands and text.startswith("/"):
            _webui_run_command(text)
        else:
            self.celune.say(text)

    @staticmethod
    def pulse_border(_selector: str) -> None:
        """Ignore a Textual-only tutorial animation in the browser host."""

    def graceful_exit(self) -> None:
        """Close the bound runtime when the browser receives `/exit`."""
        self.celune.close()


configure_webui_theme = _configure_webui_theme
webui_theme_html = _webui_theme_html
strip_webui_log_prefix = _strip_webui_log_prefix
set_webui_status = _set_webui_status
wrap_celune_callbacks = _wrap_celune_callbacks
speech_job_snapshot = _speech_job_snapshot
webui_snapshot = _webui_snapshot
webui_speak = _webui_speak


class RootResponse(BaseModel):
    """Response returned by the API root endpoint."""

    status: str


class VersionResponse(BaseModel):
    """Response returned by the API version endpoint."""

    version: str


class SpeakRequest(BaseModel):
    """Request body for asking the app to speak."""

    content: str = Field(min_length=1)
    save: bool = True


class ThinkRequest(BaseModel):
    """Request body for asking the app to think and reply."""

    content: str = Field(min_length=1)


class VoiceRequest(BaseModel):
    """Request body for changing the active voice."""

    voice_name: str = Field(min_length=1)


class ActionResponse(BaseModel):
    """Generic accepted control response."""

    status: str


class TaskCancelResponse(BaseModel):
    """Response returned after an explicit speech-task cancellation request."""

    task_id: str
    status: Literal["cancelled"] = "cancelled"


async def _cancel_speech_job(job_id: str) -> bool:
    """Ask Core to stop one active speech task and publish its terminal event."""
    status = _task_status(job_id)
    if status is None or status in {"completed", "failed", "cancelled"}:
        return False

    celune = require_celune()
    # noinspection PyBroadException
    try:
        stopped = await celune.force_stop_speech_async()
    except Exception as error:
        celune.log(
            format_error_message(
                string("api.speech_cancel_failed"),
                error,
                getattr(celune, "log_level", "info"),
            ),
            "warning",
        )
        return False
    if not stopped:
        return False

    _update_speech_job(job_id, status="cancelled")
    _publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="cancelled",
            status="cancelled",
        ),
    )
    _set_active_speech_task(None)
    return True


def _websocket_authenticated(websocket: WebSocket) -> bool:
    """Return whether a WebSocket carries the configured API token."""
    if auth_token is None:
        return True

    auth_header = websocket.headers.get("authorization", "")
    scheme, _, value = auth_header.partition(" ")
    given = value.strip() if scheme.lower() == "bearer" and value else None
    if given is None:
        given = _clean_token(websocket.headers.get("x-celune-token"))
    if given is None:
        given = _clean_token(websocket.query_params.get("token"))
    return given is not None and compare_digest(given, auth_token)


async def _send_task_events(
    websocket: WebSocket,
    subscription: TaskSubscription,
) -> None:
    """Send retained and live task events until a terminal event is observed."""
    try:
        while True:
            event = await subscription.next_event()
            await websocket.send_json(event.model_dump(mode="json", exclude_none=True))
            if event.event in {"completed", "failed", "cancelled"}:
                return
    except (WebSocketDisconnect, RuntimeError, TaskSubscriptionClosed):
        return


async def _receive_task_commands(websocket: WebSocket, job_id: str) -> None:
    """Receive API-layer commands without taking ownership of task execution."""
    try:
        while True:
            payload = await websocket.receive_json()
            try:
                command = TaskCommand.model_validate(payload)
            except (TypeError, ValueError):
                response = TaskCommandResponse(
                    task_id=job_id,
                    accepted=False,
                    status="invalid_command",
                )
            else:
                accepted = await _cancel_speech_job(job_id)
                response = TaskCommandResponse(
                    task_id=job_id,
                    command=command.command,
                    accepted=accepted,
                    status="cancelled" if accepted else "not_cancelled",
                )
            await websocket.send_json(
                response.model_dump(mode="json", exclude_none=True)
            )
    except (WebSocketDisconnect, RuntimeError):
        return


@api.websocket("/v1/ws/tasks/{job_id}")
async def speech_task_websocket(websocket: WebSocket, job_id: str) -> None:
    """Stream one accepted speech task without owning its Core execution.

    Args:
        websocket: The client WebSocket connection.
        job_id: The speech task ID to stream.
    """
    if not _websocket_authenticated(websocket):
        await websocket.close(code=4401)
        return

    subscription = _subscribe_to_speech_job(job_id)
    if subscription is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    sender = asyncio.create_task(_send_task_events(websocket, subscription))
    receiver = asyncio.create_task(_receive_task_commands(websocket, job_id))
    try:
        done, pending = await asyncio.wait(
            {sender, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await task
    finally:
        _unsubscribe_from_speech_job(job_id, subscription)
        if websocket.client_state.name != "DISCONNECTED":
            with contextlib.suppress(RuntimeError):
                await websocket.close()


@api.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Favicon endpoint.

    Returns:
        FileResponse: The app favicon file.
    """

    return FileResponse(
        project_root() / "resources" / "celune.ico",
        media_type="image/x-icon",
    )


@api.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the API root to Celune's browser UI.

    Returns:
        RedirectResponse: Redirect response pointing at the mounted WebUI.
    """
    return RedirectResponse(url="/ui")


@api.get("/v1", response_model=RootResponse)
def api_root() -> RootResponse:
    """API root endpoint.

    Returns:
        RootResponse: The response with Celune's underlying state.
    """
    try:
        celune = require_celune()
        return RootResponse(status=celune.cur_state)
    except HTTPException:
        return RootResponse(status="error")


@api.get("/v1/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """API version endpoint.

    Returns:
        VersionResponse: The underlying app version the API is connected to.
    """
    return VersionResponse(version=f"{APP_NAME} {__version__}")


@api.post("/v1/speak", response_model=None)
def speak(body: SpeakRequest) -> Union[StreamingResponse, JSONResponse]:
    """Queue speech and stream generated audio chunks back to the caller.

    Args:
        body: A speech request body.

    Returns:
        Union[StreamingResponse, JSONResponse]: The corresponding audio stream, or a JSON error payload if generation
        failed.
    """
    celune = require_celune()
    api_log("SPEAK(SYNC)", body.content)
    chunks = celune.say_stream(body.content, save=body.save)
    if chunks is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("webui.busy_try_again"),
            },
        )

    return StreamingResponse(
        audio_bytes(chunks),
        media_type="audio/flac",
        headers=stream_headers(),
    )


@api.post("/v1/speak/async", response_model=None)
def speak_async(body: SpeakRequest) -> JSONResponse:
    """Queue speech, return immediately, and expose the eventual result as a job.

    Args:
        body: A speech request body.

    Returns:
        JSONResponse: A 202 response with the created job ID, or an error payload.

    Raises:
        Exception: If speech-job setup or queueing fails unexpectedly.
    """
    celune = require_celune()
    api_log("SPEAK(ASYNC)", body.content)
    job_id = uuid.uuid4().hex
    location = f"/v1/speak/jobs/{job_id}"
    _remember_speech_job(job_id, SpeechJob(status="queued", created_at=time.time()))
    _set_active_speech_task(job_id)
    _update_speech_job(job_id, status="running")
    _publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="started",
            status="running",
        ),
    )

    try:
        chunks = celune.say_stream(body.content, save=body.save)
    except Exception:
        _set_active_speech_task(None)
        _forget_speech_job(job_id)
        raise
    if chunks is None:
        _set_active_speech_task(None)
        _forget_speech_job(job_id)
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("webui.busy_try_again"),
            },
        )

    threading.Thread(
        target=_collect_speech_job,
        args=(job_id, chunks),
        daemon=True,
        name=f"{APP_NAME}SpeechJob-{job_id[:8]}",
    ).start()

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "job_id": job_id, "location": location},
        headers={"Location": location},
    )


@api.post("/v1/think", response_model=None)
def think(body: ThinkRequest) -> JSONResponse:
    """Ask the app to think about an input and reply through Persona.

    Args:
        body: A think request body.

    Returns:
        JSONResponse: An accepted response when Persona processing starts, or a JSON error payload if the app cannot
        think right now.
    """
    celune = require_celune()
    api_log(
        "THINK",
        body.content
        if getattr(celune, "log_level", "info") == "debug"
        else f"[{string('api.content_protected')}]",
    )
    if not celune.think(body.content):
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("webui.busy_try_again"),
            },
        )

    return JSONResponse(status_code=202, content={"status": "accepted"})


@api.get("/v1/speak/jobs/{job_id}", response_model=None)
def speak_job(job_id: str) -> Union[Response, JSONResponse]:
    """Return speech job status or the completed FLAC audio payload.

    Args:
        job_id: The speech job ID returned by ``/v1/speak/async``.

    Returns:
        Union[Response, JSONResponse]: A pending/error status payload, or audio.
    """
    job = _speech_job_snapshot(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": string("api.speech_job_unknown"),
            },
        )

    if job.status != "completed":
        status_code = 500 if job.status == "failed" else 202
        content = {"status": job.status, "job_id": job_id}
        if job.error is not None:
            content["error"] = job.error
        return JSONResponse(status_code=status_code, content=content)

    return Response(
        content=job.audio or _flac_bytes(np.empty((0, 2), dtype=np.float32)),
        media_type="audio/flac",
        headers=stream_headers(),
    )


@api.post(
    "/v1/speak/jobs/{job_id}/cancel",
    response_model=None,
)
async def cancel_speech_job(job_id: str) -> Union[TaskCancelResponse, JSONResponse]:
    """Request explicit cancellation of one accepted speech task.

    Args:
        job_id: The speech task ID returned by ``/v1/speak/async``.

    Returns:
        Union[TaskCancelResponse, JSONResponse]: The cancellation result or an API error payload.
    """
    if _task_status(job_id) is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": string("api.speech_job_unknown"),
            },
        )

    if await _cancel_speech_job(job_id):
        return TaskCancelResponse(task_id=job_id)

    return JSONResponse(
        status_code=409,
        content={
            "error": "not_ready",
            "message": string("webui.busy_try_again"),
        },
    )


@api.post("/v1/voice", response_model=ActionResponse)
async def voice(body: VoiceRequest) -> Union[ActionResponse, JSONResponse]:
    """Change the active voice.

    Args:
        body: A voice change request body.

    Returns:
        Union[ActionResponse, JSONResponse]: The voice change response, or a JSON error payload if the voice change
        failed.
    """
    celune = require_celune()
    api_log("VOICE", body.voice_name)

    if body.voice_name not in celune.voices:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_value",
                "message": string("api.invalid_voice"),
            },
        )

    if not await celune.set_voice_async(body.voice_name):
        return JSONResponse(
            status_code=500,
            content={
                "error": "request_failed",
                "message": string("webui.cannot_change_voice_right_now"),
            },
        )

    return ActionResponse(status="ok")


@api.post("/v1/sfx", response_model=None)
async def sfx(
    file: UploadFile = File(...),  # noqa: B008
    keep: bool = Form(True),
) -> Union[StreamingResponse, JSONResponse]:
    """Play an uploaded sound effect file and stream the audio chunks back to the caller.

    Args:
        file: The sound effect file to use with the request.
        keep: Whether the app should hold this sound effect until the next utterance.

    Returns:
        Union[StreamingResponse, JSONResponse]: The corresponding audio stream, or a JSON error payload if playback
        failed.
    """
    celune = require_celune()
    filename = file.filename or f"sfx_{uuid.uuid4()}"
    api_log("SFX", filename, f" (keep={keep})")

    data = await file.read(max_sfx_upload_bytes + 1)
    if len(data) > max_sfx_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": "request_too_large",
                "message": string("api.sound_too_large"),
            },
        )

    # noinspection PyBroadException
    try:
        audio, sr = _decode_uploaded_audio(data)
        audio = resample_audio(audio, sr)
    except Exception as error:
        celune.log(
            format_error_message(
                string("api.invalid_input"),
                error,
                getattr(celune, "log_level", "info"),
            ),
            "warning",
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_audio",
                "message": string("api.invalid_input"),
            },
        )

    if not await run_in_threadpool(
        celune.play_audio, audio, BASE_SR, label=filename, keep=keep
    ):
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("api.cannot_play_now"),
            },
        )

    def chunks() -> Iterator[bytes]:
        yield _flac_bytes(audio)

    return StreamingResponse(
        chunks(),
        media_type="audio/flac",
        headers=stream_headers(),
    )


@api.post("/v1/convert", response_model=None)
async def convert_audio(
    file: UploadFile = File(...),  # noqa: B008
    pitch_shift: Optional[int] = Form(None),
    f0_condition: Optional[bool] = Form(None),
) -> Union[StreamingResponse, JSONResponse]:
    """Convert an uploaded source audio file through the active VC backend.

    Args:
        file: The uploaded source audio file to convert.
        pitch_shift: Optional semitone adjustment applied for this conversion only.
        f0_condition: Optional override enabling singing mode pitch conditioning.

    Returns:
        Union[StreamingResponse, JSONResponse]: The converted audio stream, or a JSON error payload if conversion
        failed.
    """
    celune = require_celune()
    if not _webui_vc_mode_active(celune):
        return _voice_conversion_unavailable_response()

    filename = file.filename or f"convert_{uuid.uuid4()}"
    api_log("CONVERT", filename)

    data = await file.read(max_sfx_upload_bytes + 1)
    if len(data) > max_sfx_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": "request_too_large",
                "message": string("api.source_audio_too_large"),
            },
        )

    # noinspection PyBroadException
    try:
        audio, sample_rate = _decode_uploaded_audio(data)
    except Exception as error:
        celune.log(
            format_error_message(
                string("api.invalid_input"),
                error,
                getattr(celune, "log_level", "info"),
            ),
            "warning",
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_audio",
                "message": string("api.invalid_input"),
            },
        )

    # noinspection PyBroadException
    try:
        output = await run_in_threadpool(
            celune.convert_audio,
            audio,
            sample_rate,
            label=filename,
            pitch_shift=pitch_shift,
            f0_condition=f0_condition,
        )
    except Exception as error:
        celune.log(
            format_error_message(
                string("api.could_not_convert"),
                error,
                getattr(celune, "log_level", "info"),
            ),
            "error",
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "request_failed",
                "message": string("api.could_not_convert"),
            },
        )

    if output is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("api.cannot_convert"),
            },
        )

    def chunks() -> Iterator[bytes]:
        prepared_audio = prepare_playback_audio(output.audio, output.sample_rate)
        yield _flac_bytes(
            prepared_audio,
            sample_rate=BASE_SR,
        )

    return StreamingResponse(
        chunks(),
        media_type="audio/flac",
        headers=stream_headers(BASE_SR),
    )


api = gr.mount_gradio_app(
    api,
    _build_webui(),
    path="/ui",
    footer_links=[],
    favicon_path=str(project_root() / "resources" / "celune.ico"),
    show_error=True,
    css=WEBUI_CSS,
    head=WEBUI_HEAD,
)


def run_api(
    celune: Optional[Celune] = None,
    host: Optional[str] = None,
    port: int = 2060,
    token: Optional[str] = None,
    requests_per_minute: int = 60,
    on_started: Optional[Callable[[str, int], None]] = None,
) -> None:
    """Start the API.

    Args:
        celune: Running app instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.
        token: Token required for API requests.
        requests_per_minute: Maximum requests allowed per client each minute.
        on_started: Callback called after the server socket is listening.
    """
    if celune is not None:
        bind_celune(celune)

    configure_api_security(token=token, requests_per_minute=requests_per_minute)
    bind_host = resolve_api_host(token=auth_token, host=host)

    def _default_started(bhost: str, bport: int) -> None:
        message = f"{APP_NAME} API has started on http://{bhost}:{bport}"
        if celune is not None:
            celune.log(message)
        else:
            print(message, flush=True)

    started_callback = on_started or _default_started
    config = uvicorn.Config(
        api,
        host=bind_host,
        port=port,
        log_level="warning",
    )
    server = StartedServer(
        config,
        on_started=lambda: started_callback(bind_host, port),
    )
    api_socket = _bind_api_socket(bind_host, port)
    global current_api_server
    current_api_server = server
    try:
        server.run(sockets=[api_socket])
    finally:
        api_socket.close()
        current_api_server = None


def start_api(
    celune: Celune,
    host: Optional[str] = None,
    port: int = 2060,
    token: Optional[str] = None,
    requests_per_minute: int = 60,
    startup_timeout: float = 5.0,
) -> threading.Thread:
    """Start the API in a background thread.

    Args:
        celune: Running app instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.
        token: Token required for API requests.
        requests_per_minute: Maximum requests allowed per client each minute.
        startup_timeout: Seconds to wait for startup confirmation.

    Returns:
        threading.Thread: The daemon thread running the API server.
    """

    started = threading.Event()
    failed = threading.Event()

    def _started(bind_host: str, bind_port: int) -> None:
        celune.log(f"{APP_NAME} API has started on http://{bind_host}:{bind_port}")
        started.set()

    def _runner() -> None:
        bind_host = resolve_api_host(token=token, host=host)
        try:
            run_api(
                celune,
                host=bind_host,
                port=port,
                token=token,
                requests_per_minute=requests_per_minute,
                on_started=_started,
            )
        except SystemExit as exc:
            if exc.code not in (0, None):
                failed.set()
                celune.log(
                    f"API runner has exited. Exit code {exc.code}",
                    "warning",
                )
        except Exception as e:
            failed.set()
            if isinstance(e, OSError) and _is_port_in_use_error(e):
                celune.log(f"API port {port} is already in use.", "warning")
            else:
                celune.log(
                    format_error_message(
                        "Could not start the API",
                        e,
                        getattr(celune, "log_level", "info"),
                    ),
                    "warning",
                )

    thread = threading.Thread(target=_runner, daemon=True, name=f"{APP_NAME}API")
    thread.start()
    deadline = time.monotonic() + max(0.0, startup_timeout)
    while not started.is_set() and not failed.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)

    if not started.is_set() and not failed.is_set():
        celune.log(
            f"API runner has not responded after {startup_timeout:.1f}s, "
            "and has timed out.",
            "warning",
        )

    return thread


webui.install(globals())
webactions.install(globals())
