# SPDX-License-Identifier: Apache-2.0
"""API layer."""

import asyncio
import contextlib
import datetime
import errno
import io
import inspect
import json
import os
import queue
import socket
import textwrap
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from hmac import compare_digest
from html import escape
from typing import (
    Literal,
    Optional,
    Union,
    cast,
)

import gradio as gr
import numpy as np
import numpy.typing as npt
import soundfile as sf
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from . import __version__, colors
from .celune import Celune
from .cevoice import default_loader
from .constants import APP_NAME, BASE_SR
from .dsp import resample_audio
from .exceptions import TaskSubscriptionClosed
from .i18n import string
from .paths import main_window_log_path, project_root
from .pipeline import (
    SpeechStreamQueue,
    current_playback_status,
    prepare_playback_audio,
)
from .typing.aliases import AudioChunk, AudioChunks, LogLevel
from .typing.api import (
    TaskCommandName,
    TaskEventName,
    TaskStatus,
    WebUiAudioValue,
    WebUiInputAudioValue,
    WebUiUpdate,
)
from .typing.common import JSONSerializable
from .ui import resources as ui_resources
from .ui.app import CeluneUI
from .utils import format_error
from .vc import VC_PITCH_SHIFT_MAX, VC_PITCH_SHIFT_MIN

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
webui_resource_page = 0
webui_last_resource_advance = 0.0
webui_last_probed_state: Optional[str] = None


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
webui_input_placeholder = string("webui.wait_placeholder")
webui_voice_locked = True
webui_theme_style = ""
webui_status_source = "probe"
webui_status_updated_at = 0.0
current_api_server: Optional["StartedServer"] = None
WEBUI_RESOURCE_ROTATE_SECONDS = 2.06
WEBUI_POLL_INTERVAL_SECONDS = WEBUI_RESOURCE_ROTATE_SECONDS / 4
WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS = 0.9


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
    """Thread-safe event queue owned by one WebSocket connection."""

    events: queue.Queue[TaskEvent] = field(default_factory=queue.Queue)
    closed: threading.Event = field(default_factory=threading.Event)

    def put(self, event: TaskEvent) -> None:
        """Queue one event for the subscribed WebSocket.

        Args:
            event: The typed event to deliver to the subscriber.
        """
        if not self.closed.is_set():
            self.events.put(event)

    def close(self) -> None:
        """Stop waiting for events without affecting the underlying task."""
        self.closed.set()

    def get(self) -> Optional[TaskEvent]:
        """Return the next event, or ``None`` while waiting for one.

        Returns:
            Optional[TaskEvent]: The next queued event, or ``None`` after a timed wait.

        Raises:
            TaskSubscriptionClosed: If the subscription was closed.
        """
        if self.closed.is_set():
            raise TaskSubscriptionClosed
        try:
            return self.events.get(timeout=0.25)
        except queue.Empty:
            return None

    async def next_event(self) -> TaskEvent:
        """Wait asynchronously for the next event without blocking the API loop.

        Returns:
            TaskEvent: The next event published for this subscription.
        """
        while True:
            event = await asyncio.to_thread(self.get)
            if event is not None:
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


class _WebUiUnset:
    """Sentinel type for optional WebUI input updates."""


_WEBUI_UNSET = _WebUiUnset()

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

      function scrollLogToBottom() {
        const logElement = document.querySelector("#celune-log-panel pre");
        if (!logElement) {
          return;
        }
        logElement.scrollTop = logElement.scrollHeight;
      }

      function installLogObserver() {
        const logElement = document.querySelector("#celune-log-panel pre");
        if (!logElement) {
          return;
        }

        if (window.__celuneLogAutoscrollTarget === logElement) {
          scrollLogToBottom();
          return;
        }

        window.__celuneLogAutoscrollTarget = logElement;
        scrollLogToBottom();

        if (window.__celuneLogAutoscrollObserver) {
          window.__celuneLogAutoscrollObserver.disconnect();
        }

        const observer = new MutationObserver(() => {
          scrollLogToBottom();
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

    button#celune-style, button#celune-send, button#celune-convert {
        background: var(--celune-button-bg, #3a304c);
        color: var(--celune-ui-accent, var(--celune-primary, #cebaff));
        border-radius: 4px;
    }

    button#celune-style:hover, button#celune-send:hover, button#celune-convert:hover {
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

    button#celune-style,
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

        button#celune-style,
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

        button#celune-style {
            border-radius: 4px 0 0 4px;
            border-right: 1px solid color-mix(
                in srgb,
                var(--celune-ui-accent, var(--celune-primary, #cebaff)) 50%,
                black
            );
        }

        button#celune-send {
            border-radius: 0 4px 4px 0;
            border-left: 1px solid color-mix(
                in srgb,
                var(--celune-ui-accent, var(--celune-primary, #cebaff)) 50%,
                black
            );
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
    webui_resource_page = 0
    webui_last_resource_advance = 0.0
    webui_last_probed_state = None
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


def _webui_status_color(severity: str) -> str:
    """Return the browser UI color for a given severity."""
    palette = colors.SEVERITY_COLORS.get("celune", colors.SEVERITY_COLORS["celune"])
    return palette.get(severity, palette["info"])


def _webui_theme_html() -> str:
    """Render runtime CSS variables for the browser UI theme."""
    severity = "info"
    accent = _webui_status_color(severity)
    background = colors.THEME.background or colors.DEFAULT_BACKGROUND
    input_bg = colors.blend(accent, background, 0.78)
    return (
        "<style>:root {"
        f"--celune-ui-accent: {accent};"
        f"--celune-ui-bg: {background};"
        f"--celune-ui-input-bg: {input_bg};"
        "}</style>"
    )


def _webui_log_line_html(message: str, severity: str = "info") -> str:
    """Render one browser log line with severity-aware coloring."""
    color = _webui_status_color(severity)
    return f'<span style="color: {color};">{escape(message)}</span>'


def _webui_audio_waveform_options() -> gr.WaveformOptions:
    """Return theme-bound waveform colors for Gradio audio components."""
    primary = colors.SEVERITY_COLORS["celune"]["info"]
    secondary = colors.THEME.secondary or colors.FADED_ACCENT
    return gr.WaveformOptions(
        waveform_color=secondary,
        waveform_progress_color=primary,
        trim_region_color=primary,
    )


def _strip_webui_log_prefix(line: str) -> str:
    """Remove persisted timestamp and severity prefixes from one log line."""
    stripped = line.strip()
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    if stripped.startswith("[") and "] " in stripped:
        stripped = stripped.split("] ", 1)[1]
    return stripped


def _seed_webui_logs() -> None:
    """Populate the browser log view from the persisted desktop log when available."""
    global webui_logs_seeded
    if webui_logs_seeded:
        return

    webui_logs_seeded = True
    path = main_window_log_path()
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-180:]
    except OSError:
        return

    for line in lines:
        lowered = line.lower()
        severity = "info"
        if "[warning]" in lowered:
            severity = "warning"
        elif "[error]" in lowered:
            severity = "error"
        webui_log_lines.append((_strip_webui_log_prefix(line), severity))


def _append_webui_log(msg: str, severity: str = "info") -> None:
    """Store one browser log line."""
    webui_log_lines.append((msg, severity))


def _set_webui_status(
    msg: str,
    severity: str = "info",
    *,
    source: str = "callback",
    updated_at: Optional[float] = None,
) -> None:
    """Update the browser UI status line."""
    global webui_status_text, webui_status_severity
    global webui_status_source, webui_status_updated_at
    webui_status_text = msg
    webui_status_severity = severity
    webui_status_source = source
    webui_status_updated_at = time.monotonic() if updated_at is None else updated_at


def _probed_status_text(celune: Celune) -> tuple[str, str]:
    """Return the best-effort footer status derived from Celune's live state."""
    if not celune.current_voice and not celune.voices:
        return string("status.could_not_start", app_name=APP_NAME), "error"

    state = (celune.cur_state or "").strip().lower()
    return {
        "idle": (string("status.idle"), "info"),
        "speaking": (string("status.speaking"), "info"),
        "thinking": (string("status.thinking"), "info"),
        "waking": (string("status.waking_up"), "info"),
        "reloading": (string("status.reloading"), "info"),
        "sleeping": (string("status.sleeping"), "sleeping"),
        "init": (string("status.initializing"), "info"),
        "generating": (string("status.generating"), "info"),
        "error": (string("status.could_not_continue", app_name=APP_NAME), "error"),
    }.get(
        state,
        (state.title() if state else string("status.initializing"), "info"),
    )


def _probe_webui_runtime() -> None:
    """Poll the live runtime so the WebUI footer updates even without new log lines."""
    global webui_last_probed_state, webui_resource_page, webui_last_resource_advance

    celune = bound_celune
    if celune is None:
        return

    now = time.monotonic()
    current_state = (celune.cur_state or "").strip().lower()
    playback_status = current_playback_status(celune)
    if playback_status is not None:
        if webui_status_text != playback_status or webui_status_source != "playback":
            _set_webui_status(
                playback_status,
                "info",
                source="playback",
                updated_at=now,
            )
        webui_last_probed_state = current_state
    elif current_state == "sleeping":
        sleeping_log = string("webui.sleeping_log", app_name=APP_NAME)
        if not any(message == sleeping_log for message, _ in webui_log_lines):
            _append_webui_log(sleeping_log, "sleeping")
        sleeping_status, sleeping_severity = _probed_status_text(celune)
        if (
            webui_status_text != sleeping_status
            or webui_status_severity != sleeping_severity
            or webui_status_source == "callback"
        ):
            _set_webui_status(
                sleeping_status,
                sleeping_severity,
                source="probe",
                updated_at=now,
            )
    if playback_status is None and current_state != webui_last_probed_state:
        status_text, severity = _probed_status_text(celune)
        should_override_status = (
            webui_last_probed_state is None
            or webui_status_text == string("status.api_starting")
            or webui_status_source != "callback"
            or now - webui_status_updated_at >= WEBUI_STATUS_PROBE_DEBOUNCE_SECONDS
            or current_state in {"idle", "sleeping", "error"}
        )
        if should_override_status:
            _set_webui_status(
                status_text,
                severity,
                source="probe",
                updated_at=now,
            )
        webui_last_probed_state = current_state

    pages = ui_resources.resource_pages(celune, "celune")
    if not pages:
        return

    if webui_last_resource_advance <= 0:
        webui_last_resource_advance = now
        return

    if now - webui_last_resource_advance >= WEBUI_RESOURCE_ROTATE_SECONDS:
        webui_resource_page = (webui_resource_page + 1) % len(pages)
        webui_last_resource_advance = now


def _wrap_celune_callbacks(celune: Celune) -> None:
    """Mirror Celune callbacks into browser UI state without replacing existing handlers."""
    if getattr(celune, "_webui_callbacks_wrapped", False):
        return

    original_log = celune.log_callback
    original_status = celune.status_callback
    original_error = cast(
        Callable[[str], None],
        getattr(celune, "error_callback", lambda _message: None),
    )
    original_progress = cast(
        Callable[[Optional[float], Optional[float]], None],
        getattr(celune, "progress_callback", lambda _progress, _total: None),
    )
    original_voice_changed = celune.voice_changed_callback
    original_input_state = celune.change_input_state_callback
    original_voice_lock_state = celune.change_voice_lock_state_callback

    def wrapped_log(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        _publish_active_task_log(msg, severity)
        _append_webui_log(msg, severity)
        _invoke_message_callback(original_log, msg, severity, loglevel)

    def wrapped_status(
        msg: str,
        severity: str = "info",
        *,
        loglevel: LogLevel = "info",
    ) -> None:
        _publish_active_task_log(msg, severity)
        _set_webui_status(msg, severity, source="callback")
        _invoke_message_callback(original_status, msg, severity, loglevel)

    def wrapped_error(msg: str) -> None:
        _publish_active_task_log(
            string("status.could_not_continue", app_name=APP_NAME),
            "error",
        )
        original_error(msg)

    def wrapped_progress(
        progress: Optional[float],
        total: Optional[float],
    ) -> None:
        _publish_active_task_progress(progress, total)
        original_progress(progress, total)

    def wrapped_voice_changed(name: str) -> None:
        _append_webui_log(string("webui.voice_changed", voice=name))
        original_voice_changed(name)

    def wrapped_input_state(locked: bool) -> None:
        global webui_input_locked, webui_input_placeholder
        has_voice = bool(celune.current_voice) or bool(celune.voices)
        webui_input_locked = locked or not has_voice
        webui_input_placeholder = _webui_input_placeholder(
            celune,
            webui_input_locked,
            has_voice,
        )
        original_input_state(locked)

    def wrapped_voice_lock_state(locked: bool) -> None:
        global webui_voice_locked
        webui_voice_locked = (
            locked
            or len(celune.voices) < 2
            or not bool(celune.current_voice or celune.voices)
        )
        original_voice_lock_state(locked)

    glow = getattr(celune, "glow", None)
    if glow is not None and hasattr(glow, "fatal"):
        original_fatal = glow.fatal

        def wrapped_fatal() -> None:
            original_fatal()
            _shutdown_api_for_fatal_error()

        glow.fatal = wrapped_fatal

    celune.log_callback = wrapped_log
    celune.status_callback = wrapped_status
    celune.error_callback = wrapped_error
    celune.progress_callback = wrapped_progress
    celune.voice_changed_callback = wrapped_voice_changed
    celune.change_input_state_callback = wrapped_input_state
    celune.change_voice_lock_state_callback = wrapped_voice_lock_state
    celune._webui_callbacks_wrapped = True


def require_celune() -> Celune:
    """Return the bound Celune instance or fail the request.

    Returns:
        Celune: The bound Celune instance set for the request.

    Raises:
        HTTPException: The user has requested an API route that required Celune, but Celune wasn't available.
    """
    if bound_celune is None:
        raise HTTPException(
            status_code=503,
            detail=string("webui.not_available"),
        )
    return bound_celune


def api_log(action: str, content: str, suffix: str = "") -> None:
    """Print the API control log line.

    Args:
        action: The request made by the user.
        content: The request body sent by the user.
        suffix: The suffix to append to the log line.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    preview = content.replace("\n", "\\n").replace("\r", "\\r")[:64]
    if len(content) > 64:
        preview += "..."
    _append_webui_log(f"{action} {preview!r}{suffix}")
    try:
        print(f"[{timestamp}] {action} {preview!r}{suffix}", flush=True)
    except ValueError:
        # Some embedded launch paths can close stdout while the WebUI stays alive.
        pass


def _normalized_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return stereo audio in frame-major form for file encoding."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[0] == 2 and normalized.shape[1] != 2:
        return normalized.transpose()
    return normalized


def _flac_bytes(audio: npt.NDArray[np.float32], sample_rate: int = BASE_SR) -> bytes:
    """Encode audio as PCM24 FLAC bytes."""
    buffer = io.BytesIO()
    sf.write(
        buffer,
        _normalized_audio(audio),
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
    )
    return buffer.getvalue()


def audio_bytes(
    chunks: SpeechStreamQueue,
    on_chunk: Optional[Callable[[int], None]] = None,
) -> Iterator[bytes]:
    """Yield one FLAC payload from queued 48 kHz stereo float32 chunks.

    Args:
        chunks: A queue of audio chunks.
        on_chunk: Optional callback invoked after each audio chunk is received.

    Returns:
        Iterator[bytes]: The audio chunk from the queue as raw bytes.

    Raises:
        item: An exception class was raised, causing the stream to be interrupted.
        Exception: The stream was interrupted by Celune.
    """
    audio_chunks: AudioChunks = []
    chunk_count = 0
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        audio_chunks.append(_normalized_audio(item))
        chunk_count += 1
        if on_chunk is not None:
            on_chunk(chunk_count)

    if audio_chunks:
        yield _flac_bytes(np.concatenate(audio_chunks))
    else:
        yield _flac_bytes(np.empty((0, 2), dtype=np.float32))


def stream_headers(sample_rate: int = BASE_SR) -> dict[str, str]:
    """Return headers describing the FLAC response.

    Args:
        sample_rate: Sample rate advertised in the response headers.

    Returns:
        dict[str, str]: Response headers for a FLAC response.
    """
    return {
        "X-Audio-Format": "flac-pcm24",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "2",
    }


def _remember_speech_job(job_id: str, job: SpeechJob) -> None:
    """Store one speech job and remove expired entries."""
    with speech_jobs_lock:
        _delete_expired_speech_jobs(time.time())
        speech_jobs[job_id] = job


def _forget_speech_job(job_id: str) -> None:
    """Remove one speech job that was rejected before it could be observed."""
    with speech_jobs_lock:
        job = speech_jobs.pop(job_id, None)
        if job is not None:
            for subscription in job.subscriptions:
                subscription.close()
            job.subscriptions.clear()


def _delete_expired_speech_jobs(now: float) -> None:
    """Remove jobs older than the in-memory job TTL."""
    expired_ids = [
        job_id
        for job_id, job in speech_jobs.items()
        if now - job.created_at > speech_job_ttl_seconds
    ]
    for job_id in expired_ids:
        speech_jobs.pop(job_id, None)


def _update_speech_job(
    job_id: str,
    *,
    status: TaskStatus,
    audio: Optional[bytes] = None,
    error: Optional[str] = None,
) -> None:
    """Update one speech job if it still exists."""
    with speech_jobs_lock:
        job = speech_jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.audio = audio
        job.error = error


def _publish_task_event(job_id: str, event: TaskEvent) -> None:
    """Append one task event and fan it out to current subscriptions."""
    with speech_jobs_lock:
        job = speech_jobs.get(job_id)
        if job is None:
            return
        job.events.append(event)
        subscriptions = tuple(job.subscriptions)

    for subscription in subscriptions:
        subscription.put(event)


def _task_status(job_id: str) -> Optional[TaskStatus]:
    """Return the current task status for API event association."""
    with speech_jobs_lock:
        job = speech_jobs.get(job_id)
        return None if job is None else job.status


def _task_event_status(job_id: str) -> Optional[TaskStatus]:
    """Return a non-terminal task status suitable for callback events."""
    status = _task_status(job_id)
    if status in {"completed", "failed", "cancelled"}:
        return None
    return status


def _publish_task_progress(
    job_id: str,
    *,
    current: Optional[float] = None,
    total: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """Publish a safe progress or status update for one task."""
    status = _task_event_status(job_id)
    if status is None:
        return
    _publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="progress",
            status=status,
            current=current,
            total=total,
            message=message,
        ),
    )


def _publish_active_task_log(message: str, severity: str = "info") -> None:
    """Mirror one safe Core status callback into the active speech task."""
    task_id = active_speech_task_id
    if task_id is None:
        return
    if severity != "error" and message.startswith("["):
        return
    status = _task_event_status(task_id)
    if status is None:
        return
    safe_message = (
        string("status.could_not_continue", app_name=APP_NAME)
        if severity == "error"
        else message
    )
    _publish_task_event(
        task_id,
        TaskEvent(
            task_id=task_id,
            event="log",
            status=status,
            message=safe_message,
            severity=severity,
        ),
    )


def _publish_active_task_progress(
    current: Optional[float],
    total: Optional[float],
) -> None:
    """Mirror one Core progress callback into the active speech task."""
    task_id = active_speech_task_id
    if task_id is not None:
        _publish_task_progress(task_id, current=current, total=total)


def _set_active_speech_task(task_id: Optional[str]) -> None:
    """Set the API task receiving Core speech callbacks."""
    global active_speech_task_id
    active_speech_task_id = task_id


def _subscribe_to_speech_job(job_id: str) -> Optional[TaskSubscription]:
    """Subscribe to one speech job and replay its retained event history."""
    subscription = TaskSubscription()
    with speech_jobs_lock:
        job = speech_jobs.get(job_id)
        if job is None:
            return None
        for event in job.events:
            subscription.put(event)
        job.subscriptions.append(subscription)
    return subscription


def _unsubscribe_from_speech_job(
    job_id: str,
    subscription: TaskSubscription,
) -> None:
    """Detach one WebSocket subscription without changing task execution."""
    with speech_jobs_lock:
        job = speech_jobs.get(job_id)
        if job is not None:
            try:
                job.subscriptions.remove(subscription)
            except ValueError:
                pass
    subscription.close()


def _speech_job_snapshot(job_id: str) -> Optional[SpeechJob]:
    """Return a copy of one speech job for response handling."""
    with speech_jobs_lock:
        _delete_expired_speech_jobs(time.time())
        job = speech_jobs.get(job_id)
        if job is None:
            return None
        return SpeechJob(
            status=job.status,
            created_at=job.created_at,
            audio=job.audio,
            error=job.error,
        )


def _collect_speech_job(job_id: str, chunks: SpeechStreamQueue) -> None:
    """Consume a speech stream queue and store its final FLAC payload."""
    try:
        audio = b"".join(
            audio_bytes(
                chunks,
                on_chunk=lambda count: _publish_task_progress(
                    job_id,
                    current=float(count),
                ),
            )
        )
    except Exception as e:
        if _task_status(job_id) == "cancelled":
            _set_active_speech_task(None)
            return
        _update_speech_job(job_id, status="failed", error=str(e))
        _publish_task_event(
            job_id,
            TaskEvent(
                task_id=job_id,
                event="failed",
                status="failed",
                error="generation_failed",
            ),
        )
        _set_active_speech_task(None)
        return

    if _task_status(job_id) == "cancelled":
        _set_active_speech_task(None)
        return

    _update_speech_job(job_id, status="completed", audio=audio)
    _publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="completed",
            status="completed",
            location=f"/v1/speak/jobs/{job_id}",
        ),
    )
    _set_active_speech_task(None)


def _webui_logs_html() -> str:
    """Render the mirrored log buffer as terminal-like HTML."""
    if not webui_log_lines:
        content = _webui_log_line_html("Waiting for response...")
    else:
        content = "\n".join(
            _webui_log_line_html(line, severity) for line, severity in webui_log_lines
        )
    return f'<div id="celune-log-panel"><pre>{content}</pre></div>'


def _webui_status_html() -> str:
    """Render the footer status cell."""
    color = _webui_status_color(webui_status_severity)
    return (
        f"{_webui_theme_html()}"
        '<div class="footer-block" '
        f'style="color: {color};">{escape(webui_status_text)}</div>'
    )


def _webui_resources_html() -> str:
    """Render the footer resource cell."""
    celune = bound_celune
    resource = ""
    if celune is not None:
        pages = ui_resources.resource_pages(celune, "celune")
        if pages:
            resource = pages[webui_resource_page % len(pages)]
    if "CTRL+" in resource:
        return (
            '<div class="footer-block">'
            f'<span class="webui-desktop-only">{escape(resource)}</span>'
            '<span class="webui-mobile-only">Use buttons for controls</span>'
            "</div>"
        )
    return f'<div class="footer-block">{escape(resource)}</div>'


def _voice_button_update() -> WebUiUpdate:
    """Return the current browser voice-button state."""
    celune = bound_celune
    if celune is None:
        return gr.update(value="Loading", interactive=False)

    has_voice = bool(celune.current_voice) or bool(celune.voices)
    voice_name = celune.current_voice or (
        celune.voices[0] if celune.voices else "No Voice Set"
    )
    interactive = (
        not webui_voice_locked and len(celune.voices) >= 2 and has_voice
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else len(celune.voices) >= 2 and not celune.is_in_tutorial
    )
    return gr.update(
        value=voice_name.capitalize(),
        interactive=interactive,
    )


def _webui_vc_mode_active(celune: Optional[Celune]) -> bool:
    """Return whether the browser UI should expose active VC controls."""
    return bool(
        celune is not None
        and getattr(celune, "input_mode", "text_to_speech") == "voice_conversion"
    )


def _webui_input_placeholder(
    celune: Celune,
    locked: bool,
    has_voice: bool,
) -> str:
    """Return the current browser input placeholder string."""
    if celune.is_in_tutorial:
        return string("webui.tutorial_placeholder")
    if locked or not has_voice:
        return string("webui.wait_placeholder")
    if _webui_vc_mode_active(celune):
        return string("webui.voice_changer_placeholder")
    return string("webui.input_placeholder")


def _input_update(
    value: Union[Optional[str], _WebUiUnset] = _WEBUI_UNSET,
) -> WebUiUpdate:
    """Return the current browser input state."""
    has_value = value is not _WEBUI_UNSET
    celune = bound_celune
    if celune is None:
        if has_value:
            return gr.update(
                value=value,
                interactive=False,
                placeholder=string("webui.wait_placeholder"),
            )
        return gr.update(
            interactive=False,
            placeholder=string("webui.wait_placeholder"),
        )
    if celune.is_in_tutorial:
        if has_value:
            return gr.update(
                value=value,
                interactive=False,
                placeholder=string("webui.tutorial_placeholder"),
            )
        return gr.update(
            interactive=False,
            placeholder=string("webui.tutorial_placeholder"),
        )
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    if getattr(celune, "_webui_callbacks_wrapped", False):
        interactive = not webui_input_locked and has_voice
        placeholder = _webui_input_placeholder(celune, webui_input_locked, has_voice)
    else:
        interactive = not celune.locked and has_voice
        placeholder = _webui_input_placeholder(celune, celune.locked, has_voice)
    if has_value:
        return gr.update(
            value=value,
            interactive=interactive,
            placeholder=placeholder,
        )
    return gr.update(
        interactive=interactive,
        placeholder=placeholder,
    )


def _send_button_update() -> WebUiUpdate:
    """Return the current browser send-button state."""
    celune = bound_celune
    if celune is None:
        return gr.update(interactive=False)
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    interactive = (
        not webui_input_locked and has_voice
        if getattr(celune, "_webui_callbacks_wrapped", False)
        else not celune.is_in_tutorial and not celune.locked and has_voice
    )
    return gr.update(interactive=interactive)


def _webui_vc_controls_update() -> tuple[
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser VC control state."""
    celune = bound_celune
    vc_enabled = _webui_vc_mode_active(celune)
    control_interactive = celune is not None and vc_enabled
    return (
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
        gr.update(interactive=control_interactive),
    )


def _webui_snapshot() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return the current browser UI snapshot."""
    _seed_webui_logs()
    _probe_webui_runtime()
    return (
        _webui_logs_html(),
        _webui_status_html(),
        _webui_resources_html(),
        _voice_button_update(),
        _send_button_update(),
        _input_update(),
    )


def _webui_submit_snapshot(
    input_value: Optional[str],
) -> tuple[
    WebUiUpdate,
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Return a browser snapshot shaped for submit/click handlers."""
    (
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
        _input,
    ) = _webui_snapshot()
    return (
        _input_update(input_value),
        logs_html,
        status_html,
        resources_html,
        voice_update,
        send_update,
    )


def _webui_run_command(text: str) -> bool:
    """Run one slash command through the main UI command path when available."""
    # noinspection PyProtectedMember
    ui = CeluneUI._instance
    if ui is None:
        _append_webui_log(
            string("webui.must_be_running_for_commands", app_name=APP_NAME),
            "warning",
        )
        return False

    try:
        parts = CeluneUI.split_command_input(text[1:])
    except ValueError as e:
        _append_webui_log(string("webui.command_parsing_error", error=e), "error")
        return False

    if not parts:
        return False

    command = parts[0].lower()
    command_args = parts[1:]
    ui.call_from_thread(ui.process_command, command, command_args)
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

    celune = require_celune()
    api_log("SPEAK(WEBUI)", text)

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
            woke = bool(_run_async_runtime_call(wake_async()))
        else:
            woke = celune.wake_from_sleep()
        if not woke:
            snapshot = _webui_submit_snapshot(text)
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
    except Exception as e:
        _append_webui_log(
            string(
                "webui.error",
                error=format_error(e, getattr(celune, "log_level", "info")),
            ),
            "error",
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

    celune = require_celune()
    if getattr(celune, "input_mode", "text_to_speech") != "voice_conversion":
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
    api_log("CONVERT(WEBUI)", "uploaded audio")
    try:
        output = celune.convert_audio(
            audio,
            sample_rate,
            label="browser audio input",
            pitch_shift=round(pitch_shift),
            f0_condition=conversion_mode.strip().lower() == "sing",
        )
    except Exception as e:
        _append_webui_log(
            string(
                "webui.error",
                error=format_error(e, getattr(celune, "log_level", "info")),
            ),
            "error",
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


configure_webui_theme = _configure_webui_theme
webui_theme_html = _webui_theme_html
strip_webui_log_prefix = _strip_webui_log_prefix
set_webui_status = _set_webui_status
wrap_celune_callbacks = _wrap_celune_callbacks
speech_job_snapshot = _speech_job_snapshot
webui_snapshot = _webui_snapshot
webui_speak = _webui_speak


def _webui_cycle_voice() -> tuple[
    str,
    str,
    str,
    WebUiUpdate,
    WebUiUpdate,
    WebUiUpdate,
]:
    """Cycle to the next available Celune voice from the browser UI."""
    celune = require_celune()
    if len(celune.voices) < 2 or not bool(celune.current_voice or celune.voices):
        return _webui_snapshot()

    current_voice = celune.current_voice or celune.voices[0]
    current_index = (
        celune.voices.index(current_voice) if current_voice in celune.voices else -1
    )
    next_voice = celune.voices[(current_index + 1) % len(celune.voices)]
    api_log("VOICE(WEBUI)", next_voice)

    set_voice_async = cast(
        Optional[Callable[[str], Awaitable[bool]]],
        getattr(celune, "set_voice_async", None),
    )
    if set_voice_async is not None:
        switched = bool(_run_async_runtime_call(set_voice_async(next_voice)))
    else:
        switched = celune.set_voice_and_wait(next_voice)

    if not switched:
        _append_webui_log(string("webui.cannot_change_voice_right_now"), "error")

    return _webui_snapshot()


def _build_webui() -> gr.Blocks:
    # pylint: disable=E1101
    """Create the browser UI mounted by the API."""
    _configure_webui_theme()
    with gr.Blocks(
        title=APP_NAME,
        fill_height=True,
    ) as demo:
        gr.HTML(webui_theme_style)
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
                            placeholder=string("webui.wait_placeholder"),
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
            timer = gr.Timer(value=WEBUI_POLL_INTERVAL_SECONDS)

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
    except Exception:
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
        else string("api.content_protected"),
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
    except Exception:
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
    if getattr(celune, "input_mode", "text_to_speech") != "voice_conversion":
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
    except Exception:
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
    except Exception:
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
        message = string(
            "api.runner_started",
            app_name=APP_NAME,
            host=bhost,
            port=bport,
        )
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
        celune.log(
            string(
                "api.runner_started",
                app_name=APP_NAME,
                host=bind_host,
                port=bind_port,
            )
        )
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
                    string("api.runner_exit_code", code=exc.code),
                    "warning",
                )
        except Exception as e:
            failed.set()
            if isinstance(e, OSError) and _is_port_in_use_error(e):
                celune.log(string("api.port_in_use", port=port), "warning")
            else:
                celune.log(
                    string(
                        "api.could_not_start",
                        error=format_error(e, getattr(celune, "log_level", "info")),
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
            string("api.runner_timeout", seconds=startup_timeout),
            "warning",
        )

    return thread
