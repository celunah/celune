# SPDX-License-Identifier: Apache-2.0
"""API layer."""

import os
import html
import json
import time
import uuid
import errno
import base64
import socket
import asyncio
import inspect
import secrets
import binascii
import datetime
import tempfile
import ipaddress
import threading
import contextlib
from typing import (
    Annotated,
    Union,
    Literal,
    Optional,
)
from pathlib import Path
from hashlib import sha256
from hmac import compare_digest
from hmac import new as hmac_new
from urllib.parse import urlsplit
from dataclasses import field, dataclass
from collections import abc, deque, defaultdict

import fastapi
import uvicorn
import numpy as np
import gradio as gr
from pydantic import Field, BaseModel
from starlette import types, concurrency, datastructures
from starlette.middleware import base as middleware_base

from .mcp import (
    configure_mcp_host_allowlist,
    mcp_app,
    mcp_lifespan,
)
from .speech import (
    prepare_playback_audio,
)
from .i18n import string
from . import __version__
from .exceptions import (
    ModelConsentDeclined,
    ModelConsentRequired,
    TaskSubscriptionClosed,
    ModelCommercialUseRestricted,
)
from .theme import colors
from .typing.api import (
    TaskStatus,
    TaskEventName,
    TaskCommandName,
)
from .celune import Celune
from .webui.actions import (
    build_webui,
    webui_run_command,
    decode_uploaded_audio,
    voice_conversion_unavailable_response,
)
from .webui.runtime import (
    api_log,
    flac_bytes,
    audio_bytes,
    task_status,
    require_celune,
    stream_headers,
    seed_webui_logs,
    append_webui_log,
    set_webui_status,
    forget_speech_job,
    update_speech_job,
    collect_speech_job,
    publish_task_event,
    remember_speech_job,
    speech_job_snapshot,
    webui_vc_mode_active,
    wrap_celune_callbacks,
    set_active_speech_task,
    subscribe_webui_events,
    publish_active_task_log,
    subscribe_to_speech_job,
    get_webui_input_placeholder,
    unsubscribe_from_speech_job,
    subscribe_webui_timed_updates,
)
from .registry.consent import (
    consent_status,
    consent_details,
    registered_models,
    consent_prompt_text,
    record_model_consent,
    exit_api_model_request,
    pending_model_consents,
    commercial_mode_enabled,
    enter_api_model_request,
    model_metadata_for_revision,
)
from .cevoice import default_loader
from .webui import runtime as webui
from .typing.aliases import LogLevel
from .audio.dsp import resample_audio
from .constants import BASE_SR, APP_NAME
from .webui import actions as webactions
from .registry.types import ModelMetadata
from .ui import resources as ui_resources
from .typing.common import JSONSerializable
from .singing import validate_singing_notation
from .extensions.events import EventDispatcher
from .utils import available, format_error_message
from .typing.events import EventName, EventCallback
from .paths import api_key_path, project_root, app_data_dir


Callable = abc.Callable
Iterator = abc.Iterator
Awaitable = abc.Awaitable
File = fastapi.File
Form = fastapi.Form
FastAPI = fastapi.FastAPI
Request = fastapi.Request
WebSocket = fastapi.WebSocket
UploadFile = fastapi.UploadFile
HTTPException = fastapi.HTTPException
WebSocketDisconnect = fastapi.WebSocketDisconnect
Response = fastapi.responses.Response
FileResponse = fastapi.responses.FileResponse
HTMLResponse = fastapi.responses.HTMLResponse
JSONResponse = fastapi.responses.JSONResponse
RedirectResponse = fastapi.responses.RedirectResponse
StreamingResponse = fastapi.responses.StreamingResponse
run_in_threadpool = concurrency.run_in_threadpool
Headers = datastructures.Headers
RequestResponseEndpoint = middleware_base.RequestResponseEndpoint
Send = types.Send
Scope = types.Scope
ASGIApp = types.ASGIApp
Receive = types.Receive

receive_webui_timed_update = webui.receive_webui_timed_update
webui_audio_waveform_options = webui.webui_audio_waveform_options
input_update = webui.input_update
webui_resources_html = webui.webui_resources_html
send_button_update = webui.send_button_update
webui_status_html = webui.webui_status_html
webui_vc_controls_update = webui.webui_vc_controls_update
webui_convert_audio = webactions.webui_convert_audio
webui_cycle_voice = webactions.webui_cycle_voice
CeluneUI = webui.CeluneUI
UiTimedUpdate = webui.UiTimedUpdate

api = FastAPI(title=f"{APP_NAME}API", lifespan=mcp_lifespan)

API_KEY_PREFIX = "C-"
API_KEY_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
API_KEY_LENGTH = 25


def generate_api_key() -> str:
    """Generate a Celune API key with 25 secure base-36 characters.

    Returns:
        str: A key with the ``C-`` prefix and 25 uppercase base-36 characters.
    """
    return API_KEY_PREFIX + "".join(
        secrets.choice(API_KEY_ALPHABET) for _ in range(API_KEY_LENGTH)
    )


def _valid_generated_key(value: str) -> bool:
    """Return whether a value matches Celune's generated-key format."""
    if not value.startswith(API_KEY_PREFIX):
        return False
    suffix = value[len(API_KEY_PREFIX) :]
    return len(suffix) == API_KEY_LENGTH and all(
        character in API_KEY_ALPHABET for character in suffix
    )


def _replace_key_file(path: Path, key: str) -> None:
    """Atomically replace a malformed generated-key file."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary_path = Path(temporary_name)
    try:
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(f"{key}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def get_or_create_api_key(
    configured_key: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Return a configured key or load and persist a generated key.

    Args:
        configured_key: Optional key from Celune's configuration.
        path: Optional storage path, primarily useful for isolated callers.

    Returns:
        str: The configured or persisted Celune API key.
    """
    for candidate in (configured_key, os.getenv("CELUNE_API_TOKEN")):
        if candidate is not None and candidate.strip():
            return candidate.strip()

    key_file = path or api_key_path(create_parent=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            key_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        for _attempt in range(20):
            existing = key_file.read_text(encoding="utf-8").strip()
            if _valid_generated_key(existing):
                return existing
            time.sleep(0.01)
        generated = generate_api_key()
        _replace_key_file(key_file, generated)
        return generated

    generated = generate_api_key()
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(f"{generated}\n")
        output.flush()
        os.fsync(output.fileno())
    return generated


@api.exception_handler(ModelConsentRequired)
async def model_consent_required_response(
    request: Request,
    error: ModelConsentRequired,
) -> JSONResponse:
    """Return structured model terms when an operation needs acknowledgement."""
    del request
    metadata = error.metadata
    return JSONResponse(
        status_code=428,
        content={
            "error": "model_consent_required",
            "model": metadata.to_dict(consented=consent_status(metadata)),
        },
    )


@api.exception_handler(ModelCommercialUseRestricted)
async def model_commercial_use_restricted_response(
    request: Request,
    error: ModelCommercialUseRestricted,
) -> JSONResponse:
    """Reject a model operation that conflicts with commercial-use mode."""
    del request
    metadata = error.metadata
    return JSONResponse(
        status_code=403,
        content={
            "error": "commercial_model_restricted",
            "message": str(error),
            "model": metadata.to_dict(consented=consent_status(metadata)),
        },
    )


@api.exception_handler(ModelConsentDeclined)
async def model_consent_declined_response(
    request: Request,
    error: ModelConsentDeclined,
) -> JSONResponse:
    """Reject an operation after the user declines model terms."""
    del request
    return JSONResponse(
        status_code=403,
        content={
            "error": "model_consent_declined",
            "message": string("registry.consent_declined"),
            "model": error.metadata.to_dict(consented=False),
        },
    )


WEBUI_AUTH_PATH = "/ui/key"
WEBUI_AUTH_COOKIE = "celune_webui_session"
WEBUI_AUTH_SESSION_SECONDS = 30 * 24 * 60 * 60
WEBUI_POLL_SNAPSHOT_API_NAME = "webui_poll_snapshot"
WEBUI_POLL_VC_API_NAME = "webui_poll_vc_controls"
WEBUI_POLL_RECORDING_API_NAME = "webui_poll_recording_control"
WEBUI_LOAD_SNAPSHOT_API_NAME = "webui_load_snapshot"
WEBUI_LOAD_VC_API_NAME = "webui_load_vc_controls"
WEBUI_LOAD_RECORDING_API_NAME = "webui_load_recording_control"
WEBUI_CYCLE_VOICE_API_NAME = "webui_cycle_voice"
WEBUI_TOGGLE_RECORDING_API_NAME = "webui_toggle_recording"
WEBUI_SPEAK_INPUT_API_NAME = "webui_speak_input"
WEBUI_SPEAK_BUTTON_API_NAME = "webui_speak_button"
WEBUI_CONVERT_API_NAME = "webui_convert_audio"
WEBUI_POLL_API_NAMES = frozenset(
    {
        WEBUI_POLL_SNAPSHOT_API_NAME,
        WEBUI_POLL_VC_API_NAME,
        WEBUI_POLL_RECORDING_API_NAME,
    }
)
WEBUI_GENERATION_API_NAMES = frozenset(
    {
        WEBUI_SPEAK_INPUT_API_NAME,
        WEBUI_SPEAK_BUTTON_API_NAME,
        WEBUI_CONVERT_API_NAME,
    }
)
_webui_generation_fn_indices: frozenset[int] = frozenset()
WEBUI_RATE_LIMIT_EXEMPT_API_NAMES = WEBUI_POLL_API_NAMES | frozenset(
    {
        WEBUI_LOAD_SNAPSHOT_API_NAME,
        WEBUI_LOAD_VC_API_NAME,
        WEBUI_LOAD_RECORDING_API_NAME,
        WEBUI_CYCLE_VOICE_API_NAME,
        WEBUI_TOGGLE_RECORDING_API_NAME,
    }
)
WEBUI_AUTH_CLIENTS_FILENAME = "webui-auth-clients.json"
WEBUI_AUTH_CLIENT_LIMIT = 1024
bound_celune: Optional[Celune] = None
auth_token: Optional[str] = None
rate_limit_per_minute = 60
rate_limit_lock = threading.Lock()
rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
webui_auth_clients_lock = threading.Lock()
webui_auth_clients: Optional[dict[str, int]] = None
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


def invoke_message_callback(
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


class ModelConsentRequest(BaseModel):
    """Exact-revision user decision submitted by an API or WebUI client."""

    model_id: str = Field(min_length=3, max_length=256)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    accepted: bool


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


def run_async_runtime_call(
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


WEBUI_HEAD = webui.read_webui_asset("head.html")
WEBUI_CSS = webui.read_webui_asset("webui.css").replace(
    "__CELUNE_CONNECTION_LOST_MESSAGE__",
    json.dumps(string("webui.connection_lost", app_name=APP_NAME), ensure_ascii=False),
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


def configure_webui_theme() -> None:
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
    theme_css = webui.read_webui_asset("theme.css")
    theme_values = {
        "BACKGROUND": background,
        "PRIMARY": primary,
        "ERROR": error,
        "FOREGROUND": foreground,
        "SECONDARY": secondary,
        "ACCENT": accent,
        "INPUT_BG": input_bg,
        "SLEEPING": sleeping,
        "BUTTON_BG": button_bg,
        "BUTTON_HOVER": button_hover,
    }
    for key, value in theme_values.items():
        theme_css = theme_css.replace(f"__{key}__", value)
    webui_theme_style = theme_css


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


def shutdown_api_for_fatal_error() -> None:
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
    return _headers_token(request.headers)


def _headers_token(headers: Headers) -> Optional[str]:
    """Extract the bearer or app token from request headers."""
    auth_header = headers.get("authorization", "")
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return _clean_token(headers.get("x-celune-token"))


def _headers_authenticated(headers: Headers) -> bool:
    """Return whether request headers carry the configured API token."""
    if auth_token is None:
        return True
    given = _headers_token(headers)
    return given is not None and compare_digest(given, auth_token)


def _authenticated(request: Request) -> bool:
    """Return whether the request carries the configured API token."""
    return _headers_authenticated(request.headers)


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

    return method == "GET" and path in {"/", "/favicon.ico"}


def _client_address(connection: Union[Request, WebSocket]) -> Optional[str]:
    """Return a client IP while accepting Cloudflare identity only from its proxy."""
    client = connection.client
    if client is None:
        return None

    peer = client.host
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return peer

    forwarded_address = connection.headers.get("cf-connecting-ip", "").strip()
    if forwarded_address:
        try:
            cloudflare_address = ipaddress.ip_address(forwarded_address)
        except ValueError:
            return peer_address.compressed
        if peer_address.is_loopback or peer_address == cloudflare_address:
            return cloudflare_address.compressed

    return peer_address.compressed


def _rate_limit_key(request: Request) -> str:
    """Return the client key used for HTTP request rate limiting."""
    address = _client_address(request) or "unknown"
    path = request.url.path.rstrip("/")
    if path == "/ui" or path.startswith("/ui/"):
        return f"webui:{address}"
    return address


def _rate_limited_client(client_key: str) -> bool:
    """Return whether a client exceeds the configured rate limit."""
    if rate_limit_per_minute <= 0:
        return False

    now = time.monotonic()
    window_start = now - 60.0

    with rate_limit_lock:
        hits = rate_limit_hits[client_key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= rate_limit_per_minute:
            return True

        hits.append(now)
        return False


def _rate_limited(request: Request) -> bool:
    """Return whether the request exceeds the configured rate limit."""
    return _rate_limited_client(_rate_limit_key(request))


def configure_webui_generation_fn_indices(
    dependencies: abc.Sequence[abc.Mapping[str, object]],
) -> None:
    """Record Gradio dependency IDs for speech and conversion actions."""
    global _webui_generation_fn_indices
    _webui_generation_fn_indices = frozenset(
        dependency_id
        for dependency in dependencies
        if isinstance(dependency.get("api_name"), str)
        and dependency.get("api_name") in WEBUI_GENERATION_API_NAMES
        and isinstance((dependency_id := dependency.get("id")), int)
    )


async def _is_webui_generation_request(request: Request) -> bool:
    """Return whether a POST can start speech or voice-conversion generation."""
    if request.method.upper() != "POST":
        return False

    path = request.url.path.rstrip("/")
    if not path.startswith("/ui/"):
        return False

    if path.endswith("/queue/join"):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        fn_index = payload.get("fn_index")
        return (
            isinstance(fn_index, int)
            and not isinstance(fn_index, bool)
            and fn_index in _webui_generation_fn_indices
        )

    return any(
        path.endswith(f"/{route}/{api_name}")
        for api_name in WEBUI_GENERATION_API_NAMES
        for route in ("run", "api", "call")
    ) or any(
        path.endswith(f"/call/v2/{api_name}") for api_name in WEBUI_GENERATION_API_NAMES
    )


def _ui_session_signature(payload: bytes) -> bytes:
    """Sign a WebUI session payload with the active application key."""
    if auth_token is None:
        return b""
    return hmac_new(
        auth_token.encode("utf-8"),
        b"Celune WebUI session v1\0" + payload,
        sha256,
    ).digest()


def _webui_auth_client_cache_path() -> Path:
    """Return the app-data path for authenticated WebUI client fingerprints."""
    return app_data_dir() / WEBUI_AUTH_CLIENTS_FILENAME


def _load_webui_auth_clients_locked() -> dict[str, int]:
    """Load unexpired client fingerprints while holding their cache lock."""
    global webui_auth_clients
    if webui_auth_clients is not None:
        return webui_auth_clients

    try:
        cached_data = json.loads(
            _webui_auth_client_cache_path().read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        cached_data = {}

    now = int(time.time())
    clients: dict[str, int] = {}
    if isinstance(cached_data, dict):
        for fingerprint, expires_at in cached_data.items():
            if (
                isinstance(fingerprint, str)
                and isinstance(expires_at, int)
                and not isinstance(expires_at, bool)
                and expires_at > now
            ):
                clients[fingerprint] = expires_at

    webui_auth_clients = clients
    return clients


def _persist_webui_auth_clients_locked(clients: dict[str, int]) -> None:
    """Write client fingerprints atomically without storing source addresses."""
    path = _webui_auth_client_cache_path()
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(clients, separators=(",", ":")),
            encoding="utf-8",
        )
        if os.name != "nt":
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _webui_auth_client_fingerprint(
    connection: Union[Request, WebSocket],
) -> Optional[str]:
    """Create a keyed fingerprint from a client's address and browser headers."""
    if auth_token is None:
        return None
    address = _client_address(connection)
    if address is None:
        return None

    client_headers = (
        connection.headers.get("user-agent", "")[:512],
        connection.headers.get("accept-language", "")[:256],
        connection.headers.get("accept", "")[:256],
        connection.headers.get("sec-ch-ua", "")[:256],
        connection.headers.get("sec-ch-ua-mobile", "")[:32],
        connection.headers.get("sec-ch-ua-platform", "")[:128],
    )
    client_data = "\0".join((address, *client_headers)).encode("utf-8")
    return hmac_new(
        auth_token.encode("utf-8"),
        b"Celune WebUI client cache v1\0" + client_data,
        sha256,
    ).hexdigest()


def _webui_auth_client_cached(connection: Union[Request, WebSocket]) -> bool:
    """Return whether this client has a persisted, unexpired WebUI grant."""
    fingerprint = _webui_auth_client_fingerprint(connection)
    if fingerprint is None:
        return False

    with webui_auth_clients_lock:
        clients = _load_webui_auth_clients_locked()
        expires_at = clients.get(fingerprint)
        if expires_at is None:
            return False
        if expires_at <= int(time.time()):
            del clients[fingerprint]
            _persist_webui_auth_clients_locked(clients)
            return False
        return True


def _remember_webui_auth_client(connection: Union[Request, WebSocket]) -> None:
    """Remember an authenticated client for the browser-session lifetime."""
    global webui_auth_clients
    fingerprint = _webui_auth_client_fingerprint(connection)
    if fingerprint is None:
        return

    now = int(time.time())
    with webui_auth_clients_lock:
        clients = _load_webui_auth_clients_locked()
        clients = {
            key: expires_at for key, expires_at in clients.items() if expires_at > now
        }
        clients[fingerprint] = now + WEBUI_AUTH_SESSION_SECONDS
        if len(clients) > WEBUI_AUTH_CLIENT_LIMIT:
            clients = dict(
                sorted(
                    clients.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:WEBUI_AUTH_CLIENT_LIMIT]
            )
        webui_auth_clients = clients
        _persist_webui_auth_clients_locked(clients)


def _encode_ui_session(client_id: str, expires_at: int) -> str:
    """Create a signed, restart-safe cookie for one browser client."""
    payload = f"v1:{expires_at}:{client_id}".encode("ascii")
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = base64.urlsafe_b64encode(_ui_session_signature(payload)).rstrip(b"=")
    return f"{encoded_payload.decode('ascii')}.{signature.decode('ascii')}"


def _ui_session_details(session: Optional[str]) -> Optional[tuple[str, int]]:
    """Return the client ID and expiry for a valid signed WebUI cookie."""
    if auth_token is None or session is None or len(session) > 512:
        return None
    try:
        encoded_payload, encoded_signature = session.split(".", 1)
        payload = base64.b64decode(
            encoded_payload.encode("ascii") + b"=" * (-len(encoded_payload) % 4),
            altchars=b"-_",
            validate=True,
        )
        signature = base64.b64decode(
            encoded_signature.encode("ascii") + b"=" * (-len(encoded_signature) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, UnicodeEncodeError, ValueError):
        return None

    if not compare_digest(signature, _ui_session_signature(payload)):
        return None

    try:
        session_version, raw_expiry, client_id = payload.decode("ascii").split(":", 2)
        expires_at = int(raw_expiry)
    except (UnicodeDecodeError, ValueError):
        return None

    now = int(time.time())
    if (
        session_version != "v1"
        or len(client_id) < 32
        or expires_at <= now
        or expires_at > now + WEBUI_AUTH_SESSION_SECONDS
    ):
        return None
    return client_id, expires_at


def _new_ui_session(client_id: Optional[str] = None) -> str:
    """Create a signed WebUI session that survives Celune restarts."""
    if auth_token is None:
        return ""
    return _encode_ui_session(
        client_id or secrets.token_urlsafe(32),
        int(time.time()) + WEBUI_AUTH_SESSION_SECONDS,
    )


def _ui_session_valid(session_id: Optional[str]) -> bool:
    """Return whether a signed WebUI session cookie is still active."""
    return _ui_session_details(session_id) is not None


def _set_ui_session_cookie(
    response: Response,
    request: Request,
    session: str,
) -> None:
    """Set a browser-scoped cookie for a validated WebUI client."""
    response.set_cookie(
        WEBUI_AUTH_COOKIE,
        session,
        max_age=WEBUI_AUTH_SESSION_SECONDS,
        path="/ui",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="strict",
    )


def _refresh_ui_session_cookie(request: Request, response: Response) -> Response:
    """Extend an active browser session near its expiry without asking again."""
    details = _ui_session_details(request.cookies.get(WEBUI_AUTH_COOKIE))
    if details is None:
        if _webui_auth_client_cached(request):
            _set_ui_session_cookie(response, request, _new_ui_session())
        return response

    client_id, expires_at = details
    if expires_at - int(time.time()) > WEBUI_AUTH_SESSION_SECONDS // 2:
        return response

    _remember_webui_auth_client(request)
    _set_ui_session_cookie(response, request, _new_ui_session(client_id))
    return response


def _webui_authenticated(connection: Union[Request, WebSocket]) -> bool:
    """Return whether a WebUI connection has an API token or active session."""
    if auth_token is None or _headers_authenticated(connection.headers):
        return True
    if _ui_session_valid(connection.cookies.get(WEBUI_AUTH_COOKIE)):
        if not _webui_auth_client_cached(connection):
            _remember_webui_auth_client(connection)
        return True
    return _webui_auth_client_cached(connection)


def _same_origin(connection: Union[Request, WebSocket]) -> bool:
    """Return whether a browser supplied origin matches this API host."""
    origin = connection.headers.get("origin")
    if origin is None:
        return False

    try:
        parsed_origin = urlsplit(origin)
        parsed_request = urlsplit(str(connection.url))
        origin_port = parsed_origin.port
        request_port = parsed_request.port
        origin_host = parsed_origin.hostname
        request_host = parsed_request.hostname
    except ValueError:
        return False

    connection_scheme = {"ws": "http", "wss": "https"}.get(
        parsed_request.scheme,
        parsed_request.scheme,
    )
    if (
        parsed_origin.username is not None
        or parsed_origin.password is not None
        or parsed_origin.path
        or parsed_origin.scheme != connection_scheme
    ):
        return False

    default_port = {"http": 80, "https": 443}.get(connection_scheme)
    return (
        origin_host is not None
        and origin_host.lower() == (request_host or "").lower()
        and (origin_port if origin_port is not None else default_port)
        == (request_port if request_port is not None else default_port)
    )


def _unauthorized_response() -> JSONResponse:
    """Build the shared unauthorized response for protected requests."""
    return JSONResponse(
        status_code=401,
        content={
            "error": "unauthorized",
            "message": string("api.unauthorized"),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _rate_limit_response() -> JSONResponse:
    """Build the shared rate limit response for protected requests."""
    current_time = datetime.datetime.now(datetime.UTC)
    next_minute = current_time.replace(second=0, microsecond=0) + datetime.timedelta(
        minutes=1
    )
    retry_after = (next_minute - current_time).total_seconds()
    return JSONResponse(
        status_code=429,
        content={
            "error": "ratelimit_exceeded",
            "message": string("api.rate_limit"),
        },
        headers={"Retry-After": str(retry_after)},
    )


def _webui_auth_page() -> str:
    """Build the standalone, blocking WebUI API-key prompt."""
    values = {
        "title": string("api.webui_auth_title"),
        "message": string("api.webui_auth_message"),
        "placeholder": string("api.webui_auth_placeholder"),
        "submit": string("api.webui_auth_submit"),
        "invalid": string("api.webui_auth_invalid"),
        "error": string("api.webui_auth_error"),
    }
    page = webui.read_webui_asset("api_key.html")
    for key, value in values.items():
        page = page.replace(f"__{key.upper()}__", html.escape(value))
    return page


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

    if is_browser_ui_request(request):
        path = request.url.path.rstrip("/")
        if path.startswith("/ui/assets/") and request.method.upper() == "GET":
            asset_name = path.removeprefix("/ui/assets/")
            asset_response = _webui_asset_response(asset_name)
            if asset_response is not None:
                return asset_response

        if auth_token is None:
            model_request_token = enter_api_model_request()
            try:
                return await call_next(request)
            finally:
                exit_api_model_request(model_request_token)

        if path.startswith("/ui/assets/") and request.method.upper() == "GET":
            return await call_next(request)
        if path == WEBUI_AUTH_PATH and request.method.upper() == "GET":
            if _webui_authenticated(request):
                return RedirectResponse(
                    "/ui",
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
            return HTMLResponse(
                _webui_auth_page(),
                headers={
                    "Cache-Control": "no-store",
                    "Content-Security-Policy": (
                        "default-src 'none'; style-src 'self'; "
                        "script-src 'self'; connect-src 'self'; font-src 'self'; "
                        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
                    ),
                    "Referrer-Policy": "no-referrer",
                    "X-Frame-Options": "DENY",
                },
            )

        if path == WEBUI_AUTH_PATH and request.method.upper() == "POST":
            if not _same_origin(request):
                return JSONResponse(status_code=403, content={"error": "forbidden"})
            if _rate_limited(request):
                return _rate_limit_response()
            return await call_next(request)

        if not _webui_authenticated(request):
            if request.method.upper() == "GET" and path == "/ui":
                return RedirectResponse(
                    "/ui/key",
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
            return _unauthorized_response()

        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            if not _same_origin(request):
                return JSONResponse(status_code=403, content={"error": "forbidden"})
            if await _is_webui_generation_request(request) and _rate_limited(request):
                return _rate_limit_response()
        model_request_token = enter_api_model_request()
        try:
            response = await call_next(request)
        finally:
            exit_api_model_request(model_request_token)
        return _refresh_ui_session_cookie(request, response)

    if not _authenticated(request):
        return _unauthorized_response()

    if _rate_limited(request):
        return _rate_limit_response()

    model_request_token = enter_api_model_request()
    try:
        return await call_next(request)
    finally:
        exit_api_model_request(model_request_token)


class WebUiWebSocketSecurityMiddleware:
    """Protect mounted WebUI WebSockets with the API token or browser session."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        path = str(scope.get("path", "")).rstrip("/")
        is_webui = path == "/ui" or path.startswith("/ui/")
        if scope["type"] == "websocket" and is_webui and auth_token is not None:
            websocket = WebSocket(scope, receive, send)
            if not _webui_authenticated(websocket) or not _same_origin(websocket):
                await send({"type": "websocket.close", "code": 4401})
                return
        await self.app(scope, receive, send)


api.add_middleware(WebUiWebSocketSecurityMiddleware)


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
    subscribe_webui_events(celune)
    subscribe_webui_timed_updates()
    configure_webui_theme()
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    webui_input_locked = celune.locked or not has_voice
    webui_input_placeholder = get_webui_input_placeholder(
        celune,
        webui_input_locked,
        has_voice,
    )
    webui_voice_locked = (
        len(celune.voices) < 2
        or celune.is_in_tutorial
        or not has_voice
        or getattr(getattr(celune, "backend", None), "singing", False)
    )
    seed_webui_logs()
    wrap_celune_callbacks(celune)
    if celune.current_voice:
        append_webui_log(string("webui.voice_ready", voice=celune.current_voice))
    set_webui_status(
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


class WebUiCommandHost:
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
        publish_active_task_log(message, severity)
        append_webui_log(message, severity)
        _ = loglevel

    def safe_status(self, message: str, severity: str = "info") -> None:
        """Forward a command status to the browser footer."""
        set_webui_status(message, severity, source="callback")

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
            webui_run_command(text)
        else:
            self.celune.say(text)

    @staticmethod
    def pulse_border(_selector: str) -> None:
        """Ignore a Textual-only tutorial animation in the browser host."""

    def graceful_exit(self) -> None:
        """Close the bound runtime when the browser receives `/exit`."""
        self.celune.close()


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


class WebUiAuthRequest(BaseModel):
    """Request body used to establish an authenticated WebUI session."""

    api_key: str = Field(min_length=1, max_length=4096)


class ActionResponse(BaseModel):
    """Generic accepted control response."""

    status: str


class TaskCancelResponse(BaseModel):
    """Response returned after an explicit speech-task cancellation request."""

    task_id: str
    status: Literal["cancelled"] = "cancelled"


@api.post(WEBUI_AUTH_PATH, include_in_schema=False)
async def authenticate_webui(
    body: WebUiAuthRequest,
    request: Request,
) -> Response:
    """Exchange the configured application key for a WebUI session cookie."""
    provided_key = _clean_token(body.api_key)
    if auth_token is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if provided_key is None or not compare_digest(provided_key, auth_token):
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": string("api.webui_auth_invalid"),
            },
            headers={"Cache-Control": "no-store"},
        )

    response = JSONResponse(
        status_code=200,
        content={"authenticated": True},
        headers={"Cache-Control": "no-store"},
    )
    _remember_webui_auth_client(request)
    _set_ui_session_cookie(response, request, _new_ui_session())
    return response


async def _cancel_speech_job(job_id: str) -> bool:
    """Ask Core to stop one active speech task and publish its terminal event."""
    status = task_status(job_id)
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

    update_speech_job(job_id, status="cancelled")
    publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="cancelled",
            status="cancelled",
        ),
    )
    set_active_speech_task(None)
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

    subscription = subscribe_to_speech_job(job_id)
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
        unsubscribe_from_speech_job(job_id, subscription)
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


def _model_consent_document(
    metadata: ModelMetadata,
    *,
    include_prompt: bool = False,
) -> dict[str, object]:
    """Serialize catalog metadata and localized confirmation details."""
    consented = not metadata.requires_consent or consent_status(metadata)
    document = metadata.to_dict(consented=consented)
    if include_prompt:
        prompt, instruction = consent_prompt_text(metadata)
        document.update(
            {
                "prompt": prompt,
                "instruction": instruction,
                "details": list(consent_details(metadata)),
                "source_url": metadata.license_url or metadata.repository_url,
                "entry_hint": string("registry.consent_entry_hint"),
                "consent_phrase": string("registry.consent_phrase"),
                "accept_label": string("registry.consent_accept"),
                "decline_label": string("registry.consent_decline"),
                "accepted_message": string("registry.consent_accepted_retry"),
                "declined_message": string("registry.consent_declined"),
                "submit_failed_message": string("registry.consent_submit_failed"),
            }
        )
    return document


@api.get("/v1/models")
def models() -> dict[str, object]:
    """List built-in models, pinned revisions, terms and consent state."""
    catalog = {metadata.model_id: metadata for metadata in registered_models()}
    catalog.update(
        {
            metadata.model_id: metadata
            for metadata in pending_model_consents()
            if metadata.model_id not in catalog
        }
    )
    return {
        "commercial_mode": commercial_mode_enabled(),
        "models": [
            _model_consent_document(metadata)
            for metadata in sorted(
                catalog.values(),
                key=lambda item: (item.category, item.model_id),
            )
        ],
    }


def _webui_asset_response(asset_name: str) -> Optional[Response]:
    """Build a response for a Celune-owned WebUI asset, if it is recognized."""
    if asset_name == "theme.css":
        return Response(
            content=webui_theme_style,
            media_type="text/css",
            headers={"Cache-Control": "no-store"},
        )
    asset = webui.webui_asset_response_data(asset_name)
    if asset is None:
        return None
    content, media_type = asset
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


def _submit_model_consent(body: ModelConsentRequest) -> dict[str, object]:
    """Apply one exact-revision acknowledgement submitted by a client."""
    metadata = model_metadata_for_revision(body.model_id, body.revision)
    record_model_consent(body.model_id, body.revision, body.accepted)
    return {
        "accepted": body.accepted,
        "model": _model_consent_document(metadata),
    }


@api.post("/v1/models/consent")
def submit_model_consent(body: ModelConsentRequest) -> dict[str, object]:
    """Acknowledge model terms for one exact Hugging Face commit."""
    return _submit_model_consent(body)


@api.get("/ui/models/pending", include_in_schema=False)
def webui_pending_model_consents() -> dict[str, object]:
    """Return localized pending prompts to the authenticated browser UI."""
    pending = pending_model_consents()
    return {
        "pending": [
            _model_consent_document(metadata, include_prompt=True)
            for metadata in pending
        ],
        "status": (string("registry.waiting_status") if pending else webui_status_text),
    }


@api.post("/ui/models/consent", include_in_schema=False)
def webui_submit_model_consent(body: ModelConsentRequest) -> dict[str, object]:
    """Submit the browser UI's decision for one exact model revision."""
    return _submit_model_consent(body)


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


@api.post("/v1/sing", response_model=None)
def sing(body: SpeakRequest) -> Union[StreamingResponse, JSONResponse]:
    """Validate CESN, queue singing, and stream generated audio to the caller."""
    celune = require_celune()
    if not getattr(getattr(celune, "backend", None), "singing", False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "unsupported_operation",
                "message": string("api.singing_backend_required"),
            },
        )

    notation_error = validate_singing_notation(celune, body.content)
    if notation_error is not None:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_notation", "message": notation_error},
        )

    api_log("SING", body.content)
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
    remember_speech_job(job_id, SpeechJob(status="queued", created_at=time.time()))
    set_active_speech_task(job_id)
    update_speech_job(job_id, status="running")
    publish_task_event(
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
        set_active_speech_task(None)
        forget_speech_job(job_id)
        raise
    if chunks is None:
        set_active_speech_task(None)
        forget_speech_job(job_id)
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("webui.busy_try_again"),
            },
        )

    threading.Thread(
        target=collect_speech_job,
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
    job = speech_job_snapshot(job_id)
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
        content=job.audio or flac_bytes(np.empty((0, 2), dtype=np.float32)),
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
    if task_status(job_id) is None:
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

    if getattr(getattr(celune, "backend", None), "singing", False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "unsupported_operation",
                "message": string("celune.singing_voice_change_unavailable"),
            },
        )

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
    file: Annotated[UploadFile, File()],
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
        audio, sr = decode_uploaded_audio(data)
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
        yield flac_bytes(audio)

    return StreamingResponse(
        chunks(),
        media_type="audio/flac",
        headers=stream_headers(),
    )


@api.post("/v1/convert", response_model=None)
async def convert_audio(
    file: Annotated[UploadFile, File()],
    f0_condition: Optional[bool] = Form(None),
) -> Union[StreamingResponse, JSONResponse]:
    """Convert an uploaded source audio file through the active VC backend.

    Args:
        file: The uploaded source audio file to convert.
        f0_condition: Optional override enabling singing mode pitch conditioning.

    Returns:
        Union[StreamingResponse, JSONResponse]: The converted audio stream, or a JSON error payload if conversion
        failed.
    """
    celune = require_celune()
    if not webui_vc_mode_active(celune):
        return voice_conversion_unavailable_response()

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
        audio, sample_rate = decode_uploaded_audio(data)
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
        yield flac_bytes(
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
    build_webui(),
    path="/ui",
    footer_links=[],
    show_error=True,
    css=WEBUI_CSS,
    head=WEBUI_HEAD,
)
api.mount("/", app=mcp_app)


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
    configure_mcp_host_allowlist(bind_host)

    def _default_started(bhost: str, bport: int) -> None:
        message = string(
            "celune.api_started",
            app_name=APP_NAME,
            url=f"http://{bhost}:{bport}",
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
    on_started: Optional[Callable[[str, int], None]] = None,
    defer_startup_log: bool = False,
) -> threading.Thread:
    """Start the API in a background thread.

    Args:
        celune: Running app instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.
        token: Token required for API requests.
        requests_per_minute: Maximum requests allowed per client each minute.
        startup_timeout: Seconds to wait for startup confirmation.
        on_started: Optional callback after the server socket begins listening.
        defer_startup_log: Leave startup URL reporting to ``on_started``.

    Returns:
        threading.Thread: The daemon thread running the API server.
    """

    started = threading.Event()
    failed = threading.Event()

    def _started(bind_host: str, bind_port: int) -> None:
        if not defer_startup_log or on_started is None:
            celune.log(
                string(
                    "celune.api_started",
                    app_name=APP_NAME,
                    url=f"http://{bind_host}:{bind_port}",
                )
            )
        if on_started is not None:
            on_started(bind_host, bind_port)
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
