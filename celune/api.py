# SPDX-License-Identifier: Apache-2.0
"""API layer."""

import os
import re
import html
import json
import time
import uuid
import errno
import atexit
import shutil
import socket
import asyncio
import threading
import contextlib
import subprocess
from hmac import compare_digest
from typing import (
    Union,
    Literal,
    Optional,
    Annotated,
)
from pathlib import Path
from collections import abc

import numpy as np
import gradio as gr
import fastapi
import uvicorn
from pydantic import Field, BaseModel
from starlette import types, concurrency, datastructures
from starlette.middleware import base as middleware_base

from . import __version__
from .ui import resources as ui_resources
from .mcp import (
    mcp_app,
    mcp_lifespan,
    configure_mcp_host_allowlist,
)
from .cesn import validate_singing_notation
from .i18n import string
from .paths import project_root
from .utils import format_error_message
from .webui import actions as webactions
from .webui import runtime as webui
from .celune import Celune
from .server import auth as server_auth
from .server import jobs as server_jobs
from .audio.dsp import resample_audio
from .constants import BASE_SR, APP_NAME
from .exceptions import (
    ModelConsentDeclined,
    ModelConsentRequired,
    TaskSubscriptionClosed,
    ModelCommercialUseRestricted,
)
from .typing.api import (
    TaskCommandName,
)
from .server.auth import (
    _clean_token,
    _same_origin,
    _rate_limited,
    _authenticated,
    _new_ui_session,
    _rate_limit_response,
    _webui_authenticated,
    is_browser_ui_request,
    _is_public_api_request,
    _set_ui_session_cookie,
    _unauthorized_response,
    _websocket_authenticated,
    _refresh_ui_session_cookie,
    _remember_webui_auth_client,
    _is_webui_generation_request,
)
from .server.jobs import SpeechJob, TaskEvent, TaskSubscription
from .webui.actions import (
    build_webui,
    decode_uploaded_audio,
    voice_conversion_unavailable_response,
)
from .webui.runtime import (
    api_log,
    flac_bytes,
    require_celune,
    seed_webui_logs,
    append_webui_log,
    set_webui_status,
    collect_speech_job,
    webui_vc_mode_active,
    configure_webui_theme,
    wrap_celune_callbacks,
    subscribe_webui_events,
    get_webui_input_placeholder,
    subscribe_webui_timed_updates,
)
from .registry.types import ModelMetadata
from .server.session import ServerSession, server_session
from .typing.pipeline import SpeechStreamQueue
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
from .server.constants import (
    WEBUI_AUTH_PATH,
)
from .pipeline.submission import prepare_playback_audio

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


def generate_api_key() -> str:
    """Generate a new API key without persisting it."""
    return server_auth.generate_api_key()


def get_or_create_api_key(
    configured_key: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Return a configured API key or load and persist a generated key."""
    return server_auth.get_or_create_api_key(configured_key, path)


def configure_api_security(
    token: Optional[str] = None,
    requests_per_minute: int = 60,
) -> None:
    """Configure API authentication and rate limiting."""
    return server_auth.configure_api_security(token, requests_per_minute)


def resolve_api_host(token: Optional[str] = None, host: Optional[str] = None) -> str:
    """Resolve the API bind host from authentication state."""
    return server_auth.resolve_api_host(token, host)


def audio_bytes(
    chunks: SpeechStreamQueue,
    on_chunk: Optional[Callable[[int], None]] = None,
) -> Iterator[bytes]:
    """Yield FLAC bytes from a speech stream for the documented API helper."""
    return webui.audio_bytes(chunks, on_chunk)


def stream_headers(sample_rate: int = BASE_SR) -> dict[str, str]:
    """Return headers for a streamed FLAC response."""
    return webui.stream_headers(sample_rate)


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


WEBUI_HEAD = webui.read_webui_asset("head.html")
WEBUI_CSS = webui.read_webui_asset("webui.css").replace(
    "__CELUNE_CONNECTION_LOST_MESSAGE__",
    json.dumps(string("webui.connection_lost", app_name=APP_NAME), ensure_ascii=False),
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

        if server_session.auth_token is None:
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
        if (
            scope["type"] == "websocket"
            and is_webui
            and server_session.auth_token is not None
        ):
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
    server_session.bound_celune = celune
    server_session.webui_resource_page = 0
    server_session.webui_last_resource_advance = 0.0
    server_session.webui_last_probed_state = None
    server_session.webui_log_lines.clear()
    server_session.webui_logs_seeded = False
    server_session.webui_active_theme_name = "celune"
    server_session.webui_caption_text = ""
    server_session.webui_caption_progress = 0.0
    server_session.webui_caption_active = False
    server_session.webui_progress_current = None
    server_session.webui_progress_total = None
    server_session.webui_timed_update_sequence = 0
    server_session.webui_timed_update_received_at = 0.0
    server_session.webui_timed_update_source = "fallback"
    _unsubscribe_webui_events(server_session)
    subscribe_webui_events(celune, server_session)
    subscribe_webui_timed_updates(server_session)
    configure_webui_theme(server_session)
    has_voice = bool(celune.current_voice) or bool(celune.voices)
    server_session.webui_input_locked = celune.locked or not has_voice
    server_session.webui_input_placeholder = get_webui_input_placeholder(
        celune,
        server_session.webui_input_locked,
        has_voice,
    )
    server_session.webui_voice_locked = (
        len(celune.voices) < 2
        or celune.is_in_tutorial
        or not has_voice
        or getattr(getattr(celune, "backend", None), "singing", False)
    )
    seed_webui_logs(server_session)
    wrap_celune_callbacks(celune, server_session)
    if celune.current_voice:
        append_webui_log(
            string("webui.voice_ready", voice=celune.current_voice),
            session=server_session,
        )
    set_webui_status(
        string("status.idle")
        if celune.cur_state == "idle"
        else celune.cur_state.title(),
        source="probe",
        session=server_session,
    )


def _unsubscribe_webui_events(session: ServerSession = server_session) -> None:
    """Remove event subscriptions owned by the browser UI bridge."""
    if session.webui_event_dispatcher is not None:
        for event_name, callback in session.webui_event_callbacks:
            session.webui_event_dispatcher.unsubscribe(event_name, callback)
    session.webui_event_dispatcher = None
    session.webui_event_callbacks = ()


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
    if server_session.auth_token is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if provided_key is None or not compare_digest(
        provided_key, server_session.auth_token
    ):
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
    status = server_jobs.task_status(job_id)
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

    server_jobs.update_speech_job(job_id, status="cancelled")
    server_jobs.publish_task_event(
        job_id,
        TaskEvent(
            task_id=job_id,
            event="cancelled",
            status="cancelled",
        ),
    )
    server_jobs.set_active_speech_task(None)
    return True


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

    subscription = server_jobs.subscribe_to_speech_job(job_id)
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
        server_jobs.unsubscribe_from_speech_job(job_id, subscription)
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
            content=server_session.webui_theme_style,
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
        "status": (
            string("registry.waiting_status")
            if pending
            else server_session.webui_status_text
        ),
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
    if getattr(getattr(celune, "backend", None), "singing", False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "unsupported_operation",
                "message": string("api.singing_backend_use_sing_endpoint"),
            },
        )

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
    if getattr(getattr(celune, "backend", None), "singing", False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "unsupported_operation",
                "message": string("api.singing_backend_use_sing_endpoint"),
            },
        )

    api_log("SPEAK(ASYNC)", body.content)
    job_id = uuid.uuid4().hex
    location = f"/v1/speak/jobs/{job_id}"
    server_jobs.remember_speech_job(
        job_id, SpeechJob(status="queued", created_at=time.time())
    )
    server_jobs.set_active_speech_task(job_id)
    server_jobs.update_speech_job(job_id, status="running")
    server_jobs.publish_task_event(
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
        server_jobs.set_active_speech_task(None)
        server_jobs.forget_speech_job(job_id)
        raise
    if chunks is None:
        server_jobs.set_active_speech_task(None)
        server_jobs.forget_speech_job(job_id)
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": string("webui.busy_try_again"),
            },
        )

    threading.Thread(
        target=collect_speech_job,
        args=(job_id, chunks, server_session),
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
    if getattr(getattr(celune, "backend", None), "singing", False):
        return JSONResponse(
            status_code=409,
            content={
                "error": "unsupported_operation",
                "message": string("celune.persona_unavailable_singing"),
            },
        )

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
    job = server_jobs.speech_job_snapshot(job_id)
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
    if server_jobs.task_status(job_id) is None:
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

    data = await file.read(server_session.max_sfx_upload_bytes + 1)
    if len(data) > server_session.max_sfx_upload_bytes:
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

    data = await file.read(server_session.max_sfx_upload_bytes + 1)
    if len(data) > server_session.max_sfx_upload_bytes:
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
    build_webui(server_session),
    path="/ui",
    footer_links=[],
    show_error=True,
    css=WEBUI_CSS,
    head=WEBUI_HEAD,
)
api.mount("/", app=mcp_app)


_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com\b", re.IGNORECASE)
_STARTUP_ANNOUNCEMENT_TIMEOUT = 10.0


class CloudflaredQuickTunnel:
    """Own one cloudflared quick-tunnel process for Celune's lifetime."""

    def __init__(self, log: Callable[[str, str], None]) -> None:
        """Create a tunnel manager.

        Args:
            log: Callback used to report tunnel status and its public URL.
        """
        self._log = log
        self._process: Optional[subprocess.Popen[str]] = None
        self._reader: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self._port: Optional[int] = None
        self._startup_announced = False
        self._startup_timer: Optional[threading.Timer] = None

    def start(self, port: int) -> bool:
        """Start the public quick tunnel and watch for its endpoint URL.

        Args:
            port: Local API port to expose through Cloudflare.

        Returns:
            bool: ``True`` when cloudflared started, otherwise ``False``.
        """
        with self._lock:
            if self._closed:
                return False
            if self._process is not None and self._process.poll() is None:
                return True

            executable = shutil.which("cloudflared")
            if executable is None:
                self._log(
                    string("celune.cloudflared_missing", app_name=APP_NAME),
                    "warning",
                )
                return False

            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            )
            try:
                process = subprocess.Popen(  # pylint: disable=R1732
                    [executable, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
            except OSError:
                self._log(
                    string("celune.cloudflared_failed", app_name=APP_NAME),
                    "warning",
                )
                return False

            self._stopping.clear()
            self._process = process
            self._port = port
            self._startup_announced = False
            atexit.register(self.stop)
            self._startup_timer = threading.Timer(
                _STARTUP_ANNOUNCEMENT_TIMEOUT,
                self._announce_local_start,
            )
            self._startup_timer.daemon = True
            self._startup_timer.start()
            self._reader = threading.Thread(
                target=self._read_output,
                args=(process,),
                daemon=True,
                name=f"{APP_NAME}Cloudflared",
            )
            self._reader.start()
        return True

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        """Read cloudflared output without blocking Celune's API thread."""
        output = process.stdout
        if output is not None:
            for line in output:
                match = _TUNNEL_URL.search(line)
                if match is not None:
                    url = match.group(0)
                    if not self._announce_startup(url):
                        self._log(
                            string(
                                "celune.public_api_endpoint",
                                app_name=APP_NAME,
                                url=url,
                            ),
                            "info",
                        )

        return_code = process.wait()
        if not self._stopping.is_set():
            self._announce_local_start()
        if return_code != 0 and not self._stopping.is_set():
            self._log(string("celune.cloudflared_failed", app_name=APP_NAME), "warning")

    def _announce_local_start(self) -> None:
        """Report the local API URL when the tunnel URL is not ready in time."""
        with self._lock:
            port = self._port
        if port is not None:
            self._announce_startup(f"http://127.0.0.1:{port}")

    def _announce_startup(self, url: str) -> bool:
        """Log the first usable API URL, returning whether this call announced it."""
        with self._lock:
            if self._startup_announced or self._stopping.is_set():
                return False
            self._startup_announced = True
            timer = self._startup_timer
            self._startup_timer = None

        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        self._log(
            string("celune.api_started", app_name=APP_NAME, url=url),
            "info",
        )
        return True

    def stop(self) -> None:
        """Terminate the quick tunnel and wait for its process to exit."""
        with self._lock:
            self._closed = True
            process = self._process
            timer = self._startup_timer
            self._startup_timer = None
            if timer is not None:
                timer.cancel()
            if process is None:
                return

            atexit.unregister(self.stop)
            self._stopping.set()
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(OSError):
                        process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=2)

            reader = self._reader
            if reader is not None and reader is not threading.current_thread():
                reader.join(timeout=1)
            self._process = None
            self._reader = None


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
    bind_host = resolve_api_host(token=server_session.auth_token, host=host)
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
    server_session.current_api_server = server
    try:
        server.run(sockets=[api_socket])
    finally:
        api_socket.close()
        server_session.current_api_server = None


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
                    string("celune.api_runner_exited", code=exc.code),
                    "warning",
                )
        except Exception as e:
            failed.set()
            if isinstance(e, OSError) and _is_port_in_use_error(e):
                celune.log(string("celune.api_port_in_use", port=port), "warning")
            else:
                celune.log(
                    format_error_message(
                        string("celune.api_start_failed"),
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
            string(
                "celune.api_runner_timeout",
                seconds=f"{startup_timeout:.1f}",
            ),
            "warning",
        )

    return thread
