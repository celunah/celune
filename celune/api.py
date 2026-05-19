# SPDX-License-Identifier: MIT
"""Celune's API layer."""

import os
import io
import time
import uuid
import queue
import datetime
import threading
from dataclasses import dataclass
from hmac import compare_digest
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Union

import uvicorn
import numpy as np
import numpy.typing as npt
import soundfile as sf
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .constants import BASE_SR
from .utils import format_error
from .dsp import _resample_audio

if TYPE_CHECKING:
    from .celune import Celune

api = FastAPI(title="CeluneAPI")
bound_celune: Optional["Celune"] = None
auth_token: Optional[str] = None
rate_limit_per_minute = 60
rate_limit_lock = threading.Lock()
rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
max_sfx_upload_bytes = 25 * 1024 * 1024
speech_jobs_lock = threading.Lock()
speech_jobs: dict[str, "SpeechJob"] = {}
speech_job_ttl_seconds = 15 * 60


@dataclass
class SpeechJob:
    """In-memory state for an accepted speech request."""

    status: str
    created_at: float
    audio: Optional[bytes] = None
    error: Optional[str] = None


class StartedServer(uvicorn.Server):
    """Uvicorn server that reports when socket binding actually succeeds."""

    def __init__(
        self,
        config: uvicorn.Config,
        on_started: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(config)
        self.on_started = on_started

    async def startup(self, sockets: Optional[list[Any]] = None) -> None:
        """Run Uvicorn startup and report only after the server is listening.

        Args:
            sockets: A list of sockets to bind the server to.

        Returns:
            None: This method starts the server and announces a startup event.
        """
        await super().startup(sockets=sockets)
        if self.started and self.on_started is not None:
            self.on_started()


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Normalize empty token values to ``None``."""
    if token is None:
        return None
    token = token.strip()
    return token or None


def _env_auth_token() -> Optional[str]:
    """Return the Celune API token from the environment, if configured."""
    return _clean_token(os.getenv("CELUNE_API_TOKEN"))


def configure_api_security(
    token: Optional[str] = None,
    requests_per_minute: int = 60,
) -> None:
    """Configure API authentication and rate limiting.

    Args:
        token: A required token to send requests.
        requests_per_minute: The max amount of requests per minute the user is allowed to send.

    Returns:
        None: This method configures API security configuration.
    """
    global auth_token, rate_limit_per_minute

    auth_token = _clean_token(token) or _env_auth_token()
    rate_limit_per_minute = max(0, int(requests_per_minute))
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
    """Extract the bearer or Celune token from a request."""
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
async def api_security(request: Request, call_next: Any) -> Any:
    """Apply token authentication and a simple per-client rate limit.

    Args:
        request: The request that should be protected.
        call_next: What to run if security checks have passed.

    Returns:
        Any: The return value of the specified function.
    """
    if not _authenticated(request):
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Who are you? Send me an authentication token.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _rate_limited(request):
        current_time = datetime.datetime.now()
        next_minute = current_time.replace(
            second=0, microsecond=0
        ) + datetime.timedelta(minutes=1)
        retry_after = (next_minute - current_time).total_seconds()

        return JSONResponse(
            status_code=429,
            content={
                "error": "ratelimit_exceeded",
                "message": "Please wait until you make me speak again.",
            },
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


def bind_celune(celune: "Celune") -> None:
    """Bind the running Celune instance to API routes.

    Args:
        celune: The instance of Celune to bind.

    Returns:
        None: This method binds Celune to an API route.
    """
    global bound_celune
    bound_celune = celune


def require_celune() -> "Celune":
    """Return the bound Celune instance or fail the request.

    Returns:
        Celune: The bound Celune instance set for the request.

    Raises:
        HTTPException: The user has requested an API route that required Celune, but Celune wasn't available.
    """
    if bound_celune is None:
        raise HTTPException(
            status_code=503,
            detail="I'm not currently available.",
        )
    return bound_celune


def api_log(action: str, content: str, suffix: str = "") -> None:
    """Print the API control log line.

    Args:
        action: The request made by the user.
        content: The request body sent by the user.
        suffix: The suffix to append to the log line.

    Returns:
        None: This method prints the log line to any configured logger or terminal.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preview = content.replace("\n", "\\n").replace("\r", "\\r")[:64]
    if len(content) > 64:
        preview += "..."
    print(f"[{timestamp}] {action} {preview!r}{suffix}", flush=True)


def _normalized_audio(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Return stereo audio in frame-major form for file encoding."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and normalized.shape[0] == 2 and normalized.shape[1] != 2:
        return normalized.T
    return normalized


def _flac_bytes(audio: npt.NDArray[np.float32]) -> bytes:
    """Encode 48 kHz audio as PCM24 FLAC bytes."""
    buffer = io.BytesIO()
    sf.write(
        buffer,
        _normalized_audio(audio),
        BASE_SR,
        format="FLAC",
        subtype="PCM_24",
    )
    return buffer.getvalue()


def audio_bytes(chunks: queue.Queue) -> Iterator[bytes]:
    """Yield one FLAC payload from queued 48 kHz stereo float32 chunks.

    Args:
        chunks: A queue of audio chunks.

    Returns:
        Iterator[bytes]: The audio chunk from the queue as raw bytes.

    Raises:
        Exception: The stream was interrupted by Celune.
    """
    audio_chunks: list[npt.NDArray[np.float32]] = []
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        audio_chunks.append(_normalized_audio(item))

    if audio_chunks:
        yield _flac_bytes(np.concatenate(audio_chunks))
    else:
        yield _flac_bytes(np.empty((0, 2), dtype=np.float32))


def stream_headers() -> dict[str, str]:
    """Return headers describing the FLAC response.

    Returns:
        dict[str, str]: Response headers for a FLAC response.
    """
    return {
        "X-Audio-Format": "flac-pcm24",
        "X-Sample-Rate": str(BASE_SR),
        "X-Channels": "2",
    }


def _remember_speech_job(job_id: str, job: SpeechJob) -> None:
    """Store one speech job and remove expired entries."""
    with speech_jobs_lock:
        _delete_expired_speech_jobs(time.time())
        speech_jobs[job_id] = job


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
    status: str,
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


def _collect_speech_job(job_id: str, chunks: queue.Queue) -> None:
    """Consume a speech stream queue and store its final FLAC payload."""
    _update_speech_job(job_id, status="running")
    try:
        audio = b"".join(audio_bytes(chunks))
    except Exception as e:
        _update_speech_job(job_id, status="failed", error=str(e))
        return

    _update_speech_job(job_id, status="completed", audio=audio)


class RootResponse(BaseModel):
    """Response returned by the API root endpoint."""

    status: str


class VersionResponse(BaseModel):
    """Response returned by the API version endpoint."""

    version: str


class SpeakRequest(BaseModel):
    """Request body for asking Celune to speak."""

    content: str = Field(min_length=1)
    save: bool = True


class VoiceRequest(BaseModel):
    """Request body for changing Celune's voice."""

    voice_name: str = Field(min_length=1)


class ActionResponse(BaseModel):
    """Generic accepted control response."""

    status: str


@api.get("/v1", response_model=RootResponse)
def root() -> RootResponse:
    """Celune API root endpoint.

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
    """Celune API version endpoint.

    Returns:
        VersionResponse: The underlying Celune version the API is connected to.
    """
    return VersionResponse(version=f"Celune {__version__}")


@api.post("/v1/speak", response_model=None)
def speak(body: SpeakRequest) -> Union[StreamingResponse, JSONResponse]:
    """Queue speech and stream generated audio chunks back to the caller.

    Args:
        body: A speech request body.

    Returns:
        Union[StreamingResponse, JSONResponse]: The corresponding audio stream, or a JSON error payload if
            generation failed.
    """
    celune = require_celune()
    api_log("SPEAK(SYNC)", body.content)
    chunks = celune.say_stream(body.content, save=body.save)
    if chunks is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": "I'm currently busy. Try again later.",
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
    """
    celune = require_celune()
    api_log("SPEAK(ASYNC)", body.content)
    chunks = celune.say_stream(body.content, save=body.save)
    if chunks is None:
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": "I'm currently busy. Try again later.",
            },
        )

    job_id = uuid.uuid4().hex
    location = f"/v1/speak/jobs/{job_id}"
    _remember_speech_job(job_id, SpeechJob(status="queued", created_at=time.time()))
    threading.Thread(
        target=_collect_speech_job,
        args=(job_id, chunks),
        daemon=True,
        name=f"CeluneSpeechJob-{job_id[:8]}",
    ).start()

    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "job_id": job_id, "location": location},
        headers={"Location": location},
    )


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
                "message": "I don't know that speech job.",
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


@api.post("/v1/voice", response_model=ActionResponse)
def voice(body: VoiceRequest) -> Union[ActionResponse, JSONResponse]:
    """Change Celune's active voice.

    Args:
        body: A voice change request body.

    Returns:
        Union[ActionResponse, JSONResponse]: The voice change response, or a JSON error payload if
            the voice change failed.
    """
    celune = require_celune()
    api_log("VOICE", body.voice_name)

    if body.voice_name not in celune.voices:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_value",
                "message": "I don't know how to speak in that voice.",
            },
        )

    if not celune.set_voice_and_wait(body.voice_name):
        return JSONResponse(
            status_code=500,
            content={
                "error": "request_failed",
                "message": "I can't change my voice right now.",
            },
        )

    return ActionResponse(status="ok")


@api.post("/v1/sfx", response_model=None)
async def sfx(
    file: UploadFile = File(...),
    keep: bool = Form(True),
) -> Union[StreamingResponse, JSONResponse]:
    """Play an uploaded sound effect file and stream the audio chunks back to the caller.

    Args:
        file: The sound effect file to use with the request.
        keep: Whether Celune should hold this sound effect until the next utterance.

    Returns:
        Union[StreamingResponse, JSONResponse]: The corresponding audio stream, or a JSON error payload if
            playback failed.
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
                "message": "That sound is too large for me to play.",
            },
        )

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
        audio = _resample_audio(np.asarray(audio, dtype=np.float32), sr)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_audio",
                "message": f"I can't understand that sound file: {format_error(e, celune.dev)}",
            },
        )

    if not celune.play_audio(audio, BASE_SR, label=filename, keep=keep):
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": "I can't play that right now.",
            },
        )

    def chunks() -> Iterator[bytes]:
        yield _flac_bytes(audio)

    return StreamingResponse(
        chunks(),
        media_type="audio/flac",
        headers=stream_headers(),
    )


def run_api(
    celune: Optional["Celune"] = None,
    host: Optional[str] = None,
    port: int = 2060,
    token: Optional[str] = None,
    requests_per_minute: int = 60,
    on_started: Optional[Callable[[str, int], None]] = None,
) -> None:
    """Start the Celune API.

    Args:
        celune: Running Celune instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.
        token: Token required for API requests.
        requests_per_minute: Maximum requests allowed per client each minute.
        on_started: Callback called after the server socket is listening.

    Returns:
        None: This function starts the API in the background.
    """
    if celune is not None:
        bind_celune(celune)

    configure_api_security(token=token, requests_per_minute=requests_per_minute)
    bind_host = resolve_api_host(token=auth_token, host=host)

    def _default_started(bhost: str, bport: int) -> None:
        http = "http"
        message = f"Celune API has started on {http}://{bhost}:{bport}"
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
    server.run()


def start_api(
    celune: "Celune",
    host: Optional[str] = None,
    port: int = 2060,
    token: Optional[str] = None,
    requests_per_minute: int = 60,
    startup_timeout: float = 5.0,
) -> threading.Thread:
    """Start the Celune API in a background thread.

    Args:
        celune: Running Celune instance to expose through the API.
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
        """Log that Uvicorn has successfully started listening."""
        http = "http"
        celune.log(f"Celune API has started on {http}://{bind_host}:{bind_port}")
        started.set()

    def _runner() -> None:
        """Run the API server without taking Celune down on failure."""
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
                celune.log(f"API runner has exited. Exit code {exc.code}", "warning")
        except Exception as e:
            failed.set()
            celune.log(
                f"Could not start the API: {format_error(e, celune.dev)}", "warning"
            )

    thread = threading.Thread(target=_runner, daemon=True, name="CeluneAPI")
    thread.start()
    deadline = time.monotonic() + max(0.0, startup_timeout)
    while not started.is_set() and not failed.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)

    if not started.is_set() and not failed.is_set():
        celune.log(
            f"API runner has not responded after {startup_timeout:.1f}s, and has timed out.",
            "warning",
        )

    return thread
