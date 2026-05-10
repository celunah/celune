import datetime
import io
import os
import queue
import struct
import threading
import time
from collections import defaultdict, deque
from hmac import compare_digest
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Union

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .dsp import _resample_audio, _split
from .utils import format_error

if TYPE_CHECKING:
    from .celune import Celune

api = FastAPI(title="CeluneAPI")
bound_celune: Optional["Celune"] = None
auth_token: Optional[str] = None
rate_limit_per_minute = 60
rate_limit_lock = threading.Lock()
rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
max_sfx_upload_bytes = 25 * 1024 * 1024


class StartedServer(uvicorn.Server):
    """Uvicorn server that reports when socket binding actually succeeds."""

    def __init__(
        self,
        config: uvicorn.Config,
        on_started: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the started-server notification wrapper.

        Args:
            config: Uvicorn server configuration.
            on_started: Optional callback invoked after successful startup.

        Returns:
            None: This constructor stores the startup callback.
        """
        super().__init__(config)
        self.on_started = on_started

    async def startup(self, sockets: Optional[list[Any]] = None) -> None:
        """Run Uvicorn startup and report only after the server is listening."""
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
    """Configure API authentication and rate limiting."""
    global auth_token, rate_limit_per_minute

    auth_token = _clean_token(token) or _env_auth_token()
    rate_limit_per_minute = max(0, int(requests_per_minute))
    with rate_limit_lock:
        rate_limit_hits.clear()


def resolve_api_host(token: Optional[str] = None, host: Optional[str] = None) -> str:
    """Resolve the API bind host from authentication state."""
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
    """Apply token authentication and a simple per-client rate limit."""
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
    """Bind the running Celune instance to API routes."""
    global bound_celune
    bound_celune = celune


def require_celune() -> "Celune":
    """Return the bound Celune instance or fail the request."""
    if bound_celune is None:
        raise HTTPException(
            status_code=503,
            detail="I'm not currently available.",
        )
    return bound_celune


def api_log(action: str, content: str, suffix: str = "") -> None:
    """Print the API control log line."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preview = content.replace("\n", "\\n").replace("\r", "\\r")[:64]
    if len(content) > 64:
        preview += "..."
    print(f"[{timestamp}] {action} {preview!r}{suffix}", flush=True)


def wav_header() -> bytes:
    """Return a streamable 48 kHz stereo PCM24 WAV header."""
    channels = 2
    sample_rate = 48000
    bits_per_sample = 24
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    unknown_size = 0xFFFFFFFF

    return b"".join(
        (
            b"RIFF",  # RIFF header
            struct.pack("<I", unknown_size),  # treat as stream
            b"WAVE",  # WAV file
            b"fmt ",  # format
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                channels,  # stereo
                sample_rate,  # 48 kHz
                byte_rate,  # 48 kHz * 2 channels * 24 bits // 8
                block_align,  # 2 channels * 24 bits // 8
                bits_per_sample,  # 24 bits
            ),
            b"data",  # data
            struct.pack("<I", unknown_size),  # PCM data
        )
    )


def float32_to_pcm24(audio: np.ndarray) -> bytes:
    """Convert normalized float32 audio to signed 24-bit little-endian PCM."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] != 2:
        audio = audio.T

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    scaled = np.where(clipped < 0.0, clipped * 8388608.0, clipped * 8388607.0)
    pcm32 = scaled.astype("<i4", copy=False).reshape(-1)
    pcm8 = pcm32.view(np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(pcm8[:, :3]).tobytes()


def audio_bytes(chunks: queue.Queue) -> Iterator[bytes]:
    """Yield a WAV stream from 48 kHz stereo float32 audio chunks."""
    yield wav_header()
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item

        yield float32_to_pcm24(item)


def stream_headers() -> dict[str, str]:
    """Return headers describing the WAV stream."""
    return {
        "X-Audio-Format": "wav-pcm24",
        "X-Sample-Rate": "48000",
        "X-Channels": "2",
    }


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
        RootResponse: The API is working.
    """
    return RootResponse(status="ok")


@api.get("/v1/version", response_model=VersionResponse)
def version() -> VersionResponse:
    """Celune API version endpoint.

    Returns:
        VersionResponse: The underlying Celune version the API is connected to.
    """
    return VersionResponse(version=f"Celune {__version__}")


@api.post("/v1/speak", response_model=None)
def speak(body: SpeakRequest) -> Union[StreamingResponse, JSONResponse]:
    """Queue speech and stream generated audio chunks back to the caller."""
    celune = require_celune()
    api_log("SPEAK", body.content)
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
        media_type="audio/wav",
        headers=stream_headers(),
    )


@api.post("/v1/voice", response_model=ActionResponse)
def voice(body: VoiceRequest) -> Union[ActionResponse, JSONResponse]:
    """Change Celune's active voice."""
    celune = require_celune()
    api_log("VOICE", body.voice_name)

    if body.voice_name not in celune.voices:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_value",
                "message": "I am not able to speak in that tone.",
            },
        )

    if not celune.set_voice_and_wait(body.voice_name):
        return JSONResponse(
            status_code=500,
            content={
                "error": "request_failed",
                "message": "I can't change my tone right now.",
            },
        )

    return ActionResponse(status="ok")


@api.post("/v1/sfx", response_model=None)
async def sfx(
    file: UploadFile = File(...),
    keep: bool = Form(True),
) -> Union[StreamingResponse, JSONResponse]:
    """Play an uploaded SFX file and stream the audio chunks back to the caller."""
    celune = require_celune()
    filename = file.filename or "uploaded SFX"
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

    if not celune.play_audio(audio, 48000, label=filename, keep=keep):
        return JSONResponse(
            status_code=409,
            content={
                "error": "not_ready",
                "message": "I can't play that right now.",
            },
        )

    def chunks() -> Iterator[bytes]:
        """Yield the uploaded SFX as 48 kHz stereo float32 chunks."""
        yield wav_header()
        for chunk in _split(audio, 48000, celune.chunk_size):
            yield float32_to_pcm24(chunk)

    return StreamingResponse(
        chunks(),
        media_type="audio/wav",
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
        """Report API startup to Celune logs or stdout.

        Args:
            bhost: Bound host address.
            bport: Bound port number.

        Returns:
            None: This callback reports startup status.
        """
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
