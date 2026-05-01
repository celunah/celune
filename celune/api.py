import datetime
import queue
import struct
import threading
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .dsp import _resample_audio, _split

if TYPE_CHECKING:
    from .celune import Celune

api = FastAPI(title="CeluneAPI")
_celune: Optional["Celune"] = None


def bind_celune(celune: "Celune") -> None:
    """Bind the running Celune instance to API routes."""
    global _celune
    _celune = celune


def require_celune() -> "Celune":
    """Return the bound Celune instance or fail the request."""
    if _celune is None:
        raise HTTPException(status_code=503, detail="Celune is not available")
    return _celune


def client_ip(request: Request) -> str:
    """Resolve the caller IP address from a FastAPI request."""
    if request.client is None:
        return "unknown"
    return request.client.host


def api_log(ip_address: str, action: str, content: str, suffix: str = "") -> None:
    """Print the API control log line."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    escaped = content.replace("\n", "\\n").replace('"', '\\"')
    print(f'[{timestamp}] {ip_address} {action} "{escaped}"{suffix}', flush=True)


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
            b"RIFF",
            struct.pack("<I", unknown_size),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ),
            b"data",
            struct.pack("<I", unknown_size),
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


class SFXRequest(BaseModel):
    """Request body for playing a sound effect."""

    sfx: str = Field(min_length=1)
    keep: bool = True


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


@api.post("/v1/speak")
def speak(body: SpeakRequest, request: Request) -> StreamingResponse:
    """Queue speech and stream generated audio chunks back to the caller."""
    celune = require_celune()
    api_log(client_ip(request), "SPEAK", body.content)
    chunks = celune.say_stream(body.content, save=body.save)
    if chunks is None:
        raise HTTPException(status_code=409, detail="Celune is currently busy")

    return StreamingResponse(
        audio_bytes(chunks),
        media_type="audio/wav",
        headers=stream_headers(),
    )


@api.post("/v1/voice", response_model=ActionResponse)
def voice(body: VoiceRequest, request: Request) -> ActionResponse:
    """Change Celune's active voice."""
    celune = require_celune()
    api_log(client_ip(request), "VOICE", body.voice_name)

    if body.voice_name not in celune.voices:
        raise HTTPException(status_code=400, detail="Unknown voice")

    if not celune.set_voice_and_wait(body.voice_name):
        raise HTTPException(
            status_code=500,
            detail="Celune could not change voice",
        )

    return ActionResponse(status="ok")


@api.post("/v1/sfx")
def sfx(body: SFXRequest, request: Request) -> StreamingResponse:
    """Play an SFX file and stream the audio chunks back to the caller."""
    celune = require_celune()
    api_log(client_ip(request), "SFX", body.sfx, f" (keep={body.keep})")

    if not celune.play(body.sfx, keep=body.keep):
        raise HTTPException(
            status_code=500, detail="Celune could not play this sound effect"
        )

    def chunks() -> Iterator[bytes]:
        """Yield the SFX file as 48 kHz stereo float32 chunks."""
        yield wav_header()
        audio, sr = sf.read(body.sfx, dtype="float32")
        audio = _resample_audio(np.asarray(audio, dtype=np.float32), sr)
        for chunk in _split(audio, 48000, celune.chunk_size):
            yield float32_to_pcm24(chunk)

    return StreamingResponse(
        chunks(),
        media_type="audio/wav",
        headers=stream_headers(),
    )


def run_api(
    celune: Optional["Celune"] = None,
    host: str = "0.0.0.0",
    port: int = 2060,
) -> None:
    """Start the Celune API.

    Args:
        celune: Running Celune instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.

    Returns:
        None: This function starts the API in the background.
    """
    if celune is not None:
        bind_celune(celune)

    uvicorn.run(
        api,
        host=host,
        port=port,
        log_level="warning",
    )


def start_api(
    celune: "Celune",
    host: str = "0.0.0.0",
    port: int = 2060,
) -> threading.Thread:
    """Start the Celune API in a background thread.

    Args:
        celune: Running Celune instance to expose through the API.
        host: The IP address to bind to.
        port: The port to bind to.

    Returns:
        threading.Thread: The daemon thread running the API server.
    """

    def _runner() -> None:
        """Run the API server without taking Celune down on failure."""
        try:
            run_api(celune, host=host, port=port)
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"Celune API could not start: {exc}", flush=True)
        except Exception as exc:
            print(f"Celune API could not start: {exc}", flush=True)

    thread = threading.Thread(target=_runner, daemon=True, name="CeluneAPI")
    thread.start()
    return thread
