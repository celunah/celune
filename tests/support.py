# SPDX-License-Identifier: MIT
"""Lightweight test fakes for Celune's unit test suite."""

from __future__ import annotations

import contextlib
import importlib
import queue
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Optional, TypedDict
from unittest import mock

import numpy as np
import numpy.typing as npt

from celune.backends.tts.base import CeluneBackend
from celune.backends.vc.base import CeluneVCBackend
from celune.constants import PipelineStates
from celune.dataclasses.pipeline import AudioOutput, VoiceConversionRequest
from celune.typing.aliases import AudioChunk
from celune.typing.common import JSONSerializable
from celune.utils import discard

if TYPE_CHECKING:
    from celune.celune import Celune


class FakeModel(TypedDict):
    """Metadata returned by the fake test backend."""

    model_id: str
    kwargs: dict[str, JSONSerializable]


class FakeBackend(CeluneBackend):
    """Tiny backend implementation used by tests without loading real models."""

    name = "fake"
    chunk_rate = 12.5
    supported_languages = ("en",)
    voice_models = MappingProxyType({"balanced": "fake/balanced", "bold": "fake/bold"})
    default_voice = "balanced"
    is_fake = True

    def __init__(
        self,
        log: Optional[Callable[[str, str], None]] = None,
        fatal: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(
            log=log or (lambda _msg, _severity="info": None),
            fatal=fatal,
        )

    def model_is_available_locally(
        self, model: str, lang: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Pretend that a model is available locally.

        Args:
            model: The model identifier requested by the caller.
            lang: The language identifier for differentiating models by language.

        Returns:
            tuple[bool, Optional[str]]: Availability and the fake local path.
        """
        discard(lang)
        return True, model

    def preload_models(self) -> None:
        """Pretend to preload models without performing work."""

    def load_model(self, model_id: str, **kwargs: JSONSerializable) -> FakeModel:
        """Return lightweight model metadata for one fake model.

        Args:
            model_id: The requested fake model identifier.
            kwargs: Backend-specific load arguments preserved for assertions.

        Returns:
            FakeModel: A dictionary describing the requested fake model.
        """
        return {"model_id": model_id, "kwargs": kwargs}

    def generate_stream(
        self, model: FakeModel, **kwargs: JSONSerializable
    ) -> Iterator[tuple[AudioChunk, int, dict[str, int]]]:
        """Yield one deterministic fake audio chunk.

        Args:
            model: The fake model passed by the caller.
            kwargs: Generation arguments accepted for interface compatibility.

        Returns:
            Iterator[tuple[npt.NDArray[np.float32], int, dict[str, int]]]: An iterator yielding one fake audio chunk.
        """
        discard(model)
        discard(kwargs)
        yield np.zeros((8, 2), dtype=np.float32), 48000, {"chunk_steps": 2}


class FakeVCBackend(CeluneVCBackend):
    """Tiny VC backend implementation used by tests without model work."""

    name = "fake-vc"

    def convert(self, request: VoiceConversionRequest) -> AudioOutput:
        """Return the source audio unchanged for one voice-conversion request.

        Args:
            request: The voice-conversion request under test.

        Returns:
            AudioOutput: Playable audio copied from the request payload.
        """
        return AudioOutput(
            audio=np.asarray(request.source_audio, dtype=np.float32).copy(),
            sample_rate=request.sample_rate,
            label=request.label,
        )


class FakeGlow:
    """Minimal RGB glow fake that records lifecycle calls."""

    def __init__(
        self,
        color: str,
        celune: Optional[Celune] = None,
        host: str = "127.0.0.1",
        port: int = 6742,
    ) -> None:
        """Initialize fake glow state."""
        self.color = color
        self.celune = celune
        self.host = host
        self.port = port
        self.connect_failed = False
        self.started = False
        self.entered = False
        self.fatal_called = False
        self.sleep_called = False
        self.wake_called = False
        self.finished = threading.Event()
        self.finished.set()
        self.scheduled: list[AudioChunk] = []
        self.reset_audio_reactivity_called = False

    def start(self) -> bool:
        """Mark the fake glow as started.

        Returns:
            bool: Always ``True`` for the fake implementation.
        """
        self.started = True
        return True

    def enter(self) -> None:
        """Record that Celune entered the ready state."""
        self.entered = True

    def leave(self) -> None:
        """Accept a leave request without performing hardware work."""

    def fatal(self) -> None:
        """Record that Celune entered a fatal glow state."""
        self.fatal_called = True

    def sleep(self) -> None:
        """Record that Celune requested sleep dimming."""
        self.sleep_called = True

    def wake(self) -> None:
        """Record that Celune requested brightness restoration."""
        self.wake_called = True

    @staticmethod
    def stop(reset: bool = True, wait: bool = False) -> None:
        """Accept a stop request without performing hardware work.

        Args:
            reset: Whether real devices would be reset.
            wait: Whether a real worker would be joined.
        """
        discard(reset)
        discard(wait)

    def schedule(self, audio: npt.NDArray[np.float32]) -> None:
        """Record audio scheduled for glow processing.

        Args:
            audio: The audio chunk scheduled by the caller.
        """
        self.scheduled.append(audio)

    def reset_audio_reactivity(self) -> None:
        """Record that live audio-reactive glow state was cleared."""
        self.reset_audio_reactivity_called = True


class FakeStream:
    """Minimal output-stream fake that records lifecycle operations."""

    def __init__(self) -> None:
        """Initialize fake stream state."""
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.written: list[AudioChunk] = []

    def start(self) -> None:
        """Record stream startup."""
        self.started = True

    def stop(self) -> None:
        """Record a graceful stream stop."""
        self.stopped = True

    def abort(self) -> None:
        """Record an immediate stream abort."""
        self.aborted = True

    def close(self) -> None:
        """Record stream closure."""
        self.closed = True

    def write(self, audio: npt.NDArray[np.float32]) -> None:
        """Record one written audio chunk.

        Args:
            audio: The audio chunk written by the caller.
        """
        self.written.append(audio)


def make_pipeline_engine() -> SimpleNamespace:
    """Build a lightweight engine-shaped object for pipeline tests.

    Returns:
        SimpleNamespace: An object exposing the pipeline attributes under test.
    """
    messages: list[tuple[str, str]] = []
    errors: list[str] = []
    statuses: list[tuple[str, str]] = []
    progress: list[tuple[Optional[float], Optional[float]]] = []
    engine = SimpleNamespace()
    engine.backend = SimpleNamespace(supported_languages=("en",))
    engine.vc_backend = None
    engine.config = {}
    engine.input_mode = "text_to_speech"
    engine.language = "Auto"
    engine.log_level = "info"
    engine.current_voice = "balanced"
    engine.current_character = None
    engine.persona_attachments = []
    engine.persona_recent_visual_context = ()
    engine.use_normalization = False
    engine.normalize = mock.Mock(return_value=None)
    engine.is_in_tutorial = False
    engine.model_ready = threading.Event()
    engine.model_ready.set()
    engine.loaded = True
    engine.locked = False
    engine.cur_state = "idle"
    engine.exit_requested = False
    engine.stream = None
    engine._stream = None
    engine.current_sr = None
    engine._current_sr = None
    engine.audio_unavailable = False
    engine._audio_unavailable = False
    engine.smart_buffer_generation_speed = None
    engine.smart_buffer_target_seconds = 0.0
    engine.total_generated_speech_seconds = 0.0
    engine.historical_generated_speech_seconds = 0.0
    engine.text_queue = queue.Queue()
    engine.audio_queue = queue.Queue()
    engine.say_lock = threading.Lock()
    engine.queue_lock = threading.Lock()
    engine.playback_done = threading.Event()
    engine.playback_done.set()
    engine.persona_queue = queue.Queue()
    engine.utterance_force_stop = threading.Event()
    engine.speech_generation = 0
    engine._playback_generation = 0
    engine.kept_sfx_audio = None
    engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
    engine.log = lambda msg, severity="info", **kwargs: messages.append((msg, severity))
    engine.error_callback = errors.append
    engine.status_callback = lambda msg, severity="info": statuses.append(
        (msg, severity)
    )
    engine.progress_callback = lambda current, total: progress.append((current, total))
    engine.idle_callback = mock.Mock()
    engine.glow = SimpleNamespace(
        schedule=mock.Mock(),
        reset_audio_reactivity=mock.Mock(),
    )
    engine.messages = messages
    engine.errors = errors
    engine.statuses = statuses
    engine.progress = progress
    return engine


def make_voice_loader(
    voice: str,
    metadata: dict[str, JSONSerializable],
) -> SimpleNamespace:
    """Return a simple CEVOICE loader stub for one named voice.

    Args:
        voice: The voice identifier to use.
        metadata: The voice metadata.

    Returns:
        SimpleNamespace: A CEVOICE loader stub for the given voice.
    """
    return SimpleNamespace(
        bundle=SimpleNamespace(voices={voice: metadata}, voice_order=(voice,)),
        materialize=lambda ref_voice, kind: Path(f"{ref_voice}.{kind}"),
    )


@contextlib.contextmanager
def mock_qwen3_backend():
    """Import the Qwen3 backend with a stub faster-qwen3-tts package."""

    class StubQwen3TTS:
        """Import-time stand-in for the FasterQwen3TTS package class."""

    with mock.patch.dict(
        sys.modules,
        {
            "faster_qwen3_tts": SimpleNamespace(
                FasterQwen3TTS=StubQwen3TTS,
                __version__="0.2.5",
            )
        },
    ):
        qwen3 = importlib.import_module("celune.backends.tts.qwen3")
        yield qwen3.Qwen3


@contextlib.contextmanager
def mock_voxcpm_backend():
    """Import the VoxCPM2 backend with a stub voxcpm package."""

    class StubVoxCPM:
        """Import-time stand-in for the VoxCPM package class."""

    with mock.patch.dict(
        sys.modules,
        {"voxcpm": SimpleNamespace(VoxCPM=StubVoxCPM)},
    ):
        voxcpm2 = importlib.import_module("celune.backends.tts.voxcpm2")
        yield voxcpm2.VoxCPM2


@contextlib.contextmanager
def mock_dotstts_backend():
    """Import the dots.tts backend with a stub dots_tts package."""

    class StubDotsTtsRuntime:
        """Import-time stand-in for the dots.tts runtime class."""

    package = ModuleType("dots_tts")
    package.__path__ = []
    runtime_module = ModuleType("dots_tts.runtime")
    runtime_module.DotsTtsRuntime = StubDotsTtsRuntime  # type: ignore[missing-attribute]
    package.runtime = runtime_module  # type: ignore[missing-attribute]

    with mock.patch.dict(
        sys.modules,
        {
            "dots_tts": package,
            "dots_tts.runtime": runtime_module,
        },
    ):
        dotstts = importlib.import_module("celune.backends.tts.dotstts")
        yield dotstts.DotsTtsMF


@contextlib.contextmanager
def mock_mini_backend():
    """Import the Mini backend with a stub pocket-tts package."""

    class StubTTSModel:
        """Import-time stand-in for the Pocket TTS package class."""

    with mock.patch.dict(
        sys.modules,
        {"pocket_tts": SimpleNamespace(TTSModel=StubTTSModel)},
    ):
        mini = importlib.import_module("celune.backends.tts.mini")
        yield mini.Mini
