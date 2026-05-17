# SPDX-License-Identifier: MIT
"""Lightweight test fakes for Celune's unit test suite."""

import queue
import threading
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

import numpy as np
import numpy.typing as npt

from celune.backends.base import CeluneBackend
from celune.utils import discard


class FakeBackend(CeluneBackend):
    """Tiny backend implementation used by tests without loading real models."""

    name = "fake"
    chunk_rate = 12.5
    supported_languages = ("en",)
    voice_models = {"balanced": "fake/balanced", "bold": "fake/bold"}
    default_voice = "balanced"

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        """Pretend that a model is available locally.

        Args:
            model: The model identifier requested by the caller.

        Returns:
            tuple[bool, Optional[str]]: Availability and the fake local path.
        """
        return True, model

    def preload_models(self) -> None:
        """Pretend to preload models without performing work.

        Returns:
            None: This fake intentionally performs no work.
        """
        return None

    def load_model(self, model_id: str, **kwargs) -> Any:
        """Return lightweight model metadata for one fake model.

        Args:
            model_id: The requested fake model identifier.
            **kwargs: Backend-specific load arguments preserved for assertions.

        Returns:
            Any: A dictionary describing the requested fake model.
        """
        return {"model_id": model_id, "kwargs": kwargs}

    def generate_stream(self, model: Any, **kwargs) -> Any:
        """Yield one deterministic fake audio chunk.

        Args:
            model: The fake model object passed by the caller.
            **kwargs: Generation arguments accepted for interface compatibility.

        Returns:
            Any: An iterator yielding one fake audio chunk.
        """
        del model, kwargs
        yield np.zeros((8, 2), dtype=np.float32), 48000, {"chunk_steps": 2}


class FakeGlow:
    """Minimal RGB glow fake that records lifecycle calls."""

    def __init__(self, color: str) -> None:
        """Initialize fake glow state.

        Args:
            color: The configured glow color.

        Returns:
            None: Constructors initialize state in place.
        """
        self.color = color
        self.connect_failed = False
        self.started = False
        self.entered = False
        self.finished = threading.Event()
        self.finished.set()
        self.scheduled: list[npt.NDArray[np.float32]] = []

    def start(self) -> bool:
        """Mark the fake glow as started.

        Returns:
            bool: Always ``True`` for the fake implementation.
        """
        self.started = True
        return True

    def enter(self) -> None:
        """Record that Celune entered the ready state.

        Returns:
            None: This helper mutates fake state in place.
        """
        self.entered = True

    def leave(self) -> None:
        """Accept a leave request without performing hardware work.

        Returns:
            None: This fake intentionally performs no work.
        """

    @staticmethod
    def stop(reset: bool = True, wait: bool = False) -> None:
        """Accept a stop request without performing hardware work.

        Args:
            reset: Whether real devices would be reset.
            wait: Whether a real worker would be joined.

        Returns:
            None: This fake intentionally performs no work.
        """
        discard(reset)
        discard(wait)

    def schedule(self, audio: npt.NDArray[np.float32]) -> None:
        """Record audio scheduled for glow processing.

        Args:
            audio: The audio chunk scheduled by the caller.

        Returns:
            None: This helper appends to fake state.
        """
        self.scheduled.append(audio)


class FakeStream:
    """Minimal output-stream fake that records lifecycle operations."""

    def __init__(self) -> None:
        """Initialize fake stream state.

        Returns:
            None: Constructors initialize state in place.
        """
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.written: list[npt.NDArray[np.float32]] = []

    def stop(self) -> None:
        """Record a graceful stream stop.

        Returns:
            None: This helper mutates fake state in place.
        """
        self.stopped = True

    def abort(self) -> None:
        """Record an immediate stream abort.

        Returns:
            None: This helper mutates fake state in place.
        """
        self.aborted = True

    def close(self) -> None:
        """Record stream closure.

        Returns:
            None: This helper mutates fake state in place.
        """
        self.closed = True

    def write(self, audio: npt.NDArray[np.float32]) -> None:
        """Record one written audio chunk.

        Args:
            audio: The audio chunk written by the caller.

        Returns:
            None: This helper appends to fake state.
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
    engine.use_normalization = False
    engine.normalize = mock.Mock(return_value=None)
    engine.is_in_tutorial = False
    engine.model_ready = threading.Event()
    engine.model_ready.set()
    engine.loaded = True
    engine.locked = False
    engine.cur_state = "idle"
    engine.text_queue = queue.Queue()
    engine.audio_queue = queue.Queue()
    engine.say_lock = threading.Lock()
    engine.queue_lock = threading.Lock()
    engine.playback_done = threading.Event()
    engine.playback_done.set()
    engine.utterance_force_stop = threading.Event()
    engine.kept_sfx_audio = None
    engine.force_stop_marker = object()
    engine.log = lambda msg, severity="info": messages.append((msg, severity))
    engine.log_dev = lambda msg, severity="info": messages.append((msg, severity))
    engine.error_callback = errors.append
    engine.status_callback = lambda msg, severity="info": statuses.append(
        (msg, severity)
    )
    engine.progress_callback = lambda current, total: progress.append((current, total))
    engine.messages = messages
    engine.errors = errors
    engine.statuses = statuses
    engine.progress = progress
    return engine
