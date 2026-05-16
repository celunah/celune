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


class FakeBackend(CeluneBackend):
    name = "fake"
    chunk_rate = 12.5
    supported_languages = ("en",)
    voice_models = {"balanced": "fake/balanced", "bold": "fake/bold"}
    default_voice = "balanced"

    @staticmethod
    def model_is_available_locally(model: str) -> tuple[bool, Optional[str]]:
        return True, model

    def preload_models(self) -> None:
        return None

    def load_model(self, model_id: str, **kwargs) -> Any:
        return {"model_id": model_id, "kwargs": kwargs}

    def generate_stream(
        self, model: Any, **kwargs
    ) -> Any:
        yield np.zeros((8, 2), dtype=np.float32), 48000, {"chunk_steps": 2}


class FakeGlow:
    def __init__(self, color: str) -> None:
        self.color = color
        self.connect_failed = False
        self.started = False
        self.entered = False
        self.scheduled: list[npt.NDArray[np.float32]] = []

    def start(self) -> bool:
        self.started = True
        return True

    def enter(self) -> None:
        self.entered = True

    def leave(self) -> None:
        return None

    def stop(self, reset: bool = True, wait: bool = False) -> None:
        return None

    def schedule(self, audio: npt.NDArray[np.float32]) -> None:
        self.scheduled.append(audio)


class FakeStream:
    def __init__(self) -> None:
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.written: list[npt.NDArray[np.float32]] = []

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True

    def write(self, audio: npt.NDArray[np.float32]) -> None:
        self.written.append(audio)


def make_pipeline_engine() -> SimpleNamespace:
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
    engine.status_callback = (
        lambda msg, severity="info": statuses.append((msg, severity))
    )
    engine.progress_callback = lambda current, total: progress.append((current, total))
    engine.messages = messages
    engine.errors = errors
    engine.statuses = statuses
    engine.progress = progress
    return engine
