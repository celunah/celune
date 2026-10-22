# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and worker helpers for pipeline tests."""

from __future__ import annotations
import threading
from pathlib import Path
from unittest import mock
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Union, Optional, cast
from collections.abc import Callable, Iterator, Sequence

import numpy as np

from celune.typing.aliases import AudioChunk
from celune.agent.runtime import AgentRuntime
from celune.agent.routing import AgentInputRouter
from celune.pipeline import playback, generation
from celune.typing.common import JSONSerializable
from celune.persona.memory import PersonaMemoryStore
from celune.core.reload import reload_model_for_language
from celune.typing.agent import AgentInputClassification, AgentClassificationResult

if TYPE_CHECKING:
    from celune.celune import Celune


class LanguageAwareBackend:
    """Tiny backend fake that reloads when the requested language changes."""

    default_voice = "balanced"

    def __init__(self) -> None:
        self.current_language = "en"
        self.unload_model = mock.Mock(side_effect=self._clear_model)
        self.load_model = mock.Mock(side_effect=self._load_model)

    def _clear_model(self) -> None:
        """Pretend to unload the active model."""

    def _load_model(self, model_id: str, **kwargs: JSONSerializable) -> mock.Mock:
        """Return a fake model and remember the requested language."""
        self.current_language = cast(str, kwargs.get("lang", "en"))
        return mock.Mock(model_id=model_id, kwargs=kwargs)

    @staticmethod
    def resolve_generation_language(lang: Optional[str]) -> str:
        """Normalize empty or unsupported language requests to English."""
        if not lang or lang == "Auto":
            return "en"
        return lang

    def should_reload_for_language(self, lang: Optional[str]) -> bool:
        """Reload when the requested language differs from the active one."""
        return self.resolve_generation_language(lang) != self.current_language

    @staticmethod
    def model_id_for_voice(_voice: str) -> str:
        """Resolve the fake voice to one model identifier."""
        return "fake/balanced"

    @staticmethod
    def generate_stream(
        model: mock.Mock, **kwargs: JSONSerializable
    ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
        """Yield one deterministic chunk and preserve kwargs for assertions."""
        del model, kwargs
        yield np.zeros((8, 2), dtype=np.float32), 48000, None


class StubEmbeddingMemoryStore(PersonaMemoryStore):
    """Memory store with deterministic test embeddings."""

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        *,
        semantic_similarity_threshold: float = 0.62,
        fallback_token_overlap_threshold: int = 1,
        automatic_max_age_days: Optional[int] = 60,
        embedding_model: str = "stub",
        embedding_map: Optional[dict[str, tuple[float, ...]]] = None,
    ) -> None:
        super().__init__(
            storage_dir=storage_dir,
            semantic_similarity_threshold=semantic_similarity_threshold,
            fallback_token_overlap_threshold=fallback_token_overlap_threshold,
            automatic_max_age_days=automatic_max_age_days,
            embedding_model=embedding_model,
        )
        self.embedding_map = embedding_map or {}
        self.return_none = False

    def _embed_texts(self, texts: Sequence[str]) -> Optional[list[np.ndarray]]:
        """Return deterministic vectors for the configured text fixtures."""
        if self.return_none:
            return None
        return [np.array(self.embedding_map[text], dtype=np.float32) for text in texts]


async def run_generation_worker(engine: Celune) -> None:
    """Run generation with the engine-bound language reload callback."""
    await generation.generation_worker_job(
        engine,
        reload_callback(engine),
    )


def reload_callback(engine: Celune) -> Callable[[Optional[str]], None]:
    """Bind the core language-reload operation to one test engine."""
    return partial(reload_model_for_language, engine)


async def run_playback_worker(engine: Celune) -> None:
    """Run the async playback worker directly inside the test loop."""
    await playback.playback_worker_job(engine)


def prepare_persona_conversation_engine(engine: Celune) -> None:
    """Install inert task routing and worker state for one Persona test."""

    def route_input(
        _text: str,
        *,
        persona_ready: bool,
    ) -> AgentClassificationResult:
        del persona_ready
        return AgentClassificationResult(
            classification=AgentInputClassification.CONVERSATION,
            confidence=1.0,
        )

    engine.agent_runtime = cast(
        AgentRuntime,
        SimpleNamespace(get_active_task=lambda _session_id: None),
    )
    engine.agent_router = cast(
        AgentInputRouter,
        SimpleNamespace(route=route_input),
    )
    engine._model_lock = threading.RLock()
    engine._persona_queue = engine.persona_queue
    engine._persona_thread = None
    engine._reload_pending = False
    engine.persona_loading = False
    engine.persona_ready = True


def wait_for_persona_worker(engine: Celune) -> None:
    """Join the Persona worker so a test observes its completed side effects."""
    worker = engine._persona_thread
    if worker is None:
        return
    worker.join(timeout=5.0)
    if worker.is_alive():
        raise AssertionError("Persona worker did not finish within five seconds")
