# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for independently collected core tests."""

import queue
import threading
from unittest import mock
from typing import Optional
from collections.abc import Callable

from celune.celune import Celune
from celune.core import lifecycle as core_lifecycle
from celune.core.lifecycle import _close_speech_pipeline as close_pipeline
from celune.dataclasses.celune import (
    CeluneAudioState,
    CeluneModelState,
    CeluneVoiceState,
    CeluneBackendState,
    CeluneRuntimeState,
    CeluneCallbackState,
    CelunePipelineState,
)

from ..support import FakeGlow, FakeBackend, CeluneTestCase


class CeluneCoreTestSupport(CeluneTestCase):
    """Shared engine construction and cleanup helpers without test methods."""

    _cached_celune: Optional[Celune] = None
    _cached_instance_keys: frozenset[str] = frozenset()

    @staticmethod
    def _close_celune(celune: Celune) -> None:
        """Close a test instance if it still owns the singleton slot."""
        if Celune._instance is celune:
            celune.close()

    @classmethod
    def _reset_cached_celune(cls, celune: Celune) -> None:
        """Restore the reusable default test instance to constructor state."""
        if celune._playback_thread is not None:
            close_pipeline(celune)
        if celune.vision is not None or celune._persona_load_thread is not None:
            core_lifecycle.unload_persona_state(celune)

        persona_thread = getattr(celune, "_persona_thread", None)
        if (
            persona_thread is not None
            and persona_thread is not threading.current_thread()
        ):
            persona_thread.join(timeout=2)
        persona_queue = getattr(celune, "_persona_queue", None)
        if persona_queue is not None:
            while True:
                try:
                    persona_queue.get_nowait()
                except queue.Empty:
                    break

        for name in set(celune.__dict__) - cls._cached_instance_keys:
            delattr(celune, name)

        backend = FakeBackend(log=core_lifecycle._noop_message, fatal=celune.fatal)
        celune._callbacks = CeluneCallbackState(
            log_callback=core_lifecycle._noop_message,
            status_callback=core_lifecycle._noop_message,
            error_callback=lambda _error: None,
            idle_callback=lambda: None,
            queue_avail_callback=lambda: None,
            voice_changed_callback=lambda _name: None,
            change_input_state_callback=core_lifecycle._noop_input_state,
            change_voice_lock_state_callback=core_lifecycle._noop_voice_lock_state,
            progress_callback=core_lifecycle._noop_progress,
            caption_progress_callback=core_lifecycle._noop_progress,
            caption_callback=core_lifecycle._noop_caption,
        )
        celune._backend_state = CeluneBackendState(
            config={},
            backend_spec=FakeBackend,
            backend_kwargs={},
            backend=backend,
            tts_backend=backend.name,
            input_mode="text_to_speech",
            chunk_size=8,
        )
        celune._model_state = CeluneModelState()
        celune._voice_state = CeluneVoiceState()
        celune._pipeline_state = CelunePipelineState(audio_queue=queue.Queue(maxsize=8))
        celune._audio_state = CeluneAudioState()
        celune._runtime_state = CeluneRuntimeState()
        celune._async_runtime_lock = threading.Lock()
        celune._voice_reload_guard = threading.Lock()
        celune._voice_reload_active = False
        celune._event_dispatcher = type(celune._event_dispatcher)(
            log_warning=celune.log,
            log_level="info",
        )
        glow = FakeGlow("#cebaff", celune=celune)
        setattr(celune, "glow", glow)  # noqa: B010
        core_lifecycle._wrap_fatal_glow(celune)
        glow.start()
        celune._model_ready.set()
        celune._playback_done.set()
        Celune._instance = None

    @classmethod
    def _cache_celune(cls, celune: Celune) -> None:
        """Install close tracking on one reusable default test instance."""
        original_close = celune.close

        def close_cached() -> None:
            """Close and invalidate the reusable default test instance."""
            if cls._cached_celune is celune:
                cls._cached_celune = None
            original_close()

        celune.close = close_cached
        cls._cached_celune = celune

    @classmethod
    def tearDownClass(cls) -> None:
        """Close the cached default test instance after the core suite."""
        if cls._cached_celune is not None:
            cached_celune = cls._cached_celune
            cls._cached_celune = None
            cached_celune.close()

    def _make_celune(
        self,
        config: dict,
        startup_callback: Optional[Callable[[str], None]] = None,
    ) -> Celune:
        """Build a Celune instance with lightweight fakes."""
        if not config and startup_callback is None:
            cached_celune = type(self)._cached_celune
            if cached_celune is not None:
                type(self)._reset_cached_celune(cached_celune)
                type(self)._cache_celune(cached_celune)
                return cached_celune

        with (
            mock.patch("celune.core.lifecycle.AudioRGBGlow", FakeGlow),
            mock.patch("celune.core.lifecycle.default_loader", return_value=None),
            mock.patch(
                "celune.core.lifecycle.persona_is_available", return_value=False
            ),
        ):
            celune = Celune(
                config=config,
                tts_backend=FakeBackend,
                startup_callback=startup_callback,
            )
            if not config and startup_callback is None:
                owner = type(self)
                owner._cached_instance_keys = frozenset(celune.__dict__)
                owner._cache_celune(celune)
                Celune._instance = None
            else:
                self.addCleanup(self._close_celune, celune)
            return celune

    @staticmethod
    def _immediate_thread(*args, **kwargs):
        """Return a thread stub whose ``start()`` runs the target immediately."""
        target = kwargs.get("target")
        if target is None and args:
            target = args[0]
        target_args = kwargs.get("args", ())

        class _ImmediateThread:
            """Immediate thread harness used by hot-reload tests."""

            @staticmethod
            def start() -> None:
                """Run the target synchronously."""
                if target is not None:
                    target(*target_args)

        return _ImmediateThread()
