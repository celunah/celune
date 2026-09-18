# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline helpers that do not perform real synthesis."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import sys
import queue
import tempfile
import threading
from types import TracebackType, SimpleNamespace
from typing import Self, Optional, cast
from pathlib import Path
from unittest import mock
from collections.abc import Iterator

import numpy as np
import pytest
import numpy.typing as npt

from celune import conversation as conversation_module, pipeline
from celune.utils import discard
from celune.celune import Celune
from celune.cevoice import (
    CEVoicePersona,
    PersonaIdentity,
    PersonaStyleValues,
)
from celune.constants import PipelineStates
from celune.typing.agent import (
    AgentTask,
    AgentContext,
    AgentRequest,
)
from celune.typing.common import JSON, JSONSerializable
from celune.typing.aliases import AudioChunk
from celune.typing.locks import ComponentLockName
from celune.dataclasses.pipeline import AudioInputRequest
from celune.persona.capabilities import PersonaCapabilities

from .support import (
    FakeStream,
    FakeVCBackend,
    CeluneTestCase,
    CeluneAsyncTestCase,
    make_voice_loader,
    make_pipeline_engine,
)
from .platform import LINUX_ONLY, WINDOWS_ONLY


class TestPipeline(CeluneTestCase):
    """Tests for lightweight pipeline behavior."""

    def test_pipeline_cpu_config_has_conservative_defaults(self) -> None:
        """Verify playback pressure protection defaults to a small bounded window."""
        engine = make_pipeline_engine()

        assert pipeline._pipeline_cpu_config(cast(Celune, engine)) == (
            True,
            12.0,
            4,
            0.001,
        )

        engine.config = {}
        assert pipeline._pipeline_cpu_config(cast(Celune, engine)) == (
            True,
            12.0,
            4,
            0.001,
        )

    def test_playback_contention_grows_reserve_after_lag_and_underflow(self) -> None:
        """Verify contention evidence expands the playback reserve target."""
        engine = make_pipeline_engine()
        monitor = pipeline._PlaybackContentionMonitor(cast(Celune, engine))

        assert monitor.target_seconds() == 2.0
        monitor.observe_scheduler_lag(0.2)
        assert monitor.target_seconds() > 2.0

        monitor.observe_write(0.05, 0.05, underflowed=True)
        assert engine.playback_underflows == 1
        assert engine.playback_contention_level == 1.0
        assert monitor.target_seconds() == 30.0
        assert monitor.capacity_seconds() == 30.0
        assert monitor.requires_rebuffer()

    def test_pipeline_cpu_config_ignores_removed_user_configuration(self) -> None:
        """Verify playback protection remains bounded regardless of engine config."""
        engine = make_pipeline_engine()
        engine.config = {"pipeline_cpu": {"enabled": False}}

        assert pipeline._pipeline_cpu_config(cast(Celune, engine)) == (
            True,
            12.0,
            4,
            0.001,
        )

    class _LanguageAwareBackend:
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
            """Normalize empty or unsupported language requests to English.

            Args:
                lang: The language identifier for differentiating models by language.

            Returns:
                str: A fake language model identifier.
            """
            if not lang or lang == "Auto":
                return "en"
            return lang

        def should_reload_for_language(self, lang: Optional[str]) -> bool:
            """Reload when the requested language differs from the active one.

            Args:
                lang: The language identifier for differentiating models by language.

            Returns:
                str: Whether a model reload would occur for this request.
            """
            return self.resolve_generation_language(lang) != self.current_language

        @staticmethod
        def model_id_for_voice(_voice: str) -> str:
            """Resolve the fake voice to one model identifier.

            Args:
                _voice: Unused voice value.

            Returns:
                str: The fake model identifier.
            """
            return "fake/balanced"

        @staticmethod
        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            """Yield one deterministic chunk and preserve kwargs for assertions.

            Args:
                model: The fake model object.
                kwargs: The keyword arguments that would be used with generation.
            """
            discard(model)
            discard(kwargs)
            yield np.zeros((8, 2), dtype=np.float32), 48000, None

    def test_queue_helpers_and_force_stop_cover_busy_and_idle_paths(self) -> None:
        """Verify queue draining, lock handling, and force-stop behavior.

        Raises:
            AssertionError: Pipeline helper behavior changes unexpectedly.
        """
        q: queue.Queue[int] = queue.Queue()
        q.put(1)
        q.put(2)
        pipeline.clear_queue(q)
        assert q.empty()

        engine = make_pipeline_engine()
        celune_engine = cast(Celune, engine)
        assert pipeline.acquire_pipeline(celune_engine, "speak")
        assert engine.locked
        assert not pipeline.acquire_pipeline(celune_engine, "speak")
        pipeline.release_pipeline(celune_engine)
        assert not engine.locked
        assert engine.cur_state == "idle"

        assert not pipeline.force_stop_speech(celune_engine)
        engine.locked = True
        engine.backend.cancel_active_request = mock.Mock()
        engine.text_queue.put("pending")
        engine.audio_queue.put("audio")
        assert pipeline.force_stop_speech(celune_engine)
        assert engine._speech_generation == 1
        assert engine.text_queue.empty()
        assert engine.persona_queue.empty()
        assert engine.audio_queue.get_nowait() is engine.force_stop_marker

    def test_close_invalidates_active_speech_before_worker_teardown(self) -> None:
        """Verify shutdown cancels speech generations before joining workers."""
        engine = make_pipeline_engine()
        engine.text_queue.put("pending speech")
        engine.persona_queue.put("pending Persona")
        engine.audio_queue.put("pending audio")
        engine.sentinel = PipelineStates.TERMINATE
        engine.generation_thread = None
        engine.playback_thread = None
        engine.glow.leave = mock.Mock()
        engine.glow.finished = threading.Event()
        engine.glow.finished.set()

        pipeline.close(cast(Celune, engine))

        self.assertTrue(engine._exit_requested)
        self.assertTrue(engine.utterance_force_stop.is_set())
        self.assertEqual(engine._speech_generation, 1)
        self.assertEqual(engine._playback_generation, 1)
        self.assertIs(engine.text_queue.get_nowait(), engine.sentinel)
        self.assertTrue(engine.persona_queue.empty())
        self.assertIs(engine.audio_queue.get_nowait(), engine.force_stop_marker)
        self.assertIs(engine.audio_queue.get_nowait(), engine.sentinel)

    def test_cancelled_speech_generation_cannot_queue_playback(self) -> None:
        """Verify a backend chunk racing with stop is rejected atomically."""
        engine = make_pipeline_engine()
        pipeline.register_playback_source(cast(Celune, engine), 1, kind="speech")
        engine._speech_generation = 2
        engine._active_speech_generation = 1
        engine.utterance_force_stop.set()

        queued = pipeline._queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.zeros((8, 2), dtype=np.float32),
            48000,
        )

        assert not queued
        assert engine.audio_queue.empty()

    def test_playback_queue_trace_includes_reserve_state(self) -> None:
        """Verify playback queue traces expose current reserve diagnostics."""
        engine = make_pipeline_engine()
        engine.playback_buffer_seconds = 3.25
        engine.playback_contention_level = 0.75
        engine.playback_underflows = 2

        assert pipeline._queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.zeros((8, 2), dtype=np.float32),
            48000,
        )

        message = engine.messages[-1][0]
        assert "reserve=3.25s" in message
        assert "contention=0.75" in message
        assert "underflows=2" in message
        assert "queue_wait=" in message
        assert "generation_gap=0.000s" in message
        assert "rebuffer_wait=0.000s" in message

    def test_playback_queue_trace_records_generation_and_queue_wait(self) -> None:
        """Verify playback traces record producer timing around the bounded queue."""
        engine = make_pipeline_engine()
        audio = np.zeros((8, 2), dtype=np.float32)

        assert pipeline._queue_playback_chunk(cast(Celune, engine), 1, audio, 48000)
        assert pipeline._queue_playback_chunk(cast(Celune, engine), 1, audio, 48000)

        assert engine.playback_queue_wait_seconds >= 0.0
        assert engine.playback_generation_gap_seconds >= 0.0
        assert "generation_gap=" in engine.messages[-1][0]

    def test_playback_writer_records_wait_gap_and_write_duration(self) -> None:
        """Verify the persistent writer exposes each application-side timing stage."""
        engine = make_pipeline_engine()
        monitor = pipeline._PlaybackContentionMonitor(cast(Celune, engine))
        writer = pipeline._PlaybackWriter(cast(Celune, engine), monitor)
        pipeline.register_playback_source(cast(Celune, engine), 1, kind="speech")
        pipeline._playback_source_meta(cast(Celune, engine))[1]["total_frames"] = 8.0
        engine.caption_progress_callback = mock.Mock()

        with mock.patch(
            "celune.pipeline._write_playback_block",
            side_effect=lambda _engine, _audio: False,
        ):
            writer.start()
            writer.submit(np.zeros((8, 2), dtype=np.float32), (1,))
            writer.wait_empty()
            writer.stop()

        assert engine.playback_writer_wait_seconds >= 0.0
        assert engine.playback_writer_gap_seconds == 0.0
        assert engine.playback_writer_write_seconds >= 0.0
        assert (
            pipeline._playback_source_meta(cast(Celune, engine))[1]["played_frames"]
            == 8.0
        )
        engine.caption_progress_callback.assert_called_once_with(8.0, 8.0)
        assert not any(
            "[PLAY] playback write" in message for message, _ in engine.messages
        )

    def test_force_stop_queues_worker_stop_and_invalidates_old_sources(
        self,
    ) -> None:
        """Verify stop delegates stream teardown and rejects old mixer audio."""
        engine = make_pipeline_engine()
        engine.locked = True
        engine.cur_state = "speaking"
        engine.playback_done.clear()
        fake_stream = FakeStream()
        engine.stream = fake_stream
        celune_engine = cast(Celune, engine)
        pipeline.register_playback_source(celune_engine, 1, kind="sfx")
        pipeline.set_playback_source_status(celune_engine, 1, "Playing fixture")
        old_generation = engine._playback_generation

        assert pipeline.force_stop_speech(celune_engine)

        assert engine._playback_generation == old_generation + 1
        assert not fake_stream.aborted
        assert engine.stream is fake_stream
        assert engine.audio_queue.get_nowait() == engine.force_stop_marker
        assert not pipeline._queue_playback_chunk(
            celune_engine,
            1,
            np.zeros((8, 2), dtype=np.float32),
            48000,
            generation=old_generation,
        )

    def test_working_signal_completion_does_not_notify_idle(self) -> None:
        """Verify the transitional working cue is not treated as a readiness idle event."""
        engine = make_pipeline_engine()
        engine.cur_state = "reloading"

        assert pipeline.play_signal(cast(Celune, engine), "working")

        queued = list(engine.audio_queue.queue)
        done_markers = [
            item for item in queued if isinstance(item, pipeline.PlaybackSourceDone)
        ]
        assert len(done_markers) == 1
        assert not done_markers[0].notify_idle
        assert engine.cur_state == "reloading"

    def test_sleeping_signal_preserves_sleeping_state(self) -> None:
        """Verify the sleeping cue does not classify Celune as speaking."""
        engine = make_pipeline_engine()

        assert pipeline.play_signal(cast(Celune, engine), "sleeping")

        assert engine.cur_state == "sleeping"

    def test_current_playback_status_returns_latest_active_source(self) -> None:
        """Verify polling can recover the latest active playback status."""
        engine = make_pipeline_engine()
        pipeline.set_playback_source_status(
            cast(Celune, engine),
            1,
            "Playing first",
        )
        pipeline.set_playback_source_status(
            cast(Celune, engine),
            2,
            "Playing second",
        )

        assert (
            pipeline.current_playback_status(cast(Celune, engine)) == "Playing second"
        )

    def test_readiness_signal_does_not_block_concurrent_speech_queueing(self) -> None:
        """Verify the readiness cue does not briefly reject speech as busy."""
        engine = make_pipeline_engine()
        queued_during_signal: list[bool] = []
        original_register = pipeline.register_playback_source

        def register_and_queue(
            engine_arg: Celune,
            source_id: int,
            *,
            kind: str,
            base_gain: float = 1.0,
        ) -> None:
            with mock.patch(
                "celune.speech.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                queued_during_signal.append(
                    pipeline.queue_speech(cast(Celune, engine), "hello")
                )
            original_register(
                engine_arg,
                source_id,
                kind=kind,
                base_gain=base_gain,
            )

        with mock.patch(
            "celune.pipeline._register_playback_source",
            side_effect=register_and_queue,
        ):
            assert pipeline.play_signal(cast(Celune, engine), "readiness")

        assert queued_during_signal == [True]
        request = engine.text_queue.get_nowait()
        assert request.text == "hello"


@pytest.mark.anyio
class TestPipelineAsync(CeluneAsyncTestCase):
    """Tests for async pipeline entry points."""

    _LanguageAwareBackend = TestPipeline._LanguageAwareBackend

    @staticmethod
    async def _run_generation_worker(engine: Celune) -> None:
        """Run the async generation worker directly inside the test loop."""
        await pipeline.generation_worker_job(engine)

    @staticmethod
    async def _run_playback_worker(engine: Celune) -> None:
        """Run the async playback worker directly inside the test loop."""
        await pipeline.playback_worker_job(engine)

    async def test_playback_input_reader_reuses_one_persistent_thread(self) -> None:
        """Verify playback input waits do not create one thread per timeout."""
        engine = make_pipeline_engine()
        engine.sentinel = PipelineStates.TERMINATE
        reader = pipeline._PlaybackInputReader(cast(Celune, engine))
        reader.start()
        reader_thread = reader._thread

        engine.audio_queue.put("first")
        assert await reader.get() == "first"
        assert reader._thread is reader_thread

        reader.stop()
        assert reader._thread is None

    async def test_queue_speech_async_waits_for_model_readiness_in_daemon_thread(
        self,
    ) -> None:
        """Verify speech queueing offloads model-ready waits from the event loop."""
        engine = make_pipeline_engine()
        engine.model_ready.clear()

        def mark_ready(*, timeout: Optional[float]) -> bool:
            del timeout
            engine.model_ready.set()
            return True

        engine.model_ready.wait = mock.Mock(side_effect=mark_ready)
        run_in_daemon_thread = mock.AsyncMock(side_effect=lambda function: function())

        with mock.patch("celune.speech._run_in_daemon_thread", run_in_daemon_thread):
            queued = await pipeline.queue_speech_async(
                cast(Celune, engine),
                "hello",
                display_text="shown",
            )

        assert queued
        engine.model_ready.wait.assert_called_once_with(timeout=0.1)
        assert run_in_daemon_thread.await_count == 1
        request = engine.text_queue.get_nowait()
        assert request.text == "hello"
        assert request.display_text == "shown"

    async def test_pipeline_blocking_work_uses_daemon_threads(self) -> None:
        """Verify blocked pipeline work cannot hold asyncio executor shutdown open."""
        assert await pipeline._run_in_daemon_thread(
            lambda: threading.current_thread().daemon
        )

    def test_queue_speech_handles_success_and_failure_paths(self) -> None:
        """Verify speech queueing success and rejection paths.

        Raises:
            AssertionError: Speech queueing behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        celune_engine = cast(Celune, engine)
        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.queue_speech(celune_engine, "hello", display_text="shown")
        request = engine.text_queue.get_nowait()
        assert request.text == "hello"
        assert request.display_text == "shown"
        assert request.language == "en"
        assert engine.statuses[-1] == ("Generating", "info")

    def test_queue_speech_does_not_admit_work_during_model_reload(self) -> None:
        """Verify a speech request is rejected at reload admission instead of deadlocking."""
        engine = make_pipeline_engine()
        engine._reload_pending = True
        engine.model_ready.clear()

        queued = pipeline.queue_speech(cast(Celune, engine), "hello")

        assert not queued
        assert not engine.locked
        assert engine._last_component_busy is not None
        assert engine._last_component_busy.components == (
            ComponentLockName.MODEL_LOADING,
        )

    def test_persona_wait_does_not_cross_a_model_reload_boundary(self) -> None:
        """Verify Persona waits for reload completion before changing engine state."""
        engine = make_pipeline_engine()
        engine._reload_pending = True
        wait_calls = 0

        def wait_for_playback(*, timeout: Optional[float]) -> bool:
            nonlocal wait_calls
            del timeout
            wait_calls += 1
            if wait_calls == 2:
                engine._reload_pending = False
                engine.cur_state = "idle"
            return True

        engine.playback_done.wait = mock.Mock(side_effect=wait_for_playback)

        assert conversation_module._wait_for_persona_playback(cast(Celune, engine))
        assert wait_calls == 2

        engine = make_pipeline_engine()
        engine.use_normalization = True
        engine.normalize = mock.Mock(return_value="normalized")
        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.queue_speech(cast(Celune, engine), "raw")
        engine.normalize.assert_not_called()
        request = engine.text_queue.get_nowait()
        assert request.text == "raw"
        assert request.language == "en"
        assert request.normalize

        engine = make_pipeline_engine()
        engine.language = "fr"
        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.queue_speech(cast(Celune, engine), "hello")
        request = engine.text_queue.get_nowait()
        assert request.language == "fr"

        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(name="qwen3", supported_languages=("en",))
        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.queue_speech(cast(Celune, engine), "hello")
        request = engine.text_queue.get_nowait()
        assert request.language == "Auto"

        engine = make_pipeline_engine()
        engine.is_in_tutorial = True
        assert not pipeline.queue_speech(cast(Celune, engine), "hello")
        assert engine.messages[-1][1] == "warning"

        engine = make_pipeline_engine()
        engine.loaded = False
        assert not pipeline.queue_speech(cast(Celune, engine), "hello")
        assert engine.errors == ["Celune is not currently ready"]

    def test_queue_speech_normalizes_tts_text_without_changing_display_text(
        self,
    ) -> None:
        """Normalize technical speech input while preserving the visible text."""
        engine = make_pipeline_engine()
        raw_text = r'{status: "ok"} C:\Users\user foo_bar.py'

        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            self.assertTrue(
                pipeline.queue_speech(
                    cast(Celune, engine),
                    raw_text,
                    display_text=raw_text,
                )
            )

        request = engine.text_queue.get_nowait()
        self.assertEqual(
            request.text,
            "status, ok C drive, Users, user foo underscore bar dot py",
        )
        self.assertEqual(request.display_text, raw_text)

    def test_handle_audio_input_accepts_and_ignores_audio_by_default(self) -> None:
        """Verify engine-level audio input is a safe explicit no-op in TTS mode."""
        engine = make_pipeline_engine()
        engine.log = mock.Mock()
        engine.loaded = True
        engine.locked = False
        engine.cur_state = "idle"
        audio = np.ones((16, 2), dtype=np.float32)
        request = AudioInputRequest(audio=audio, sample_rate=48000, label="mic test")

        result = pipeline.handle_audio_input(cast(Celune, engine), request)

        assert result
        assert engine.text_queue.empty()
        assert engine.audio_queue.empty()
        assert engine.cur_state == "idle"
        engine.log.assert_called_once()
        assert engine.log.call_args.kwargs["loglevel"] == "verbose"

    def test_handle_audio_input_routes_to_vc_backend_in_voice_conversion_mode(
        self,
    ) -> None:
        """Verify VC mode sends audio input through the configured VC backend."""
        engine = make_pipeline_engine()
        engine.input_mode = "voice_conversion"
        engine.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        engine.current_voice = "balanced"
        engine.current_character = "Celune"
        audio = np.ones((16, 2), dtype=np.float32)
        request = AudioInputRequest(audio=audio, sample_rate=48000, label="mic test")

        convert_mock = mock.Mock(
            return_value=SimpleNamespace(
                audio=np.asarray(audio, dtype=np.float32).copy(),
                sample_rate=48000,
                label="mic test",
            )
        )
        engine.vc_backend.convert = convert_mock
        loader = make_voice_loader("balanced", {"reference_text": "Pack reference."})

        with (
            mock.patch("celune.speech.default_loader", return_value=loader),
            mock.patch("celune.speech.queue_sfx_audio", return_value=True) as q,
        ):
            result = pipeline.handle_audio_input(cast(Celune, engine), request)

        assert result
        convert_mock.assert_called_once()
        vc_request = convert_mock.call_args.args[0]
        assert vc_request.target_references == (Path("balanced.wav"),)
        assert vc_request.pitch_shift == 0
        assert not vc_request.f0_condition
        q.assert_called_once()
        queued_audio = q.call_args.args[1]
        assert q.call_args.args[2] == 48000
        assert q.call_args.args[3] == "mic test"
        assert q.call_args.kwargs["status_label_key"] == "pipeline.revoicing_label"
        assert queued_audio.shape == (16, 2)
        assert queued_audio is not audio
        assert np.array_equal(queued_audio, audio)
        assert engine.text_queue.empty()

    def test_handle_audio_input_reports_missing_vc_backend_cleanly(self) -> None:
        """Verify VC mode surfaces a clean error when no VC backend is configured."""
        engine = make_pipeline_engine()
        engine.input_mode = "voice_conversion"
        engine.vc_backend = None
        engine.log = mock.Mock()
        audio = np.ones((8, 2), dtype=np.float32)

        result = pipeline.handle_audio_input(
            cast(Celune, engine),
            AudioInputRequest(audio=audio, sample_rate=24000, label="fixture"),
        )

        assert not result
        engine.log.assert_called_once()
        assert engine.errors == ["Voice conversion backend is not configured."]
        assert engine.audio_queue.empty()

    def test_handle_audio_input_normalizes_audio_before_vc_boundary(self) -> None:
        """Verify VC requests cannot send non-finite or out-of-range samples."""
        engine = make_pipeline_engine()
        engine.input_mode = "voice_conversion"
        engine.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        engine.current_voice = "balanced"
        engine.current_character = "Celune"
        source_audio = np.array([np.nan, 2.0, -2.0, np.inf], dtype=np.float32)
        convert_mock = mock.Mock(
            return_value=SimpleNamespace(
                audio=np.zeros(4, dtype=np.float32),
                sample_rate=48000,
                label="mic test",
            )
        )
        engine.vc_backend.convert = convert_mock
        loader = make_voice_loader("balanced", {"reference_text": "Pack reference."})

        with (
            mock.patch("celune.speech.default_loader", return_value=loader),
            mock.patch("celune.speech.queue_sfx_audio", return_value=True),
        ):
            result = pipeline.handle_audio_input(
                cast(Celune, engine),
                AudioInputRequest(
                    audio=source_audio,
                    sample_rate=48000,
                    label="mic test",
                ),
            )

        assert result
        request = convert_mock.call_args.args[0]
        assert np.all(np.isfinite(request.source_audio))
        assert float(np.max(np.abs(request.source_audio))) <= 0.95

    def test_handle_audio_input_applies_engine_vc_pitch_shift_to_output(
        self,
    ) -> None:
        """Verify VC routing applies the configured pitch shift to converted output."""
        engine = make_pipeline_engine()
        engine.input_mode = "voice_conversion"
        engine.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        engine.current_voice = "balanced"
        engine.current_character = "Celune"
        engine.vc_pitch_shift = -5
        request = AudioInputRequest(
            audio=np.ones((12, 2), dtype=np.float32),
            sample_rate=48000,
            label="mic test",
        )
        convert_mock = mock.Mock(
            return_value=SimpleNamespace(
                audio=np.ones((12, 2), dtype=np.float32),
                sample_rate=48000,
                label="mic test",
            )
        )
        engine.vc_backend.convert = convert_mock
        loader = make_voice_loader("balanced", {"reference_text": "Pack reference."})

        with (
            mock.patch("celune.speech.default_loader", return_value=loader),
            mock.patch("celune.speech.queue_sfx_audio", return_value=True),
            mock.patch(
                "celune.speech.pitch_shift_audio",
                return_value=np.ones((12, 2), dtype=np.float32) * 0.25,
            ) as shift_audio,
        ):
            result = pipeline.handle_audio_input(cast(Celune, engine), request)

        assert result
        assert convert_mock.call_args.args[0].pitch_shift == 0
        shift_audio.assert_called_once_with(mock.ANY, 48000, -5)

    def test_handle_audio_input_passes_engine_vc_f0_condition_to_vc_backend(
        self,
    ) -> None:
        """Verify VC routing carries the configured engine conversion mode."""
        engine = make_pipeline_engine()
        engine.input_mode = "voice_conversion"
        engine.vc_backend = FakeVCBackend(log=lambda _msg, _severity="info": None)
        engine.current_voice = "balanced"
        engine.current_character = "Celune"
        engine.vc_f0_condition = True
        request = AudioInputRequest(
            audio=np.ones((12, 2), dtype=np.float32),
            sample_rate=48000,
            label="mic test",
        )
        convert_mock = mock.Mock(
            return_value=SimpleNamespace(
                audio=np.ones((12, 2), dtype=np.float32),
                sample_rate=48000,
                label="mic test",
            )
        )
        engine.vc_backend.convert = convert_mock
        loader = make_voice_loader("balanced", {"reference_text": "Pack reference."})

        with (
            mock.patch("celune.speech.default_loader", return_value=loader),
            mock.patch("celune.speech.queue_sfx_audio", return_value=True),
        ):
            result = pipeline.handle_audio_input(cast(Celune, engine), request)

        assert result
        assert convert_mock.call_args.args[0].f0_condition

    def test_tts_mode_does_not_route_audio_to_vc_backend(self) -> None:
        """Verify the default TTS mode ignores audio instead of invoking VC routing."""
        engine = make_pipeline_engine()
        engine.input_mode = "text_to_speech"
        engine.vc_backend = mock.Mock()

        result = pipeline.handle_audio_input(
            cast(Celune, engine),
            AudioInputRequest(
                audio=np.ones((4, 2), dtype=np.float32),
                sample_rate=16000,
                label="fixture",
            ),
        )

        assert result
        engine.vc_backend.convert.assert_not_called()

    def test_download_youtube_sfx_writes_expected_temp_wav(self) -> None:
        """Verify yt-dlp downloads to Celune's fixed temporary WAV path."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            expected = temp_root / "temp" / "temporary_audio.wav"

            def fake_run(*args, **kwargs):
                discard(args)
                discard(kwargs)
                expected.write_bytes(b"RIFFdemoWAVE")
                return SimpleNamespace(
                    returncode=0,
                    stdout="Fixture Video Title\n",
                    stderr="",
                )

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.playback.subprocess.run", side_effect=fake_run
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved == (expected, "Fixture Video Title")
        command = run.call_args.args[0]
        assert command[0] == sys.executable
        assert command[1:3] == ["-m", "yt_dlp"]
        assert "--print" not in command
        assert str(temp_root / "temp" / "temporary_audio.%(ext)s") in command

    def _assert_download_youtube_sfx_uses_repo_venv_python(
        self, expected_python: str
    ) -> None:
        """Verify a compiled yt-dlp launch uses the expected venv Python."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            expected = temp_root / "temp" / "temporary_audio.wav"

            def fake_run(*args, **kwargs):
                discard(args)
                discard(kwargs)
                expected.write_bytes(b"RIFFdemoWAVE")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch("celune.playback.running_compiled", return_value=True),
                mock.patch("celune.playback.project_root", return_value=Path("/repo")),
                mock.patch(
                    "celune.playback.subprocess.run", side_effect=fake_run
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved == (expected, "Fixture Video Title")
        command = run.call_args.args[0]
        assert command[0] == expected_python
        assert command[1:3] == ["-m", "yt_dlp"]

    @LINUX_ONLY
    def test_download_youtube_sfx_uses_repo_venv_python_on_linux(self) -> None:
        """Verify Linux compiled launches use the repository venv Python."""
        self._assert_download_youtube_sfx_uses_repo_venv_python(
            "/repo/.venv/bin/python"
        )

    @WINDOWS_ONLY
    def test_download_youtube_sfx_uses_repo_venv_python_on_windows(self) -> None:
        """Verify Windows compiled launches use the repository venv Python."""
        self._assert_download_youtube_sfx_uses_repo_venv_python(
            r"\repo\.venv\Scripts\python.exe"
        )

    def test_download_youtube_sfx_passes_optional_authentication_settings(self) -> None:
        """Verify optional YouTube cookies, tokens, and runtime settings reach yt-dlp."""
        engine = make_pipeline_engine()
        engine.config = {
            "youtube": {
                "cookies_file": "C:/private/youtube-cookies.txt",
                "cookies_from_browser": "chrome",
                "po_token": ["web.gvs+gvs-token", "web.player+player-token"],
                "player_client": ["web_embedded", "android_vr"],
                "js_runtimes": "node:C:/Program Files/nodejs/node.exe",
                "remote_components": ["ejs:npm"],
                "extractor_args": ["youtube:player_skip=webpage"],
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            expected = temp_root / "temp" / "temporary_audio.wav"

            def fake_run(*args, **kwargs):
                discard(args)
                discard(kwargs)
                expected.write_bytes(b"RIFFdemoWAVE")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.playback.subprocess.run", side_effect=fake_run
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved == (expected, "Fixture Video Title")
        command = run.call_args.args[0]
        assert (
            command[command.index("--cookies") + 1] == "C:/private/youtube-cookies.txt"
        )
        assert "--cookies-from-browser" not in command
        assert command[command.index("--js-runtimes") + 1] == (
            "node:C:/Program Files/nodejs/node.exe"
        )
        assert command[command.index("--remote-components") + 1] == "ejs:npm"
        assert "youtube:po_token=web.gvs+gvs-token,web.player+player-token" in command
        assert "youtube:player_client=web_embedded,android_vr" in command
        assert "youtube:player_skip=webpage" in command

    def test_download_youtube_sfx_logs_missing_file_state(self) -> None:
        """Verify missing yt-dlp output uses the current no-file warning messages."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.playback.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="postprocessor said something",
                        stderr="",
                    ),
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved is None
        warnings = [msg for msg, severity in engine.messages if severity == "warning"]
        assert run.call_count == 4
        assert warnings[-1] == "Could not download audio: downloader returned no file"
        assert all(
            "postprocessor said something" not in message for message in warnings
        )
        assert engine.errors[-1] == "Could not download YouTube audio"

    def test_download_youtube_sfx_logs_download_failure_state(self) -> None:
        """Verify yt-dlp failures use the current download-failed warning messages."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.playback.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="yt-dlp exploded",
                    ),
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved is None
        warnings = [msg for msg, severity in engine.messages if severity == "warning"]
        assert run.call_count == 4
        assert warnings[-1] == "Could not download audio: yt-dlp exploded"
        assert all("yt-dlp exploded" in warning for warning in warnings)
        assert engine.errors[-1] == "Could not download YouTube audio"

    def test_download_youtube_sfx_compresses_yt_dlp_error_output(self) -> None:
        """Verify noisy yt-dlp warnings collapse to the actionable error reason."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output = (
                "WARNING: [youtube] No supported JavaScript runtime could be found.\n"
                "ERROR: unable to download video data: HTTP Error 403: Forbidden\n"
            )

            with (
                mock.patch(
                    "celune.playback.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.playback.available",
                    return_value=True,
                ),
                mock.patch(
                    "celune.playback._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.playback.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr=output,
                    ),
                ) as run,
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        assert resolved is None
        warnings = [msg for msg, severity in engine.messages if severity == "warning"]
        assert run.call_count == 4
        assert (
            warnings[-1]
            == "Could not download audio: unable to download video data: HTTP Error 403: Forbidden"
        )
        assert all("JavaScript runtime" not in warning for warning in warnings)

    def test_youtube_sfx_title_reads_oembed_title(self) -> None:
        """Verify YouTube titles can be resolved without yt-dlp title output."""

        class FakeResponse:
            """Minimal urlopen response stub."""

            def __enter__(self) -> Self:
                return self

            def __exit__(
                self,
                exc_type: Optional[type[BaseException]],
                exc: Optional[BaseException],
                traceback: Optional[TracebackType],
            ) -> None:
                discard(exc_type)
                discard(exc)
                discard(traceback)

            @staticmethod
            def read() -> bytes:
                """Read a mock video title.

                Returns:
                    bytes: Mock JSON payload returned by the fake HTTP response.
                """
                return b'{"title":"Fixture Video Title"}'

        with mock.patch("celune.playback.urlopen", return_value=FakeResponse()):
            title = pipeline.youtube_sfx_title("https://youtu.be/demo")

        assert title == "Fixture Video Title"

    def test_play_accepts_youtube_url_via_downloaded_wav(self) -> None:
        """Verify YouTube URLs are resolved to a WAV and played as SFX."""
        engine = make_pipeline_engine()
        downloaded = Path("C:/Users/user/AppData/Local/Celune/temporary_audio.wav")
        audio = np.ones((8, 2), dtype=np.float32)
        volume = 0.4

        with (
            mock.patch(
                "celune.speech._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ) as download,
            mock.patch("celune.speech.os.path.exists", return_value=True),
            mock.patch("celune.speech.sf.read", return_value=(audio, 48000)) as read,
            mock.patch(
                "celune.speech.queue_sfx_audio", return_value=True
            ) as queue_audio,
        ):
            ok = pipeline.play(
                cast(Celune, engine),
                "https://www.youtube.com/watch?v=demo",
                keep=True,
                volume=volume,
            )

        assert ok
        download.assert_called_once()
        read.assert_called_once_with(str(downloaded), dtype="float32")
        queued_args = queue_audio.call_args.args
        queued_kwargs = queue_audio.call_args.kwargs
        assert queued_args[0] == cast(Celune, engine)
        np.testing.assert_allclose(queued_args[1], np.asarray(audio, dtype=np.float32))
        assert queued_args[2:] == (48000, "Fixture Video Title", True)
        assert queued_kwargs == {"volume": volume * 0.5}

    def test_play_reports_playing_after_youtube_download(self) -> None:
        """Verify a successful YouTube download replaces the download status."""
        engine = make_pipeline_engine()
        downloaded = Path("C:/Users/user/AppData/Local/Celune/temporary_audio.wav")

        with (
            mock.patch(
                "celune.speech._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ),
            mock.patch("celune.speech.os.path.exists", return_value=True),
            mock.patch(
                "celune.speech.sf.read",
                return_value=(np.ones((8, 2), dtype=np.float32), 48000),
            ),
            mock.patch("celune.speech.queue_sfx_audio", return_value=True),
        ):
            assert pipeline.play(
                cast(Celune, engine),
                "https://www.youtube.com/watch?v=demo",
            )

        assert engine.statuses[-1] == ("Playing Fixture Video Title", "info")

    def test_play_calls_started_callback_before_queueing_audio(self) -> None:
        """Verify callers can report playback before synchronous SFX enqueueing."""
        engine = make_pipeline_engine()
        downloaded = Path("C:/Users/user/AppData/Local/Celune/temporary_audio.wav")
        events: list[str] = []

        def queue_audio(*_args, **_kwargs) -> bool:
            events.append("queued")
            return True

        with (
            mock.patch(
                "celune.speech._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ),
            mock.patch("celune.speech.os.path.exists", return_value=True),
            mock.patch(
                "celune.speech.sf.read",
                return_value=(np.ones((8, 2), dtype=np.float32), 48000),
            ),
            mock.patch("celune.speech.queue_sfx_audio", side_effect=queue_audio),
        ):
            assert pipeline.play(
                cast(Celune, engine),
                "https://www.youtube.com/watch?v=demo",
                on_started=lambda: events.append("started"),
            )

        assert events == ["started", "queued"]

    def test_queue_sfx_audio_allows_overlay_while_speech_pipeline_is_locked(
        self,
    ) -> None:
        """Verify SFX sources can be queued while speech already owns the pipeline."""
        engine = make_pipeline_engine()
        engine.locked = True
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        audio = np.ones((4800, 2), dtype=np.float32) * 0.25

        ok = pipeline.queue_sfx_audio(
            cast(Celune, engine),
            audio,
            48000,
            "fixture",
        )

        assert ok
        assert not engine.playback_done.is_set()
        queued = list(engine.audio_queue.queue)
        assert any(isinstance(item, pipeline.PlaybackChunk) for item in queued)
        assert any(isinstance(item, pipeline.PlaybackSourceDone) for item in queued)

    def test_blocked_playback_put_does_not_block_speech_queueing(self) -> None:
        """Verify a full playback queue cannot prevent a new speech request."""
        engine = make_pipeline_engine()
        put_started = threading.Event()
        release_put = threading.Event()

        class BlockingQueue(queue.Queue):
            """Pause one playback put while allowing other pipeline locks to run."""

            def put(
                self, item: object, block: bool = True, timeout: Optional[float] = None
            ) -> None:
                if not put_started.is_set():
                    put_started.set()
                    release_put.wait(timeout=1.0)
                super().put(item, block=block, timeout=timeout)

        engine.audio_queue = BlockingQueue(maxsize=1)
        pipeline.register_playback_source(cast(Celune, engine), 1, kind="sfx")
        producer = threading.Thread(
            target=lambda: pipeline.queue_playback_chunk(
                cast(Celune, engine),
                1,
                np.zeros((8, 2), dtype=np.float32),
                48000,
            ),
            daemon=True,
        )
        producer.start()
        assert put_started.wait(timeout=1.0)

        speech_result: list[bool] = []

        def queue_speech_request() -> None:
            with mock.patch(
                "celune.speech.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                speech_result.append(
                    pipeline.queue_speech(cast(Celune, engine), "hello")
                )

        speech_thread = threading.Thread(target=queue_speech_request, daemon=True)
        speech_thread.start()
        speech_thread.join(timeout=1.0)

        release_put.set()
        producer.join(timeout=1.0)

        assert not speech_thread.is_alive()
        assert speech_result == [True]
        assert not producer.is_alive()

    async def test_playback_worker_mixes_sources_and_glow_receives_mixed_audio(
        self,
    ) -> None:
        """Verify the DSP mixer sums overlapping sources before playback/probing."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        glow_calls: list[AudioChunk] = []
        engine.glow = SimpleNamespace(
            schedule=lambda audio: glow_calls.append(np.asarray(audio))
        )
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        fake_stream = FakeStream()

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.2, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            2,
            np.full((2400, 2), 0.3, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)
        pipeline.queue_playback_done(cast(Celune, engine), 2)
        engine.audio_queue.put(engine.sentinel)

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        assert fake_stream.started
        assert len(fake_stream.written) == 1
        mixed_audio = np.concatenate(fake_stream.written)
        assert mixed_audio.shape == (2400, 2)
        np.testing.assert_allclose(mixed_audio, 0.5, atol=1e-6)
        assert len(glow_calls) == len(fake_stream.written)
        np.testing.assert_allclose(np.concatenate(glow_calls), 0.5, atol=1e-6)
        assert engine.playback_done.is_set()

    async def test_playback_worker_uses_configured_output_device(self) -> None:
        """Verify playback streams honor the configured output device override."""
        engine = make_pipeline_engine()
        engine.config = {"output_recording_device": "VB-Cable Output"}
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        fake_stream = FakeStream()

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.2, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)
        engine.audio_queue.put(engine.sentinel)

        with mock.patch(
            "celune.pipeline.sd.OutputStream",
            return_value=fake_stream,
        ) as mock_stream:
            await self._run_playback_worker(cast(Celune, engine))

        assert mock_stream.call_args.kwargs["device"] == "VB-Cable Output"
        assert mock_stream.call_args.kwargs["latency"] == "high"

    async def test_playback_worker_records_output_underflow(self) -> None:
        """Verify PortAudio underflows raise the adaptive reserve target."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END

        class UnderflowStream(FakeStream):
            """Output stream fake that reports one inserted underflow."""

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                super().write(audio)
                return True

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.2, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)
        engine.audio_queue.put(engine.sentinel)

        with mock.patch(
            "celune.pipeline.sd.OutputStream",
            return_value=UnderflowStream(),
        ):
            await self._run_playback_worker(cast(Celune, engine))

        assert engine.playback_underflows == 1
        assert engine.playback_contention_level == 1.0

    async def test_playback_worker_logs_friendly_output_device_match_warnings(
        self,
    ) -> None:
        """Verify ambiguous output devices are downgraded to warnings."""
        engine = make_pipeline_engine()
        engine.config = {"output_recording_device": "CABLE-B Input"}
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.2, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)
        engine.audio_queue.put(engine.sentinel)

        with mock.patch(
            "celune.pipeline.resolve_audio_device",
            side_effect=ValueError(
                "the specified output device name has multiple matches for "
                "'CABLE-B Input (VB-Audio Cable B)':\n"
                "- [22] CABLE-B Input (VB-Audio Cable B), Windows DirectSound\n"
                "- [28] CABLE-B Input (VB-Audio Cable B), Windows WASAPI\n\n"
                "please specify one of the above devices, then restart Celune"
            ),
        ):
            await self._run_playback_worker(cast(Celune, engine))

        assert engine.errors[-1] == "No suitable audio devices"
        warning_messages = [
            msg for msg, severity in engine.messages if severity == "warning"
        ]
        assert warning_messages
        assert (
            "the specified output device name has multiple matches"
            in warning_messages[-1]
        )

    async def test_playback_worker_does_not_emit_idle_for_non_idle_completion_marker(
        self,
    ) -> None:
        """Verify non-readiness completion markers cannot snap the runtime back to idle."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.cur_state = "reloading"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        fake_stream = FakeStream()

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.2, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(
            cast(Celune, engine),
            1,
            notify_idle_when_finished=False,
        )
        engine.audio_queue.put(engine.sentinel)

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        engine.idle_callback.assert_not_called()
        assert engine.cur_state == "reloading"
        assert engine.playback_done.is_set()

    async def test_playback_worker_reports_live_audio_progress(self) -> None:
        """Verify playback progress follows audio position without flooding updates."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        fake_stream = FakeStream()

        assert pipeline.queue_sfx_audio(
            cast(Celune, engine),
            np.full((2400 * 8, 2), 0.25, dtype=np.float32),
            48000,
            "progress.wav",
        )
        engine.audio_queue.put(engine.sentinel)

        monotonic_values = iter(i * 0.01 for i in range(500))
        with (
            mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream),
            mock.patch(
                "celune.pipeline._monotonic_time",
                side_effect=lambda: next(monotonic_values),
            ),
        ):
            await self._run_playback_worker(cast(Celune, engine))

        in_flight = [
            (current, total)
            for current, total in engine.progress
            if current is not None
            and total is not None
            and total > 1
            and current < total
        ]
        assert in_flight
        assert len(in_flight) < len(fake_stream.written)
        assert engine.progress[-1] == (1, 1)

    async def test_playback_worker_admits_speech_after_sfx_has_already_started(
        self,
    ) -> None:
        """Verify late-arriving speech reaches the DSP while SFX is still active."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()

        class InjectingStream(FakeStream):
            """A fake injecting stream."""

            def __init__(self) -> None:
                super().__init__()
                self.injected = False

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                super().write(audio)
                if not self.injected:
                    self.injected = True
                    pipeline.queue_playback_chunk(
                        cast(Celune, engine),
                        2,
                        np.full((2400, 2), 0.4, dtype=np.float32),
                        48000,
                    )
                    pipeline.queue_playback_done(
                        cast(Celune, engine),
                        2,
                        release_pipeline_when_finished=True,
                    )
                    engine.audio_queue.put(engine.sentinel)
                return False

        fake_stream = InjectingStream()
        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((9600, 2), 0.1, dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        blocks = fake_stream.written
        assert len(blocks) >= 3
        assert any(np.max(block) > 0.45 for block in blocks[1:])
        assert engine.playback_done.is_set()

    async def test_playback_status_restores_prior_sfx_label_after_speech_finishes(
        self,
    ) -> None:
        """Verify mixed playback restores the prior SFX status after speech ends."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        engine.caption_progress_callback = mock.Mock()

        class InjectingStream(FakeStream):
            """A fake injecting stream."""

            def __init__(self) -> None:
                super().__init__()
                self.injected = False

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                super().write(audio)
                if not self.injected:
                    self.injected = True
                    pipeline.register_playback_source(
                        cast(Celune, engine), 2, kind="speech"
                    )
                    pipeline.set_playback_source_status(
                        cast(Celune, engine), 2, "Speaking"
                    )
                    pipeline.queue_playback_chunk(
                        cast(Celune, engine),
                        2,
                        np.full((2400, 2), 0.4, dtype=np.float32),
                        48000,
                    )
                    pipeline.queue_playback_done(
                        cast(Celune, engine),
                        2,
                        release_pipeline_when_finished=True,
                    )
                    engine.audio_queue.put(engine.sentinel)
                return False

        fake_stream = InjectingStream()
        assert pipeline.queue_sfx_audio(
            cast(Celune, engine),
            np.full((9600, 2), 0.1, dtype=np.float32),
            48000,
            "loop.wav",
        )

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        statuses = [msg for msg, _ in engine.statuses]
        assert "Playing loop.wav" in statuses
        assert "Speaking" in statuses
        speaking_index = statuses.index("Speaking")
        assert "Playing loop.wav" in statuses[speaking_index + 1 :]
        engine.caption_progress_callback.assert_called_with(2400.0, 2400.0)

    async def test_playback_worker_ducks_sfx_to_quarter_and_restores_with_fades(
        self,
    ) -> None:
        """Verify speech ducks SFX to 25 percent, then fades it back up."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()

        class InjectingStream(FakeStream):
            """A fake stream that injects speech after SFX has started."""

            def __init__(self) -> None:
                super().__init__()
                self.injected = False

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                super().write(audio)
                if not self.injected:
                    self.injected = True
                    pipeline.register_playback_source(
                        cast(Celune, engine), 2, kind="speech"
                    )
                    pipeline.set_playback_source_status(
                        cast(Celune, engine), 2, "Speaking"
                    )
                    for _ in range(3):
                        pipeline.queue_playback_chunk(
                            cast(Celune, engine),
                            2,
                            np.zeros((2400, 2), dtype=np.float32),
                            48000,
                        )
                    pipeline.queue_playback_done(
                        cast(Celune, engine),
                        2,
                        release_pipeline_when_finished=True,
                    )
                    engine.audio_queue.put(engine.sentinel)
                return False

        fake_stream = InjectingStream()
        assert pipeline.queue_sfx_audio(
            cast(Celune, engine),
            np.ones((2400 * 12, 2), dtype=np.float32),
            48000,
            "duck.wav",
            volume=0.8,
        )

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        means = [float(np.mean(block)) for block in fake_stream.written]
        assert len(means) >= 6
        assert means[0] > 0.79
        assert min(means) < 0.45
        min_index = means.index(min(means))
        assert min_index > 0
        assert means[min_index] < means[0]
        assert means[-1] > means[min_index] + 0.25
        assert means[-1] > 0.7

    async def test_force_stop_resets_glow_audio_reactivity(self) -> None:
        """Verify forced playback stop clears the glow's audio-reactive state."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.current_voice = "balanced"
        engine.idle_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        fake_stream = FakeStream()

        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.3, dtype=np.float32),
            48000,
        )
        engine.audio_queue.put(engine.force_stop_marker)
        engine.audio_queue.put(engine.sentinel)

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        engine.glow.reset_audio_reactivity.assert_called_once_with()
        assert engine.playback_done.is_set()
        engine.idle_callback.assert_called_once_with()

    async def test_force_stop_during_write_releases_pipeline_once(self) -> None:
        """Verify a stop racing with output writes performs one cleanup only."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.cur_state = "speaking"
        engine.locked = True
        engine.playback_done.clear()
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        stop_results: list[bool] = []

        class StoppingStream(FakeStream):
            """Stop Celune from the output-write callback."""

            def __init__(self) -> None:
                super().__init__()
                self.stop_requested = False

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                super().write(audio)
                if not self.stop_requested:
                    self.stop_requested = True
                    self.assert_stop()
                    engine.audio_queue.put(engine.sentinel)
                return False

            def assert_stop(self) -> None:
                """Request the same stop that the UI command uses."""
                stop_results.append(pipeline.force_stop_speech(cast(Celune, engine)))

        fake_stream = StoppingStream()
        engine.glow = SimpleNamespace(schedule=mock.Mock())
        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.full((2400, 2), 0.3, dtype=np.float32),
            48000,
        )

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        engine.idle_callback.assert_called_once_with()
        assert not engine.locked
        assert stop_results == [True]

    async def test_playback_error_releases_pipeline_after_sfx_output_failure(
        self,
    ) -> None:
        """Verify an output failure cannot leave an SFX pipeline lease held."""
        engine = make_pipeline_engine()
        engine.stream = None
        engine._stream = None
        engine._current_sr = None
        engine.current_sr = None
        engine.dev = False
        engine.sentinel = PipelineStates.TERMINATE
        engine.force_stop_marker = PipelineStates.UTTERANCE_FORCE_END
        engine.text_queue = queue.Queue()
        engine.audio_queue = queue.Queue()
        assert pipeline.acquire_pipeline(cast(Celune, engine), "sfx")
        pipeline.register_playback_source(cast(Celune, engine), 1, kind="sfx")
        pipeline.queue_playback_chunk(
            cast(Celune, engine),
            1,
            np.zeros((2400, 2), dtype=np.float32),
            48000,
        )
        pipeline.queue_playback_done(cast(Celune, engine), 1)
        engine.audio_queue.put(engine.sentinel)

        class FailingStream(FakeStream):
            """Raise when the playback worker writes the SFX block."""

            def write(self, audio: npt.NDArray[np.float32]) -> bool:
                del audio
                raise RuntimeError("simulated SFX output failure")

        with mock.patch(
            "celune.pipeline.sd.OutputStream", return_value=FailingStream()
        ):
            await self._run_playback_worker(cast(Celune, engine))

        assert not engine.locked
        assert engine._pipeline_lock_owner is None
        assert engine.playback_done.is_set()

    def test_finalize_playback_idle_resets_glow_audio_reactivity(self) -> None:
        """Verify normal playback completion restores the resting glow."""
        engine = make_pipeline_engine()
        engine.locked = False
        engine.cur_state = "speaking"
        engine.dev = False

        pipeline.finalize_playback_idle(cast(Celune, engine))

        engine.glow.reset_audio_reactivity.assert_called_once_with()
        assert engine.playback_done.is_set()
        assert engine.cur_state == "idle"
        engine.idle_callback.assert_called_once_with()

    def test_finalize_playback_idle_does_not_announce_readiness_while_reloading(
        self,
    ) -> None:
        """Verify transitional playback does not announce readiness mid-reload."""
        engine = make_pipeline_engine()
        engine.locked = True
        engine.loaded = False
        engine.cur_state = "reloading"

        pipeline.finalize_playback_idle(cast(Celune, engine))

        assert ("Ready to speak.", "info") not in engine.messages
        assert not getattr(engine, "_ready_announced", False)
        assert engine.cur_state == "reloading"

    def test_finalize_playback_idle_does_not_emit_idle_callback_while_locked(
        self,
    ) -> None:
        """Verify locked non-readiness playback cannot unlock the UI through idle callbacks."""
        engine = make_pipeline_engine()
        engine.locked = True
        engine.loaded = False
        engine.cur_state = "reloading"

        pipeline.finalize_playback_idle(cast(Celune, engine))

        engine.idle_callback.assert_not_called()
        assert engine.playback_done.is_set()
        assert engine.cur_state == "reloading"

    def test_finalize_playback_idle_does_not_unlock_voice_reload(self) -> None:
        """Verify a voice reload remains transitional after pending playback drains."""
        engine = make_pipeline_engine()
        engine.locked = False
        engine.loaded = True
        engine.cur_state = "reloading"

        pipeline.finalize_playback_idle(cast(Celune, engine))

        engine.idle_callback.assert_not_called()
        assert engine.playback_done.is_set()
        assert engine.cur_state == "reloading"

    def test_think_builds_persona_payload_and_queues_response(self) -> None:
        """Verify Persona request formatting without loading a Persona model.

        Raises:
            AssertionError: Persona request behavior changes unexpectedly.
        """

        class FakeResponse:
            """Fake API response class."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "I can help with that."}

        class FakeVision:
            """Fake vision API class object."""

            def __init__(self) -> None:
                self.payload: Optional[JSON] = None

            def post(self, json: JSON) -> FakeResponse:
                """Post a fake request.

                Args:
                    json: The JSON body to be posted.

                Returns:
                    FakeResponse: A fake response object.
                """
                self.payload = json
                return FakeResponse()

        engine = make_pipeline_engine()
        engine.config = {
            "vram": "high",
            "persona": {"model_id": "fixture/persona-test"},
            "persona_persona": "The active character is gentle and observant.",
            "persona_context": "The user is testing request formatting.",
        }
        engine.current_character = "Celune"
        engine.current_voice = "calm"
        engine.voice_prompt = "small pauses, soft delivery"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Celune",
                profile="A quietly attentive nocturnal presence with emotional continuity.",
            ),
            speaking_style="Soft-spoken, intimate, and reflective without sounding timid.",
            boundaries=("Do not drift into customer-support phrasing.",),
            prompt_rules=(
                "Treat the user as someone already in conversation with the character.",
            ),
            example_dialogue=(
                "User: i think i fixed it",
                "Celune: Sounds like you finally wrestled it into behaving.",
            ),
            style=PersonaStyleValues(
                warmth="high",
                directness="mid",
                humor="low",
                detail="mid",
                formality="low",
                enthusiasm="low",
            ),
        )
        engine.persona_history = [{"role": "assistant", "content": "Earlier reply."}]
        engine.vision = FakeVision()
        engine.dev = False

        with mock.patch(
            "celune.speech.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "What now?")

        request = engine.text_queue.get_nowait()
        assert request.text == "I can help with that."
        assert not request.save

        payload = cast(JSON, engine.vision.payload)
        self.assertEqual(payload["model"], "fixture/persona-test")
        self.assertEqual(payload["quantization"], "4bit")
        self.assertEqual(payload["quantized"], True)
        self.assertEqual(payload["request"], "What now?")
        self.assertEqual(payload["user"], "What now?")
        self.assertEqual(payload["character"], "Celune")

        character_card = cast(str, payload["character_card"])
        system_prompt = cast(str, payload["system"])
        messages = cast(list[dict[str, str]], payload["messages"])
        assert "Name: Celune" in character_card
        assert "The active character is gentle and observant." in character_card
        assert (
            "A quietly attentive nocturnal presence with emotional continuity."
            in character_card
        )
        assert "Soft-spoken, intimate, and reflective" in character_card
        assert "Prompt Rules:" in character_card
        assert "Example Dialogue:" in character_card
        assert "<profile>" in system_prompt
        assert "<behavior>" in system_prompt
        assert "## Identity" in system_prompt
        assert "Name: Celune" in system_prompt
        assert "<history>" not in system_prompt
        assert "Earlier reply." not in system_prompt
        assert messages[0] == {"role": "system", "content": system_prompt}
        assert messages[-1] == {"role": "user", "content": "What now?"}
        assert messages[1] == {"role": "assistant", "content": "Earlier reply."}
        assert len(messages) == 3
        assert engine.persona_history[-2:] == [
            {"role": "user", "content": "What now?"},
            {"role": "assistant", "content": "I can help with that."},
        ]

    def test_agent_classification_request_uses_a_disposable_prompt(
        self,
    ) -> None:
        """Build routing input without retaining conversational Persona history."""
        engine = make_pipeline_engine()
        engine.config = {
            "vram": "high",
            "persona": {"model_id": "fixture/persona-test"},
        }
        engine.persona_history = [{"role": "assistant", "content": "Earlier."}]

        payload = pipeline.build_agent_classification_request(
            cast(Celune, engine),
            "Please handle this.",
        )

        self.assertEqual(payload["format"], "celune_agent_classification")
        self.assertEqual(payload["request"], "Please handle this.")
        self.assertEqual(payload["context_space"], 8192)
        self.assertEqual(payload["max_new_tokens"], 96)
        system_prompt = payload["system"]
        self.assertIsInstance(system_prompt, str)
        assert isinstance(system_prompt, str)
        self.assertIn("Classify the latest user input", system_prompt)
        self.assertIn(
            "internal routing request, not a character response", system_prompt
        )
        self.assertIn("Do not add a preamble", system_prompt)
        self.assertIn("a task classification must use route task", system_prompt)
        self.assertIn("inspect or retrieve live/local state", system_prompt)
        self.assertIn(
            "Judge the intended operation and target semantically", system_prompt
        )
        messages = cast(list[JSON], payload["messages"])
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "Please handle this."},
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please handle this."},
            ],
        )

    def test_agent_classification_request_includes_active_routing_context(self) -> None:
        """Expose typed active-task context to semantic routing without a second prompt path."""
        engine = make_pipeline_engine()
        engine.config = {"vram": "high"}

        payload = pipeline.build_agent_classification_request(
            cast(Celune, engine),
            "That is fine, continue.",
            routing_context={
                "active_task": {"task_id": "task-1", "state": "awaiting_approval"},
                "pending_approval": {
                    "request_id": "approval-1",
                    "prompt": "Allow the change?",
                },
            },
        )

        self.assertEqual(
            payload["routing_context"],
            {
                "active_task": {"task_id": "task-1", "state": "awaiting_approval"},
                "pending_approval": {
                    "request_id": "approval-1",
                    "prompt": "Allow the change?",
                },
            },
        )
        system_prompt = payload["system"]
        self.assertIsInstance(system_prompt, str)
        assert isinstance(system_prompt, str)
        self.assertIn("approval_response", system_prompt)
        self.assertIn("awaiting_approval", system_prompt)

    def test_persona_request_uses_xhigh_quantization(self) -> None:
        """Verify xhigh VRAM presets request Persona in 8-bit mode."""
        engine = make_pipeline_engine()
        engine.config = {"vram": "xhigh", "persona": {"model_id": "fixture/persona"}}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_prompt = None
        engine.persona_history = []

        with mock.patch("celune.vram.torch.cuda.is_available", return_value=False):
            payload = pipeline.build_persona_request(cast(Celune, engine), "Hello")

        assert payload["quantization"] == "8bit"

    def test_persona_context_space_uses_agent_budget_only_for_agent_requests(
        self,
    ) -> None:
        """Use the smaller VLM context for conversation and larger agent context for tasks."""
        engine = make_pipeline_engine()
        engine.config = {"vram": "high"}

        conversation = pipeline.build_persona_request(cast(Celune, engine), "Hello")
        self.assertEqual(conversation["context_space"], 8192)

        task = AgentTask(
            task_id="task-context",
            session_id="session-context",
            request=AgentRequest(request="Check the status."),
        )
        agent_context = AgentContext(
            request=task.request,
            mode="agent",
            persona_capabilities=PersonaCapabilities(),
            task=task,
        )
        agent = pipeline.build_persona_request(
            cast(Celune, engine),
            task.request.request,
            agent_context=agent_context,
        )
        self.assertEqual(agent["context_space"], 32768)
