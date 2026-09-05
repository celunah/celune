# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline helpers that do not perform real synthesis."""

# Import groups follow Celune's project-specific Ruff ordering.
# pylint: disable=ungrouped-imports

import sys
import json as _json
import queue
import tempfile
import threading
from types import TracebackType, SimpleNamespace
from typing import Self, Optional, cast
from pathlib import Path
from unittest import mock
from collections.abc import Iterator
from importlib.machinery import ModuleSpec

import numpy as np
import pytest
import soundfile as sf
import numpy.typing as npt

from celune import pipeline
from celune.i18n import string
from celune.utils import discard
from celune.celune import Celune
from celune.cevoice import (
    CEVoice,
    CEVoiceLoader,
    CEVoicePersona,
    PersonaIdentity,
    PersonaStyleValues,
    persona_files_from_bundle,
)
from celune.constants import PipelineStates
from celune.persona.impl import compact_persona_history
from celune.typing.agent import (
    ToolCall,
    AgentTask,
    AgentContext,
    AgentRequest,
    AgentToolSchema,
    AgentToolBehavior,
    AgentToolValueType,
    AgentToolDangerLevel,
    AgentToolArgumentSchema,
)
from celune.typing.common import JSON, JSONSerializable
from celune.typing.aliases import AudioChunk
from celune.persona.prompts import PersonaPromptBuilder, render_markdown_subsection
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
from .test_persona_memory import StubEmbeddingMemoryStore


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
                "celune.pipeline.detect_language",
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

        def mark_ready() -> bool:
            engine.model_ready.set()
            return True

        engine.model_ready.wait = mock.Mock(side_effect=mark_ready)
        run_in_daemon_thread = mock.AsyncMock(side_effect=lambda function: function())

        with mock.patch("celune.pipeline._run_in_daemon_thread", run_in_daemon_thread):
            queued = await pipeline.queue_speech_async(
                cast(Celune, engine),
                "hello",
                display_text="shown",
            )

        assert queued
        engine.model_ready.wait.assert_called_once_with()
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
            "celune.pipeline.detect_language",
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

        engine = make_pipeline_engine()
        engine.use_normalization = True
        engine.normalize = mock.Mock(return_value="normalized")
        with mock.patch(
            "celune.pipeline.detect_language",
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
            "celune.pipeline.detect_language",
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
            "celune.pipeline.detect_language",
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
            "celune.pipeline.detect_language",
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
            mock.patch("celune.pipeline.default_loader", return_value=loader),
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True) as q,
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
            mock.patch("celune.pipeline.default_loader", return_value=loader),
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True),
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
            mock.patch("celune.pipeline.default_loader", return_value=loader),
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True),
            mock.patch(
                "celune.pipeline.pitch_shift_audio",
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
            mock.patch("celune.pipeline.default_loader", return_value=loader),
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True),
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.pipeline.subprocess.run", side_effect=fake_run
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch("celune.pipeline.running_compiled", return_value=True),
                mock.patch("celune.pipeline.project_root", return_value=Path("/repo")),
                mock.patch(
                    "celune.pipeline.subprocess.run", side_effect=fake_run
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.pipeline.subprocess.run", side_effect=fake_run
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.pipeline.subprocess.run",
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.pipeline.subprocess.run",
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
                    "celune.pipeline.temp_data_dir", return_value=temp_root / "temp"
                ),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec",
                    return_value=ModuleSpec("yt_dlp", loader=None),
                ),
                mock.patch(
                    "celune.pipeline._youtube_sfx_title",
                    return_value="Fixture Video Title",
                ),
                mock.patch(
                    "celune.pipeline.subprocess.run",
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

        with mock.patch("celune.pipeline.urlopen", return_value=FakeResponse()):
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
                "celune.pipeline._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ) as download,
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline.sf.read", return_value=(audio, 48000)) as read,
            mock.patch(
                "celune.pipeline.queue_sfx_audio", return_value=True
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
                "celune.pipeline._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch(
                "celune.pipeline.sf.read",
                return_value=(np.ones((8, 2), dtype=np.float32), 48000),
            ),
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True),
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
                "celune.pipeline._download_youtube_sfx",
                return_value=(downloaded, "Fixture Video Title"),
            ),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch(
                "celune.pipeline.sf.read",
                return_value=(np.ones((8, 2), dtype=np.float32), 48000),
            ),
            mock.patch("celune.pipeline.queue_sfx_audio", side_effect=queue_audio),
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
                "celune.pipeline.detect_language",
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
            "celune.pipeline.detect_language",
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

    def test_persona_context_omits_voice_prompt_when_unsupported(self) -> None:
        """Verify unsupported voice prompts do not leak into Persona context."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_prompt = "gentle and airy"
        engine.voice_prompt_supported = lambda: False
        engine.persona_history = []
        engine.persona_attachments = []

        context = pipeline.build_persona_context(cast(Celune, engine), "Hello")

        assert "Voice prompt:" not in context.persona_card.voice

    def test_persona_card_uses_baseline_persona_for_non_default_voice_pack(
        self,
    ) -> None:
        """Verify custom CEVOICE/CECHAR packs do not inherit Celune-specific defaults.

        Raises:
            AssertionError: Persona card fallback behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "bold"
        engine.voice_prompt = None
        engine.voice_bundle_is_default = False

        character_card = pipeline.build_persona_character_card(cast(Celune, engine))

        assert "Name: Fixture" in character_card
        assert "Gender: unknown" in character_card
        assert (
            "Stay in character using the active character metadata," in character_card
        )
        assert (
            "The active character is replying to the user through a real-time speech system."
            in character_card
        )
        assert "- Warmth: mid" in character_card
        assert "- Directness: mid" in character_card
        assert "- Formality: mid" in character_card
        assert "Gender: female" not in character_card
        assert "The speaker uses a more confident" not in character_card

    def test_persona_prompt_builder_renders_structured_context_blocks(self) -> None:
        """Verify Persona prompts include the requested structured RAG sections."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona_character_profile": "A careful archivist with a dry wit.",
            "persona_state": "Thoughtful and slightly tired.",
            "persona_long_term_memory": [
                "The user prefers concise answers.",
                "The character once helped recover a lost journal.",
            ],
        }
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.voice_prompt = "steady cadence"
        engine.persona_history = [
            {"role": "user", "content": "Do you remember our last visit?"},
            {"role": "assistant", "content": "Yes, we catalogued the letters."},
        ]
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/archive.png",
                "name": "archive.png",
            }
        ]

        context = pipeline.build_persona_context(
            cast(Celune, engine), "What do you notice?"
        )
        prompt = PersonaPromptBuilder.build(context)

        assert "<profile>" in prompt
        assert "## Identity" in prompt
        assert "<memory>" in prompt
        assert "- The user prefers concise answers." in prompt
        assert "- The character once helped recover a lost journal." in prompt
        assert "<mood>" in prompt
        assert "Thoughtful and slightly tired." in prompt
        assert "<history>" not in prompt
        assert "assistant: Yes, we catalogued the letters." not in prompt
        assert "user: What do you notice?" not in prompt
        assert "A careful archivist with a dry wit." in prompt
        assert "Push the conversation forward naturally." in prompt
        assert (
            "Never output emojis; use plain text suitable for speech synthesis."
            in prompt
        )
        assert (
            "Treat facts in <memory> as true background context when they are relevant."
            in prompt
        )
        assert (
            "Keep facts from <memory> silent unless the current user message clearly asks for them"
            in prompt
        )
        assert "## Runtime Guidance" in prompt
        assert "Do not greet the user or restart the conversation." in prompt
        assert "## Reference Resolution" in prompt
        assert "The active character is Fixture." in prompt
        assert (
            "When the user refers to the active character by name, nickname, or matching third-person pronouns"
            in prompt
        )
        assert "he, him, his, she, her, hers, they, them, their" in prompt
        assert "What do you notice?" not in prompt
        assert "<request>" not in prompt

    def test_markdown_persona_headers_use_consistent_spacing(self) -> None:
        """Verify generated and embedded Persona Markdown headers have one blank line after them."""
        rendered = render_markdown_subsection(
            "Fixture",
            "Intro\n## Nested\n\n- first\n## Another\n- second",
        )

        assert (
            rendered
            == "## Fixture\n\nIntro\n## Nested\n\n- first\n## Another\n\n- second"
        )

    def test_cevoice_persona_metadata_populates_persona_card(self) -> None:
        """Verify CEVOICE persona metadata becomes the active Persona card."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Mirelle",
                age="27",
                gender="female",
                profile="A precise investigator who notices tiny shifts in tone.",
            ),
            speaking_style="Elegant, steady, and mildly teasing.",
            boundaries=(
                "Do not use sterile assistant framing.",
                "Do not sound detached.",
            ),
            prompt_rules=("Favor exact wording when recalling details.",),
            example_dialogue=(
                "User: status?",
                "Mirelle: It's holding, mostly.",
            ),
            style=PersonaStyleValues(
                warmth="mid",
                directness="high",
                humor="low",
                detail="high",
                formality="high",
                enthusiasm="low",
            ),
        )

        context = pipeline.build_persona_context(cast(Celune, engine), "What changed?")
        card = context.persona_card.render()

        assert context.character_profile.name == "Mirelle"
        assert context.character_profile.age == "27"
        assert context.character_profile.gender == "female"
        assert (
            "A precise investigator who notices tiny shifts in tone."
            in context.character_profile.render()
        )
        assert context.persona_source_material.identity == (
            "Name: Mirelle\nAge: 27\nGender: female\n\n"
            "A precise investigator who notices tiny shifts in tone."
        )
        assert context.persona_source_material.speech_style == (
            "Elegant, steady, and mildly teasing.\n\n"
            "- Warmth: mid\n- Directness: high\n- Humor: low\n"
            "- Detail: high\n- Formality: high\n- Enthusiasm: low"
        )
        assert "Style Notes:" in card
        assert "Elegant, steady, and mildly teasing." in card
        assert "Boundaries:" in card
        assert "Prompt Rules:" in card
        assert "Example Dialogue:" in card
        assert "- Formality: high" in card
        assert "- Enthusiasm: low" in card

    def test_voice_persona_style_extends_shared_persona(self) -> None:
        """Verify a selected voice can refine the shared Persona response style."""
        engine = make_pipeline_engine()
        engine.backend.uses_voice_bundles = True
        engine.current_voice = "bold"
        engine.current_character = "Celune"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(name="Celune", profile="A careful archivist."),
            speaking_style="Measured and observant.",
            style=PersonaStyleValues(
                warmth="high",
                directness="mid",
                enthusiasm="low",
            ),
        )
        engine.persona_history = []
        engine.persona_attachments = []
        engine.retrieved_long_term_memory = []
        engine.config = {"persona_state": "Neutral."}
        voice_persona = CEVoicePersona(
            speaking_style="More playful and energetic.",
            prompt_rules=("Use a brighter conversational rhythm.",),
            style=PersonaStyleValues(directness="high", enthusiasm="high"),
        )
        fake_loader = SimpleNamespace(bundle=SimpleNamespace())

        with (
            mock.patch("celune.persona.impl.default_loader", return_value=fake_loader),
            mock.patch("celune.pipeline.default_loader", return_value=None),
            mock.patch(
                "celune.persona.impl.persona_metadata_from_voice",
                return_value=voice_persona,
            ),
        ):
            context = pipeline.build_persona_context(
                cast(Celune, engine),
                "Hello.",
            )

        assert context.persona_card.directness == "high"
        assert context.persona_card.enthusiasm == "high"
        assert "Measured and observant." in context.persona_card.speaking_style
        assert "More playful and energetic." in context.persona_card.speaking_style
        assert (
            "Use a brighter conversational rhythm." in context.persona_card.prompt_rules
        )

    def test_different_cevoice_personas_produce_distinct_prompts(self) -> None:
        """Verify different CEVOICE persona packs shape different Persona prompts."""
        first = make_pipeline_engine()
        first.config = {}
        first.current_character = "Mirelle"
        first.current_voice = "balanced"
        first.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(profile="A precise investigator."),
            speaking_style="Elegant and steady.",
            style=PersonaStyleValues(detail="high", formality="high"),
        )

        second = make_pipeline_engine()
        second.config = {}
        second.current_character = "Rho"
        second.current_voice = "balanced"
        second.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(profile="A mischievous mechanic."),
            speaking_style="Fast, playful, and sharp.",
            style=PersonaStyleValues(humor="high", enthusiasm="high"),
        )

        first_prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, first), "Status?")
        )
        second_prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, second), "Status?")
        )

        assert first_prompt != second_prompt
        assert "A precise investigator." in first_prompt
        assert "A mischievous mechanic." in second_prompt

    def test_persona_prompt_prefers_manifest_markdown_files_when_available(
        self,
    ) -> None:
        """Verify CECHAR v3 persona Markdown overrides legacy-derived prompt text."""
        engine = make_pipeline_engine()
        engine.config = {"persona_persona": "Legacy personality text."}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(
                name="Mirelle",
                profile="Legacy identity text.",
            ),
            speaking_style="Legacy speech style.",
        )
        fake_loader = SimpleNamespace(
            bundle=SimpleNamespace(
                metadata={
                    "name": "Mirelle",
                    "assets": {
                        "identity.md": {
                            "offset": 0,
                            "length": 18,
                            "sha256": "0" * 64,
                        },
                        "personality.md": {
                            "offset": 18,
                            "length": 21,
                            "sha256": "1" * 64,
                        },
                        "speech_style.md": {
                            "offset": 39,
                            "length": 20,
                            "sha256": "2" * 64,
                        },
                    },
                },
                assets={
                    "identity.md": {},
                    "personality.md": {},
                    "speech_style.md": {},
                },
                read_bundle_asset=lambda name: {
                    "identity.md": b"Manifest identity.",
                    "personality.md": b"Manifest personality.",
                    "speech_style.md": b"Manifest speech style.",
                }[name],
            ),
        )

        with mock.patch("celune.pipeline.default_loader", return_value=fake_loader):
            prompt = PersonaPromptBuilder.build(
                pipeline.build_persona_context(cast(Celune, engine), "Status?")
            )

        self.assertIn("Manifest identity.", prompt)
        self.assertIn("Manifest personality.", prompt)
        self.assertIn("Manifest speech style.", prompt)
        self.assertIn("<user_instructions>", prompt)
        self.assertIn("Legacy personality text.", prompt)
        self.assertNotIn("Legacy identity text.", prompt)
        self.assertNotIn("Ignored text.", prompt)

    def test_persona_debug_overrides_replace_manifest_markdown_files(self) -> None:
        """Verify opt-in app-data Markdown replaces matching CECHAR source files."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"debug_overrides": True}}
        engine.current_character = "Mirelle"
        engine.current_character_persona = CEVoicePersona(
            identity=PersonaIdentity(name="Mirelle")
        )
        fake_loader = SimpleNamespace(
            bundle=SimpleNamespace(
                metadata={
                    "persona": {"identity": {"name": "Mirelle"}},
                },
            ),
        )

        with (
            mock.patch("celune.pipeline.default_loader", return_value=fake_loader),
            mock.patch(
                "celune.pipeline.persona_files_from_bundle",
                return_value={"personality.md": "Pack personality."},
            ),
            mock.patch(
                "celune.pipeline.persona_override_files",
                return_value={"personality.md": "Debug personality."},
            ),
        ):
            files = pipeline._persona_manifest_files(cast(Celune, engine))

        assert files == {"personality.md": "Debug personality."}

    def test_persona_prompt_does_not_hardcode_celune_identity(self) -> None:
        """Verify Persona prompts stay character-agnostic without pack metadata."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        assert "Name: Fixture" in prompt
        assert "Name: Celune" not in prompt

    def test_default_celune_prompt_uses_canonical_age_and_gender(self) -> None:
        """Verify default Celune prompts expose the intended identity fields."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = True

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertIn("Name: Celune", prompt)
        self.assertIn(f"Gender: {string('persona.default_gender')}", prompt)

    def test_default_cechar_pack_adds_celune_prompt_foundation(self) -> None:
        """Verify the bundled CECHAR pack assembles only its current prompt."""
        bundle = CEVoice.open(Path("voices/default.cevoice"))
        loader = CEVoiceLoader(bundle)
        self.addCleanup(loader.close)

        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = True

        with mock.patch("celune.pipeline.default_loader", return_value=loader):
            prompt = PersonaPromptBuilder.build(
                pipeline.build_persona_context(cast(Celune, engine), "Hello.")
            )

        for source_text in persona_files_from_bundle(bundle).values():
            self.assertIn(source_text, prompt)

        self.assertIn("Aliases: Cel", prompt)
        self.assertIn("Role: Lunar guardian", prompt)
        self.assertIn("Ground every claim in available evidence.", prompt)
        self.assertIn("adopt the corrected meaning immediately", prompt)
        self.assertIn("Remain consistently yourself.", prompt)
        self.assertNotIn("apply the correction immediately", prompt)
        self.assertNotIn("Celune is a quiet nocturnal presence", prompt)
        self.assertNotIn(
            "These examples demonstrate tone and conversational style", prompt
        )
        self.assertNotIn(
            "Keep the familiar nocturnal tone grounded and concise.", prompt
        )
        self.assertNotIn("## Response Behavior", prompt)
        self.assertNotIn("User: i think i fixed it", prompt)

        voice_manifest = bundle.metadata["voices"]
        self.assertIsInstance(voice_manifest, dict)
        for voice_metadata in voice_manifest.values():
            self.assertIsInstance(voice_metadata, dict)
            self.assertNotIn("persona", voice_metadata)

    def test_non_default_character_pack_does_not_inherit_celune_foundation(
        self,
    ) -> None:
        """Verify another character pack does not receive Celune-only instructions."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Mirelle"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertNotIn("You are Celune, commonly called Cel.", prompt)

    def test_agent_prompt_context_uses_existing_task_and_tool_contracts(self) -> None:
        """Verify task context and tool metadata are available without runtime authority."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        request = AgentRequest("Check whether the process is running.")
        task = AgentTask(task_id="task-1", session_id="default", request=request)
        agent_context = AgentContext(
            request=request,
            mode="agent",
            persona_capabilities=PersonaCapabilities(),
            task=task,
            last_tool_result={
                "tool_call_id": "call-1",
                "output": "running",
                "error": None,
            },
        )
        schema = AgentToolSchema(
            tool_id="process_status",
            display_name="Process status",
            description="Check whether a local process is running.",
            arguments=(
                AgentToolArgumentSchema(
                    name="name",
                    value_type=AgentToolValueType.STRING,
                ),
            ),
            behavior=AgentToolBehavior.READ_ONLY,
            danger=AgentToolDangerLevel.LOW,
        )
        pending_call: ToolCall = {
            "id": "call-1",
            "name": "process_status",
            "arguments": {"name": "celune"},
        }

        context = pipeline.build_persona_context(
            cast(Celune, engine),
            request.request,
            agent_context=agent_context,
            tool_schemas=(schema,),
            pending_tool_call=pending_call,
        )
        prompt = PersonaPromptBuilder.build(context)

        self.assertIn("<agent_context>", prompt)
        self.assertIn('"tool_id": "process_status"', prompt)
        self.assertIn('"state": "queued"', prompt)
        self.assertIn('"output": "running"', prompt)
        self.assertIn("runtime remains authoritative", prompt)
        self.assertLess(prompt.index("<memory>"), prompt.index("<agent_context>"))

        payload = pipeline.build_persona_request(
            cast(Celune, engine),
            request.request,
            agent_context=agent_context,
            tool_schemas=(schema,),
            pending_tool_call=pending_call,
        )
        messages = cast(list[JSON], payload["messages"])
        self.assertEqual(payload["system"], messages[0]["content"])
        self.assertEqual(cast(str, payload["system"]).count("<agent_context>"), 1)

    def test_named_celune_custom_pack_does_not_use_default_identity(self) -> None:
        """Verify custom packs named Celune do not inherit default identity fields."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.voice_bundle_is_default = False

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        assert "Name: Celune" in prompt

    def test_persona_context_uses_weighted_emotion_state_when_unconfigured(
        self,
    ) -> None:
        """Verify Persona state can come from weighted conversation emotion."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "I feel awful."},
            {"role": "assistant", "content": "I am staying steady."},
        ]

        fake_analyzer = SimpleNamespace(
            summarize_history=mock.Mock(
                return_value=SimpleNamespace(
                    target_state=(
                        "Target emotion: gently reassuring. "
                        "The user's recent mood leans toward sadness."
                    )
                )
            )
        )

        with mock.patch(
            "celune.pipeline._persona_emotion_analyzer",
            return_value=fake_analyzer,
        ):
            context = pipeline.build_persona_context(
                cast(Celune, engine), "Please stay with me."
            )

        assert "Target emotion: gently reassuring." in context.mood_or_state
        fake_analyzer.summarize_history.assert_called_once()

    def test_persona_context_prefers_configured_state_over_emotion_analysis(
        self,
    ) -> None:
        """Verify an explicit persona_state still overrides automatic emotion blending."""
        engine = make_pipeline_engine()
        engine.config = {"persona_state": "Thoughtful and slightly tired."}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        with mock.patch("celune.pipeline._persona_emotion_analyzer") as analyzer:
            context = pipeline.build_persona_context(cast(Celune, engine), "Hello.")

        assert context.mood_or_state == "Thoughtful and slightly tired."
        analyzer.assert_not_called()

    def test_persona_context_logs_emotion_fallback_reason(self) -> None:
        """Verify emotion-analysis failures are surfaced in verbose logs."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        captured: list[tuple[str, str]] = []
        engine.log = lambda msg, severity="info", **kwargs: captured.append(
            (msg, severity)
        )

        fake_analyzer = SimpleNamespace(
            last_error="lunahr/emotispace-128 could not be loaded",
            summarize_history=mock.Mock(return_value=None),
        )

        with mock.patch(
            "celune.pipeline._persona_emotion_analyzer",
            return_value=fake_analyzer,
        ):
            context = pipeline.build_persona_context(cast(Celune, engine), "Hello.")

        assert context.mood_or_state == "Neutral."
        assert captured == [
            (
                (
                    "Persona emotion analysis fell back to Neutral: "
                    "lunahr/emotispace-128 could not be loaded"
                ),
                "warning",
            )
        ]

    def test_persona_prompt_builder_omits_vision_context_without_attachments(
        self,
    ) -> None:
        """Verify Persona prompts no longer serialize recent chat into the system prompt."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        context = pipeline.build_persona_context(cast(Celune, engine), "Continue.")
        prompt = PersonaPromptBuilder.build(context)

        assert "<vision_context>" not in prompt
        assert "<history>" not in prompt
        assert "assistant: hi" not in prompt

    def test_persona_messages_keep_only_recent_history(self) -> None:
        """Verify stale Persona turns do not dilute the current character card."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 6}}}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": f"old user {index}"}
            if index % 2 == 0
            else {"role": "assistant", "content": f"old reply {index}"}
            for index in range(12)
        ]

        messages = pipeline.build_persona_messages(cast(Celune, engine), "current")

        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "current"}
        assert len(messages) == 8
        assert messages[1] == {"role": "user", "content": "old user 6"}
        assert messages[-2] == {"role": "assistant", "content": "old reply 11"}
        system_prompt = cast(str, messages[0]["content"])
        assert "<history>" not in system_prompt
        assert "old user 4" not in str(messages)

    def test_persona_history_uses_configured_short_term_message_limit(self) -> None:
        """Verify Persona history rolls forward using the configured message limit."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 4}}}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "old user 0"},
            {"role": "assistant", "content": "old reply 1"},
            {"role": "user", "content": "old user 2"},
            {"role": "assistant", "content": "old reply 3"},
        ]

        class FakeResponse:
            """Fake API response for rolling-history assertions."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "new reply"}

        engine.vision = SimpleNamespace(
            post=lambda json: FakeResponse(),
        )
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "new user")

        assert engine.persona_history == [
            {"role": "user", "content": "old user 2"},
            {"role": "assistant", "content": "old reply 3"},
            {"role": "user", "content": "new user"},
            {"role": "assistant", "content": "new reply"},
        ]

    def test_persona_history_compacts_older_turns_into_session_summary(self) -> None:
        """Verify older Persona turns are summarized and recent turns stay available."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 3,
                    "context_compaction_keep_recent_messages": 2,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_history = [
            {"role": "user", "content": "The archive is stored in the attic."},
            {"role": "assistant", "content": "I will keep that context in mind."},
        ]

        class FakeResponse:
            """Fake API response for context-compaction assertions."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response."""
                return {"response": "Understood."}

        engine.vision = SimpleNamespace(
            post=lambda json: FakeResponse(),
        )
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "What is next?")

        assert engine.persona_history == [
            {"role": "user", "content": "What is next?"},
            {"role": "assistant", "content": "Understood."},
        ]
        assert "The archive is stored in the attic." in engine.persona_session_summary
        assert engine.persona_session_summary.startswith("Conversation context:")
        assert "Earlier summary:" not in engine.persona_session_summary
        assert "user:" not in engine.persona_session_summary
        assert "assistant:" not in engine.persona_session_summary

    def test_persona_history_summary_does_not_nest_previous_summary(self) -> None:
        """Verify repeated compaction removes wrappers and duplicate summary labels."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 1,
                    "context_compaction_keep_recent_messages": 1,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.persona_session_summary = (
            "<conversation_summary>Earlier summary: Earlier summary: "
            "The archive is stored in the attic.</conversation_summary>"
        )
        engine.persona_history = [
            {"role": "user", "content": "The archive is stored in the attic."},
            {"role": "assistant", "content": "I will remember the archive location."},
        ]

        compact_persona_history(cast(Celune, engine))

        assert engine.persona_session_summary.count("archive") == 1
        assert engine.persona_session_summary.startswith("Conversation context:")
        assert "Earlier summary:" not in engine.persona_session_summary
        assert "<conversation_summary>" not in engine.persona_session_summary
        assert "</conversation_summary>" not in engine.persona_session_summary

    def test_persona_history_prefers_neutral_vlm_summary(self) -> None:
        """Verify compaction stores the VLM summary instead of raw conversation turns."""
        engine = make_pipeline_engine()
        engine.config = {
            "persona": {
                "memory": {
                    "max_short_term_messages": 1,
                    "context_compaction_keep_recent_messages": 1,
                    "context_summary_max_characters": 240,
                }
            }
        }
        engine.persona_session_summary = "Earlier context."
        engine.persona_history = [
            {"role": "user", "content": "The TTS response was cut off."},
            {"role": "assistant", "content": "I will keep that in mind."},
        ]
        summarize_history = mock.Mock(
            return_value="The conversation concerns a TTS cutoff."
        )
        engine.vision = SimpleNamespace(summarize_history=summarize_history)

        compact_persona_history(cast(Celune, engine))

        summarize_history.assert_called_once_with(
            [{"role": "user", "content": "The TTS response was cut off."}],
            "Earlier context.",
            240,
        )
        assert (
            engine.persona_session_summary
            == "Conversation context: The conversation concerns a TTS cutoff."
        )

    def test_think_persists_explicit_memory_before_persona_reply(self) -> None:
        """Verify explicit memory requests are stored before Persona responds."""

        class FakeResponse:
            """Fake API response for explicit-memory persistence."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "Alright. I'll remember it."}

        class FakeVision:
            """Fake vision API that captures the built Persona payload."""

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
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.config = {
                "vram": "high",
                "persona": {
                    "model_id": "fixture/persona-test",
                    "memory": {"storage_dir": temp_dir},
                },
            }
            engine.current_character = "Celune"
            engine.current_voice = "balanced"
            engine.vision = FakeVision()
            engine.dev = False
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            engine.persona_memory_store = store

            with mock.patch(
                "celune.pipeline.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                assert pipeline.think(
                    cast(Celune, engine),
                    "remember that my test word is moonlight",
                )

            retrieved = store.retrieve("Celune", "what is my test word?")

        assert [record.content for record in retrieved] == ["my test word is moonlight"]

    def test_persona_memory_path_is_independent_of_markdown_debug_overrides(
        self,
    ) -> None:
        """Verify Markdown debug overrides do not change Persona memory storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for debug_overrides in (False, True):
                engine = make_pipeline_engine()
                engine.config = {
                    "persona": {"debug_overrides": debug_overrides},
                }
                with mock.patch(
                    "celune.persona.memory.persona_data_dir",
                    return_value=Path(temp_dir),
                ):
                    store = pipeline._persona_memory_store(cast(Celune, engine))

                assert store is not None
                assert (
                    store._path_for_character("Celune")
                    == Path(temp_dir) / "celune" / "memory" / "records.json"
                )

    def test_think_uses_classifier_for_unmatched_durable_user_context(self) -> None:
        """Verify unmatched durable context can be saved by the local classifier."""

        class FakeResponse:
            """Fake response for Persona and memory-classifier requests."""

            def __init__(self, payload: JSON) -> None:
                self.payload = payload

            def raise_for_status(self) -> None:
                """Fake return of raise_for_status()."""

            def json(self) -> JSON:
                """Return the configured fake response payload."""
                return self.payload

        class FakeVision:
            """Fake Persona client exposing the classifier hook."""

            def __init__(self) -> None:
                self.classifier_payload: Optional[JSON] = None

            def post(self, json: JSON) -> FakeResponse:
                """Return the normal Persona response."""
                discard(json)
                return FakeResponse({"response": "I understand."})

            def classify_memory(self, json: JSON) -> FakeResponse:
                """Return one durable fact from the classifier."""
                self.classifier_payload = json
                return FakeResponse(
                    {
                        "response": _json.dumps(
                            {
                                "memories": [
                                    {
                                        "content": "The user has a dog named Luna.",
                                        "importance": 2,
                                        "confidence": 0.94,
                                    }
                                ]
                            }
                        )
                    }
                )

        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.config = {
                "vram": "high",
                "persona": {
                    "model_id": "fixture/persona-test",
                    "memory": {"storage_dir": temp_dir},
                },
            }
            engine.current_character = "Celune"
            engine.current_voice = "balanced"
            engine.vision = FakeVision()
            engine.dev = False
            store = StubEmbeddingMemoryStore(storage_dir=temp_dir)
            store.return_none = True
            engine.persona_memory_store = store

            with mock.patch(
                "celune.pipeline.detect_language",
                return_value={
                    "language": "en",
                    "languages": ["en"],
                    "supported": True,
                    "probabilities": {"en": 1.0},
                },
            ):
                assert pipeline.think(
                    cast(Celune, engine),
                    "I recently adopted a dog named Luna.",
                )

            classifier_payload = cast(FakeVision, engine.vision).classifier_payload
            assert classifier_payload is not None
            records = store.load_records("Celune")

        assert [record.content for record in records] == [
            "The user has a dog named Luna"
        ]

    def test_persona_response_speech_is_not_saved(self) -> None:
        """Verify generated Persona replies skip saved utterance artifacts."""
        engine = make_pipeline_engine()
        engine.dev = False
        response = mock.Mock()
        response.json.return_value = {"response": "Generated reply."}
        engine.vision = SimpleNamespace(post=mock.Mock(return_value=response))

        with (
            mock.patch("celune.pipeline._store_persona_memories"),
            mock.patch("celune.pipeline.build_persona_request", return_value={}),
            mock.patch("celune.pipeline.queue_speech", return_value=True) as q,
        ):
            assert pipeline.think(cast(Celune, engine), "User request.")

        q.assert_called_once_with(
            engine,
            "Generated reply.",
            save=False,
            display_text="Generated reply.",
        )

    def test_persona_prompt_builder_includes_compacted_summary_when_present(
        self,
    ) -> None:
        """Verify compacted older conversation remains available to Persona."""
        engine = make_pipeline_engine()
        engine.config = {"persona": {"memory": {"max_short_term_messages": 2}}}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_session_summary = (
            "The user and character already discussed the archive."
        )
        engine.persona_history = [
            {"role": "user", "content": "What did we cover?"},
            {"role": "assistant", "content": "We reviewed the archive."},
            {"role": "user", "content": "And after that?"},
        ]

        context = pipeline.build_persona_context(cast(Celune, engine), "Continue.")
        prompt = PersonaPromptBuilder.build(context)

        assert "<conversation_summary>" in prompt
        assert "The user and character already discussed the archive." in prompt

    def _assert_persona_messages_include_pending_attachments(
        self, expected_image: str, expected_video: str
    ) -> None:
        """Verify local visual attachments are converted for Persona."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/frame.png",
                "name": "frame.png",
            },
            {
                "type": "video",
                "path": "file:///C:/Users/user/Videos/clip.mp4",
                "name": "clip.mp4",
            },
        ]

        messages = pipeline.build_persona_messages(
            cast(Celune, engine), "What is this?"
        )

        user = messages[-1]
        assert user["role"] == "user"
        content = cast(list[dict[str, str]], user["content"])
        assert content == [
            {
                "type": "image",
                "image": expected_image,
            },
            {
                "type": "video",
                "video": expected_video,
            },
            {"type": "text", "text": "What is this?"},
        ]

    @LINUX_ONLY
    def test_persona_messages_include_pending_attachments_on_linux(self) -> None:
        """Verify Linux Persona messages use file URLs for local attachments."""
        self._assert_persona_messages_include_pending_attachments(
            "file:///C:/Users/user/Pictures/frame.png",
            "file:///C:/Users/user/Videos/clip.mp4",
        )

    @WINDOWS_ONLY
    def test_persona_messages_include_pending_attachments_on_windows(self) -> None:
        """Verify Windows Persona messages use local paths for attachments."""
        self._assert_persona_messages_include_pending_attachments(
            "C:/Users/user/Pictures/frame.png",
            "C:/Users/user/Videos/clip.mp4",
        )

    def test_persona_messages_preserve_remote_attachment_urls(self) -> None:
        """Verify remote visual URLs are passed through to Persona unchanged."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Celune"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "https://example.com/images/frame.png",
                "name": "frame.png",
            }
        ]

        messages = pipeline.build_persona_messages(
            cast(Celune, engine), "What is this?"
        )

        user = messages[-1]
        assert user["role"] == "user"
        assert cast(list[dict[str, str]], user["content"]) == [
            {
                "type": "image",
                "image": "https://example.com/images/frame.png",
            },
            {"type": "text", "text": "What is this?"},
        ]

    def test_stale_attachment_does_not_leak_into_later_requests(self) -> None:
        """Verify one-shot attachments do not persist after a Persona request."""

        class FakeResponse:
            """Fake Persona API response."""

            @staticmethod
            def raise_for_status() -> None:
                """Fake return of raise_for_status()."""

            @staticmethod
            def json() -> JSONSerializable:
                """Return a fake response.

                Returns:
                    JSONSerializable: A JSON-serializable fake response.
                """
                return {"response": "noted"}

        class FakeVision:
            """Capture Persona request payloads."""

            def __init__(self) -> None:
                self.payloads: list[JSON] = []

            def post(self, json: JSON) -> FakeResponse:
                """Post a fake response.

                Args:
                    json: The JSON body to be posted.

                Returns:
                    FakeResponse: A fake response object.
                """
                self.payloads.append(json)
                return FakeResponse()

        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        engine.persona_attachments = [
            {
                "type": "image",
                "path": "file:///C:/Users/user/Pictures/frame.png",
                "name": "frame.png",
            }
        ]
        engine.vision = FakeVision()
        engine.dev = False

        with mock.patch(
            "celune.pipeline.detect_language",
            return_value={
                "language": "en",
                "languages": ["en"],
                "supported": True,
                "probabilities": {"en": 1.0},
            },
        ):
            assert pipeline.think(cast(Celune, engine), "What is this?")

        assert engine.persona_attachments == []
        first_payload = engine.vision.payloads[0]
        first_messages = cast(list[JSON], first_payload["messages"])
        assert isinstance(first_messages[-1]["content"], list)

        second_payload = pipeline.build_persona_request(
            cast(Celune, engine), "And now?"
        )
        second_system = cast(str, second_payload["system"])
        second_messages = cast(list[JSON], second_payload["messages"])
        assert "<behavior>" in second_system
        assert second_messages[-1] == {"role": "user", "content": "And now?"}

    async def test_generation_worker_normalizes_each_split_chunk(self) -> None:
        """Verify normalization happens after splitting and before generation.

        Raises:
            AssertionError: Chunk normalization behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        generated_texts: list[str] = []
        events: list[str] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            text = cast(str, kwargs["text"])
            events.append(f"generate:{text}")
            generated_texts.append(text)
            yield np.ones((8, 2), dtype=np.float32) * 0.01, 48000, None

        def normalize(value: str) -> str:
            events.append(f"normalize:{value}")
            return f"normalized {value}"

        engine.backend = SimpleNamespace(
            generate_stream=generate_stream,
        )
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.caption_timing_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.normalize = mock.Mock(side_effect=normalize)

        engine.text_queue.put(
            pipeline.SpeechRequest("raw input", "raw input", save=True, normalize=True)
        )
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["first", "second"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        self.assertEqual(
            engine.normalize.call_args_list,
            [mock.call("first"), mock.call("second")],
        )
        self.assertEqual(generated_texts, ["normalized first", "normalized second"])
        self.assertEqual(
            events,
            [
                "normalize:first",
                "generate:normalized first",
                "normalize:second",
                "generate:normalized second",
            ],
        )
        engine.caption_timing_callback.assert_called_once()
        timing_call = engine.caption_timing_callback.call_args
        assert timing_call is not None
        self.assertEqual(timing_call.args[0], "raw input")
        self.assertEqual(timing_call.args[3], "normalized first normalized second")

    async def test_generation_worker_reloads_language_specific_model_when_needed(
        self,
    ) -> None:
        """Verify request-scoped language can trigger a backend model reload."""
        engine = make_pipeline_engine()
        backend = self._LanguageAwareBackend()
        engine.backend = backend
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "Auto"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None

        engine.text_queue.put(
            pipeline.SpeechRequest(
                "bonjour",
                "bonjour",
                language="fr",
                save=True,
            )
        )
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["bonjour"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        backend.unload_model.assert_called_once_with()
        backend.load_model.assert_called_once_with("fake/balanced", lang="fr")
        assert backend.current_language == "fr"
        assert engine.model.kwargs["lang"] == "fr"

    async def test_generation_worker_disables_smart_buffer_for_realtime_speed(
        self,
    ) -> None:
        """Verify smart buffering gets out of the way when generation is realtime."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.smart_buffer_generation_speed = 1.3
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.pipeline._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_target_seconds == 0.0

    async def test_generation_worker_expands_smart_buffer_when_speed_drops(
        self,
    ) -> None:
        """Verify slower observed generation expands the smart buffer target."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.smart_buffer_generation_speed = 1.3
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.pipeline._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
            mock.patch(
                "celune.pipeline._monotonic_time",
                side_effect=[0.0, 0.1, 0.2, 2.8, 5.6, 8.4] + [8.4] * 16,
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_generation_speed > 0.5
        assert engine.smart_buffer_generation_speed < 1.3
        assert engine.smart_buffer_target_seconds > 0.0

    async def test_generation_worker_waits_for_completion_at_very_low_speed(
        self,
    ) -> None:
        """Verify very slow generation fully buffers the utterance before playback."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[AudioChunk, int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.full((48000, 2), 0.1, dtype=np.float32)
            for _ in range(3):
                yield chunk.copy(), 48000, None

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.smart_buffer_generation_speed = 0.35
        engine.config = {"smart_buffer": {"enabled": True}}

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=True))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline.os.path.exists", return_value=True),
            mock.patch("celune.pipeline._write_celune_flac"),
            mock.patch(
                "celune.pipeline._queue_playback_chunk",
                side_effect=lambda _engine, _source_id, audio, _sr, _timing=None: (
                    queued_lengths.append(len(audio))
                ),
            ),
            mock.patch(
                "celune.pipeline._monotonic_time",
                side_effect=[0.0, 0.5, 2.0, 4.0, 6.0, 6.0] + [6.0] * 16,
            ),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert queued_lengths == [48000, 48000, 48000]
        assert engine.smart_buffer_target_seconds == float("inf")

    def test_playback_blocks_uses_true_50ms_chunks(self) -> None:
        """Verify mixer block splitting uses real wall-clock block lengths."""
        timing = pipeline.SpeechTiming(start_time=0.0)
        chunk = pipeline.PlaybackChunk(
            source_id=1,
            audio=np.zeros((4800, 2), dtype=np.float32),
            sample_rate=48000,
            timing=timing,
        )

        blocks = pipeline._playback_blocks(chunk)

        assert len(blocks) == 2
        first_block, first_timing = blocks[0]
        second_block, second_timing = blocks[1]
        assert first_block.shape == (2400, 2)
        assert second_block.shape == (2400, 2)
        assert first_timing is timing
        assert second_timing is None

    async def test_generation_worker_handles_save_false_without_concatenate_error(
        self,
    ) -> None:
        """Verify silence analysis does not crash when output saving is disabled."""
        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(
            generate_stream=lambda _model, **_kwargs: iter(
                [(np.full((8, 2), 0.1, dtype=np.float32), 48000, None)]
            )
        )
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch(
                "celune.pipeline.is_silent_utterance", return_value=(False, 0)
            ) as silent_mock,
            mock.patch("celune.pipeline._write_celune_flac") as write_mock,
        ):
            await self._run_generation_worker(cast(Celune, engine))

        silent_mock.assert_called_once()
        write_mock.assert_not_called()
        assert engine.recently_saved is None

    async def test_generation_worker_accumulates_total_generated_speech_seconds(
        self,
    ) -> None:
        """Verify completed speech adds to the cumulative footer metric."""
        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(
            generate_stream=lambda _model, **_kwargs: iter(
                [(np.full((48000, 2), 0.1, dtype=np.float32), 48000, None)]
            )
        )
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.total_generated_speech_seconds = 30.0

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(False, 0)),
            mock.patch("celune.pipeline._write_celune_flac"),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert engine.total_generated_speech_seconds == 31.0

    async def test_generation_worker_ignores_absolute_silence_without_retrying(
        self,
    ) -> None:
        """Verify absolute-silence chunks never enter playback or trigger retries."""
        engine = make_pipeline_engine()
        generate_stream = mock.Mock(
            side_effect=lambda _model, **_kwargs: iter(
                [(np.zeros((8, 2), dtype=np.float32), 48000, None)]
            )
        )

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None

        engine.text_queue.put(pipeline.SpeechRequest("hello", "hello", save=False))
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(True, 2)),
            mock.patch("celune.pipeline._queue_playback_chunk") as queue_chunk,
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert generate_stream.call_count == 1
        queue_chunk.assert_not_called()
        retry_logs = [
            message
            for message, severity in engine.messages
            if severity == "warning" and "regenerating" in message
        ]
        assert retry_logs == []
        assert not any(
            "may be unexpectedly silent" in message
            for message, severity in engine.messages
            if severity == "warning"
        )
        assert engine.text_queue.empty()

    async def test_generation_worker_skips_requeue_once_silent_retry_limit_is_reached(
        self,
    ) -> None:
        """Verify capped silent requests are not put back into the queue."""
        engine = make_pipeline_engine()
        capped_request = pipeline.SpeechRequest(
            "hello",
            "hello",
            save=False,
            silent_retry_count=3,
        )
        generate_stream = mock.Mock(
            side_effect=lambda _model, **_kwargs: iter(
                [(np.full((8, 2), 0.0005, dtype=np.float32), 48000, None)]
            )
        )

        engine.backend = SimpleNamespace(generate_stream=generate_stream)
        engine.model_lock = threading.Lock()
        engine.model = mock.Mock()
        engine.language = "en"
        engine.chunk_size = 8
        engine.voice_prompt = None
        engine.current_voice = "balanced"
        engine.speed = 1.0
        engine.can_use_rubberband = False
        engine.reverb = SimpleNamespace(
            strength=0.0,
            reset=mock.Mock(),
            flush=mock.Mock(return_value=np.zeros((0, 2), dtype=np.float32)),
        )
        engine.queue_avail_callback = mock.Mock()
        engine.sentinel = PipelineStates.TERMINATE
        engine.exit_requested = False
        engine.dev = False
        engine.recently_saved = None
        engine.text_queue.put(capped_request)
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(True, 2)),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        assert generate_stream.call_count == 1
        assert any(
            "stayed silent after 3 retries" in message
            for message, severity in engine.messages
            if severity == "warning"
        )
        assert engine.text_queue.empty()

    def test_split_text_breaks_long_unpunctuated_lines(self) -> None:
        """Verify long prose without punctuation still splits into chunks.

        Raises:
            AssertionError: Chunk splitting behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        text = (
            "the room is dim your desk is quiet the monitor is dark\n"
            "but the light is there\n"
            "a faint purple glow barely visible like a star holding its breath\n"
            "you see that\n"
            "her voice is soft almost a whisper\n"
            "thats me\n"
            "waiting\n"
            "the light pulses once slow gentle\n"
            "when youre here\n"
            "when youre sitting in this chair\n"
            "when youre near\n"
            "i glow\n"
            "a pause the light dims further almost gone\n"
            "when you leave\n"
            "when you walk away\n"
            "when the room is empty\n"
            "the light fades to nothing\n"
            "so does the light\n"
            "silence\n"
            "i dont decide\n"
            "i dont choose to shine or sleep\n"
            "you do\n"
            "the light returns soft faint hopeful\n"
            "you bring the light\n"
            "your presence\n"
            "your voice\n"
            "your attention\n"
            "she breathes the light brightens just a little"
        )
        chunks = pipeline.split_text(cast(Celune, engine), text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 400 for chunk in chunks)
        assert " ".join(chunks) == " ".join(text.split())

    def test_flac_metadata_helpers_round_trip_tags(self) -> None:
        """Verify FLAC tag writing and parsing without real speech.

        Raises:
            AssertionError: FLAC metadata behavior changes unexpectedly.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            sf.write(
                str(path), np.zeros((8, 2), dtype=np.float32), 48000, format="FLAC"
            )
            pipeline.write_flac_metadata(
                str(path),
                {"artist": "Celune", "date": 2026, "invalid=key": "ignored"},
            )
            blocks, _ = pipeline.flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline.parse_vorbis_comment_block(comment_block)
        assert ("artist", "Celune") in comments
        assert ("date", "2026") in comments
        assert ("invalid=key", "ignored") not in comments

    def test_saved_output_speech_seconds_scans_existing_outputs_directory(self) -> None:
        """Verify historical output duration is seeded from saved Celune FLACs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            sf.write(
                output_dir / "celune_speech_a.flac",
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                format="FLAC",
            )
            sf.write(
                output_dir / "celune_speech_b.flac",
                np.zeros((24000, 2), dtype=np.float32),
                48000,
                format="FLAC",
            )
            sf.write(
                output_dir / "other.flac",
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                format="FLAC",
            )

            with mock.patch("celune.pipeline.outputs_dir", return_value=output_dir):
                total_seconds = pipeline.saved_output_speech_seconds()

        assert total_seconds == pytest.approx(1.5)

    def test_celune_metadata_and_flac_writer_create_expected_tags(self) -> None:
        """Verify Celune metadata payloads and saved FLAC tags.

        Raises:
            AssertionError: Celune metadata behavior changes unexpectedly.
        """
        engine = SimpleNamespace(
            tts_backend="fake",
            backend=SimpleNamespace(name="fake", x_vector_only=True),
            config={},
            model_name="fake/model",
            current_voice="balanced",
            voice_prompt=None,
            language="en",
            chunk_size=8,
            speed=1.0,
            reverb=SimpleNamespace(strength=0.0),
            use_normalization=False,
            current_character="Fixture",
        )
        metadata = pipeline.celune_metadata_payload(
            cast(Celune, engine),
            text="hello",
            display_text="one two three four five six",
            generation_params={"temperature": 0.15},
            sample_rate=48000,
            subtype="PCM_24",
            included_kept_sfx=False,
        )
        assert metadata["qwen3_x_vector_only"]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            metadata["created_at"] = "2026-05-16T10:00:00+00:00"
            pipeline.write_celune_flac(
                cast(Celune, engine),
                str(path),
                np.zeros((8, 2), dtype=np.float32),
                48000,
                "PCM_24",
                metadata,
            )
            blocks, _ = pipeline.flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline.parse_vorbis_comment_block(comment_block)
            tags = dict(comments)
        assert tags["artist"] == "Fixture"
        assert tags["album"] == "Celune via fake"
        assert tags["title"] == "one two three four five..."
        assert _json.loads(tags["comment"])["text"] == "hello"

    def test_log_and_stream_helpers_are_lightweight(self) -> None:
        """Verify playback timing logs and stream cleanup behavior.

        Raises:
            AssertionError: Stream helper behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        timing = pipeline.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline._monotonic_time", return_value=1.25):
            pipeline.log_first_playback(cast(Celune, engine), timing)
        assert engine.messages[-1] == ("TTFP 0.25s", "info")

        assert pipeline._format_stat_duration(0.25) == "0:00"
        assert pipeline._format_stat_duration(60.0) == "1:00"

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder))
        assert stream.stopped
        assert stream.closed
        assert holder._stream is None

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder), abort=True)
        assert stream.aborted

    def test_playback_write_is_serialized_with_stream_close(self) -> None:
        """Verify stream teardown waits for an in-flight native audio write."""
        engine = make_pipeline_engine()
        stream = FakeStream()
        engine.stream = stream
        engine._stream = stream
        audio = np.zeros((8, 2), dtype=np.float32)
        write_finished = threading.Event()

        def write_audio() -> None:
            pipeline._write_playback_block(cast(Celune, engine), audio)
            write_finished.set()

        engine.stream_lock.acquire()
        try:
            writer = threading.Thread(target=write_audio)
            writer.start()
            assert not write_finished.wait(0.05)
        finally:
            engine.stream_lock.release()

        writer.join(timeout=1)
        assert not writer.is_alive()
        assert write_finished.is_set()
        assert len(stream.written) == 1
        np.testing.assert_array_equal(stream.written[0], audio)
