# SPDX-License-Identifier: MIT
"""Tests for pipeline helpers that do not perform real synthesis."""

import os
import sys
import queue
import tempfile
import threading
import json as _json
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import cast, Optional
from unittest import mock, IsolatedAsyncioTestCase, TestCase
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import soundfile as sf

from celune import pipeline
from celune.celune import Celune
from celune.dataclasses.pipeline import AudioInputRequest
from celune.utils import discard
from celune.persona.prompts import PersonaPromptBuilder
from celune.constants import JSON, JSONSerializable, PipelineStates
from celune.cevoice import CEVoicePersona, PersonaIdentity, PersonaStyleValues
from .support import FakeStream, FakeVCBackend, make_pipeline_engine, make_voice_loader
from .test_persona_memory import StubEmbeddingMemoryStore


class PipelineTests(TestCase):
    """Tests for lightweight pipeline behavior."""

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
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
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
        self.assertEqual(q.empty(), True)

        engine = make_pipeline_engine()
        celune_engine = cast(Celune, engine)
        self.assertEqual(pipeline.acquire_pipeline(celune_engine, "speak"), True)
        self.assertEqual(engine.locked, True)
        self.assertEqual(pipeline.acquire_pipeline(celune_engine, "speak"), False)
        pipeline.release_pipeline(celune_engine)
        self.assertEqual(engine.locked, False)
        self.assertEqual(engine.cur_state, "idle")

        self.assertEqual(pipeline.force_stop_speech(celune_engine), False)
        engine.locked = True
        engine.text_queue.put("pending")
        engine.audio_queue.put("audio")
        self.assertEqual(pipeline.force_stop_speech(celune_engine), True)
        self.assertEqual(engine.text_queue.empty(), True)
        self.assertIs(engine.audio_queue.get_nowait(), engine.force_stop_marker)

    def test_working_signal_completion_does_not_notify_idle(self) -> None:
        """Verify the transitional working cue is not treated as a readiness idle event."""
        engine = make_pipeline_engine()
        engine.cur_state = "reloading"

        self.assertEqual(pipeline.play_signal(cast(Celune, engine), "working"), True)

        queued = list(engine.audio_queue.queue)
        done_markers = [
            item for item in queued if isinstance(item, pipeline.PlaybackSourceDone)
        ]
        self.assertEqual(len(done_markers), 1)
        self.assertEqual(done_markers[0].notify_idle, False)
        self.assertEqual(engine.cur_state, "reloading")

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
            self.assertEqual(
                pipeline.play_signal(cast(Celune, engine), "readiness"), True
            )

        self.assertEqual(queued_during_signal, [True])
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "hello")


class PipelineAsyncTests(IsolatedAsyncioTestCase):
    """Tests for async pipeline entry points."""

    _LanguageAwareBackend = PipelineTests._LanguageAwareBackend

    async def _run_generation_worker(self, engine: Celune) -> None:
        """Run the async generation worker directly inside the test loop."""
        await pipeline.generation_worker_job(engine)

    async def _run_playback_worker(self, engine: Celune) -> None:
        """Run the async playback worker directly inside the test loop."""
        await pipeline.playback_worker_job(engine)

    async def test_queue_speech_async_waits_for_model_readiness_via_to_thread(
        self,
    ) -> None:
        """Verify async speech queueing offloads model-ready waits from the event loop."""
        engine = make_pipeline_engine()
        engine.model_ready.clear()

        def mark_ready() -> bool:
            engine.model_ready.set()
            return True

        engine.model_ready.wait = mock.Mock(side_effect=mark_ready)
        to_thread = mock.AsyncMock(side_effect=lambda func, *args: func(*args))

        with mock.patch("celune.pipeline.asyncio.to_thread", to_thread):
            queued = await pipeline.queue_speech_async(
                cast(Celune, engine),
                "hello",
                display_text="shown",
            )

        self.assertEqual(queued, True)
        engine.model_ready.wait.assert_called_once_with()
        self.assertEqual(to_thread.await_count, 1)
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "hello")
        self.assertEqual(request.display_text, "shown")

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
            self.assertEqual(
                pipeline.queue_speech(celune_engine, "hello", display_text="shown"),
                True,
            )
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "hello")
        self.assertEqual(request.display_text, "shown")
        self.assertEqual(request.language, "en")
        self.assertEqual(engine.statuses[-1], ("Generating", "info"))

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
            self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "raw"), True)
        engine.normalize.assert_not_called()
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "raw")
        self.assertEqual(request.language, "en")
        self.assertEqual(request.normalize, True)

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
            self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), True)
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.language, "fr")

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
            self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), True)
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.language, "Auto")

        engine = make_pipeline_engine()
        engine.is_in_tutorial = True
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.messages[-1][1], "warning")

        engine = make_pipeline_engine()
        engine.loaded = False
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.errors, ["Celune is not currently ready"])

    def test_handle_audio_input_accepts_and_ignores_audio_by_default(self) -> None:
        """Verify engine-level audio input is a safe explicit no-op in TTS mode."""
        engine = make_pipeline_engine()
        engine.log = mock.Mock()
        engine.log_dev = mock.Mock()
        engine.loaded = True
        engine.locked = False
        engine.cur_state = "idle"
        audio = np.ones((16, 2), dtype=np.float32)
        request = AudioInputRequest(audio=audio, sample_rate=48000, label="mic test")

        result = pipeline.handle_audio_input(cast(Celune, engine), request)

        self.assertEqual(result, True)
        self.assertEqual(engine.text_queue.empty(), True)
        self.assertEqual(engine.audio_queue.empty(), True)
        self.assertEqual(engine.cur_state, "idle")
        engine.log.assert_not_called()
        engine.log_dev.assert_called_once()

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
            mock.patch("celune.pipeline.queue_sfx_audio", return_value=True) as queue,
        ):
            result = pipeline.handle_audio_input(cast(Celune, engine), request)

        self.assertEqual(result, True)
        convert_mock.assert_called_once()
        vc_request = convert_mock.call_args.args[0]
        self.assertEqual(vc_request.target_references, (Path("balanced.wav"),))
        self.assertEqual(vc_request.pitch_shift, 0)
        self.assertEqual(vc_request.f0_condition, False)
        queue.assert_called_once()
        queued_audio = queue.call_args.args[1]
        self.assertEqual(queue.call_args.args[2], 48000)
        self.assertEqual(queue.call_args.args[3], "mic test")
        self.assertEqual(
            queue.call_args.kwargs["status_label_key"],
            "pipeline.revoicing_label",
        )
        self.assertEqual(queued_audio.shape, (16, 2))
        self.assertIsNot(queued_audio, audio)
        self.assertEqual(np.array_equal(queued_audio, audio), True)
        self.assertEqual(engine.text_queue.empty(), True)

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

        self.assertEqual(result, False)
        engine.log.assert_called_once()
        self.assertEqual(
            engine.errors,
            ["Voice conversion backend is not configured."],
        )
        self.assertEqual(engine.audio_queue.empty(), True)

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

        self.assertEqual(result, True)
        self.assertEqual(convert_mock.call_args.args[0].pitch_shift, 0)
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

        self.assertEqual(result, True)
        self.assertEqual(convert_mock.call_args.args[0].f0_condition, True)

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

        self.assertEqual(result, True)
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
                mock.patch("celune.pipeline.app_data_dir", return_value=temp_root),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec", return_value=object()
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

        self.assertEqual(resolved, (expected, "Fixture Video Title"))
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:3], ["-m", "yt_dlp"])
        self.assertNotIn("--print", command)
        self.assertIn(str(temp_root / "temp" / "temporary_audio.%(ext)s"), command)

    def test_download_youtube_sfx_uses_repo_venv_python_when_compiled(self) -> None:
        """Verify compiled launches call yt-dlp through the repo venv Python."""
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
                mock.patch("celune.pipeline.app_data_dir", return_value=temp_root),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec", return_value=object()
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

        self.assertEqual(resolved, (expected, "Fixture Video Title"))
        command = run.call_args.args[0]
        expected_python = (
            r"/repo/.venv/bin/python"
            if os.name != "nt"
            else r"\repo\.venv\Scripts\python.exe"
        )
        self.assertEqual(command[0], expected_python)
        self.assertEqual(command[1:3], ["-m", "yt_dlp"])

    def test_download_youtube_sfx_logs_missing_file_state(self) -> None:
        """Verify missing yt-dlp output uses the current no-file warning messages."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with (
                mock.patch("celune.pipeline.app_data_dir", return_value=temp_root),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec", return_value=object()
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
                ),
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        self.assertIsNone(resolved)
        warnings = [msg for msg, severity in engine.messages if severity == "warning"]
        self.assertIn("Downloader returned no file.", warnings)
        self.assertIn("postprocessor said something", warnings)
        self.assertNotIn("Could not download audio.", warnings)
        self.assertNotIn("Audio downloading failed:", warnings)
        self.assertTrue(any("postprocessor said something" in msg for msg in warnings))
        self.assertEqual(engine.errors[-1], "Could not download YouTube audio")

    def test_download_youtube_sfx_logs_download_failure_state(self) -> None:
        """Verify yt-dlp failures use the current download-failed warning messages."""
        engine = make_pipeline_engine()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            with (
                mock.patch("celune.pipeline.app_data_dir", return_value=temp_root),
                mock.patch(
                    "celune.pipeline.importlib_util.find_spec", return_value=object()
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
                ),
            ):
                resolved = pipeline.download_youtube_sfx(
                    cast(Celune, engine),
                    "https://youtu.be/demo",
                )

        self.assertIsNone(resolved)
        warnings = [msg for msg, severity in engine.messages if severity == "warning"]
        self.assertIn("Could not download audio.", warnings)
        self.assertIn("yt-dlp exploded", warnings)
        self.assertNotIn("Downloader returned no file.", warnings)
        self.assertEqual(engine.errors[-1], "Could not download YouTube audio")

    def test_youtube_sfx_title_reads_oembed_title(self) -> None:
        """Verify YouTube titles can be resolved without yt-dlp title output."""

        class FakeResponse:
            """Minimal urlopen response stub."""

            def __enter__(self) -> "FakeResponse":
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

        self.assertEqual(title, "Fixture Video Title")

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

        self.assertEqual(ok, True)
        download.assert_called_once()
        read.assert_called_once_with(str(downloaded), dtype="float32")
        queued_args = queue_audio.call_args.args
        queued_kwargs = queue_audio.call_args.kwargs
        self.assertEqual(queued_args[0], cast(Celune, engine))
        np.testing.assert_allclose(queued_args[1], np.asarray(audio, dtype=np.float32))
        self.assertEqual(queued_args[2:], (48000, "Fixture Video Title", True))
        self.assertEqual(queued_kwargs, {"volume": volume * 0.5})

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

        self.assertEqual(ok, True)
        self.assertEqual(engine.playback_done.is_set(), False)
        queued = list(engine.audio_queue.queue)
        self.assertTrue(
            any(isinstance(item, pipeline.PlaybackChunk) for item in queued)
        )
        self.assertTrue(
            any(isinstance(item, pipeline.PlaybackSourceDone) for item in queued)
        )

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
        glow_calls: list[npt.NDArray[np.float32]] = []
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

        self.assertEqual(fake_stream.started, True)
        self.assertEqual(len(fake_stream.written), 1)
        mixed_audio = np.concatenate(fake_stream.written)
        self.assertEqual(mixed_audio.shape, (2400, 2))
        np.testing.assert_allclose(mixed_audio, 0.5, atol=1e-6)
        self.assertEqual(len(glow_calls), len(fake_stream.written))
        np.testing.assert_allclose(np.concatenate(glow_calls), 0.5, atol=1e-6)
        self.assertEqual(engine.playback_done.is_set(), True)

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

        self.assertEqual(mock_stream.call_args.kwargs["device"], "VB-Cable Output")

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
                "The specified output device name has multiple matches for "
                "'CABLE-B Input (VB-Audio Cable B)':\n"
                "- [22] CABLE-B Input (VB-Audio Cable B), Windows DirectSound\n"
                "- [28] CABLE-B Input (VB-Audio Cable B), Windows WASAPI\n\n"
                "Please specify one of the above devices, then restart Celune."
            ),
        ):
            await self._run_playback_worker(cast(Celune, engine))

        self.assertEqual(engine.errors[-1], "No suitable audio devices")
        warning_messages = [
            msg for msg, severity in engine.messages if severity == "warning"
        ]
        self.assertTrue(warning_messages)
        self.assertIn(
            "The specified output device name has multiple matches",
            warning_messages[-1],
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
        self.assertEqual(engine.cur_state, "reloading")
        self.assertEqual(engine.playback_done.is_set(), True)

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

        self.assertEqual(
            pipeline.queue_sfx_audio(
                cast(Celune, engine),
                np.full((2400 * 8, 2), 0.25, dtype=np.float32),
                48000,
                "progress.wav",
            ),
            True,
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
        self.assertTrue(in_flight)
        self.assertLess(len(in_flight), len(fake_stream.written))
        self.assertEqual(engine.progress[-1], (1, 1))

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

            def write(self, audio: npt.NDArray[np.float32]) -> None:
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
        self.assertGreaterEqual(len(blocks), 3)
        self.assertTrue(any(np.max(block) > 0.45 for block in blocks[1:]))
        self.assertEqual(engine.playback_done.is_set(), True)

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

        class InjectingStream(FakeStream):
            """A fake injecting stream."""

            def __init__(self) -> None:
                super().__init__()
                self.injected = False

            def write(self, audio: npt.NDArray[np.float32]) -> None:
                super().write(audio)
                if not self.injected:
                    self.injected = True
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

        fake_stream = InjectingStream()
        self.assertEqual(
            pipeline.queue_sfx_audio(
                cast(Celune, engine),
                np.full((9600, 2), 0.1, dtype=np.float32),
                48000,
                "loop.wav",
            ),
            True,
        )

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        statuses = [msg for msg, _ in engine.statuses]
        self.assertIn("Playing loop.wav", statuses)
        self.assertIn("Speaking", statuses)
        speaking_index = statuses.index("Speaking")
        self.assertIn("Playing loop.wav", statuses[speaking_index + 1 :])

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

            def write(self, audio: npt.NDArray[np.float32]) -> None:
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

        fake_stream = InjectingStream()
        self.assertEqual(
            pipeline.queue_sfx_audio(
                cast(Celune, engine),
                np.ones((2400 * 12, 2), dtype=np.float32),
                48000,
                "duck.wav",
                volume=0.8,
            ),
            True,
        )

        with mock.patch("celune.pipeline.sd.OutputStream", return_value=fake_stream):
            await self._run_playback_worker(cast(Celune, engine))

        means = [float(np.mean(block)) for block in fake_stream.written]
        self.assertGreaterEqual(len(means), 6)
        self.assertGreater(means[0], 0.79)
        self.assertLess(min(means), 0.45)
        min_index = means.index(min(means))
        self.assertGreater(min_index, 0)
        self.assertLess(means[min_index], means[0])
        self.assertGreater(means[-1], means[min_index] + 0.25)
        self.assertGreater(means[-1], 0.7)

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
        self.assertEqual(engine.playback_done.is_set(), True)
        engine.idle_callback.assert_called_once_with()

    def test_finalize_playback_idle_resets_glow_audio_reactivity(self) -> None:
        """Verify normal playback completion restores the resting glow."""
        engine = make_pipeline_engine()
        engine.locked = False
        engine.cur_state = "speaking"
        engine.dev = False

        pipeline.finalize_playback_idle(cast(Celune, engine))

        engine.glow.reset_audio_reactivity.assert_called_once_with()
        self.assertEqual(engine.playback_done.is_set(), True)
        self.assertEqual(engine.cur_state, "idle")
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

        self.assertNotIn(("Ready to speak.", "info"), engine.messages)
        self.assertEqual(getattr(engine, "_ready_announced", False), False)
        self.assertEqual(engine.cur_state, "reloading")

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
        self.assertEqual(engine.playback_done.is_set(), True)
        self.assertEqual(engine.cur_state, "reloading")

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
            self.assertEqual(pipeline.think(cast(Celune, engine), "What now?"), True)

        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "I can help with that.")

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
        self.assertIn("Name: Celune", character_card)
        self.assertIn("The active character is gentle and observant.", character_card)
        self.assertIn(
            "A quietly attentive nocturnal presence with emotional continuity.",
            character_card,
        )
        self.assertIn("Soft-spoken, intimate, and reflective", character_card)
        self.assertIn("Prompt Rules:", character_card)
        self.assertIn("Example Dialogue:", character_card)
        self.assertIn("<history>", system_prompt)
        self.assertIn("<profile>", system_prompt)
        self.assertIn("<behavior>", system_prompt)
        self.assertIn("Earlier reply.", system_prompt)
        self.assertIn("user: What now?", system_prompt)
        self.assertIn("The assistant has already acknowledged", system_prompt)
        self.assertIn("You are Celune", system_prompt)
        self.assertIn("refer to yourself as Celune", system_prompt)
        self.assertIn("Celune:", system_prompt)
        self.assertEqual(messages[0], {"role": "system", "content": system_prompt})
        self.assertEqual(messages[-1], {"role": "user", "content": "What now?"})
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            engine.persona_history[-2:],
            [
                {"role": "user", "content": "What now?"},
                {"role": "assistant", "content": "I can help with that."},
            ],
        )

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

        self.assertEqual(payload["quantization"], "8bit")

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

        self.assertNotIn("Voice prompt:", context.persona_card.voice)

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

        self.assertIn("Name: Fixture", character_card)
        self.assertIn("Gender: unknown", character_card)
        self.assertIn(
            "Stay in character using the active character metadata,", character_card
        )
        self.assertIn(
            "The active character is replying to the user through a real-time speech system.",
            character_card,
        )
        self.assertIn("- Warmth: mid", character_card)
        self.assertIn("- Directness: mid", character_card)
        self.assertIn("- Formality: mid", character_card)
        self.assertNotIn("Gender: female", character_card)
        self.assertNotIn("The speaker uses a more confident", character_card)

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

        self.assertIn("<profile>", prompt)
        self.assertIn("<memories>", prompt)
        self.assertIn("- The user prefers concise answers.", prompt)
        self.assertIn(
            "- The character once helped recover a lost journal.",
            prompt,
        )
        self.assertIn("<mood>", prompt)
        self.assertIn("Thoughtful and slightly tired.", prompt)
        self.assertIn("<history>", prompt)
        self.assertIn("assistant: Yes, we catalogued the letters.", prompt)
        self.assertIn("user: What do you notice?", prompt)
        self.assertIn("You are Fixture", prompt)
        self.assertIn(
            "Push the conversation forward instead of returning to earlier turns.",
            prompt,
        )
        self.assertIn(
            "Treat facts in <memories> as true context when they are relevant.",
            prompt,
        )
        self.assertIn(
            "Keep items from <memories> silent unless the current user message clearly asks for them",
            prompt,
        )
        self.assertIn(
            "The assistant has already acknowledged",
            prompt,
        )
        self.assertIn(
            "Do not greet the user. Do not ask what they need. Just respond.",
            prompt,
        )
        self.assertIn(
            "Do not bring up older messages, stored facts, or resolved topics on your own.",
            prompt,
        )
        self.assertIn("Fixture:", prompt)
        self.assertIn("What do you notice?", prompt)
        self.assertNotIn("<request>", prompt)

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

        self.assertEqual(context.character_profile.name, "Mirelle")
        self.assertEqual(context.character_profile.age, "27")
        self.assertEqual(context.character_profile.gender, "female")
        self.assertIn(
            "A precise investigator who notices tiny shifts in tone.",
            context.character_profile.render(),
        )
        self.assertEqual(
            context.character_profile.render_identity_summary(),
            "\n".join(
                (
                    "You are Mirelle, a precise investigator who notices tiny shifts in tone.",
                    "When asked for an introduction, refer to yourself as Mirelle.",
                )
            ),
        )
        self.assertIn("Style Notes:", card)
        self.assertIn("Elegant, steady, and mildly teasing.", card)
        self.assertIn("Boundaries:", card)
        self.assertIn("Prompt Rules:", card)
        self.assertIn("Example Dialogue:", card)
        self.assertIn("- Formality: high", card)
        self.assertIn("- Enthusiasm: low", card)
        self.assertEqual(
            context.persona_card.behavior_cues(),
            (
                "Elegant, steady, and mildly teasing.",
                "Do not use sterile assistant framing.\n- Do not sound detached.",
            ),
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

        self.assertNotEqual(first_prompt, second_prompt)
        self.assertIn("Mirelle:", first_prompt)
        self.assertIn("Rho:", second_prompt)

    def test_persona_prompt_does_not_hardcode_celune_identity(self) -> None:
        """Verify Persona prompts stay character-agnostic without pack metadata."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"

        prompt = PersonaPromptBuilder.build(
            pipeline.build_persona_context(cast(Celune, engine), "Hello.")
        )

        self.assertIn("Fixture:", prompt)
        self.assertNotIn("Name: Celune", prompt)
        self.assertIn("You are Fixture", prompt)

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

        self.assertIn("Celune:", prompt)
        self.assertIn("You are Celune", prompt)
        self.assertNotIn("Gender: female", prompt)
        self.assertNotIn("The speaker uses a more confident", prompt)

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

        self.assertIn("Celune:", prompt)
        self.assertIn("You are Celune", prompt)

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

        self.assertIn("Target emotion: gently reassuring.", context.mood_or_state)
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

        self.assertEqual(context.mood_or_state, "Thoughtful and slightly tired.")
        analyzer.assert_not_called()

    def test_persona_context_logs_emotion_fallback_reason(self) -> None:
        """Verify emotion-analysis failures are surfaced in developer logs."""
        engine = make_pipeline_engine()
        engine.config = {}
        engine.current_character = "Fixture"
        engine.current_voice = "balanced"
        captured: list[tuple[str, str]] = []
        engine.log_dev = lambda msg, severity="info": captured.append((msg, severity))

        fake_analyzer = SimpleNamespace(
            last_error="lunahr/emotispace-128 could not be loaded",
            summarize_history=mock.Mock(return_value=None),
        )

        with mock.patch(
            "celune.pipeline._persona_emotion_analyzer",
            return_value=fake_analyzer,
        ):
            context = pipeline.build_persona_context(cast(Celune, engine), "Hello.")

        self.assertEqual(context.mood_or_state, "Neutral.")
        self.assertEqual(
            captured,
            [
                (
                    "Persona emotion analysis fell back to Neutral: "
                    "lunahr/emotispace-128 could not be loaded",
                    "warning",
                )
            ],
        )

    def test_persona_prompt_builder_omits_vision_context_without_attachments(
        self,
    ) -> None:
        """Verify Persona prompts omit vision context when no media is attached."""
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

        self.assertNotIn("<vision_context>", prompt)
        self.assertIn("<history>", prompt)
        self.assertIn("assistant: hi", prompt)

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

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1], {"role": "user", "content": "current"})
        self.assertEqual(len(messages), 2)
        system_prompt = cast(str, messages[0]["content"])
        self.assertIn("<history>", system_prompt)
        self.assertIn("user: old user 6", system_prompt)
        self.assertIn("assistant: old reply 11", system_prompt)
        self.assertNotIn("old user 4", system_prompt)

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
            post=lambda json: FakeResponse(),  # noqa: ARG005
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
            self.assertEqual(pipeline.think(cast(Celune, engine), "new user"), True)

        self.assertEqual(
            engine.persona_history,
            [
                {"role": "user", "content": "old user 2"},
                {"role": "assistant", "content": "old reply 3"},
                {"role": "user", "content": "new user"},
                {"role": "assistant", "content": "new reply"},
            ],
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
                self.assertEqual(
                    pipeline.think(
                        cast(Celune, engine),
                        "remember that my test word is moonlight",
                    ),
                    True,
                )

            retrieved = store.retrieve("Celune", "what is my test word?")

        self.assertEqual(
            [record.content for record in retrieved],
            ["my test word is moonlight"],
        )

    def test_persona_prompt_builder_includes_short_term_summary_when_present(
        self,
    ) -> None:
        """Verify short-term memory can include a session summary for later use."""
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

        self.assertIn("<history>", prompt)
        self.assertIn("Summary:", prompt)
        self.assertIn(
            "The user and character already discussed the archive.",
            prompt,
        )
        self.assertIn("assistant: We reviewed the archive.", prompt)
        self.assertIn("user: And after that?", prompt)

    def test_persona_messages_include_pending_attachments(self) -> None:
        """Verify visual attachments are sent in the next persona user turn."""
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
        self.assertEqual(user["role"], "user")
        content = cast(list[dict[str, str]], user["content"])
        self.assertEqual(
            content,
            [
                {
                    "type": "image",
                    "image": (
                        "C:/Users/user/Pictures/frame.png"
                        if os.name == "nt"
                        else "file:///C:/Users/user/Pictures/frame.png"
                    ),
                },
                {
                    "type": "video",
                    "video": (
                        "C:/Users/user/Videos/clip.mp4"
                        if os.name == "nt"
                        else "file:///C:/Users/user/Videos/clip.mp4"
                    ),
                },
                {"type": "text", "text": "What is this?"},
            ],
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
        self.assertEqual(user["role"], "user")
        self.assertEqual(
            cast(list[dict[str, str]], user["content"]),
            [
                {
                    "type": "image",
                    "image": "https://example.com/images/frame.png",
                },
                {"type": "text", "text": "What is this?"},
            ],
        )

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
            self.assertEqual(
                pipeline.think(cast(Celune, engine), "What is this?"), True
            )

        self.assertEqual(engine.persona_attachments, [])
        first_payload = engine.vision.payloads[0]
        first_messages = cast(list[JSON], first_payload["messages"])
        self.assertIsInstance(first_messages[-1]["content"], list)

        second_payload = pipeline.build_persona_request(
            cast(Celune, engine), "And now?"
        )
        second_system = cast(str, second_payload["system"])
        second_messages = cast(list[JSON], second_payload["messages"])
        self.assertIn("<behavior>", second_system)
        self.assertEqual(second_messages[-1], {"role": "user", "content": "And now?"})

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
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
            discard(model)
            text = cast(str, kwargs["text"])
            events.append(f"generate:{text}")
            generated_texts.append(text)
            yield np.zeros((8, 2), dtype=np.float32), 48000, None

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
        self.assertEqual(backend.current_language, "fr")
        self.assertEqual(engine.model.kwargs["lang"], "fr")

    async def test_generation_worker_disables_smart_buffer_for_realtime_speed(
        self,
    ) -> None:
        """Verify smart buffering gets out of the way when generation is realtime."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.zeros((48000, 2), dtype=np.float32)
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

        self.assertEqual(queued_lengths, [48000, 48000, 48000])
        self.assertEqual(engine.smart_buffer_target_seconds, 0.0)

    async def test_generation_worker_expands_smart_buffer_when_speed_drops(
        self,
    ) -> None:
        """Verify slower observed generation expands the smart buffer target."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.zeros((48000, 2), dtype=np.float32)
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

        self.assertEqual(queued_lengths, [48000, 48000, 48000])
        self.assertGreater(engine.smart_buffer_generation_speed, 0.5)
        self.assertLess(engine.smart_buffer_generation_speed, 1.3)
        self.assertGreater(engine.smart_buffer_target_seconds, 0.0)

    async def test_generation_worker_waits_for_completion_at_very_low_speed(
        self,
    ) -> None:
        """Verify very slow generation fully buffers the utterance before playback."""
        engine = make_pipeline_engine()
        queued_lengths: list[int] = []

        def generate_stream(
            model: mock.Mock, **kwargs: JSONSerializable
        ) -> Iterator[tuple[npt.NDArray[np.float32], int, Optional[dict]]]:
            discard(model)
            discard(kwargs)
            chunk = np.zeros((48000, 2), dtype=np.float32)
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

        self.assertEqual(queued_lengths, [48000, 48000, 48000])
        self.assertEqual(engine.smart_buffer_target_seconds, float("inf"))

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

        self.assertEqual(len(blocks), 2)
        first_block, first_timing = blocks[0]
        second_block, second_timing = blocks[1]
        self.assertEqual(first_block.shape, (2400, 2))
        self.assertEqual(second_block.shape, (2400, 2))
        self.assertIs(first_timing, timing)
        self.assertIsNone(second_timing)

    async def test_generation_worker_handles_save_false_without_concatenate_error(
        self,
    ) -> None:
        """Verify silence analysis does not crash when output saving is disabled."""
        engine = make_pipeline_engine()
        engine.backend = SimpleNamespace(
            generate_stream=lambda _model, **_kwargs: iter(
                [(np.zeros((8, 2), dtype=np.float32), 48000, None)]
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
        self.assertIsNone(engine.recently_saved)

    async def test_generation_worker_requeues_silent_utterance_until_retry_limit(
        self,
    ) -> None:
        """Verify fully silent utterances are retried only up to the configured cap."""
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
        ):
            await self._run_generation_worker(cast(Celune, engine))

        self.assertEqual(generate_stream.call_count, 4)
        retry_logs = [
            message
            for message, severity in engine.messages
            if severity == "warning" and "regenerating" in message
        ]
        self.assertEqual(len(retry_logs), 3)
        self.assertIn("(1/3)", retry_logs[0])
        self.assertIn("(2/3)", retry_logs[1])
        self.assertIn("(3/3)", retry_logs[2])
        self.assertTrue(
            any(
                "stayed silent after 3 retries" in message
                for message, severity in engine.messages
                if severity == "warning"
            )
        )
        self.assertEqual(engine.text_queue.empty(), True)

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
        engine.text_queue.put(capped_request)
        engine.text_queue.put(engine.sentinel)

        with (
            mock.patch("celune.pipeline.split_text", return_value=["hello"]),
            mock.patch("celune.pipeline.is_silent_utterance", return_value=(True, 2)),
        ):
            await self._run_generation_worker(cast(Celune, engine))

        self.assertEqual(generate_stream.call_count, 1)
        self.assertTrue(
            any(
                "stayed silent after 3 retries" in message
                for message, severity in engine.messages
                if severity == "warning"
            )
        )
        self.assertEqual(engine.text_queue.empty(), True)

    def test_split_text_breaks_long_unpunctuated_lines(self) -> None:
        """Verify long prose without punctuation still splits into chunks.

        Raises:
            AssertionError: Chunk splitting behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        text = "\n".join(
            [
                "the room is dim your desk is quiet the monitor is dark",
                "but the light is there",
                "a faint purple glow barely visible like a star holding its breath",
                "you see that",
                "her voice is soft almost a whisper",
                "thats me",
                "waiting",
                "the light pulses once slow gentle",
                "when youre here",
                "when youre sitting in this chair",
                "when youre near",
                "i glow",
                "a pause the light dims further almost gone",
                "when you leave",
                "when you walk away",
                "when the room is empty",
                "the light fades to nothing",
                "so does the light",
                "silence",
                "i dont decide",
                "i dont choose to shine or sleep",
                "you do",
                "the light returns soft faint hopeful",
                "you bring the light",
                "your presence",
                "your voice",
                "your attention",
                "she breathes the light brightens just a little",
            ]
        )

        chunks = pipeline.split_text(cast(Celune, engine), text)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 400 for chunk in chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

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
        self.assertIn(("artist", "Celune"), comments)
        self.assertIn(("date", "2026"), comments)
        self.assertNotIn(("invalid=key", "ignored"), comments)

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
        self.assertEqual(metadata["qwen3_x_vector_only"], True)

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
        self.assertEqual(tags["artist"], "Fixture")
        self.assertEqual(tags["album"], "Celune via fake")
        self.assertEqual(tags["title"], "one two three four five...")
        self.assertEqual(_json.loads(tags["comment"])["text"], "hello")

    def test_log_and_stream_helpers_are_lightweight(self) -> None:
        """Verify playback timing logs and stream cleanup behavior.

        Raises:
            AssertionError: Stream helper behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        timing = pipeline.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline._monotonic_time", return_value=1.25):
            pipeline.log_first_playback(cast(Celune, engine), timing)
        self.assertEqual(engine.messages[-1], ("TTFP: 0.25 seconds", "info"))

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder))
        self.assertEqual(stream.stopped, True)
        self.assertEqual(stream.closed, True)
        self.assertIsNone(holder._stream)

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(cast(Celune, holder), abort=True)
        self.assertEqual(stream.aborted, True)
