# SPDX-License-Identifier: MIT
"""Tests for pipeline helpers that do not perform real synthesis."""

import json
import queue
import threading
import tempfile
from typing import cast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock, TestCase

import numpy as np
import soundfile as sf

from celune import pipeline
from celune.celune import Celune
from celune.constants import JSON

from tests.support import FakeStream, make_pipeline_engine


class PipelineTests(TestCase):
    """Tests for lightweight pipeline behavior."""

    def test_queue_helpers_and_force_stop_cover_busy_and_idle_paths(self) -> None:
        """Verify queue draining, lock handling, and force-stop behavior.

        Returns:
            None: Assertions verify pipeline state changes.

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

    def test_queue_speech_handles_success_and_failure_paths(self) -> None:
        """Verify speech queueing success and rejection paths.

        Returns:
            None: Assertions verify queueing behavior.

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
        self.assertEqual(request.normalize, True)

        engine = make_pipeline_engine()
        engine.is_in_tutorial = True
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.messages[-1][1], "warning")

        engine = make_pipeline_engine()
        engine.loaded = False
        self.assertEqual(pipeline.queue_speech(cast(Celune, engine), "hello"), False)
        self.assertEqual(engine.errors, ["Celune is not currently ready"])

    def test_generation_worker_normalizes_each_split_chunk(self) -> None:
        """Verify normalization happens after splitting and before generation.

        Returns:
            None: Assertions verify per-chunk normalization behavior.

        Raises:
            AssertionError: Chunk normalization behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        generated_texts: list[str] = []
        events: list[str] = []

        def generate_stream(model: object, **kwargs: object) -> object:
            del model
            text = cast(str, kwargs["text"])
            events.append(f"generate:{text}")
            generated_texts.append(text)
            yield np.zeros((8, 2), dtype=np.float32), 48000, None

        def normalize(value: str) -> str:
            events.append(f"normalize:{value}")
            return f"normalized {value}"

        engine.backend = SimpleNamespace(
            generate_stream=generate_stream,
            generation_progress_total=lambda text: None,
            generation_progress_steps=lambda timing: 1,
        )
        engine.model_lock = threading.Lock()
        engine.model = object()
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
        engine.sentinel = object()
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
            pipeline.generation_worker(cast(Celune, engine))

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

    def test_split_text_breaks_long_unpunctuated_lines(self) -> None:
        """Verify long prose without punctuation still splits into chunks.

        Returns:
            None: Assertions verify long newline-delimited input is chunked.

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

        Returns:
            None: Assertions verify metadata behavior.

        Raises:
            AssertionError: FLAC metadata behavior changes unexpectedly.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            sf.write(
                str(path), np.zeros((8, 2), dtype=np.float32), 48000, format="FLAC"
            )
            pipeline._write_flac_metadata(
                str(path),
                {"artist": "Celune", "date": 2026, "invalid=key": "ignored"},
            )
            blocks, _ = pipeline._flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline._parse_vorbis_comment_block(comment_block)
        self.assertIn(("artist", "Celune"), comments)
        self.assertIn(("date", "2026"), comments)
        self.assertNotIn(("invalid=key", "ignored"), comments)

    def test_celune_metadata_and_flac_writer_create_expected_tags(self) -> None:
        """Verify Celune metadata payloads and saved FLAC tags.

        Returns:
            None: Assertions verify Celune metadata behavior.

        Raises:
            AssertionError: Celune metadata behavior changes unexpectedly.
        """
        engine = SimpleNamespace(
            tts_backend="fake",
            backend=SimpleNamespace(name="fake", x_vector_only=True),
            config={"qwen3_mode": "clone"},
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
        metadata = pipeline._celune_metadata_payload(
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
            pipeline._write_celune_flac(
                cast(Celune, engine),
                str(path),
                np.zeros((8, 2), dtype=np.float32),
                48000,
                "PCM_24",
                metadata,
            )
            blocks, _ = pipeline._flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == pipeline._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = pipeline._parse_vorbis_comment_block(comment_block)
            tags = dict(comments)
        self.assertEqual(tags["artist"], "Fixture")
        self.assertEqual(tags["album"], "Celune via fake")
        self.assertEqual(tags["title"], "one two three four five...")
        self.assertEqual(json.loads(tags["comment"])["text"], "hello")

    def test_log_and_stream_helpers_are_lightweight(self) -> None:
        """Verify playback timing logs and stream cleanup behavior.

        Returns:
            None: Assertions verify helper behavior.

        Raises:
            AssertionError: Stream helper behavior changes unexpectedly.
        """
        engine = make_pipeline_engine()
        timing = pipeline.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline.time.monotonic", return_value=1.25):
            pipeline.log_first_playback(cast(Celune, engine), cast(JSON, timing))
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
