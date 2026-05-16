# SPDX-License-Identifier: MIT
"""Tests for pipeline helpers that do not perform real synthesis."""

import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

from celune import pipeline

from tests.support import FakeStream, make_pipeline_engine


class PipelineTests(unittest.TestCase):
    def test_queue_helpers_and_force_stop_cover_busy_and_idle_paths(self) -> None:
        q: queue.Queue[int] = queue.Queue()
        q.put(1)
        q.put(2)
        pipeline.clear_queue(q)
        self.assertEqual(q.empty(), True)

        engine = make_pipeline_engine()
        self.assertEqual(pipeline.acquire_pipeline(engine, "speak"), True)
        self.assertEqual(engine.locked, True)
        self.assertEqual(pipeline.acquire_pipeline(engine, "speak"), False)
        pipeline.release_pipeline(engine)
        self.assertEqual(engine.locked, False)
        self.assertEqual(engine.cur_state, "idle")

        self.assertEqual(pipeline.force_stop_speech(engine), False)
        engine.locked = True
        engine.text_queue.put("pending")
        engine.audio_queue.put("audio")
        self.assertEqual(pipeline.force_stop_speech(engine), True)
        self.assertEqual(engine.text_queue.empty(), True)
        self.assertIs(engine.audio_queue.get_nowait(), engine.force_stop_marker)

    def test_queue_speech_handles_success_and_failure_paths(self) -> None:
        engine = make_pipeline_engine()
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
                pipeline.queue_speech(engine, "hello", display_text="shown"),
                True,
            )
        request = engine.text_queue.get_nowait()
        self.assertEqual(request.text, "hello")
        self.assertEqual(request.display_text, "shown")
        self.assertEqual(engine.statuses[-1], ("Generating", "info"))

        engine = make_pipeline_engine()
        engine.is_in_tutorial = True
        self.assertEqual(pipeline.queue_speech(engine, "hello"), False)
        self.assertEqual(engine.messages[-1][1], "warning")

        engine = make_pipeline_engine()
        engine.loaded = False
        self.assertEqual(pipeline.queue_speech(engine, "hello"), False)
        self.assertEqual(engine.errors, ["Celune is not currently ready"])

    def test_flac_metadata_helpers_round_trip_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "voice.flac"
            sf.write(str(path), np.zeros((8, 2), dtype=np.float32), 48000, format="FLAC")
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
            engine,
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
                engine,
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
        engine = make_pipeline_engine()
        timing = pipeline.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline.time.monotonic", return_value=1.25):
            pipeline.log_first_playback(engine, timing)
        self.assertEqual(engine.messages[-1], ("TTFP: 0.25 seconds", "info"))

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(holder)
        self.assertEqual(stream.stopped, True)
        self.assertEqual(stream.closed, True)
        self.assertIsNone(holder._stream)

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        pipeline.close_stream(holder, abort=True)
        self.assertEqual(stream.aborted, True)
