# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline helpers that do not perform real synthesis."""

import tempfile
import threading
import json as _json
from typing import cast
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import pytest
import numpy as np
import soundfile as sf

from celune import cesn
from celune.celune import Celune
from celune.constants import APP_SLUG
from celune import metadata as celune_metadata
from celune.pipeline import output, playback, submission
from celune.dataclasses import pipeline as pipeline_types

from ..support import (
    FakeStream,
    CeluneAsyncTestCase,
    make_pipeline_engine,
)


@pytest.mark.anyio
class TestPipelineAsync(CeluneAsyncTestCase):
    """Exercise pipeline output and playback behavior."""

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
        chunks = submission.split_text(cast(Celune, engine), text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 400 for chunk in chunks)
        assert " ".join(chunks) == " ".join(text.split())

    def test_singing_backend_converts_complete_text_before_splitting(self) -> None:
        """Convert CESN once before buffering splits a long request."""
        engine = make_pipeline_engine()
        notation = (
            "[lang:en][bpm:120,key:c,scale:major]"
            "[c3:4d,d3:8,e3:16][tɛkst.tu.sɪŋ] Text to sing."
        )
        converted = "backend-format text"
        engine.backend.cesn_to_model = mock.Mock(return_value=converted)

        with mock.patch.object(
            engine.backend,
            "cesn_to_model",
            return_value=converted,
        ) as convert:
            assert (
                cesn._prepare_synthesis_text(cast(Celune, engine), notation) == notation
            )
            convert.assert_not_called()

            engine.backend.singing = True
            assert (
                cesn._prepare_synthesis_text(cast(Celune, engine), notation)
                == converted
            )

        convert.assert_called_once_with(notation)

    def test_singing_backend_keeps_converted_score_in_one_chunk(self) -> None:
        """Preserve the full backend score payload as one generation request."""
        engine = make_pipeline_engine()
        engine.backend.singing = True
        payload = '{"sections":[{"text":"moon light"}]}'

        with (
            mock.patch.object(
                cesn,
                "_prepare_synthesis_text",
                return_value=payload,
            ),
            mock.patch.object(submission, "split_text") as split,
        ):
            assert cesn.prepare_synthesis_chunks(
                cast(Celune, engine), "score", submission.split_text
            ) == [payload]

        split.assert_not_called()

    def test_singing_input_detection_requires_a_note_event(self) -> None:
        """Recognize CESN events while ignoring prose and comment-only input."""
        assert not cesn.contains_cesn_note_event("plain lyrics")
        assert not cesn.contains_cesn_note_event("[#only a comment]")
        assert not cesn.contains_cesn_note_event("[lang:en][bpm:120,key:c,scale:major]")
        assert cesn.contains_cesn_note_event("[a4:4] Ah")
        assert cesn.contains_cesn_note_event("[r:1]")

    def test_singing_output_filename_uses_plain_cesn_lyrics(self) -> None:
        """Use lyric words instead of CESN tags in the output filename."""
        notation = "[lang:en][slur:2][c4:4,d4:4][ˈtʃɛl.oʊ]cello!"

        assert (
            output._output_filename_fragment(notation, singing_backend=True) == "cello"
        )

    def test_output_filename_prefix_distinguishes_speech_and_singing(self) -> None:
        """Use distinct output filename markers for speech and singing."""
        assert output._output_filename_prefix(singing_backend=False) == (
            f"{APP_SLUG}_speech"
        )
        assert output._output_filename_prefix(singing_backend=True) == (
            f"{APP_SLUG}_singing"
        )

    def test_singing_backend_logs_out_of_scale_cesn_warning(self) -> None:
        """Show parser warnings through the active runtime log."""
        engine = make_pipeline_engine()
        engine.backend.singing = True
        engine.backend.cesn_to_model = mock.Mock(side_effect=lambda text: text)
        cesn._prepare_synthesis_text(
            cast(
                Celune,
                engine,
            ),
            "[lang:en][bpm:120,key:c,scale:major][c#3:4][wɜːd]Word.",
        )

        assert any("outside c major" in message for message, _ in engine.messages)

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
            celune_metadata.write_flac_metadata(
                str(path),
                {"artist": "Celune", "date": 2026, "invalid=key": "ignored"},
            )
            blocks, _ = celune_metadata.flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == celune_metadata._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = celune_metadata.parse_vorbis_comment_block(comment_block)
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
                output_dir / "celune_singing_c.flac",
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                format="FLAC",
            )
            sf.write(
                output_dir / "other.flac",
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                format="FLAC",
            )

            with mock.patch(
                "celune.pipeline.output.outputs_dir", return_value=output_dir
            ):
                total_seconds = output.saved_output_speech_seconds()

        assert total_seconds == pytest.approx(2.5)

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
        metadata = celune_metadata.celune_metadata_payload(
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
            celune_metadata.write_celune_flac(
                cast(Celune, engine),
                str(path),
                np.zeros((8, 2), dtype=np.float32),
                48000,
                "PCM_24",
                metadata,
            )
            blocks, _ = celune_metadata.flac_metadata_blocks(path.read_bytes())
            comment_block = next(
                payload
                for block_type, payload in blocks
                if block_type == celune_metadata._FLAC_VORBIS_COMMENT_BLOCK
            )
            _, comments = celune_metadata.parse_vorbis_comment_block(comment_block)
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
        timing = pipeline_types.SpeechTiming(start_time=1.0, first_playback_time=1.25)
        with mock.patch("celune.pipeline.playback.monotonic_time", return_value=1.25):
            playback.log_first_playback(cast(Celune, engine), timing)
        assert engine.messages[-1] == ("TTFP 0.25s", "info")

        assert output.format_stat_duration(0.25) == "0:00"
        assert output.format_stat_duration(60.0) == "1:00"

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        playback.close_stream(cast(Celune, holder))
        assert stream.stopped
        assert stream.closed
        assert holder._stream is None

        stream = FakeStream()
        holder = SimpleNamespace(stream=stream, _stream=stream, _current_sr=48000)
        playback.close_stream(cast(Celune, holder), abort=True)
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
            playback._write_playback_block(cast(Celune, engine), audio)
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
