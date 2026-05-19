# SPDX-License-Identifier: MIT
"""Tests for API audio encoding helpers."""

import io
import json
import queue
import time
from typing import cast
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import soundfile as sf
from starlette.responses import Response

from celune import api


class ApiAudioTests(TestCase):
    """Tests for API audio payload formatting."""

    def test_audio_bytes_encode_flac_from_stream_chunks(self) -> None:
        """Verify queued speech audio is returned as PCM24 FLAC."""
        chunks: queue.Queue[object] = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)

        payload = b"".join(api.audio_bytes(chunks))
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")

        self.assertEqual(payload[:4], b"fLaC")
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(audio.shape, (8, 2))

    def test_stream_headers_describe_flac(self) -> None:
        """Verify API metadata matches the encoded response format."""
        self.assertEqual(
            api.stream_headers(),
            {
                "X-Audio-Format": "flac-pcm24",
                "X-Sample-Rate": "48000",
                "X-Channels": "2",
            },
        )

    def test_async_speak_returns_accepted_job_and_later_audio(self) -> None:
        """Verify async speech accepts immediately and exposes completed audio."""
        chunks: queue.Queue[object] = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        previous_celune = api.bound_celune

        try:
            api.bound_celune = SimpleNamespace(
                say_stream=lambda content, save=True: chunks
            )
            response = api.speak_async(api.SpeakRequest(content="hello"))
            payload = json.loads(bytes(response.body))

            self.assertEqual(response.status_code, 202)
            self.assertEqual(payload["status"], "accepted")
            self.assertEqual(response.headers["location"], payload["location"])

            result: Response
            for _ in range(20):
                result = api.speak_job(payload["job_id"])
                if result.status_code == 200:
                    break
                time.sleep(0.01)
            else:
                self.fail("async speech job did not complete")

            result = cast(Response, result)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(bytes(result.body)[:4], b"fLaC")
        finally:
            api.bound_celune = previous_celune
            api.speech_jobs.clear()

    def test_speech_job_lookup_expires_old_jobs(self) -> None:
        """Verify async speech jobs are removed after their in-memory TTL."""
        previous_ttl = api.speech_job_ttl_seconds
        api.speech_job_ttl_seconds = 10
        api.speech_jobs.clear()
        try:
            api.speech_jobs["old"] = api.SpeechJob(
                status="completed",
                created_at=time.time() - 11,
                audio=b"old",
            )
            api.speech_jobs["fresh"] = api.SpeechJob(
                status="completed",
                created_at=time.time(),
                audio=b"fresh",
            )

            self.assertIsNone(api._speech_job_snapshot("old"))
            self.assertNotIn("old", api.speech_jobs)
            self.assertIsNotNone(api._speech_job_snapshot("fresh"))
        finally:
            api.speech_job_ttl_seconds = previous_ttl
            api.speech_jobs.clear()
