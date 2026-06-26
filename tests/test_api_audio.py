# SPDX-License-Identifier: MIT
"""Tests for API audio encoding helpers."""

import io
import json
import time
import queue
import asyncio
from unittest import TestCase, mock
from types import SimpleNamespace
from typing import cast, Optional

import numpy as np
import soundfile as sf
from fastapi import UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from celune import api
from celune.celune import Celune
from celune.pipeline import SpeechStreamQueue


class ApiAudioTests(TestCase):
    """Tests for API audio payload formatting."""

    @staticmethod
    def _wav_bytes(
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> bytes:
        """Encode one in-memory WAV fixture for upload tests."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    @staticmethod
    async def _response_bytes(response: Response) -> bytes:
        """Collect one response body for direct route-call tests."""
        if getattr(response, "body", None) is not None:
            return bytes(response.body)

        chunks: list[bytes] = []
        async for chunk in cast(StreamingResponse, response).body_iterator:
            if isinstance(chunk, str):
                chunks.append(chunk.encode("utf-8"))
            else:
                chunks.append(bytes(chunk))
        return b"".join(chunks)

    def test_audio_bytes_encode_flac_from_stream_chunks(self) -> None:
        """Verify queued speech audio is returned as PCM24 FLAC."""
        chunks: SpeechStreamQueue = queue.Queue()
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
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        previous_celune = api.bound_celune

        try:
            api.bound_celune = cast(
                Celune, SimpleNamespace(say_stream=lambda content, save=True: chunks)
            )
            response = api.speak_async(api.SpeakRequest(content="hello"))
            payload = json.loads(bytes(response.body))

            self.assertEqual(response.status_code, 202)
            self.assertEqual(payload["status"], "accepted")
            self.assertEqual(response.headers["location"], payload["location"])

            result: Optional[Response]
            for _ in range(20):
                result = api.speak_job(payload["job_id"])
                if result.status_code == 200:
                    break
                time.sleep(0.01)
            else:
                result = None
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

            self.assertIsNone(api.speech_job_snapshot("old"))
            self.assertNotIn("old", api.speech_jobs)
            self.assertIsNotNone(api.speech_job_snapshot("fresh"))
        finally:
            api.speech_job_ttl_seconds = previous_ttl
            api.speech_jobs.clear()

    def test_convert_route_rejects_requests_outside_voice_conversion_mode(self) -> None:
        """Verify VC conversion uploads are rejected while running in TTS mode."""
        previous_celune = api.bound_celune
        audio = np.zeros((24, 2), dtype=np.float32)

        try:
            api.bound_celune = cast(
                Celune,
                SimpleNamespace(
                    input_mode="text_to_speech",
                    dev=False,
                ),
            )
            response = asyncio.run(
                api.convert_audio(
                    UploadFile(
                        file=io.BytesIO(self._wav_bytes(audio)),
                        filename="fixture.wav",
                    )
                )
            )
        finally:
            api.bound_celune = previous_celune

        self.assertEqual(response.status_code, 409)
        payload = json.loads(bytes(cast(JSONResponse, response).body))
        self.assertIn("I am not currently able", payload["message"])

    def test_convert_route_returns_converted_audio(self) -> None:
        """Verify VC conversion uploads return FLAC audio in Celune's playback format."""
        previous_celune = api.bound_celune
        source_audio = np.zeros((24, 2), dtype=np.float32)
        converted_audio = np.ones((12, 2), dtype=np.float32) * 0.25

        try:
            api.bound_celune = cast(
                Celune,
                SimpleNamespace(
                    input_mode="voice_conversion",
                    dev=False,
                    convert_audio=mock.Mock(
                        return_value=SimpleNamespace(
                            audio=converted_audio,
                            sample_rate=24000,
                            label="fixture.wav",
                        )
                    ),
                ),
            )
            response = asyncio.run(
                api.convert_audio(
                    UploadFile(
                        file=io.BytesIO(
                            self._wav_bytes(source_audio, sample_rate=44100)
                        ),
                        filename="fixture.wav",
                    )
                )
            )
        finally:
            api.bound_celune = previous_celune

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-sample-rate"], "48000")
        payload = asyncio.run(self._response_bytes(response))
        self.assertEqual(payload[:4], b"fLaC")
        decoded_audio, sample_rate = sf.read(
            io.BytesIO(payload),
            dtype="float32",
        )
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(decoded_audio.shape, (24, 2))
