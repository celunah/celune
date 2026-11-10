# SPDX-License-Identifier: Apache-2.0
"""Tests for API audio encoding helpers."""

import io
import json
import time
import queue
import asyncio
import threading
from types import SimpleNamespace
from typing import cast
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from celune import api
from celune.celune import Celune
from celune.server import jobs
from celune.typing.pipeline import SpeechStreamQueue

from ..celtest import celtest
from ..support import CeluneTestCase


class TestApiAudio(CeluneTestCase):
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

    @pytest.mark.e2e
    @celtest("E2E: API accepts speech and serves the completed audio job")
    def test_async_speak_returns_accepted_job_and_later_audio(self) -> None:
        """Submit speech over HTTP and retrieve its completed FLAC job."""
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        previous_celune = api.server_session.bound_celune
        previous_auth_token = api.server_session.auth_token
        model_ready = threading.Event()
        model_ready.set()

        try:
            api.server_session.auth_token = None
            api.server_session.bound_celune = cast(
                Celune,
                SimpleNamespace(
                    input_mode="text_to_speech",
                    exit_requested=False,
                    model_ready=model_ready,
                    say_lock=threading.Lock(),
                    locked=False,
                    cur_state="idle",
                    say_stream=lambda content, save=True: chunks,
                ),
            )
            with TestClient(api.api) as client:
                response = client.post("/v1/speak/async", json={"content": "hello"})
                payload = response.json()

                assert response.status_code == 202
                assert payload["status"] == "accepted"
                assert response.headers["location"] == payload["location"]

                deadline = time.monotonic() + 3
                while True:
                    result = client.get(payload["location"])
                    if result.status_code == 200:
                        break
                    if time.monotonic() >= deadline:
                        pytest.fail("async speech job did not complete")
                    time.sleep(0.01)

                assert result.status_code == 200
                assert result.content[:4] == b"fLaC"
                assert result.headers["X-Audio-Format"] == "flac-pcm24"
                assert result.headers["X-Sample-Rate"] == "48000"
                assert result.headers["X-Channels"] == "2"
                assert sf.info(io.BytesIO(result.content)).subtype == "PCM_24"
                audio, sample_rate = sf.read(
                    io.BytesIO(result.content),
                    dtype="float32",
                )
                assert sample_rate == 48000
                assert audio.shape == (8, 2)
        finally:
            api.server_session.bound_celune = previous_celune
            api.server_session.auth_token = previous_auth_token
            api.server_session.speech_jobs.clear()

    def test_speech_routes_reject_active_singing_backend_without_queueing(self) -> None:
        """Require CESN requests to use the dedicated singing endpoint."""
        previous_celune = api.server_session.bound_celune
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            say_stream=mock.Mock(),
            dev=False,
        )
        api.server_session.bound_celune = cast(Celune, engine)
        try:
            with mock.patch.object(api.server_jobs, "remember_speech_job") as remember:
                responses = (
                    api.speak(api.SpeakRequest(content="hello")),
                    api.speak_async(api.SpeakRequest(content="hello")),
                )

            for response in responses:
                payload = json.loads(bytes(response.body))
                assert response.status_code == 409
                assert payload["error"] == "unsupported_operation"
                assert payload["message"] == (
                    "Use /v1/sing with CESN while a singing backend is active."
                )
            remember.assert_not_called()
        finally:
            api.server_session.bound_celune = previous_celune

        engine.say_stream.assert_not_called()

    def test_speech_job_lookup_expires_old_jobs(self) -> None:
        """Verify async speech jobs are removed after their in-memory TTL."""
        previous_ttl = api.server_session.speech_job_ttl_seconds
        api.server_session.speech_job_ttl_seconds = 10
        api.server_session.speech_jobs.clear()
        try:
            api.server_session.speech_jobs["old"] = api.SpeechJob(
                status="completed",
                created_at=time.time() - 11,
                audio=b"old",
            )
            api.server_session.speech_jobs["fresh"] = api.SpeechJob(
                status="completed",
                created_at=time.time(),
                audio=b"fresh",
            )

            assert jobs.speech_job_snapshot("old") is None
            assert "old" not in api.server_session.speech_jobs
            assert jobs.speech_job_snapshot("fresh") is not None
        finally:
            api.server_session.speech_job_ttl_seconds = previous_ttl
            api.server_session.speech_jobs.clear()

    def test_convert_route_rejects_requests_outside_voice_conversion_mode(self) -> None:
        """Verify VC conversion uploads are rejected while running in TTS mode."""
        previous_celune = api.server_session.bound_celune
        audio = np.zeros((24, 2), dtype=np.float32)

        try:
            api.server_session.bound_celune = cast(
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
            api.server_session.bound_celune = previous_celune

        assert response.status_code == 409
        payload = json.loads(bytes(cast(JSONResponse, response).body))
        assert "I am not currently able" in payload["message"]

    @pytest.mark.e2e
    @celtest("E2E: API converts an uploaded WAV into playable FLAC")
    def test_convert_route_returns_converted_audio(self) -> None:
        """Upload audio over HTTP and receive Celune's playable FLAC format."""
        previous_celune = api.server_session.bound_celune
        previous_auth_token = api.server_session.auth_token
        source_audio = np.zeros((24, 2), dtype=np.float32)
        converted_audio = np.ones((12, 2), dtype=np.float32) * 0.25

        try:
            api.server_session.auth_token = None
            api.server_session.bound_celune = cast(
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
            with TestClient(api.api) as client:
                response = client.post(
                    "/v1/convert",
                    files={
                        "file": (
                            "fixture.wav",
                            self._wav_bytes(source_audio, sample_rate=44100),
                            "audio/wav",
                        )
                    },
                )
        finally:
            api.server_session.bound_celune = previous_celune
            api.server_session.auth_token = previous_auth_token

        assert response.status_code == 200
        assert response.headers["x-sample-rate"] == "48000"
        payload = response.content
        assert payload[:4] == b"fLaC"
        decoded_audio, sample_rate = sf.read(
            io.BytesIO(payload),
            dtype="float32",
        )
        assert sample_rate == 48000
        assert decoded_audio.shape == (24, 2)
