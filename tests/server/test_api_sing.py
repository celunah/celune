# SPDX-License-Identifier: Apache-2.0
"""Tests for the CESN singing REST endpoint."""

import json
import queue
import asyncio
from typing import cast
from unittest import mock
from types import SimpleNamespace

import numpy as np
from starlette.responses import Response

from celune import api
from celune.celune import Celune
from celune.typing.pipeline import SpeechStreamQueue

from ..support import CeluneTestCase


class TestApiSing(CeluneTestCase):
    """Verify `/v1/sing` validates notation and returns the audio stream."""

    def test_sing_requires_an_active_singing_backend(self) -> None:
        """Reject singing requests when the active backend does not support it."""
        previous_celune = api.server_session.bound_celune
        api.server_session.bound_celune = cast(
            Celune,
            SimpleNamespace(backend=SimpleNamespace(singing=False), dev=False),
        )
        try:
            response = api.sing(api.SpeakRequest(content="[c4:4][a] Ah"))
            payload = json.loads(bytes(response.body))
        finally:
            api.server_session.bound_celune = previous_celune

        assert response.status_code == 409
        assert payload["error"] == "unsupported_operation"

    def test_sing_returns_notation_errors_without_queueing(self) -> None:
        """Report malformed CESN before it enters Celune's speech queue."""
        previous_celune = api.server_session.bound_celune
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            say_stream=mock.Mock(),
            dev=False,
        )
        api.server_session.bound_celune = cast(Celune, engine)
        try:
            with mock.patch.object(
                api,
                "validate_singing_notation",
                return_value="Parse error in CESN notation (line 1, column 2):",
            ):
                response = api.sing(api.SpeakRequest(content="[bad]"))
            payload = json.loads(bytes(response.body))
        finally:
            api.server_session.bound_celune = previous_celune

        assert response.status_code == 422
        assert payload["error"] == "invalid_notation"
        engine.say_stream.assert_not_called()

    def test_sing_streams_flac_after_validating_notation(self) -> None:
        """Return normalized FLAC audio after valid CESN is accepted."""
        previous_celune = api.server_session.bound_celune
        chunks: SpeechStreamQueue = queue.Queue()
        chunks.put(np.zeros((2, 8), dtype=np.float32))
        chunks.put(None)
        engine = SimpleNamespace(
            backend=SimpleNamespace(singing=True),
            say_stream=mock.Mock(return_value=chunks),
            dev=False,
        )
        api.server_session.bound_celune = cast(Celune, engine)
        try:
            with mock.patch.object(api, "validate_singing_notation", return_value=None):
                response = api.sing(
                    api.SpeakRequest(content="[c4:4][a] Ah", save=False)
                )
            payload = asyncio.run(api_response_bytes(response))
        finally:
            api.server_session.bound_celune = previous_celune

        assert response.status_code == 200
        assert response.media_type == "audio/flac"
        assert payload[:4] == b"fLaC"
        engine.say_stream.assert_called_once_with("[c4:4][a] Ah", save=False)


async def api_response_bytes(response: Response) -> bytes:
    """Collect a direct API response for assertions."""
    if getattr(response, "body", None) is not None:
        return bytes(response.body)
    chunks: list[bytes] = []
    async for chunk in cast(api.StreamingResponse, response).body_iterator:
        chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk))
    return b"".join(chunks)
