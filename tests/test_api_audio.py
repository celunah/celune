# SPDX-License-Identifier: MIT
"""Tests for API audio encoding helpers."""

import io
import queue
from unittest import TestCase

import numpy as np
import soundfile as sf

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
